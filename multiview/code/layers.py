import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from einops import rearrange, reduce, repeat
from torch import Tensor
import einops

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        att_w = nn.functional.softmax(self.W(x).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x

class Project(nn.Module):
    def __init__(self, feats):
        super(Project, self).__init__()
        self.feats = feats
        self.conv2d_0 = nn.Conv2d(feats, 256, kernel_size=3, stride=2, padding=1)
        self.conv2d_1 = nn.Conv2d(feats, 256, kernel_size=3, stride=2, padding=1)
        self.conv2d_2 = nn.Conv2d(feats, 256, kernel_size=3, stride=2, padding=1)
        # self.proj_0 = nn.Sequential(
        #     nn.Linear(feats, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256)
        #     )
        # self.proj_1 = nn.Sequential(
        #     nn.Linear(feats, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256)
        #     )
        # self.proj_2 = nn.Sequential(
        #     nn.Linear(feats, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256)
        #     )
    def forward(self, f0, f1, f2):
        f0 = self.conv2d_0(f0)
        b = f0.shape[0]//8
        s = 8
        # f0 = f0.reshape(b, s, f0.shape[1], f0.shape[2], f0.shape[3])
        # print(f0.shape)
        f0 = einops.rearrange(f0, '(b s) c h w -> b (s h w) c', b=b, s=s, h=f0.shape[2], w=f0.shape[3])
        # f0 = f0.reshape(f0.shape[0], f0.shape[1] * f0.shape[2], f0.shape[3])
        # f0 = self.proj_0(f0)
        # f1 = f1.reshape(f1.shape[0], f1.shape[1] * f1.shape[2], f1.shape[3])
        f1 = self.conv2d_1(f1)
        # f1 = f1.reshape(b, s, f1.shape[1], f1.shape[2], f1.shape[3])
        f1 = einops.rearrange(f1, '(b s) c h w -> b (s h w) c', b=b, s=s, h=f1.shape[2], w=f1.shape[3])
        f2 = self.conv2d_2(f2)
        # f2 = f2.reshape(b, s, f2.shape[1], f2.shape[2], f2.shape[3])
        f2 = einops.rearrange(f2, '(b s) c h w -> b (s h w) c', b=b, s=s, h=f2.shape[2], w=f2.shape[3])
        # f1 = self.proj_1(f1)
        # f2 = f2.reshape(f2.shape[0], f2.shape[1] * f2.shape[2], f2.shape[3])
        # f2 = self.proj_2(f2)
        return f0, f1, f2

class EmbeddingLayer(nn.Module):
    def __init__(self, emb_size: int = 256, total_tokens: int = 24*25):
        super(EmbeddingLayer, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.sep_token = nn.Parameter(torch.randn(1,1, emb_size))

        self.plane_0 = nn.Parameter(torch.randn(1, emb_size))
        self.plane_1 = nn.Parameter(torch.randn(1, emb_size))
        self.plane_2 = nn.Parameter(torch.randn(1, emb_size))

        self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))

    def forward(self, inp):
        x0, x1, x2 = inp
        b, _, _ = x0.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        sep_token = repeat(self.sep_token, '() n e -> b n e', b=b)

        x = torch.cat((cls_tokens, x0, sep_token, x1, sep_token, x2, sep_token), dim=1)

        x[:, :2+8*25] += self.plane_0
        x[:, 2+8*25:3+16*25] += self.plane_1
        x[:, 3+16*25:] += self.plane_2

        x += self.positions
        # print(x.shape)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 256, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 2, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 256,
                 drop_p: float = 0.,
                 forward_expansion: int = 2,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 8, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 256, n_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        cls_token = x[:, 0]
        return self.linear(cls_token)