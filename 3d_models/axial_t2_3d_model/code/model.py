import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from layers import Project, EmbeddingLayer, TransformerEncoder, MultiHeadAttentionPooling

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, vocab):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = nn.Parameter(torch.randn(1, vocab, d_model))
        self.register_parameter('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_slice,
        nhead=8,
        nhid=384,
        num_layers=1,
        dropout=0.1,
    ):

        super().__init__()

        self.d_model = d_model
        assert (
            self.d_model % nhead == 0
        ), "nheads must divide evenly into d_model"

        self.pos_encoder = PositionalEncoding(
            self.d_model, dropout=dropout, vocab=3*n_slice
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=nhid, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # self.src_mask)

        return x


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        att_w = nn.functional.softmax(self.W(x).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x

class ImageHead(nn.Module):
    def __init__(self, input_dim):
        super(ImageHead, self).__init__()
        self.classification = nn.Linear(input_dim, 1)
        self.objectdetection = nn.Linear(input_dim, 2)
        
    def forward(self, x):
        class_logits = self.classification(x)
        localization_logits = self.objectdetection(x)
        return class_logits, localization_logits

class Model(nn.Module):
    def __init__(self, back_bone, n_head, device_id):
        super().__init__()
        print("N head: ", n_head)
        self.back_bone = back_bone
        
        if "maxvit" in back_bone or "convnext" in back_bone:
            self.model = timm.create_model(back_bone, pretrained=True, num_classes=3)
            self.model.head = nn.Identity()
        else:
            self.model = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
            self.model.fc_norm = nn.Identity()
            self.model.head_drop = nn.Identity()
            self.model.head = nn.Identity()
        self.model.set_grad_checkpointing()
        
        feats = 512
        if "convnext" in back_bone:
            self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
        
        self.project = Project(feats)
        # self.metadata_project = nn.Linear(11, 12)
        self.encoder = nn.Sequential(
                                    EmbeddingLayer(),
                                    TransformerEncoder()
                                )
        
        self.n_head = n_head
        # self.direction_heads = nn.Linear(feats, 2).to(device_id)
        self.level_heads = nn.Linear(feats, 10).to(device_id)
        # self.coord_heads = nn.ModuleList([nn.Sequential(
        #                             nn.Linear(feats, 2),
        #                         ) for i in range(10)]).to(device_id)
        # self.severity_left_heads = nn.Linear(feats, 3).to(device_id)
        # self.severity_right_heads = nn.Linear(feats, 3).to(device_id)
        self.volume_heads = nn.ModuleList([nn.Sequential(
                                    SelfAttentionPooling(256),
                                    nn.Linear(256, 3),
                                ) for i in range(n_head)]).to(device_id)

        self.device_id = device_id

    def extract_feature(self, x0):
        x0 = x0/255.0
        x0 = x0.transpose(2, 3).transpose(2, 4).contiguous()
        
        bs, N_EVAL_0, in_chans, h0, w0 = x0.shape

        n_slice_per_c = N_EVAL_0
        x0 = x0.reshape(bs * N_EVAL_0, in_chans, h0, w0)
        # x = torch.cat([x1, x2], 0)
        
        features0 = self.model.forward_features(x0)

        return features0, bs, n_slice_per_c
    
    def get_image_logits(self, f, bs, n_slice_per_c):
        N_EVAL = n_slice_per_c
        f = self.global_pool(f)
        f = f.view(bs, N_EVAL, -1)

        features = f.reshape(bs * n_slice_per_c, -1)
        # coord_logits = torch.zeros(10, bs * n_slice_per_c, 2).to(self.device_id)
        # for i, l in enumerate(self.coord_heads):
        #     coord_logits[i] = self.coord_heads[i](features)
        
        level_logits = self.level_heads(features)
        # severity_left_logits = self.severity_left_heads(features)
        # severity_right_logits = self.severity_right_heads(features)
        # return obj_image_logits, severity_left_logits, severity_right_logits
        return level_logits, level_logits
        
    def forward(self, x0):
        features0, bs, n_slice_per_c = self.extract_feature(x0)
        coord_logits, level_logits = self.get_image_logits(features0, bs, n_slice_per_c)
        features0 = self.project(features0)
        features = self.encoder(features0)
        # print(features.shape)
        
        volume_logits = torch.zeros(self.n_head, bs, 3).to(self.device_id)
        for i, l in enumerate(self.volume_heads):
            volume_logits[i] = self.volume_heads[i](features)
        
        return volume_logits, coord_logits, level_logits
    
if __name__ == '__main__':
    model = Model("convnext_pico_ols.d1_in1k", 10, "cuda:0").cuda()
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    x0 = torch.rand((8, 8, 512, 512, 3)).cuda()
    print(model(x0)[0].shape)