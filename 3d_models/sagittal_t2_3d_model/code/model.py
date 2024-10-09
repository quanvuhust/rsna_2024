import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from layers import Project, EmbeddingLayer, TransformerEncoder, ClassificationHead, SelfAttentionPooling

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

class Neck(nn.Module):
    def __init__(self, input_dim):
        super(Neck, self).__init__()
        self.head = nn.Linear(input_dim, input_dim)
        
        
    def forward(self, x):
        x = self.head(x)
        # x = torch.mean(x, 1)
        return x  


class Model(nn.Module):
    def __init__(self, back_bone, n_head, device_id):
        super().__init__()
        print("N head: ", n_head)
        self.back_bone = back_bone
        
        if "maxvit" in back_bone or "convnext" in back_bone:
            self.model0 = timm.create_model(back_bone, pretrained=True, num_classes=3)
            self.model0.head = nn.Identity()
        else:
            self.model0 = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
            self.model0.fc_norm = nn.Identity()
            self.model0.head_drop = nn.Identity()
            self.model0.head = nn.Identity()
        self.model0.set_grad_checkpointing()
        
        # if "maxvit" in back_bone or "convnext" in back_bone:
        #     self.model1 = timm.create_model(back_bone, pretrained=True, num_classes=3)
        #     self.model1.head = nn.Identity()
        # else:
        #     self.model1 = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
        #     self.model1.fc_norm = nn.Identity()
        #     self.model1.head_drop = nn.Identity()
        #     self.model1.head = nn.Identity()
        # self.model1.set_grad_checkpointing()


        # self.model2 = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
        # self.model2.set_grad_checkpointing()
        # self.model2.fc_norm = nn.Identity()
        # self.model2.head_drop = nn.Identity()
        # self.model2.head = nn.Identity()
        feats = 768
        drop = 0.0
        if "convnext" in back_bone:
            self.global_pool0 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
            self.global_pool1 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
            self.global_pool2 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
        
        lstm_embed = feats * 1
        self.project = Project(feats)
        self.encoder = nn.Sequential(
                                    EmbeddingLayer(),
                                    TransformerEncoder()
                                )
        
        # self.lstm3 = TransformerBlock(feats, 8, 8, 512, 1, 0.1)
        # self.lstm3 = nn.GRU(lstm_embed, lstm_embed, num_layers=1, dropout=drop, bidirectional=True, batch_first=True).to(device_id)
        # self.lstm0 = nn.GRU(lstm_embed, lstm_embed, num_layers=1, dropout=drop, bidirectional=True, batch_first=True).to(device_id)
        # self.lstm1 = nn.GRU(lstm_embed, lstm_embed, num_layers=1, dropout=drop, bidirectional=True, batch_first=True).to(device_id)
        # self.neck2 = Neck(lstm_embed, device_id)
        self.n_head = n_head
        self.image_heads = nn.ModuleList([nn.Sequential(
                                    nn.Linear(feats, 1),
                                ) for i in range(5)]).to(device_id)
        self.volume_heads = nn.ModuleList([nn.Sequential(
                                    SelfAttentionPooling(256),
                                    nn.Linear(256, 3),
                                ) for i in range(n_head)]).to(device_id)
        # self.volume_heads_0 = nn.ModuleList([nn.Sequential(
        #                             nn.Dropout(0.1),
        #                             SelfAttentionPooling(2*lstm_embed),
        #                             nn.Linear(2*lstm_embed, 3),
        #                         ) for i in range(10)]).to(device_id)
        # self.volume_heads_1 = nn.ModuleList([nn.Sequential(
        #                             nn.Dropout(0.1),
        #                             SelfAttentionPooling(2*lstm_embed),
        #                             nn.Linear(2*lstm_embed, 3),
        #                         ) for i in range(15)]).to(device_id)
        
        # self.n_patch = (294//14)*(294//14)
        # self.heat_map = nn.Linear(self.n_patch, self.n_patch, bias=False)
        # torch.nn.init.constant_(self.heat_map.weight, -10)
        # self.heat_map.bias.data.fill_(0.01)
        self.device_id = device_id

    def extract_feature(self, x0):
        x0 = x0/255.0
        x0 = x0.transpose(2, 3).transpose(2, 4).contiguous()
        
        bs, N_EVAL_0, in_chans, h0, w0 = x0.shape

        n_slice_per_c = N_EVAL_0
        x0 = x0.reshape(bs * N_EVAL_0, in_chans, h0, w0)
        # x = torch.cat([x1, x2], 0)
        
        features0 = self.model0.forward_features(x0)

        return features0, bs, n_slice_per_c
    
    def get_image_logits(self, f0, bs, n_slice_per_c):
        N_EVAL = n_slice_per_c
        f0 = self.global_pool0(f0)
        f0 = f0.view(bs, N_EVAL, -1)

        features = f0.reshape(bs * n_slice_per_c, -1)
        
        image_logits = torch.zeros(self.n_head, bs * n_slice_per_c, 3).to(self.device_id)
        for i, l in enumerate(self.image_heads):
            image_logits[i] = self.image_heads[i](features)
        return image_logits
        
    def forward(self, x0):
        features0, bs, n_slice_per_c = self.extract_feature(x0)
        image_logits = self.get_image_logits(features0, bs, n_slice_per_c)
        features0 = self.project(features0)
        features = self.encoder(features0)
        
        volume_logits = torch.zeros(self.n_head, bs, 3).to(self.device_id)
        for i, l in enumerate(self.volume_heads):
            volume_logits[i] = self.volume_heads[i](features)
        
        return volume_logits, image_logits
    
if __name__ == '__main__':
    model = Model("convnext_small.fb_in22k_ft_in1k", 25, "cuda:0").cuda()
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    x0 = torch.rand((8, 8, 288, 288, 3)).cuda()
    x1 = torch.rand((8, 8, 288, 288, 3)).cuda()
    x2 = torch.rand((8, 8, 288, 288, 3)).cuda()
    print(model(x0, x1, x2)[0].shape)