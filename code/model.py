import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


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
            self.d_model, dropout=dropout, vocab=n_slice*3
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
    def __init__(self, input_dim, device_id):
        super(Neck, self).__init__()
        self.lstm = nn.GRU(input_dim, input_dim, num_layers=1, dropout=0.0, bidirectional=True, batch_first=True).to(device_id)
        self.atten_pool = SelfAttentionPooling(2*input_dim)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.atten_pool(x)
        return x  


class Model(nn.Module):
    def __init__(self, back_bone, n_head, device_id):
        super().__init__()
        self.model0 = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
        self.model0.set_grad_checkpointing()
        self.model0.fc_norm = nn.Identity()
        self.model0.head_drop = nn.Identity()
        self.model0.head = nn.Identity()

        self.model1 = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
        self.model1.set_grad_checkpointing()
        self.model1.fc_norm = nn.Identity()
        self.model1.head_drop = nn.Identity()
        self.model1.head = nn.Identity()

        self.model2 = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
        self.model2.set_grad_checkpointing()
        self.model2.fc_norm = nn.Identity()
        self.model2.head_drop = nn.Identity()
        self.model2.head = nn.Identity()
        feats = 768
        drop = 0.0
        # self.global_pool0 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
        # self.global_pool1 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
        # self.global_pool2 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
        
        lstm_embed = feats * 1
        
        # self.lstm2 = TransformerBlock(feats, 24, 8, 384, 1, 0.1)
        self.lstm3 = nn.GRU(lstm_embed, lstm_embed, num_layers=1, dropout=drop, bidirectional=True, batch_first=True).to(device_id)
        # self.neck0 = Neck(lstm_embed, device_id)
        # self.neck1 = Neck(lstm_embed, device_id)
        # self.neck2 = Neck(lstm_embed, device_id)
        self.n_head = n_head
        self.image_heads = nn.ModuleList([nn.Sequential(
                                    nn.Linear(feats, 1),
                                ) for i in range(25)]).to(device_id)
        self.volume_heads_3 = nn.ModuleList([nn.Sequential(
                                    nn.Dropout(0.1),
                                    SelfAttentionPooling(2*lstm_embed),
                                    nn.Linear(2*lstm_embed, 3),
                                ) for i in range(n_head)]).to(device_id)
        # self.volume_heads_0 = nn.ModuleList([nn.Sequential(
        #                             nn.Linear(2*lstm_embed*3, 3),
        #                         ) for i in range(n_head)]).to(device_id)
        
        self.n_patch = (294//14)*(294//14)
        self.heat_map = nn.Linear(self.n_patch, self.n_patch, bias=False)
        # torch.nn.init.constant_(self.heat_map.weight, -10)
        # self.heat_map.bias.data.fill_(0.01)
        self.device_id = device_id

        
    def forward(self, x):
        x = x/255.0
        # print(x.shape)
        x = x.transpose(2, 3).transpose(2, 4).contiguous()
        # print(x.shape)
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        N_EVAL = n_slice_per_c//3
        x0 = x[:, :N_EVAL, :, :, :]
        x1 = x[:, N_EVAL:2*N_EVAL, :, :, :]
        x2 = x[:, 2*N_EVAL:, :, :, :]
        x0 = x0.reshape(bs * N_EVAL, in_chans, image_size, image_size)
        x1 = x1.reshape(bs * N_EVAL, in_chans, image_size, image_size)
        x2 = x2.reshape(bs * N_EVAL, in_chans, image_size, image_size)
        # x = (x - self.IMAGENET_DEFAULT_MEAN[None,:, None, None])/self.IMAGENET_DEFAULT_STD[None,:, None, None]
        
        features_map0 = self.model0.forward_features(x0)
        features0 = features_map0[:, 0]
        features_map1 = self.model1.forward_features(x1)
        features1 = features_map1[:, 0]
        features_map2 = self.model2.forward_features(x2)
        features2 = features_map2[:, 0]

        # features_neck0 = self.neck0(features0.view(bs, N_EVAL, -1))
        # features_neck1 = self.neck1(features1.view(bs, N_EVAL, -1))
        # features_neck2 = self.neck2(features2.view(bs, N_EVAL, -1))
        # features_neck = torch.cat([features_neck0, features_neck1, features_neck2], 1)
        # volume_logits_0 = torch.zeros(self.n_head, bs, 3).to(self.device_id)
        # for i, l in enumerate(self.volume_heads_0):
        #     volume_logits_0[i] = self.volume_heads_0[i](features_neck)

        features_map = torch.cat([features_map0, features_map1, features_map2], 0)
        features_map = torch.mean(features_map, 2)[:, 5:]
        heat_maps = self.heat_map(features_map)
        
        features0 = features0.view(bs, N_EVAL, -1)
        features1 = features1.view(bs, N_EVAL, -1)
        features2 = features2.view(bs, N_EVAL, -1)
        features = torch.cat([features0, features1, features2], 1)
        # print(features.shape)
        features = features.reshape(bs * n_slice_per_c, -1)
        # features = self.global_pool(features)
        # print(avg_feat.shape)
        image_logits = torch.zeros(self.n_head, bs * n_slice_per_c, 1).to(self.device_id)
        for i, l in enumerate(self.image_heads):
            image_logits[i] = self.image_heads[i](features)
        
        features = features.view(bs, n_slice_per_c, -1)
        features, _ = self.lstm3(features)
        volume_logits_3 = torch.zeros(self.n_head, bs, 3).to(self.device_id)
        for i, l in enumerate(self.volume_heads_3):
            volume_logits_3[i] = self.volume_heads_3[i](features)
        
        return volume_logits_3, image_logits, heat_maps
    
if __name__ == '__main__':
    model = Model("vit_base_patch14_reg4_dinov2.lvd142m", 25, "cuda:0").cuda()
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    x = torch.rand((8, 24, 294, 294, 3)).cuda()
    print(model(x)[0].shape)