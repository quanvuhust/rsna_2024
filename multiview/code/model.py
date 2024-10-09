import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from layers import Project, EmbeddingLayer, TransformerEncoder, ClassificationHead, SelfAttentionPooling

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout, vocab):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         # Compute the positional encodings once in log space.
#         pe = nn.Parameter(torch.randn(1, vocab, d_model))
#         self.register_parameter('pe', pe)
        
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)


class Model(nn.Module):
    def __init__(self, back_bone, n_head, device_id):
        super().__init__()
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
        
        if "maxvit" in back_bone or "convnext" in back_bone:
            self.model1 = timm.create_model(back_bone, pretrained=True, num_classes=3)
            self.model1.head = nn.Identity()
        else:
            self.model1 = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
            self.model1.fc_norm = nn.Identity()
            self.model1.head_drop = nn.Identity()
            self.model1.head = nn.Identity()
        self.model1.set_grad_checkpointing()


        self.model2 = timm.create_model(back_bone, pretrained=True, num_classes=3)
        
        # self.model2.fc_norm = nn.Identity()
        # self.model2.head_drop = nn.Identity()
        self.model2.head = nn.Identity()
        self.model2.set_grad_checkpointing()
        feats = 768
        drop = 0.0
        if "convnext" in back_bone:
            self.global_pool0 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
            self.global_pool1 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
            self.global_pool2 = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
        
        self.project = Project(feats)
        self.encoder = nn.Sequential(
                                    EmbeddingLayer(),
                                    TransformerEncoder()
                                )
        self.n_head = n_head
        self.image_heads = nn.ModuleList([nn.Sequential(
                                    nn.Linear(feats, 1),
                                ) for i in range(25)]).to(device_id)
        self.volume_heads = nn.ModuleList([nn.Sequential(
                                    SelfAttentionPooling(256),
                                    nn.Linear(256, 3),
                                ) for i in range(n_head)]).to(device_id)
        self.device_id = device_id

    def extract_feature(self, x0, x1, x2):
        x0 = x0/255.0
        x0 = x0.transpose(2, 3).transpose(2, 4).contiguous()
        x1 = x1/255.0
        x1 = x1.transpose(2, 3).transpose(2, 4).contiguous()
        x2 = x2/255.0
        x2 = x2.transpose(2, 3).transpose(2, 4).contiguous()
        
        bs, N_EVAL_0, in_chans, h0, w0 = x0.shape
        bs, N_EVAL_1, in_chans, h1, w1 = x1.shape
        bs, N_EVAL_2, in_chans, h2, w2 = x2.shape
        n_slice_per_c = N_EVAL_0+N_EVAL_1+N_EVAL_2
        x0 = x0.reshape(bs * N_EVAL_0, in_chans, h0, w0)
        x1 = x1.reshape(bs * N_EVAL_1, in_chans, h1, w1)
        x2 = x2.reshape(bs * N_EVAL_2, in_chans, h2, w2)
        # x = torch.cat([x1, x2], 0)
        
        features0 = self.model0.forward_features(x0)
        # features0 = self.global_pool0(features0)
        features1 = self.model1.forward_features(x1)
        # features1 = self.global_pool1(features1)
        features2 = self.model2.forward_features(x2)
        # features2 = self.global_pool2(features2)
        
        # features0 = features0.view(bs, N_EVAL_0, features0.shape[1], features0.shape[2])
        # features1 = features1.view(bs, N_EVAL_1, features1.shape[1], features1.shape[2])
        # features2 = features2.view(bs, N_EVAL_2, features2.shape[1], features2.shape[2])
        return features0, features1, features2, bs, n_slice_per_c

    def get_image_logits(self, f0, f1, f2, bs, n_slice_per_c):
        N_EVAL = n_slice_per_c//3
        f0 = self.global_pool0(f0)
        f0 = f0.view(bs, N_EVAL, -1)
        f1 = self.global_pool1(f1)
        f1 = f1.view(bs, N_EVAL, -1)
        f2 = self.global_pool2(f2)
        f2 = f2.view(bs, N_EVAL, -1)
        features = torch.cat([f0, f1, f2], 1)
        features = features.reshape(bs * n_slice_per_c, -1)
        
        image_logits = torch.zeros(self.n_head, bs * n_slice_per_c, 3).to(self.device_id)
        for i, l in enumerate(self.image_heads):
            image_logits[i] = self.image_heads[i](features)
        return image_logits

        
    def forward(self, x0, x1, x2):
        features0, features1, features2, bs, n_slice_per_c = self.extract_feature(x0, x1, x2)
        image_logits = self.get_image_logits(features0, features1, features2, bs, n_slice_per_c)
        features0, features1, features2 = self.project(features0, features1, features2)
        features = self.encoder((features0, features1, features2))
        

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