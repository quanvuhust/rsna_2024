import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


class Model(nn.Module):
    def __init__(self, back_bone, n_head, device_id):
        super().__init__()
        print("Number head: ", n_head)
        
        self.model = timm.create_model(back_bone, pretrained=True, num_classes=3)
        self.model.fc_norm = nn.Identity()
        self.model.head_drop = nn.Identity()
        self.model.head = nn.Identity()
        self.model.set_grad_checkpointing()

        feats = 768
        self.n_head = n_head
        self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
        
        self.heads = nn.ModuleList([nn.Sequential(
                                    nn.Linear(feats, 1),
                                ) for i in range(n_head)]).to(device_id)
        self.device_id = device_id
        self.back_bone = back_bone
        # self.heat_map = nn.Conv2d(feats, 1, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = x/255.0
        # x = x.transpose(1, 2).transpose(1, 3).contiguous()
        bs, _, _, _ = x.shape
        # x = (x - self.IMAGENET_DEFAULT_MEAN[None,:, None, None])/self.IMAGENET_DEFAULT_STD[None,:, None, None]
        
        features = self.model.forward_features(x)
        # heat_map = self.heat_map(features)
        features = self.global_pool(features)
        
        # features = self.global_pool(features)
        # print(avg_feat.shape)
        logits = torch.zeros(self.n_head, bs, 1).to(self.device_id)
        for i, l in enumerate(self.heads):
            logits[i] = self.heads[i](features)
        return logits
    
if __name__ == '__main__':
    model = Model("convnext_tiny.in12k_ft_in1k", 5, "cuda:0").cuda()
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    x = torch.rand((8, 3, 384, 384)).cuda()
    print(model(x)[1].shape)
