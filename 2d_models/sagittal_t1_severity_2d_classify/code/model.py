import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


class Model(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        if "maxvit" in back_bone or "convnext" in back_bone:
            self.model = timm.create_model(back_bone, pretrained=True, num_classes=3)
            # self.model.head = nn.Identity()
        else:
            self.model = timm.create_model(back_bone, pretrained=True, num_classes=3, dynamic_img_pad=True, dynamic_img_size=True)
            # self.model.fc_norm = nn.Identity()
            # self.model.head_drop = nn.Identity()
            # self.model.head = nn.Identity()
        self.model.set_grad_checkpointing()

        self.device_id = device_id
        self.back_bone = back_bone
        
    def forward(self, x):
        x = x/255.0
        logits = self.model(x)

        return logits
    
if __name__ == '__main__':
    model = Model("mobilenetv4_hybrid_medium.e500_r224_in1k", 5, "cuda:0").cuda()
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    x = torch.rand((8, 8, 3, 518, 518)).cuda()
    print(model(x).shape)
