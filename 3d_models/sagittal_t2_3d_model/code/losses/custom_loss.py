import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.nn.functional as F

class AnySevereSpinal(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = torch.argmax(y[:, :, :], dim=2)  
        w = 2 ** y

        y_spinal_prob = y_pred[:, :, :].softmax(dim=1)             # (batch_size, 3,  5)
        w_max = torch.amax(w[:, :], dim=1)                         # (batch_size, )
        y_max = torch.amax(y[:, :] == 2, dim=1).to(y_pred.dtype)   # 0 or 1
        y_pred_max = y_spinal_prob[:, 2, :].amax(dim=1)

        loss_max = -torch.log(y_pred_max)*y_max
        
        loss = (w_max * loss_max).sum() / y.size(0)
        return loss
    
if __name__ == '__main__':
    loss = AnySevereSpinal()
    y_pred = torch.rand((8, 3, 5))
    y = torch.rand((8, 5, 3))
    # y = torch.randint(0, 2, (8, 5))
    print(loss(y_pred, y))