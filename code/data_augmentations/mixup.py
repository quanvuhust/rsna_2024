import torch
import numpy as np

def mixup(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x1.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x1.size()[0])
    mixed_x1 = lam * x1 + (1 - lam) * x1[rand_index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[rand_index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[rand_index, :]
    target_a_1, target_b_1 = y1, y1[rand_index]
    # print(y2.shape, rand_index)
    target_a_2, target_b_2 = y2, y2[rand_index]
    # print("HEHE: ",target_a_1.shape, target_b_1.shape, target_a_2.shape, target_b_2.shape)
    return mixed_x1, mixed_x2, mixed_x3, target_a_1, target_b_1, target_a_2, target_b_2, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(imgs: torch.Tensor, real_labels_1: torch.Tensor, real_labels_2: torch.Tensor, real_labels_3: torch.Tensor, device_id, alpha: float = 1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(imgs.size()[0]).to(device_id)
    target_a_1 = real_labels_1
    target_b_1 = real_labels_1[rand_index]
    target_a_2 = real_labels_2
    target_b_2 = real_labels_2[rand_index]
    target_a_3 = real_labels_3
    target_b_3 = real_labels_3[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
    return imgs, target_a_1, target_b_1, target_a_2, target_b_2, target_a_3, target_b_3, lam