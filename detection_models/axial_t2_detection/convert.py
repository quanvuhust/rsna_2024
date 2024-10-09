import torch
import os

root = "./"
os.makedirs("checkpoints", exist_ok=True)
for filename in os.listdir("{}".format(root)):
    if "best_bbox_mAP_epoch_10.pth" in filename:
        filepath = os.path.join("{}".format(root), filename)
        print(filepath)
        state_dict = torch.load(filepath, map_location="cpu")
        print(state_dict.keys())
        ema_state_dict = state_dict['state_dict']
        ema_filepath = os.path.join("{}/checkpoints".format(root), filename)
        torch.save(ema_state_dict, ema_filepath)
