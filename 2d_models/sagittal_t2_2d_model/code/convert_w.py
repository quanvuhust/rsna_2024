from model import Model
import torch

def load_old_weight(model, weight_path):
    if weight_path is not None:
        pretrained_dict = torch.load(weight_path)
        print(pretrained_dict.keys())
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model.load_state_dict(pretrained_dict)
    return model

model = Model("convnext_small.fb_in22k_ft_in1k", 25, "cuda:0").to("cuda:0")
model = load_old_weight(model, "/root/2d_model/label_coordinates_weights/exp_4/4/auc_exp_4_pretrain_fold4.pt")
torch.save(model.model.state_dict(), "/root/2d_model/label_coordinates_weights/exp_4/4/auc_exp_4_pretrain_fold4_bk.pt")
