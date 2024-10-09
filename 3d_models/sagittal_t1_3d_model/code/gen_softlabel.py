import os
# import optuna
# import wandb
import mlflow
# from losses.label_smoothing import LabelSmoothingCrossEntropy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.random_erasing import RandomErasing
from data_augmentations.mixup import mixup, cutmix
from optimizer.sam import SAM
from optimizer.adan import Adan
from optimizer.ranger21.ranger21 import Ranger21
from optimizer.lion_pytorch.lion_pytorch import Lion
from timm.layers import convert_sync_batchnorm

from datetime import datetime
import schedulefree
import json
from sklearn import metrics
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"]='3'

# importing the libraries
import pandas as pd
import numpy as np
import time
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
#matplotlib inline
from torch import nn
# for creating validation set
from torch.optim import lr_scheduler


import logging 
import torchvision.transforms as transforms
from timm.utils import ModelEmaV3
from model import Model

#import cv2

from torch.optim import Adam, SGD
from tqdm import tqdm

from utils import seed_torch, count_parameters
from datasets.build_loader import build_dataloader
from datasets.dataset import ImageFolder

from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch
from torchvision.transforms import Compose 

from eval import eval
import argparse
import gc

from timm.utils import get_state_dict

default_configs = {}

from metrics import score

def load_old_weight(model, weight_path):
    if weight_path is not None:
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict)
    return model


def build_net(default_configs, col_names, device_id, fold):
    model = Model(default_configs["backbone"], len(col_names), device_id).to(device_id)
    load_old_weight(model, "weights/{}/sagittal_t1/{}/sagittal_t1_loss_{}_fold{}.pt".format(default_configs["a_name"], fold, default_configs["a_name"], fold))
    return model


def val_one_fold(fold, test_loader, col_names, axis):
    device_id = "cuda:0"
    model = build_net(default_configs, col_names, device_id, fold)

    k = 0
    model.eval()
    predictions = []; ground_truths = []
    probs = []
    for i in range(len(col_names)):
        probs.append([])
        ground_truths.append([])
    

    patient_id_list = []
    pred_rows = []
    gt_rows = []

    with torch.no_grad():
        for series_stacks_ax, volume_labels, any_labels, frame_labels, patient_ids, series_ids in tqdm(test_loader):   
            series_stacks_ax = series_stacks_ax.to(device_id).float()
            volume_labels = volume_labels.to(device_id).long()

            volume_logits, image_logits = model(series_stacks_ax)
            for i in range(volume_logits.shape[1]):
                prob = torch.nn.Softmax(dim=1)(volume_logits[:, i, :])
                study_id = patient_ids[i].item()
                series_id = series_ids[i].item()
                pred_rows.append({"study_id": study_id, "series_id": series_id})
                for j, col_name in enumerate(col_names):
                    if volume_labels[i][j].item() == 0:
                        gt = np.array([1, 0, 0])
                    elif volume_labels[i][j].item() == 1:
                        gt = np.array([0, 1, 0])
                    elif volume_labels[i][j].item() == 2:
                        gt = np.array([0, 0, 1]) 
                    if volume_labels[i][j].item() == -100: 
                        gt = np.array([-100, -100, -100]) 
                        # pred = np.array([prob[j][0].item(), prob[j][1].item(), prob[j][2].item()])
                        pred_rows[-1][col_name] = ' '.join(map(str, (gt).tolist()))
                    else:   
                        pred = np.array([prob[j][0].item(), prob[j][1].item(), prob[j][2].item()])
                        pred_rows[-1][col_name] = ' '.join(map(str, (0.7*gt+ 0.3*pred).tolist()))
                    
                pred_rows[-1]["fold"] = fold
            # print(pred_rows[-1])
            for j in range(len(patient_ids)):
                patient_id_list.append(patient_ids[j].item())
                
    return pred_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--exp", type=str, default="exp_2")
    args = parser.parse_args()
 
    # experiment_id = mlflow.create_experiment(args.exp)
 
    f = open(os.path.join('configs', "{}.json".format(args.exp)))
    default_configs = json.load(f)
    f.close()

    DATA_PATH = "train"
    
    folds = [0, 1, 2, 3, 4]
    n_fold = len(folds)

 
    train_loader_list = {}
    test_loader_list = {}
    axis_list = ["sagittal_t1"]
    for axis in axis_list:
        df = pd.read_csv("data/train_all.csv")
        col_names = df.columns.tolist()[1:11]
        oof = []
        for fold in folds:
            print("FOLD: ", fold)
            val_df = df[df["fold"] == fold]
            
            print(col_names)
            
            test_data = ImageFolder(val_df, col_names, axis, default_configs, "soft_label")
            # test_data_2 = ImageFolder(val_df, DATA_PATH, default_configs["image_size"], 11, "test", None)
            
            test_loader = DataLoader(test_data, batch_size=int(default_configs["batch_size"]), 
                    pin_memory=True, num_workers=default_configs["num_workers"], drop_last=False)
    
            pred_rows = val_one_fold(fold, test_loader, col_names, axis) 
            oof += pred_rows
            pred_df = pd.DataFrame.from_dict(oof) 
            pred_df.to_csv('data/oof_{}.csv'.format(default_configs["a_name"]), index=False)


