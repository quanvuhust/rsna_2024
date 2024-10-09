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
    load_old_weight(model, "weights/{}/all/{}/all_loss_{}_fold{}.pt".format(default_configs["a_name"], fold, default_configs["a_name"], fold))
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
        for series_stack, volume_labels, _, frame_labels, __, patient_ids in tqdm(test_loader):   
            series_stack = series_stack.to(device_id).float()
            volume_labels = volume_labels.to(device_id).long()

            volume_logits, image_logits, _ = model(series_stack)
            for i in range(volume_logits.shape[1]):
                prob = torch.nn.Softmax(dim=1)(volume_logits[:, i, :])
                study_id = patient_ids[i].item()
                for j, col_name in enumerate(col_names):
                    row_id = str(study_id) + "_" + col_name
                    pred_rows.append({"row_id": row_id, "normal_mild": prob[j][0].item(), "moderate": prob[j][1].item(), "severe": prob[j][2].item()})
                    if volume_labels[i][j].item() == 0:
                        gt_rows.append({"row_id": row_id, "normal_mild": 1, "moderate": 0, "severe": 0, "sample_weight": 1})
                    elif volume_labels[i][j].item() == 1:
                        gt_rows.append({"row_id": row_id, "normal_mild": 0, "moderate": 1, "severe": 0, "sample_weight": 2})
                    elif volume_labels[i][j].item() == 2:
                        gt_rows.append({"row_id": row_id, "normal_mild": 0, "moderate": 0, "severe": 1, "sample_weight": 4})    

            for j in range(len(patient_ids)):
                patient_id_list.append(patient_ids[j].item())
                
       
    pred_df = pd.DataFrame.from_dict(pred_rows) 
    
    gt_df = pd.DataFrame.from_dict(gt_rows) 
    
    
    pred_df.to_csv('results/pred_{}_fold{}.csv'.format(default_configs["a_name"], fold), index=False)
    gt_df.to_csv('results/gt_{}_fold{}.csv'.format(default_configs["a_name"], fold), index=False)
    lb_loss = score(gt_df, pred_df, "row_id", 1.0)
    print("lb loss: ", lb_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--exp", type=str, default="exp_0")
    args = parser.parse_args()
 
    # experiment_id = mlflow.create_experiment(args.exp)
 
    f = open(os.path.join('configs', "{}.json".format(args.exp)))
    default_configs = json.load(f)
    f.close()

    DATA_PATH = "train"
    
    folds = [0, 1, 2]
    n_fold = len(folds)

 
    train_loader_list = {}
    test_loader_list = {}
    axis_list = ["all"]
    for axis in axis_list:
        df = pd.read_csv("data/train_{}.csv".format(axis))
        col_names = df.columns.tolist()[1:-1]
        for fold in folds:
            print("FOLD: ", fold)
            train_df = df[df["fold"] != fold]
            val_df =  df[df["fold"] == fold]
            
            print(col_names)
            train_data = ImageFolder(train_df, col_names, axis, default_configs, "train")
            
            test_data = ImageFolder(val_df, col_names, axis, default_configs, "test")
            # test_data_2 = ImageFolder(val_df, DATA_PATH, default_configs["image_size"], 11, "test", None)
            
            test_loader = DataLoader(test_data, batch_size=int(default_configs["batch_size"]//4), 
                    pin_memory=True, num_workers=default_configs["num_workers"], drop_last=False)
    
            val_one_fold(fold, test_loader, col_names, axis) 


