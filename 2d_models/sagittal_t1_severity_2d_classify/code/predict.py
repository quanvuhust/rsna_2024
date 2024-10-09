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

def load_old_weight(model, weight_path):
    if weight_path is not None:
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict)
    return model


def build_net(default_configs, col_names, device_id, fold):
    model = Model(default_configs["backbone"], len(col_names), device_id).to(device_id)
    load_old_weight(model, "label_coordinates_weights/{}/{}/loss_{}_axial_fold{}.pt".format(default_configs["a_name"], fold, default_configs["a_name"], fold))
    return model


def val_one_fold(fold, test_loader, col_names):
    device_id = "cuda:0"
    model = build_net(default_configs, col_names, device_id, fold)

    k = 0
    model.eval()


    patient_id_list = []
    pred_rows = []
    gt_rows = []

    with torch.no_grad():
        k = 0
        for images, obj_labels in tqdm(test_loader):   
            images = images.to(device_id).int()
            obj_labels = obj_labels.to(device_id).float()

            logits = model(images)
            for j in range(logits.shape[1]):
                image = images[j].detach().cpu().numpy()
                image = image.transpose(1, 2, 0).astype(np.uint8).copy() 
                or_image = image.copy()
                h, w, _ = or_image.shape
                min_x = 1000
                min_y = 1000
                max_x = 0
                max_y = 0
                for i in range(logits.shape[0]):
                    # prob = torch.nn.Softmax(dim=1)(logits[i][:, 0])
                    x_batch = torch.sigmoid(logits[i][:, 1])
                    y_batch = torch.sigmoid(logits[i][:, 2])
                    x = x_batch[j]
                    y = y_batch[j]
                    
                    label = obj_labels[j][i]
                    
                    image = cv2.circle(image, (int(x*w), int(y*h)), 10, (255, 0, 0), 3)
                    if int(x*w) > max_x:
                        max_x = int(x*w)
                    if int(x*w) < min_x:
                        min_x = int(x*w)
                    if int(y*h) > max_y:
                        max_y = int(y*h)
                    if int(y*h) < min_y:
                        min_y = int(y*h)    
                    if label[0] == 1:
                        image = cv2.circle(image, (int(label[1]*w), int(label[2]*h)), 10, (0, 255, 0), 3)
                cv2.imwrite("results/{}.jpg".format(k), image)
                w_crop = max_x - min_x
                h_crop = max_y - min_y
                x_c = (min_x + max_x)//2
                y_c = (min_y + max_y)//2
                print(x_c, y_c, h_crop, w_crop)
                w_crop = max(w_crop, h_crop)*3
                h_crop = w_crop
                crop_image = or_image[y_c-h_crop//2:y_c+h_crop//2,x_c-w_crop//2:x_c+w_crop//2]
                cv2.imwrite("results/{}_crop.jpg".format(k), crop_image)
                k += 1
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--exp", type=str, default="exp_0")
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
    df = pd.read_csv("data/train_label_axial_coordinates.csv")
    avg_score = {"auc": 0}
    folds = [0]
        
    col_names = df.columns.tolist()[1:-1]
    oof = []
    for fold in folds:
        print("FOLD: ", fold)
        val_df = df[df["fold"] == fold]
        test_data = ImageFolder(val_df, default_configs, "test") 
        # test_data_2 = ImageFolder(val_df, DATA_PATH, default_configs["image_size"], 11, "test", None)
        
        test_loader = DataLoader(test_data, batch_size=int(default_configs["batch_size"])*4, 
                pin_memory=True, num_workers=default_configs["num_workers"], drop_last=False)

        val_one_fold(fold, test_loader, ["left", "right"]) 
