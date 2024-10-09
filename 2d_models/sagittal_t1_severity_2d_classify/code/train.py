import os
# import optuna
# import wandb
import mlflow
# from losses.label_smoothing import LabelSmoothingCrossEntropy
from losses.focal_bce_loss import FocalLossBCE
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

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.distributed as dist

def remove_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def make_weight_folder(default_configs, fold):
    weight_path = os.path.join("label_coordinates_weights", default_configs["a_name"])
    os.makedirs(weight_path, exist_ok=True)
    weight_path = os.path.join(weight_path, str(fold))
    os.makedirs(weight_path, exist_ok=True)
    return weight_path

def load_old_weight(model, weight_path):
    if weight_path is not None:
        pretrained_dict = torch.load(weight_path)
        print(pretrained_dict.keys())
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model.load_state_dict(pretrained_dict)
    return model

def load_pretrain(model, weight_path):
    if weight_path is not None:
        pretrained_dict = torch.load(weight_path)
        print(pretrained_dict.keys())
        model_dict = model.model.state_dict()
        pretrained_dict0 = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model.model.load_state_dict(pretrained_dict0)
        print("Load success model 0: ")
       

    return model

class DistanceLoss(nn.Module):
    def __init__(self, device_id):
        super(DistanceLoss, self).__init__()
        self.loss_x = torch.nn.MSELoss(reduction='none')
        self.loss_y = torch.nn.MSELoss(reduction='none')
        self.device_id = device_id

    def forward(self, output, target):
        return self.loss_x(torch.sigmoid(output[:, 0]), target[:, 0]) + self.loss_y(torch.sigmoid(output[:, 1]), target[:, 1])

def build_criterion(default_configs, device_id):
    obj_criterion = FocalLossBCE().to(device_id)
    coord_criterion = DistanceLoss(device_id).to(device_id)
    severity_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 2, 4]).to(device_id)).to(device_id)
    return obj_criterion, coord_criterion, severity_criterion

def build_net(default_configs, device_id):
    model = Model(default_configs["backbone"], device_id).to(device_id)

    return model

def build_optimizer(default_configs, model_without_ddp, device_id, num_steps):
    num_tasks = dist.get_world_size()
    # lr = default_configs["lr"]*num_tasks
    lr = default_configs["lr"]
    if default_configs["optimizer"] == "schedulefree":
        optimizer_model = schedulefree.AdamWScheduleFree(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"], warmup_steps=num_steps*2)
    elif default_configs["optimizer"] == "Adan":
        optimizer_model = Adan(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"])
    
    return optimizer_model

def log_metric_mlflow(rank, metric_name, metric_value, step):
    if rank == 0:
        mlflow.log_metric(metric_name, metric_value, step=step)

def train_one_fold(fold, train_loader, test_loader, rank, num_tasks):
    random_erase = RandomErasing(probability=0.5)
    if rank == 0:
        print("FOLD: ", fold)
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    
    DATA_PATH = "train"
    start_epoch = 0

    scaler = torch.cuda.amp.GradScaler()
    weight_path = make_weight_folder(default_configs, fold)
    obj_criterion, coord_criterion, severity_criterion = build_criterion(default_configs, device_id)
    model = build_net(default_configs, device_id)
    if "mobilenetv4" in default_configs["backbone"]:
        model = convert_sync_batchnorm(model)
    
    if rank == 0:
        model_ema = ModelEmaV3(
            model,
            decay=default_configs["model_ema_decay"],
            use_warmup=True,
            device=device_id)
        # model_ema = torch.compile(model_ema)
     
    # model = torch.compile(model)
    ddp_model = NativeDDP(model, device_ids=[device_id])
    # ddp_model = torch.compile(ddp_model)
    
    optimizer_model = build_optimizer(default_configs, ddp_model, device_id, len(train_loader))
    
    best_metric_ema = {"auc": {"score": 0, "list": []}, "f1_score": {"score": 0, "list": []}, "loss": {"score": 10000, "list": []}}
    best_model_path = ""

    input_list, output_list = [], []
    iter_size = 1

    for epoch in range(start_epoch, default_configs["num_epoch"]):
        if rank == 0:
            print("\n-----------------Epoch: " + str(epoch) + " -----------------")
        train_loader.sampler.set_epoch(epoch)
        # grid.set_prob(epoch, default_configs["num_epoch"])
        for param_group in optimizer_model.param_groups:
            log_metric_mlflow(rank, "lr", param_group['lr'], step=epoch)
            
        start = time.time()
        optimizer_model.zero_grad()
        if default_configs["optimizer"] == "schedulefree":
            optimizer_model.train()

        batch_idx = 0
        updates_per_epoch = (len(train_loader) + default_configs["accumulation_steps"] - 1) // default_configs["accumulation_steps"]
        num_updates = epoch * updates_per_epoch
        loss_severity_train = 0
        for images, severity_labels in tqdm(train_loader):
            ddp_model.train()
            images = images.to(device_id).float()
            severity_labels = severity_labels.to(device_id).long()

            if torch.rand(1)[0] < 0.5 and default_configs["use_mixup"]:
                mix_images, target_a, target_b, lam = mixup(images, severity_labels, alpha=default_configs["mixup_alpha"])
                with torch.cuda.amp.autocast():
                    logits = ddp_model(mix_images)
                    # print(logits.shape, target_a.shape)
                    loss = lam*severity_criterion(logits, target_a) + \
                            (1-lam)*severity_criterion(logits, target_b)

                    loss_severity_train += loss.item()
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()
                
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer_model)
                    scaler.update()
                    optimizer_model.zero_grad()
                    if rank == 0:
                        model_ema.update(ddp_model, step=num_updates)
            else:
                with torch.cuda.amp.autocast():
                    images = random_erase(images)
                    logits = ddp_model(images)
                    loss_severity = severity_criterion(logits, severity_labels)
                    loss = loss_severity
                    loss_severity_train += loss_severity.item()
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer_model)
                    scaler.update()
                    optimizer_model.zero_grad()
                    if rank == 0:
                        model_ema.update(ddp_model, step=num_updates)
                
        
            batch_idx += 1
        
        end = time.time()
        log_metric_mlflow(rank, "train_elapsed_time", end - start, step=epoch)
        print("Train severity loss: ", loss_severity_train/batch_idx)
        if rank == 0:
            print("train elapsed time", end - start)
        dist.barrier()
        if rank == 0:
            start = time.time()
            if default_configs["optimizer"] == "schedulefree":
                optimizer_model.eval()
            val_metric = eval(test_loader, model_ema.module, device_id, epoch, True, default_configs, coord_criterion, severity_criterion)
            end = time.time()
            print("val elapsed time", end - start)
            for val_metric_type in val_metric.keys():
                print("Val ema {}: {}".format(val_metric_type, val_metric[val_metric_type]))
                mlflow.log_metric("val_{}_ema".format(val_metric_type), val_metric[val_metric_type], step=epoch)
                flag = False
                if val_metric_type == "loss":
                    if(val_metric[val_metric_type] < best_metric_ema[val_metric_type]["score"]):
                        best_metric_ema[val_metric_type] = {"score": val_metric[val_metric_type], "list": []}
                        flag = True

                    if flag == True:
                        best_model_path = os.path.join(weight_path, '{}_{}_{}_fold{}.pt'.format(val_metric_type, default_configs["a_name"], "axial", fold))
                        best_pretrain_path = os.path.join(weight_path, '{}_{}_pretrain_{}_fold{}.pt'.format(val_metric_type, default_configs["a_name"], "axial", fold))
                        try:
                            os.remove(best_model_path)
                        except Exception as e:
                            print(e)
                        # exported_model = torch._dynamo.export(get_state_dict(model_ema), input)
                        exported_model = get_state_dict(model_ema)
                        torch.save(exported_model, best_model_path)
                        torch.save(model_ema.module.model.state_dict(), best_pretrain_path)
                        best_metric_ema[val_metric_type] = {"score": val_metric[val_metric_type], "list": []}
                        # print("Save best model ema: ", best_model_path, val_metric[val_metric_type])
        dist.barrier()

    del ddp_model
    torch.cuda.empty_cache()
    gc.collect()

    return best_metric_ema


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--exp", type=str, default="exp_0")
    args = parser.parse_args()
    dist.init_process_group("nccl")
 
    # experiment_id = mlflow.create_experiment(args.exp)
 
    rank = dist.get_rank()
    num_tasks = dist.get_world_size()
    print(f"Start running basic DDP example on rank {rank}.")
 
    existing_exp = mlflow.get_experiment_by_name(args.exp)
    if rank == 0:
        if not existing_exp:
            mlflow.create_experiment(args.exp)
 
    dist.barrier()
    experiment = mlflow.set_experiment(args.exp)
    experiment_id = experiment.experiment_id
 
    f = open(os.path.join('configs', "{}.json".format(args.exp)))
    default_configs = json.load(f)
    f.close()
    if rank == 0:
        print(default_configs)
 
    seed_torch()
    DATA_PATH = "train"
    
    folds = [0, 1, 2, 3, 4]
    n_fold = len(folds)

 
    train_loader_list = {}
    test_loader_list = {}
        
    # df = pd.read_csv("data/train_label_axial_coordinates.csv")
    avg_score = {"loss": 0}
    for fold in folds:
        print("FOLD: ", fold)
        train_df = pd.read_csv("data/train_fold{}.csv".format(fold))
        val_df =  pd.read_csv("data/val_fold{}.csv".format(fold))
        
        train_data = ImageFolder(train_df, default_configs, "train")
        test_data = ImageFolder(val_df, default_configs, "test")
        # test_data_2 = ImageFolder(val_df, DATA_PATH, default_configs["image_size"], 11, "test", None)
        sampler_train = torch.utils.data.DistributedSampler(
            train_data, num_replicas=num_tasks, rank=rank, shuffle=True, seed=2023,
        )
        # sampler_val = torch.utils.data.DistributedSampler(
        #     dataset_val, num_replicas=num_tasks, rank=rank, shuffle=False)
        print("Sampler_train = %s" % str(sampler_train))
        train_loader = DataLoader(train_data, sampler=sampler_train, batch_size=default_configs["batch_size"], pin_memory=False, 
            num_workers=default_configs["num_workers"], drop_last=True)

        test_loader = DataLoader(test_data, batch_size=int(default_configs["batch_size"])*4, 
                pin_memory=True, num_workers=default_configs["num_workers"], drop_last=False)
        train_loader_list[fold] = train_loader
        test_loader_list[fold] = test_loader

    if rank == 0:
        with mlflow.start_run(
            experiment_id=experiment_id,
        ) as parent_run:
            mlflow.set_tag("mlflow.runName", "rsna_label")
            mlflow.log_params(default_configs)
            mlflow.log_artifacts("code") 
            mlflow.log_artifacts("configs")
            for fold in folds:
                with mlflow.start_run(experiment_id=experiment_id,
                    description="fold_{}".format(fold),
                    tags={
                        mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: parent_run.info.run_id
                    }, nested=True):
                    mlflow.set_tag("mlflow.runName", "fold_{}".format(fold))
                    score = train_one_fold(fold, train_loader_list[fold], test_loader_list[fold], rank, num_tasks) 
                    for k, v in avg_score.items():
                        avg_score[k] += score[k]["score"]
                        mlflow.log_metric("{}".format(k), score[k]["score"])
                        # mlflow.log_metric("{}_onnx".format(k), onnx_metric[k])
                        print("{}: ".format(k), score[k]["score"])
            for k, v in avg_score.items():
                print("CV_{}: ".format(k), avg_score[k]/n_fold)
                mlflow.log_metric("CV_{}".format(k), avg_score[k]/n_fold)    
            mlflow.end_run()
    else:
        for fold in folds:
            score = train_one_fold(fold, train_loader_list[fold], test_loader_list[fold], rank, num_tasks) 

