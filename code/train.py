import os
# import optuna
# import wandb
import mlflow
from losses.focal_bce_loss import FocalLossBCE
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
from utils import scheduler_lr

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

def make_weight_folder(default_configs, axis, fold):
    weight_path = os.path.join("weights", default_configs["a_name"])
    os.makedirs(weight_path, exist_ok=True)
    weight_path = os.path.join(weight_path, axis)
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

class DistanceLoss(nn.Module):
    def __init__(self, device_id):
        super(DistanceLoss, self).__init__()
        self.loss_x = torch.nn.MSELoss(reduction='none')
        self.loss_y = torch.nn.MSELoss(reduction='none')
        self.device_id = device_id

    def forward(self, output, target):
        return self.loss_x(output[:, 0], target[:, 0]) + self.loss_y(output[:, 1], target[:, 1])

def build_criterion(default_configs, device_id):
    coord_criterion = DistanceLoss(device_id).to(device_id)
    obj_criterion = FocalLossBCE().to(device_id)
    volume_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 2, 4]).to(device_id), label_smoothing=default_configs["smoothing_value"]).to(device_id)
    any_criterion = nn.NLLLoss().to(device_id)
    heatmap_criterion = FocalLossBCE().to(device_id)
    return coord_criterion, obj_criterion, volume_criterion, any_criterion, heatmap_criterion

def build_net(default_configs, col_names, device_id):
    model = Model(default_configs["backbone"], len(col_names), device_id).to(device_id)
    # load_old_weight(model, weights_path[fold])
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

def train_one_fold(fold, train_loader, test_loader, rank, num_tasks, col_names, axis):
    random_erase = RandomErasing(probability=0.5)
    if rank == 0:
        print("FOLD: ", fold)
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    
    DATA_PATH = "train"
    start_epoch = 0

    scaler = torch.cuda.amp.GradScaler()
    weight_path = make_weight_folder(default_configs, axis, fold)
    coord_criterion, obj_criterion, volume_criterion, any_criterion, heatmap_criterion = build_criterion(default_configs, device_id)
    model = build_net(default_configs, col_names, device_id)
    if "mobilenetv4" in default_configs["backbone"]:
        model = convert_sync_batchnorm(model)
    
    # if rank == 0:
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
    # scheduler = lr_scheduler.OneCycleLR(optimizer_model, default_configs["lr"], steps_per_epoch=len(train_loader), epochs=default_configs["num_epoch"])
    
    best_metric_ema = {"loss": {"score": 10000, "list": []}}
    for col_name in col_names:
        best_metric_ema[col_name] = {"score": 10000, "list": []}
    best_metric_ema['subarticular_loss'] = {"score": 10000, "list": []}
    best_metric_ema['foraminal_loss'] = {"score": 10000, "list": []}
    best_metric_ema['spinal_loss'] = {"score": 10000, "list": []}
    best_metric_ema['any_severe_spinal_loss'] = {"score": 10000, "list": []}
    best_model_path = ""

    input_list, output_list = [], []
    iter_size = 1

    for epoch in range(start_epoch, default_configs["num_epoch"]):
        if rank == 0:
            print("\n-----------------Epoch: " + str(epoch) + " -----------------")
        train_loader.sampler.set_epoch(epoch)
        # scheduler_lr(optimizer_model, epoch, default_configs["lr"])
        for param_group in optimizer_model.param_groups:
            log_metric_mlflow(rank, "lr", param_group['lr'], step=epoch)
            print("LR: ", param_group['lr'])
        
            
        start = time.time()
        optimizer_model.zero_grad()
        if default_configs["optimizer"] == "schedulefree":
            optimizer_model.train()

        batch_idx = 0
        updates_per_epoch = (len(train_loader) + default_configs["accumulation_steps"] - 1) // default_configs["accumulation_steps"]
        num_updates = epoch * updates_per_epoch
        loss_volume_train = 0; loss_obj_train = 0; loss_coord_train = 0; loss_any_train = 0
        for series_stacks, volume_labels, any_labels, frame_labels, heat_map_labels, img_paths in tqdm(train_loader):
            ddp_model.train()
            series_stacks = series_stacks.to(device_id).float()
            frame_labels = frame_labels.to(device_id).float()
            heat_map_labels = heat_map_labels.to(device_id).float()
            
            any_labels = any_labels.to(device_id).long()
            volume_labels = volume_labels.to(device_id).long()
            # print("HEHE: ", series_stacks.shape, frame_labels.shape, volume_labels.shape)
            if torch.rand(1)[0] < 0.5 and default_configs["use_mixup"]:
                mix_series_stacks, volume_target_a, volume_target_b, frame_target_a, frame_target_b, heat_map_labels_a, heat_map_labels_b, lam = mixup(series_stacks, volume_labels, frame_labels, heat_map_labels, alpha=default_configs["mixup_alpha"])
                frame_target_a = frame_target_a.view(frame_target_a.shape[0] * frame_target_a.shape[1], frame_target_a.shape[2])
                frame_target_b = frame_target_b.view(frame_target_b.shape[0] * frame_target_b.shape[1], frame_target_b.shape[2])
                heat_map_labels_a = heat_map_labels_a.view(heat_map_labels_a.shape[0] * heat_map_labels_a.shape[1], heat_map_labels_a.shape[2])
                heat_map_labels_b = heat_map_labels_b.view(heat_map_labels_b.shape[0] * heat_map_labels_b.shape[1], heat_map_labels_b.shape[2])
                with torch.cuda.amp.autocast():
                    volume_logits, image_logits, heat_maps = ddp_model(mix_series_stacks)
                    loss_volume = 0
                    for i_head in range(volume_logits.shape[0]):
                        loss_volume += lam*volume_criterion(volume_logits[i_head], volume_target_a[:,i_head]) + \
                            (1-lam)*volume_criterion(volume_logits[i_head], volume_target_b[:,i_head])
                    loss_volume = loss_volume/volume_logits.shape[0]
                    loss_image = 0; loss_obj = 0; loss_coord = 0
                    
                    for i_head in range(image_logits.shape[0]):
                        loss_obj += lam*obj_criterion(image_logits[i_head][:, 0], frame_target_a[:,i_head]) + \
                            (1-lam)*obj_criterion(image_logits[i_head][:, 0], frame_target_b[:,i_head])
                        
                    loss_obj = loss_obj/image_logits.shape[0]
                    loss_image = loss_obj

                    # probs = torch.nn.Softmax(dim=2)(volume_logits[20:,:,:])[:,:,2]
                    # any_probs, _ = torch.max(probs, 0)
                    # any_probs = any_probs.unsqueeze(1)
                    # any_probs = torch.cat((1-any_probs, any_probs), dim=1)
                    # loss_any = lam*any_criterion(torch.log(any_probs), any_labels_a) + (1-lam)*any_criterion(torch.log(any_probs), any_labels_b)
                    # print("3: ", heat_map_labels_a.shape, heat_map_labels_b.shape)
                    loss_heat_map = lam*heatmap_criterion(heat_maps, heat_map_labels_a) + (1-lam)*heatmap_criterion(heat_maps, heat_map_labels_b)
                    loss = loss_volume + loss_image +  0.125*loss_heat_map
                    loss_volume_train += loss_volume.item()
                    loss_obj_train += loss_obj.item()
                    loss_coord_train += loss_heat_map.item()
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()
                
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    # scaler.unscale_(optimizer_model)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), default_configs["max_norm"])
                    scaler.step(optimizer_model)
                    scaler.update()
                    optimizer_model.zero_grad()
                    model_ema.update(ddp_model, step=num_updates)
            else:
                frame_labels = frame_labels.view(frame_labels.shape[0] * frame_labels.shape[1], frame_labels.shape[2])
                heat_map_labels = heat_map_labels.view(heat_map_labels.shape[0] * heat_map_labels.shape[1], heat_map_labels.shape[2])
                with torch.cuda.amp.autocast():
                    # bs, n_slice_per_c, in_chans, image_size, _ = series_stacks.shape
                    # images = series_stacks.view(bs * n_slice_per_c, in_chans, image_size, image_size)
                    # images = random_erase(images)
                    # series_stacks = images.view(bs, n_slice_per_c, in_chans, image_size, image_size)
                    volume_logits, image_logits, heat_maps = ddp_model(series_stacks)
                    loss_volume = 0
                    for i_head in range(volume_logits.shape[0]):
                        loss_volume += volume_criterion(volume_logits[i_head], volume_labels[:,i_head])
                    loss_volume = loss_volume/volume_logits.shape[0]
                    loss_image = 0; loss_obj = 0; loss_coord = 0
                    for i_head in range(image_logits.shape[0]):
                        loss_obj += obj_criterion(image_logits[i_head][:, 0], frame_labels[:,i_head])
                    loss_obj = loss_obj/image_logits.shape[0]
                    # loss_coord = loss_coord/image_logits.shape[0]
                    loss_image = loss_obj
                    # print(loss_volume.item(), loss_obj.item(), loss_coord.item())
                    # probs = torch.nn.Softmax(dim=2)(volume_logits[20:,:,:])[:,:,2]
                    # any_probs, _ = torch.max(probs, 0)
                    # any_probs = any_probs.unsqueeze(1)
                    # any_probs = torch.cat((1-any_probs, any_probs), dim=1)
                    # # print(any_probs.shape, any_labels.shape, volume_labels.shape)
                    loss_heat_map = heatmap_criterion(heat_maps, heat_map_labels)
                    
                    loss = loss_volume + loss_image + 0.125*loss_heat_map
                    loss_volume_train += loss_volume.item()
                    loss_obj_train += loss_obj.item()
                    # loss_any_train += loss_any.item()
                    loss_coord_train += loss_heat_map.item()
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer_model)
                    scaler.update()
                    optimizer_model.zero_grad()
                    # if rank == 0:
                    model_ema.update(ddp_model, step=num_updates)
            # scheduler.step()    
            batch_idx += 1
        print("Train volume loss: ", loss_volume_train/batch_idx)
        print("Train obj loss: ", loss_obj_train/batch_idx)
        # print("Train any loss: ", loss_any_train/batch_idx)
        print("Train coord loss: ", loss_coord_train/batch_idx)
        end = time.time()
        log_metric_mlflow(rank, "train_elapsed_time", end - start, step=epoch)
        if rank == 0:
            print("train elapsed time", end - start)
        dist.barrier()
        if rank == 0:
            start = time.time()
            if default_configs["optimizer"] == "schedulefree":
                optimizer_model.eval()
            val_metric = eval(test_loader, model_ema.module, device_id, epoch, True, default_configs, col_names)
            end = time.time()
            print("val elapsed time", end - start)
            for val_metric_type in val_metric.keys():
                print("Val ema {}: {}".format(val_metric_type, val_metric[val_metric_type]))
                mlflow.log_metric("val_{}_ema".format(val_metric_type), val_metric[val_metric_type], step=epoch)
                flag = False
                if(val_metric[val_metric_type] < best_metric_ema[val_metric_type]["score"]):
                    best_metric_ema[val_metric_type] = {"score": val_metric[val_metric_type], "list": []}
                    flag = True

                if flag == True and val_metric_type == "loss":
                    best_model_path = os.path.join(weight_path, '{}_{}_{}_fold{}.pt'.format(axis, val_metric_type, default_configs["a_name"], fold))
                    try:
                        os.remove(best_model_path)
                    except Exception as e:
                        print(e)
                    # exported_model = torch._dynamo.export(get_state_dict(model_ema), input)
                    exported_model = get_state_dict(model_ema)
                    torch.save(exported_model, best_model_path)
                    # mlflow.log_artifact(best_model_path)
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
    axis_list = ["all"]
    for axis in axis_list:
        df = pd.read_csv("data/train_{}.csv".format(axis))
        col_names = df.columns.tolist()[1:-1]
        avg_score = {"loss": 0}
        for col_name in col_names:
            avg_score[col_name] = 0
        avg_score['subarticular_loss'] = 0
        avg_score['foraminal_loss'] = 0
        avg_score['spinal_loss'] = 0
        avg_score['any_severe_spinal_loss'] = 0
        for fold in folds:
            print("FOLD: ", fold)
            train_df = df[df["fold"] != fold]
            val_df =  df[df["fold"] == fold]
            
            print(col_names)
            train_data = ImageFolder(train_df, col_names, axis, default_configs, "train")
            test_data = ImageFolder(val_df, col_names, axis, default_configs, "test")
            # test_data_2 = ImageFolder(val_df, DATA_PATH, default_configs["image_size"], 11, "test", None)
            sampler_train = torch.utils.data.DistributedSampler(
                train_data, num_replicas=num_tasks, rank=rank, shuffle=True, seed=2023,
            )
            # sampler_val = torch.utils.data.DistributedSampler(
            #     dataset_val, num_replicas=num_tasks, rank=rank, shuffle=False)
            print("Sampler_train = %s" % str(sampler_train))
            train_loader = DataLoader(train_data, sampler=sampler_train, batch_size=default_configs["batch_size"], pin_memory=False, 
                num_workers=default_configs["num_workers"], drop_last=True)
    
            test_loader = DataLoader(test_data, batch_size=int(default_configs["batch_size"]), 
                    pin_memory=True, num_workers=default_configs["num_workers"], drop_last=False)
            train_loader_list[fold] = train_loader
            test_loader_list[fold] = test_loader
    
        if rank == 0:
            with mlflow.start_run(
                experiment_id=experiment_id,
            ) as parent_run:
                mlflow.set_tag("mlflow.runName", "rsna_" + axis)
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
                        score = train_one_fold(fold, train_loader_list[fold], test_loader_list[fold], rank, num_tasks, col_names, axis) 
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
                score = train_one_fold(fold, train_loader_list[fold], test_loader_list[fold], rank, num_tasks, col_names, axis) 
