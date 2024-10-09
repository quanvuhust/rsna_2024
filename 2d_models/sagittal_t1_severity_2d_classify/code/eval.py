from torch.autograd import Variable
import torch
import numpy as np
import time
import torch.nn.functional as F
import sklearn
from sklearn import metrics
import os
import shutil
# from plot_confusion_matrix import cm_analysis
import mlflow
import pandas as pd
from tqdm import tqdm

counting = 0
def eval(val_loader, model, device, epoch, is_ema, default_configs, coord_criterion, severity_criterion):
    if is_ema:
        print("EMA EVAL")
    else:
        print("NORMAL EVAL")

    k = 0
    model.eval()
    predictions = []; ground_truths = []
    probs = []

    # extravasation_predictions = []; extravasation_ground_truths = []

    val_metric = {"auc": 0, "loss": 0}
    batch_idx = 0
    val_loss_severity_left = 0
    val_loss_severity_right = 0
    val_loss = 0
    with torch.no_grad():
        for images, severity_labels in tqdm(val_loader):   
            images = images.to(device).float()
            severity_labels = severity_labels.to(device).long()
            logits = model(images)
            # loss_coord = 0
            # for i_head in range(logits.shape[0]):
            #     loss_coord += torch.sum(obj_labels[:,i_head, 0]*coord_criterion(logits[i_head][:, 1:], obj_labels[:,i_head, 1:]))/torch.sum(obj_labels[:,i_head, 0])
            loss_severity= severity_criterion(logits, severity_labels)
            # loss_coord = loss_coord/logits.shape[0]
            val_loss += loss_severity.item()
            batch_idx += 1

        print("Severity loss: ", val_loss/batch_idx)
    val_metric['auc'] = 0 
    val_metric['loss'] = val_loss/batch_idx
         
    return val_metric
