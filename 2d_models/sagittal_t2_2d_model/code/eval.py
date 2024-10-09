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
def eval(val_loader, model, device, epoch, is_ema, default_configs, col_names, coord_criterion):
    if is_ema:
        print("EMA EVAL")
    else:
        print("NORMAL EVAL")

    k = 0
    model.eval()
    predictions = []; ground_truths = []
    probs = []
    for i in range(len(col_names)):
        probs.append([])
        ground_truths.append([])
        predictions.append([])
    # extravasation_predictions = []; extravasation_ground_truths = []

    val_metric = {"auc": 0, "eer": 0, "loss": 0}
    batch_idx = 0
    val_loss = 0
    with torch.no_grad():
        for images, heat_maps, obj_labels in tqdm(val_loader):   
            images = images.to(device).float()
            obj_labels = obj_labels.to(device).float()
            heat_maps = heat_maps.to(device).float()
            logits = model(images)
            # loss_coord = coord_criterion(heat_map_logits, heat_maps)
            # val_loss += loss_coord.item()
            batch_idx += 1
            for i in range(logits.shape[0]):
                prob = torch.nn.Sigmoid()(logits[i][:,0])
                preds = prob > 0.5
                predictions[i] += [preds]
                probs[i] += [prob]
                ground_truths[i] += [obj_labels[:, i, 0].detach().cpu()]
      
        total_roc_auc = 0
        total_f1_score = 0
        tt = 0
        for i in range(len(col_names)):
            try:
                ground_truths[i] = torch.cat(ground_truths[i]).cpu().numpy() 
                predictions[i] = torch.cat(predictions[i]).cpu().numpy()
                probs[i] = torch.cat(probs[i]).cpu().numpy()
                f1_score = metrics.f1_score(ground_truths[i], predictions[i])
                roc_auc = metrics.roc_auc_score(ground_truths[i], probs[i])

                fpr, tpr, threshold = metrics.roc_curve(ground_truths[i], probs[i], pos_label=1)
                fnr = 1 - tpr
                acer_threshold = threshold[np.nanargmin(np.absolute((fnr + fpr)/2))]
                eer_threshold = threshold[np.nanargmin(np.absolute(fnr - fpr))]
                EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
                print('roc_auc {}: '.format(col_names[i]), roc_auc)
                print('eer {}: '.format(col_names[i]), EER, eer_threshold)
                val_metric[col_names[i]] = roc_auc
                total_roc_auc += roc_auc
                total_f1_score += EER
                tt += 1
            except:
                print("Missing: ", col_names[i])

    val_metric['auc'] = total_roc_auc/tt
    val_metric['eer'] = total_f1_score/tt
    val_metric['loss'] = val_loss/batch_idx
         
    return val_metric

def eval_pretrain(val_loader, model, device, epoch, is_ema, default_configs, col_names, coord_criterion):
    if is_ema:
        print("EMA EVAL")
    else:
        print("NORMAL EVAL")

    k = 0
    model.eval()
    probs = []

    # extravasation_predictions = []; extravasation_ground_truths = []

    val_metric = {"loss": 0}
    batch_idx = 0
    val_loss = 0
    with torch.no_grad():
        for images, obj_labels in tqdm(val_loader):   
            images = images.to(device).float()
            obj_labels = obj_labels.to(device).float()
            logits = model(images)
            loss_coord = 0
            for i_head in range(logits.shape[0]):
                loss_coord += coord_criterion(logits[i_head][:, 1:], obj_labels[:,i_head, 1:])

            loss_coord = loss_coord/logits.shape[0]
            val_loss += loss_coord.item()
            batch_idx += 1

    val_metric['loss'] = val_loss/batch_idx
         
    return val_metric