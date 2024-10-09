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
from metrics import score

counting = 0

def cal_logloss(gts, probs):
    gts = torch.cat(gts).cpu().numpy() 
    probs = torch.cat(probs).cpu().numpy()
    # print(gts.shape, probs.shape)
    sample_weight = []
    for j in range(gts.shape[0]):
        if(gts[j] == 0):
            sample_weight.append(1)
        elif(gts[j] == 1):
            sample_weight.append(2)
        else:
            sample_weight.append(4)
    logloss = sklearn.metrics.log_loss(gts, probs, sample_weight=sample_weight)
    return logloss


def eval(val_loader, model, device, epoch, is_ema, default_configs, col_names):
    if is_ema:
        print("EMA EVAL")
    else:
        print("NORMAL EVAL")

    k = 0
    model.eval()
    ground_truths = {0: [], 1: [], 3: []}
    probs = {0: [], 1: [], 3: []}
    for i in range(len(col_names)):
        probs[0].append([])
        ground_truths[0].append([])
        probs[1].append([])
        ground_truths[1].append([])
        probs[3].append([])
        ground_truths[3].append([])
    
    subarticular_gts = {0: [], 1: [], 3: []}; subarticular_probs = {0: [], 1: [], 3: []}


    n_images = 0
    patient_id_list = []
    val_metric = {"loss": 0}
    pred_rows = {0: [], 1: [], 3: []}
    gt_rows = {0: [], 1: [], 3: []}

    with torch.no_grad():
        for batch_idx, (series_stacks_ax, volume_labels, any_labels, frame_labels, patient_ids, series_ids) in enumerate(val_loader):   
            series_stacks_ax = series_stacks_ax.to(device).float()
            volume_labels = volume_labels.to(device).long()

            volume_logits_3, image_logits = model(series_stacks_ax)
            for volume_logits, k in zip([volume_logits_3], [3]):
                for i in range(volume_logits.shape[1]):
                    study_id = patient_ids[i]
                    prob = torch.nn.Softmax(dim=1)(volume_logits[:, i, :])
                    for j, col_name in enumerate(col_names):
                        row_id = str(study_id) + "_" + col_name
                        
                        if volume_labels[i][j].item() == 0:
                            gt_rows[k].append({"row_id": row_id, "normal_mild": 1, "moderate": 0, "severe": 0, "sample_weight": 1})
                        elif volume_labels[i][j].item() == 1:
                            gt_rows[k].append({"row_id": row_id, "normal_mild": 0, "moderate": 1, "severe": 0, "sample_weight": 2})
                        elif volume_labels[i][j].item() == 2:
                            gt_rows[k].append({"row_id": row_id, "normal_mild": 0, "moderate": 0, "severe": 1, "sample_weight": 4})   
                        elif volume_labels[i][j].item() == -100:
                            continue
                        pred_rows[k].append({"row_id": row_id, "normal_mild": prob[j][0].item(), "moderate": prob[j][1].item(), "severe": prob[j][2].item()})
                for i in range(volume_logits.shape[0]):
                    prob = torch.nn.Softmax(dim=1)(volume_logits[i])
                    probs[k][i] += [prob]
                    ground_truths[k][i] += [volume_labels[:, i].detach().cpu()]
                    
                    subarticular_gts[k] += [volume_labels[:, i].detach().cpu()]
                    subarticular_probs[k] += [prob]

            for j in range(len(patient_ids)):
                patient_id_list.append(patient_ids[j])
                
    for k in [3]:    
        total_logloss = 0
        for i in range(len(col_names)):
            ground_truths[k][i] = torch.cat(ground_truths[k][i]).cpu().numpy() 
            max_label = np.max(ground_truths[k][i])
            probs[k][i] = torch.cat(probs[k][i]).cpu().numpy()
            gts = []
            prbs = []
            for index in range(ground_truths[k][i].shape[0]):
                if ground_truths[k][i][index] != -100:
                    gts.append(ground_truths[k][i][index])
                    prbs.append(probs[k][i][index])
            sample_weight = []
            for j in range(len(gts)):
                if(gts[j] == 0):
                    sample_weight.append(1)
                elif(gts[j] == 1):
                    sample_weight.append(2)
                else:
                    sample_weight.append(4)
            gts = np.array(gts)
            prbs = np.array(prbs)
            if max_label == 2:
                logloss = sklearn.metrics.log_loss(gts, prbs, sample_weight=sample_weight)
            else:
                logloss = sklearn.metrics.log_loss(gts, prbs[:, :2], sample_weight=sample_weight)
            print('{}. logloss {}: '.format(k, col_names[i]), logloss)
            if k == 3:
                val_metric[col_names[i]] = logloss
            total_logloss += logloss

        pred_df = pd.DataFrame.from_dict(pred_rows[k]) 
        gt_df = pd.DataFrame.from_dict(gt_rows[k]) 
        lb_loss = score(gt_df, pred_df, "row_id", 1.0)

        print("{}. lb loss: ".format(k), lb_loss)
        print("{}. Average loss: ".format(k), total_logloss/(len(col_names)))
        if k == 3:
            val_metric['loss'] = lb_loss

         
    return val_metric