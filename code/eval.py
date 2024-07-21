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
    predictions = []; ground_truths = []
    probs = []
    for i in range(len(col_names)):
        probs.append([])
        ground_truths.append([])
    
    subarticular_gts = []; subarticular_probs = []
    foraminal_gts = []; foraminal_probs = []
    spinal_gts = []; spinal_probs = []
    any_severe_spinal_gts = []; any_severe_spinal_probs = []


    n_images = 0
    patient_id_list = []
    val_metric = {"loss": 0}
    pred_rows = []
    gt_rows = []

    with torch.no_grad():
        for batch_idx, (series_stack, volume_labels, any_labels, frame_labels, heat_map_labels, patient_ids) in enumerate(val_loader):   
            series_stack = series_stack.to(device).float()
            volume_labels = volume_labels.to(device).long()

            volume_logits, image_logits, heat_maps = model(series_stack)
            for i in range(volume_logits.shape[1]):
                study_id = patient_ids[i]
                prob = torch.nn.Softmax(dim=1)(volume_logits[:, i, :])
                for j, col_name in enumerate(col_names):
                    row_id = str(study_id) + "_" + col_name
                    pred_rows.append({"row_id": row_id, "normal_mild": prob[j][0].item(), "moderate": prob[j][1].item(), "severe": prob[j][2].item()})
                    if volume_labels[i][j].item() == 0:
                        gt_rows.append({"row_id": row_id, "normal_mild": 1, "moderate": 0, "severe": 0, "sample_weight": 1})
                    elif volume_labels[i][j].item() == 1:
                        gt_rows.append({"row_id": row_id, "normal_mild": 0, "moderate": 1, "severe": 0, "sample_weight": 2})
                    elif volume_labels[i][j].item() == 2:
                        gt_rows.append({"row_id": row_id, "normal_mild": 0, "moderate": 0, "severe": 1, "sample_weight": 4})   
            for i in range(volume_logits.shape[0]):
                prob = torch.nn.Softmax(dim=1)(volume_logits[i])
                probs[i] += [prob]
                ground_truths[i] += [volume_labels[:, i].detach().cpu()]
                if i >=0 and i < 10:
                    subarticular_gts += [volume_labels[:, i].detach().cpu()]
                    subarticular_probs += [prob]
                elif i < 20:
                    foraminal_gts += [volume_labels[:, i].detach().cpu()]
                    foraminal_probs += [prob]
                else:
                    spinal_gts += [volume_labels[:, i].detach().cpu()]
                    spinal_probs += [prob]
            
            for i in range(volume_labels.shape[0]):
                any_severe_spinal_label = volume_labels[i, 20:].max().item()
                if any_severe_spinal_label == 2:
                    any_severe_spinal_label = 1
                else:
                    any_severe_spinal_label = 0
                any_severe_spinal_gts += [torch.tensor([any_severe_spinal_label])]

            prob = torch.nn.Softmax(dim=2)(volume_logits[20:,:,:])[:,:,2]
            any_prob, _ = torch.max(prob, 0)
            any_severe_spinal_probs += [any_prob]

            for j in range(len(patient_ids)):
                patient_id_list.append(patient_ids[j])
                
        
        total_logloss = 0
        for i in range(len(col_names)):
            ground_truths[i] = torch.cat(ground_truths[i]).cpu().numpy() 
            max_label = np.max(ground_truths[i])
            probs[i] = torch.cat(probs[i]).cpu().numpy()
            sample_weight = []
            for j in range(ground_truths[i].shape[0]):
                if(ground_truths[i][j] == 0):
                    sample_weight.append(1)
                elif(ground_truths[i][j] == 1):
                    sample_weight.append(2)
                else:
                    sample_weight.append(4)
            if max_label == 2:
                logloss = sklearn.metrics.log_loss(ground_truths[i], probs[i], sample_weight=sample_weight)
            else:
                logloss = sklearn.metrics.log_loss(ground_truths[i], probs[i][:, :2], sample_weight=sample_weight)
            print('logloss {}: '.format(col_names[i]), logloss)
            val_metric[col_names[i]] = logloss
            total_logloss += logloss
    
    subarticular_loss = cal_logloss(subarticular_gts, subarticular_probs)
    foraminal_loss = cal_logloss(foraminal_gts, foraminal_probs)
    spinal_loss = cal_logloss(spinal_gts, spinal_probs)
    any_severe_spinal_loss = cal_logloss(any_severe_spinal_gts, any_severe_spinal_probs)
    logloss = (subarticular_loss + foraminal_loss + spinal_loss) /3

    pred_df = pd.DataFrame.from_dict(pred_rows) 
    gt_df = pd.DataFrame.from_dict(gt_rows) 
    lb_loss = score(gt_df, pred_df, "row_id", 1.0)

    print("subarticular_loss: ", subarticular_loss)
    print("foraminal_loss: ", foraminal_loss)
    print("spinal_loss: ", spinal_loss)
    print("any_severe_spinal_loss: ", any_severe_spinal_loss)
    print("log loss: ", logloss)
    print("lb loss: ", lb_loss)
    print("Average loss: ", total_logloss/(len(col_names)))
    val_metric['loss'] = lb_loss
    val_metric['subarticular_loss'] = subarticular_loss
    val_metric['foraminal_loss'] = foraminal_loss
    val_metric['spinal_loss'] = spinal_loss
    val_metric['any_severe_spinal_loss'] = any_severe_spinal_loss
         
    return val_metric