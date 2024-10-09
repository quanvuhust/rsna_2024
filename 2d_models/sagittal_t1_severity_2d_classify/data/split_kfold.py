import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from sklearn import model_selection as sk_model_selection
import numpy as np
import os
from PIL import Image
import pandas as pd
import hashlib
import cv2
import shutil

classes = [0, 1, 2]
directory = "/ssd/kaggle/RSNA_2024/rsna_2024/crop_data/sagittal_t1/images"
file_paths = []
labels = []
study_ids = []
RANDOM_SEED = 2024

for label in classes:
    class_directory = os.path.join(directory, str(label))
    for videoname in os.listdir(class_directory):
        f = os.path.join(class_directory, videoname)
        study_id, series_id, instance_number, _ = videoname.split("_")
        file_paths.append(f)
        labels.append(label)
        study_ids.append(int(study_id))
file_paths = np.array(file_paths)
labels = np.array(labels)
skf = StratifiedGroupKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
for fold, (train_idx, valid_idx) in enumerate(skf.split(file_paths, labels, groups=study_ids)):
    x_train, y_train = file_paths[train_idx], labels[train_idx]
    x_train = np.expand_dims(x_train, 1)
    y_train = np.expand_dims(y_train, 1)
    print(x_train.shape)
    x_val, y_val = file_paths[valid_idx], labels[valid_idx]
    x_val = np.expand_dims(x_val, 1)
    y_val = np.expand_dims(y_val, 1)
    train_df = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=["file_path", "label"])
    train_df.to_csv('train_fold{}.csv'.format(fold), index=False)
    val_df = pd.DataFrame(np.concatenate((x_val, y_val), axis=1), columns=["file_path", "label"])
    val_df.to_csv('val_fold{}.csv'.format(fold), index=False) 