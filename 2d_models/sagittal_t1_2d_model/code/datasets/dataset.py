import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
from data_augmentations.rand_augment import preprocess_input
from torchvision.io import read_image
from torch.utils.data import Sampler, RandomSampler, SequentialSampler
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.random_erasing import RandomErasing
from typing import Union, Tuple, List, Dict
import random
import math
import pandas as pd
# import torchio as tio
import math
from .transforms import hlfip_with_heatmap

def first(n):  
    return int(n[0]) 

class ImageFolder(data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, default_configs, mode):
        super().__init__()
        self.data_path = "/root/images"
        self.study_set = {}
        cond_list = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing"]
        level_list = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        level_map = {"L1/L2": 0, "L2/L3": 1, "L3/L4": 2, "L4/L5": 3, "L5/S1": 4}
        self.axis_list = ["Sagittal T1"]
        self.axis_map = {"Sagittal T1": "sagittal_t1", "Sagittal T2": "sagittal_t2", "Axial T2": "axial_t2"}
        self.train_series_df = pd.read_csv("data/train_series_descriptions.csv")
        self.train_labels_df = pd.read_csv("data/train_label_coordinates.csv")
        self.metadata_df = pd.read_csv("data/metadata.csv")
        self.train_series_df.set_index("series_id", inplace = True)

        self.study_list = set()
        for index, row in df.iterrows():
            study_id = row["study_id"]
            self.study_list.add(study_id)
        
        col_maps = {}
        col_i = 0
        self.col_names = []
        for cond in cond_list:
            for level in level_list:
                if cond not in col_maps.keys():
                    col_maps[cond] = {}
                col_maps[cond][level] = col_i
                col_i += 1
                self.col_names.append(cond + " " + level)
        for index, row in self.metadata_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            instance_number = row["image_id"]
            axis = self.train_series_df.loc[int(series_id)]['series_description'].replace("/STIR", "")
            axis = self.axis_map[axis]
            h, w = int(row["h"]), int(row["w"])
            
            if default_configs["axis"] in axis and study_id in self.study_list:
                if study_id not in self.study_set.keys():
                    self.study_set[study_id] = {}
                if series_id not in self.study_set[study_id].keys():
                    self.study_set[study_id][series_id] = {}    
                self.study_set[study_id][series_id][instance_number] = (h, w)
        
        label_coordinates = {}
        for index, row in self.train_labels_df.iterrows():
            study_id = row["study_id"]
            if study_id in self.study_set.keys():
                series_id = row["series_id"]
                if series_id in self.study_set[study_id].keys():
                    instance_number = row["instance_number"]
                    condition = row["condition"]
                    if condition in col_maps.keys():
                        level = row["level"]
                        if study_id not in label_coordinates.keys():
                            label_coordinates[study_id] = {}
                        if series_id not in label_coordinates[study_id].keys():
                            label_coordinates[study_id][series_id] = {}
                        if instance_number not in label_coordinates[study_id][series_id].keys():
                            label_coordinates[study_id][series_id][instance_number] = np.zeros((5, 3))
                        # label_id = col_maps[condition][level] 
                        level_id = level_map[level]
                        h, w = self.study_set[study_id][series_id][instance_number]
                        if row["x"] >= 6 and row["y"] >= 6:
                            label_coordinates[study_id][series_id][instance_number][level_id][0] = 1
                            label_coordinates[study_id][series_id][instance_number][level_id][1] = row["x"]/w
                            label_coordinates[study_id][series_id][instance_number][level_id][2] = row["y"]/h
        
        self.negative_frames = {}
        for study_id in self.study_set.keys():
            if study_id in label_coordinates.keys():
                for series_id in self.study_set[study_id].keys():
                    if series_id in label_coordinates[study_id].keys():
                        for instance_number in self.study_set[study_id][series_id].keys():
                            if instance_number not in label_coordinates[study_id][series_id]:
                                if study_id not in self.negative_frames.keys():
                                    self.negative_frames[study_id] = {}
                                if series_id not in self.negative_frames[study_id].keys():
                                    self.negative_frames[study_id][series_id] = []
                                self.negative_frames[study_id][series_id].append(instance_number)
        
        self.datasets = []
        if mode == "train":
            i = 0
            for study_id in self.study_set.keys():
                for series_id in self.study_set[study_id].keys():
                    axis = self.train_series_df.loc[int(series_id)]['series_description'].replace("/STIR", "")
                    if axis in self.axis_list:
                        for instance_number in self.study_set[study_id][series_id].keys():
                            h, w = self.study_set[study_id][series_id][instance_number]
                            if study_id in label_coordinates.keys() and series_id in label_coordinates[study_id].keys() and instance_number in label_coordinates[study_id][series_id].keys():
                                self.datasets.append((study_id, series_id, instance_number, label_coordinates[study_id][series_id][instance_number], h, w))
                            else:
                                self.datasets.append((study_id, series_id, instance_number, np.zeros((5, 3)), h, w))
                                i += 1
        else:
            i = 0
            for study_id in self.study_set.keys():
                for series_id in self.study_set[study_id].keys():
                    axis = self.train_series_df.loc[int(series_id)]['series_description'].replace("/STIR", "")
                    if axis in self.axis_list:
                        for instance_number in self.study_set[study_id][series_id].keys():
                            h, w = self.study_set[study_id][series_id][instance_number]
                            if study_id in label_coordinates.keys() and series_id in label_coordinates[study_id].keys() and instance_number in label_coordinates[study_id][series_id].keys():
                                self.datasets.append((study_id, series_id, instance_number, label_coordinates[study_id][series_id][instance_number], h, w))
                            else:
                                self.datasets.append((study_id, series_id, instance_number, np.zeros((5, 3)), h, w))
                                i += 1

        self.mode = mode
        self.N_EVAL = default_configs["n_eval"]
        self.df = df.reset_index(drop=True)
        self.df = self.df.groupby(['study_id'])

        self.labels = {}
        self.image_size = default_configs["image_size"]
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.8, 1.0), p=1),
            # A.Resize(height=self.image_size, width=self.image_size, p=1),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.Perspective(p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.5),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                # A.ElasticTransform(alpha=3),
            ], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, border_mode=0, p=0.5),
            # A.Resize(height=self.image_size, width=self.image_size, p=1),
        ])

        self.test_transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1),
        ])

        self.mode = mode
        
    def __len__(self):
        return len(self.datasets)
    
    def gen_heat_map(self, label):
        heat_map = np.zeros((12, 12))
        step = 1/12
        for i in range(label.shape[0]):
            prob, x, y = label[i]
            if prob == 1:
                x_heatmap = int(math.floor(x/step))
                y_heatmap = int(math.floor(y/step))
                heat_map[y_heatmap, x_heatmap] = 1
        return heat_map
                
    def __getitem__(self, index):
        study_id, series_id, instance_number, label, h, w = self.datasets[index]
        axis = self.train_series_df.loc[int(series_id)]['series_description'].replace("/STIR", "")
        # axis = self.axis_map[axis]
        data_path = os.path.join(self.data_path, axis)
        study_id_path = os.path.join(data_path, str(study_id))
        series_path = os.path.join(study_id_path, str(series_id))
        image_path = os.path.join(series_path, str(instance_number) + ".jpg")
        # print(image_path)
        pil_img = Image.open(image_path).convert('RGB')
        img = np.asarray(pil_img)
        heat_map = self.gen_heat_map(label)
        if self.mode == 'train':
            img = self.train_transform(image=img)["image"]
            # img, heat_map = hlfip_with_heatmap(img, heat_map)
        else:
            img = self.test_transform(image=img)["image"]
        # cv2.imwrite("test.png", img)
        img = img.transpose(2, 0, 1)
        
        return img, heat_map, label

class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, default_configs, mode):
        self.data_dir = "data"
        self.level_list = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        self.level_map = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
        self.image_size = default_configs["image_size"]
        self.records= self.load_coords(df)
        self.ids = list(self.records.keys())
        
        self.train_transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.5)
        ])

        self.test_transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1),
        ])
        self.mode = mode
        

    def load_coords(self, df):
        # Convert to dict
        
        # d = df.groupby("series_id")[["relative_x", "relative_y"]].apply(lambda x: list(x.itertuples(index=False, name=None)))
        records = {}

        for index, row in df.iterrows():
            series_id = row["source"] + "_" + row["filename"].replace(".jpg", "")
            if series_id not in records.keys():
                records[series_id] = np.zeros((len(self.level_list), 3))
            records[series_id][self.level_map[row["level"]]] = np.array([1, row["relative_x"], row["relative_y"]])

        return records
    
    
    def load_img(self, source, series_id):
        fname= os.path.join(self.data_dir, "processed_{}_jpgs/{}.jpg".format(source, series_id))
        pil_img = Image.open(fname).convert('RGB')
        img = np.asarray(pil_img)
        if self.mode == 'train':
            img = self.train_transform(image=img)["image"]
        else:
            img = self.test_transform(image=img)["image"]
        img= np.transpose(img, (2, 0, 1))
        return img
        
        
    def __getitem__(self, idx):
        series_id = self.ids[idx]
        label= self.records[series_id]
        source= series_id.split("_")[0]
        series_id= "_".join(series_id.split("_")[1:])     
                
        img= self.load_img(source, series_id)
        # print(img.shape)
        return img, label
    
    def __len__(self,):
        return len(self.ids)