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

def first(n):  
    return int(n[0]) 


def crop_black(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(grayscale, 32, 255, cv2.THRESH_BINARY)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    foreground = img[y:y+h, x:x+w]
    return foreground


class ImageFolder(data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, col_names, dataset, default_configs, mode):
        super().__init__()
        self.data_path = os.path.join("/root/sagittal_model/images", dataset)
        self.axial_crop_data_path = "/root/sagittal_model/images/axial_t2"
        self.study_set = {}
        self.dataset = dataset
        cond_list = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Left Subarticular Stenosis", "Right Subarticular Stenosis", "Spinal Canal Stenosis"]
        level_list = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        self.train_series_df = pd.read_csv("data/train_series_descriptions.csv")
        # self.train_series_df.set_index("series_id", inplace = True)
        self.study_list = set()
        for index, row in df.iterrows():
            study_id = row["study_id"]
            self.study_list.add(study_id)
        
        self.axis_map = {}
        for index, row in self.train_series_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            series_description = row["series_description"]
            if series_description == "Sagittal T2/STIR":
                axis = "sagittal_t2"
            elif series_description == "Sagittal T1":
                axis = "sagittal_t1"
            elif series_description == "Axial T2":
                axis = "axial_t2"
            if study_id not in self.axis_map.keys():
                self.axis_map[study_id] = {}
            if axis not in self.axis_map[study_id].keys():
                self.axis_map[study_id][axis] = []
            self.axis_map[study_id][axis].append(series_id)
        self.label_coordinates_df = pd.read_csv("data/train_label_coordinates.csv")
        self.axial_coordinates_df = pd.read_csv("data/train_label_axial_coordinates.csv")
        
                
        col_maps = {}
        col_i = 0
        
        for cond in cond_list:
            for level in level_list:
                if cond not in col_maps.keys():
                    col_maps[cond] = {}
                col_maps[cond][level] = col_i
                col_i += 1
        for root, dirs, files in os.walk(self.data_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                study_id = int(file_path.split("/")[-3])
                if study_id in self.study_list:
                    series_id = int(file_path.split("/")[-2])
                    instance_number, h, w = list(map(int, name.replace(".jpg", "").split("_")))
                    if study_id not in self.study_set.keys():
                        self.study_set[study_id] = {}
                    if series_id not in self.study_set[study_id].keys():
                        self.study_set[study_id][series_id] = {}    
                    self.study_set[study_id][series_id][instance_number] = (h, w)
        # print("HEHE: ", self.study_set.keys())
        self.axial_labels = {}
        for index, row in self.axial_coordinates_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            is_invert = row["is_invert"]
            if study_id in self.study_set.keys() and series_id in self.study_set[study_id].keys():
                if study_id not in self.axial_labels.keys():
                    self.axial_labels[study_id] = {}
                self.axial_labels[study_id][series_id] = is_invert
        self.label_coordinates = {}
        self.label_levels = {}
        self.n_patch = default_configs["image_size"] // 14
        for index, row in self.label_coordinates_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            instance_number = row["instance_number"]
            condition = row["condition"]
            level = row["level"]
            if study_id in self.study_set.keys() and series_id in self.study_set[study_id].keys():
                if study_id not in self.label_coordinates.keys():
                    self.label_coordinates[study_id] = {}
                    self.label_levels[study_id] = {}
                if series_id not in self.label_coordinates[study_id].keys():
                    self.label_coordinates[study_id][series_id] = {}
                    self.label_levels[study_id][series_id] = {}
                if instance_number not in self.label_coordinates[study_id][series_id].keys():
                    self.label_coordinates[study_id][series_id][instance_number] = np.zeros((col_i,))
                if level not in self.label_levels[study_id][series_id].keys():
                    self.label_levels[study_id][series_id][level] = set()  
                self.label_levels[study_id][series_id][level].add(instance_number)  
                label_id = col_maps[condition][level] 
                
                h, w = self.study_set[study_id][series_id][instance_number]
                if row["x"] >= 6 and row["y"] >= 6:
                    self.label_coordinates[study_id][series_id][instance_number][label_id] = 1
        
        self.negative_frames = {}
        for study_id in self.study_set.keys():
            if study_id in self.label_coordinates.keys():
                for series_id in self.study_set[study_id].keys():
                    # if series_id in self.label_coordinates[study_id].keys():
                    for instance_number in self.study_set[study_id][series_id].keys():
                        if study_id not in self.negative_frames.keys():
                            self.negative_frames[study_id] = {}
                        if series_id not in self.negative_frames[study_id].keys():
                            self.negative_frames[study_id][series_id] = []
                        if study_id not in self.label_coordinates.keys() or series_id not in self.label_coordinates[study_id].keys() or instance_number not in self.label_coordinates[study_id][series_id].keys():
                            self.negative_frames[study_id][series_id].append(instance_number)
                            # self.heat_maps[study_id][series_id][instance_number] = np.zeros((self.n_patch*self.n_patch))
        
        self.mode = mode
        self.N_EVAL = default_configs["n_eval"]
        self.df = df.reset_index(drop=True)
        self.df = self.df.groupby(['study_id'])
        temp_list = np.unique(df["study_id"].values)
        self.study_ids = set()
        for study_id in temp_list:
            if len(self.axis_map[study_id]) >= 3:
                self.study_ids.add(study_id)
        self.study_ids = list(self.study_ids)
        
        self.labels = {}
        self.image_sizes = {"sagittal_t2": (default_configs["image_size"], default_configs["image_size"]), 
                            "axial_t2": (default_configs["image_size"], default_configs["image_size"]), 
                            "sagittal_t1": (default_configs["image_size"], default_configs["image_size"])}

        self.train_normal_transforms = {}
        self.train_instance_transforms = {}
        for axis in ["axial_t2", "sagittal_t2", "sagittal_t1"]:
            if "axial" in axis:
                normal_aug_list = [
                    A.RandomResizedCrop(height=self.image_sizes[axis][0], width=self.image_sizes[axis][1], scale=(0.7, 1.0), p=1),
                    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.75),
                    A.Perspective(p=0.75),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ], p=0.75),

                    A.OneOf([
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.),
                        # A.ElasticTransform(alpha=3),
                    ], p=0.75),
                    # A.CoarseDropout(max_holes=2, max_height=int(288 * 0.25), max_width=int(288 * 0.25), p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.75),
                ]
                instance_aug_list = [
                    A.RandomResizedCrop(height=self.image_sizes[axis][0], width=self.image_sizes[axis][1], scale=(0.8, 1.0), p=1),
                    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
                    A.Perspective(p=0.5),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ], p=0.5),
                    A.OneOf([
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.),
                        # A.ElasticTransform(alpha=3),
                    ], p=0.5),
                    # A.CoarseDropout(max_holes=1, max_height=int(288 * 0.25), max_width=int(288 * 0.25), p=0.3),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
                ]
            else:
                normal_aug_list = [
                    A.RandomResizedCrop(height=self.image_sizes[axis][0], width=self.image_sizes[axis][1], scale=(0.7, 1.0), p=1),
                    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.75),
                    A.Perspective(p=0.75),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ], p=0.75),

                    A.OneOf([
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.),
                        # A.ElasticTransform(alpha=3),
                    ], p=0.75),

                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.75),
                ]
                instance_aug_list = [
                    A.RandomResizedCrop(height=self.image_sizes[axis][0], width=self.image_sizes[axis][1], scale=(0.8, 1.0), p=1),
                    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
                    A.Perspective(p=0.5),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ], p=0.5),
                    A.OneOf([
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.),
                        # A.ElasticTransform(alpha=3),
                    ], p=0.5),

                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
                ]
            
            self.train_normal_transforms[axis] = A.Compose(normal_aug_list)
            self.train_instance_transforms[axis] = A.Compose(instance_aug_list)
        self.hflip_transform = A.Compose([
                A.HorizontalFlip(p=1),
            ])
        self.test_transforms = {}
        for axis in ["axial_t2", "sagittal_t2", "sagittal_t1"]:
            self.test_transforms[axis] = A.Compose([
                # A.CenterCrop(288, 288),
                A.Resize(height=self.image_sizes[axis][0], width=self.image_sizes[axis][1], p=1),
            ])
        self.mode = mode
        for index, row in df.iterrows():
            study_id = row["study_id"]
            label = row[col_names].values
            # if self.mode == "train":
            #     one_hot_label = np.zeros((len(col_names)*3,))
            #     for i in range(label.shape[0]):
            #         one_hot_label[i*3:(i+1)*3][label[i]] = 1
                
            #     self.labels[study_id] = one_hot_label
            # else:
            new_label = []
            for i in range(label.shape[0]):
                if type(label[i]) == str:
                    new_label.append(np.array([float(num) for num in label[i].split(' ')]))
                else:
                    new_label.append(label[i])
            label = np.array(new_label)
            self.labels[study_id] = label
            
        
        
    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        
        if self.dataset == "all":
            axis_list = ["axial_t2", "sagittal_t2", "sagittal_t1"]
        else:
            axis_list = [self.dataset]
        series_stacks = {}
        for axis in axis_list:
            if axis == "axial_t2":
                n_eval = self.N_EVAL
            else:
                n_eval = self.N_EVAL
            series_stacks[axis] = np.zeros((n_eval, self.image_sizes[axis][0], self.image_sizes[axis][1], 3)).astype("uint8")
        
        multiview_frame_labels = np.zeros((len(axis_list)*self.N_EVAL, 25))
        
        multiview_label = self.labels[study_id]
        
        # print(study_id, injure_labels)
        for i_view, axis in enumerate(axis_list):
            series_list = self.axis_map[study_id][axis]
            series_id = int(random.choices(series_list, k=1)[0])
            if axis == "axial_t2":
                study_id_path = os.path.join(self.axial_crop_data_path, str(study_id))
                n_eval = self.N_EVAL
            else:
                study_id_path = os.path.join(self.data_path, str(study_id))
                n_eval = self.N_EVAL
                
            series_path = os.path.join(study_id_path, str(series_id))
            flag = False
            frame_labels = np.zeros((n_eval, 25))
            instance_select_list = []
            if study_id in self.label_coordinates.keys():
                if series_id in self.label_coordinates[study_id].keys():
                    instance_number_list = set()
                    instance_number_test_list = set()
                    for level in self.label_levels[study_id][series_id].keys():
                        l = list(self.label_levels[study_id][series_id][level])
                        instance_number = random.sample(l, 1)[0]
                        instance_number_list.add(instance_number)
                        instance_number_test_list.add(l[0])
                    instance_number_list = list(instance_number_list)
                    instance_number_test_list = list(instance_number_test_list)
                    for level in self.label_levels[study_id][series_id].keys():
                        for ins in self.label_levels[study_id][series_id][level]:
                            if ins not in instance_number_test_list:
                                instance_number_test_list.append(ins)
                    total_instance = len(instance_number_list)
                    if total_instance > 0:
                        if self.mode == "train":
                            instance_select_list = instance_number_list
                        else:
                            instance_select_list = instance_number_test_list[:n_eval]
                    n_left = n_eval - len(instance_select_list)
                    # remain_frames = []
                    # for instance_number in self.study_set[study_id][series_id].keys():
                    #     if instance_number not in instance_select_list:
                    #         remain_frames.append(instance_number)
                    normal_select_list = []
                    if n_left > 0:
                        n_slect = min(n_left, len(self.negative_frames[study_id][series_id]))
                        normal_select_list = random.sample(self.negative_frames[study_id][series_id], n_slect)
                        n_left -= n_slect
                    
                    select_list = instance_select_list + normal_select_list
            if study_id not in self.label_coordinates.keys() or series_id not in self.label_coordinates[study_id].keys():
                select_list = random.sample(self.negative_frames[study_id][series_id], n_eval)
            if self.mode == "train":
                # if axis == "axial_t2":
                #     if study_id in self.axial_labels.keys() and series_id in self.axial_labels[study_id].keys():
                #         # is_invert = self.axial_labels[study_id][series_id]
                #         if is_invert == 1:
                #             select_list.sort(reverse=True)
                #         else:
                #             select_list.sort()
                #     else:
                #         select_list.sort()
                # else:
                if np.random.rand() < 0.5:
                    select_list.sort()
                else:
                    select_list.sort(reverse=True)
            else:
                if axis == "axial_t2":
                    is_invert = self.axial_labels[study_id][series_id]
                    if is_invert == 1:
                        select_list.sort(reverse=True)
                    else:
                        select_list.sort()
                else:
                    select_list.sort()
            
            for i, instance_number in enumerate(select_list):
                if instance_number in instance_select_list:
                    frame_label = self.label_coordinates[study_id][series_id][instance_number]
                else:
                    frame_label = np.zeros((25,))
                frame_labels[i] = frame_label
                
            image_paths = []
            for instance_number in select_list:
                # if axis == "axial_t2":
                #     image_path = os.path.join(series_path, str(instance_number) + ".jpg")
                # else:
                h, w = self.study_set[study_id][series_id][instance_number]
                image_path = os.path.join(series_path, str(instance_number) + "_" + str(h) + "_" + str(w) + ".jpg")
                image_paths.append(image_path)

            total_features = len(image_paths)
            # print(total_images, step)
            series_stack = np.zeros((n_eval, self.image_sizes[axis][0], self.image_sizes[axis][1], 3)).astype("uint8")
            if self.mode == "train":
                
                for j in range(total_features):
                    image_path = image_paths[j]
                    instance_number = select_list[j]
                    pil_img = Image.open(image_path).convert('RGB')
                    img = np.asarray(pil_img)
                    
                    if np.sum(frame_labels[j]) == 0:
                        img = self.train_normal_transforms[axis](image=img)["image"]
                    else:    
                        img = self.train_instance_transforms[axis](image=img)["image"]
                    # if flag_flip == 1:
                    #     img = self.hflip_transform(image=img)["image"]
                    
                    series_stack[j] = img
            else:
                
                for j in range(total_features):
                    image_path = image_paths[j]
                    pil_img = Image.open(image_path).convert('RGB')
                    img = np.asarray(pil_img)
                    img = self.test_transforms[axis](image=img)["image"]
                    series_stack[j] = img
                
            series_stacks[axis][:, :, :, :] = series_stack
            if axis == "axial_t2":
                multiview_frame_labels[i_view*n_eval:(i_view+1)*n_eval, :] = frame_labels
            else:
                multiview_frame_labels[(i_view-1)*n_eval+self.N_EVAL:i_view*n_eval+self.N_EVAL, :] = frame_labels
            # multiview_heat_map_labels[i_view*self.N_EVAL:(i_view+1)*self.N_EVAL, :] = heat_map_labels
        # multiview_series_stack = multiview_series_stack.transpose(0, 3, 1, 2)
        
        any_severe_spinal_label = 0
        # print(multiview_label.shape, type(multiview_label), type(any_severe_spinal_label), type(multiview_frame_labels), type(study_id), type(series_stacks["axial_t2"]), type(series_stacks["sagittal_t2"]), type(series_stacks["sagittal_t1"]))
        return series_stacks["axial_t2"], series_stacks["sagittal_t2"], series_stacks["sagittal_t1"], multiview_label, any_severe_spinal_label, multiview_frame_labels, study_id