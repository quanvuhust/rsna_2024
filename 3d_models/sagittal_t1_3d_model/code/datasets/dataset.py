import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
from typing import Union, Tuple, List, Dict
import random
import math
import pandas as pd
from .transforms import get_train_transform


class ImageFolder(data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, col_names, dataset, default_configs, mode):
        super().__init__()
        self.data_path = "/root/images/Sagittal T1"
        self.axial_crop_data_path = "/root/images/Sagittal T1"
        
        self.study_set = {}
        self.dataset = dataset
        cond_list = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing"]
        level_list = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        self.train_series_df = pd.read_csv("data/train_series_descriptions.csv")
        self.metadata_df = pd.read_csv("data/position_orders.csv")

        self.frame_orders = {}
        for index, row in self.metadata_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            instance_list = row["instance_list"].replace("[", "").replace("]", "").replace(", ", " ")
            if study_id not in self.frame_orders.keys():
                self.frame_orders[study_id] = {}
            if series_id not in self.frame_orders[study_id].keys():
                self.frame_orders[study_id][series_id] = {}
            for i, num in enumerate(instance_list.split(' ')):
                self.frame_orders[study_id][series_id][int(num)] = i

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
        
        col_maps = {}
        col_i = 0
        
        for cond in cond_list:
            for level in level_list:
                if cond not in col_maps.keys():
                    col_maps[cond] = {}
                col_maps[cond][level] = col_i
                col_i += 1
        i_to_level = {}
        for cond in col_maps.keys():
            for level in col_maps[cond].keys():
                i_to_level[col_maps[cond][level]] = cond+level
        print(i_to_level)
        for root, dirs, files in os.walk(self.data_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                study_id = int(file_path.split("/")[-3])
                if study_id in self.study_list:
                    series_id = int(file_path.split("/")[-2])
                    instance_number = int(name.replace(".jpg", ""))
                    # instance_number, h, w = list(map(int, name.replace(".jpg", "").split("_")))
                    if study_id not in self.study_set.keys():
                        self.study_set[study_id] = {}
                    if series_id not in self.study_set[study_id].keys():
                        self.study_set[study_id][series_id] = {}    
                    # self.study_set[study_id][series_id][instance_number] = (instance_number, instance_number)
                    self.study_set[study_id][series_id][instance_number] = (instance_number, instance_number)

        self.label_coordinates = {}
        self.label_levels = {}
        self.positive_frames = {}
        self.n_patch = default_configs["image_size"] // 14
        for index, row in self.label_coordinates_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            instance_number = row["instance_number"]
            condition = row["condition"]
            level = row["level"]
            condition_level = condition + level
            if "Foraminal" in condition:
                if study_id in self.study_set.keys() and series_id in self.study_set[study_id].keys():
                    if study_id not in self.label_coordinates.keys():
                        self.label_coordinates[study_id] = {}
                        self.positive_frames[study_id] = {}
                        self.label_levels[study_id] = {}
                    if series_id not in self.label_coordinates[study_id].keys():
                        self.label_coordinates[study_id][series_id] = {}
                        self.positive_frames[study_id][series_id] = set()
                        self.label_levels[study_id][series_id] = {}
                    self.positive_frames[study_id][series_id].add(instance_number)
                    if instance_number not in self.label_coordinates[study_id][series_id].keys():
                        self.label_coordinates[study_id][series_id][instance_number] = np.zeros((10,))
                    if condition_level not in self.label_levels[study_id][series_id].keys():
                        self.label_levels[study_id][series_id][condition_level] = set()
                    self.label_levels[study_id][series_id][condition_level].add(instance_number)
                    
                    label_id = col_maps[condition][level]
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
        
        self.labels = {}

        self.image_sizes = [default_configs["image_size"], default_configs["image_size"]]
        normal_aug_list, instance_aug_list = get_train_transform(default_configs["image_size"])
        self.train_normal_transforms = A.Compose(normal_aug_list)
        self.train_instance_transforms = A.Compose(instance_aug_list)
        self.test_transforms = A.Compose([
            # A.CenterCrop(288, 288),
            A.Resize(height=default_configs["image_size"], width=default_configs["image_size"]),
        ])
        self.mode = mode
        raw_labels = {}
        for index, row in df.iterrows():
            study_id = row["study_id"]
            label = row[col_names].values
            if default_configs["finetune"] == True and self.mode == "train":
                series_id = row["series_id"]
                if study_id not in raw_labels.keys():
                    raw_labels[study_id] = {}
                raw_labels[study_id][series_id] = label
            else:
                raw_labels[study_id] = label

        self.study_ids = []
        for study_id in self.study_set.keys():
            for series_id in self.study_set[study_id].keys():
                if default_configs["finetune"] == True and self.mode == "train":
                    label = raw_labels[study_id][series_id]
                else:
                    label = raw_labels[study_id]
                
                new_label = []
                flag = False
                for i in range(label.shape[0]):
                    condition_level = i_to_level[i]
                    if condition_level not in self.label_levels[study_id][series_id].keys():
                        if type(label[i]) == str:
                            new_label.append(np.array([-100, -100, -100]) )
                        else:
                            new_label.append(-100)
                        # flag = True
                    else:
                        if type(label[i]) == str:
                            new_label.append(np.array([float(num) for num in label[i].split(' ')]))
                        else:
                            new_label.append(label[i])
                
                label = np.array(new_label)
                # print(label)
                self.labels[series_id] = label
                self.study_ids.append((study_id, series_id))
        
        
    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, index):
        study_id, series_id = self.study_ids[index]
        
        n_eval = self.N_EVAL
        series_stacks = np.zeros((n_eval, self.image_sizes[0], self.image_sizes[1], 3)).astype("uint8")
        
        multiview_frame_labels = np.zeros((self.N_EVAL, 10))
        multiview_label = self.labels[series_id]

        # print(study_id, injure_labels)
        study_id_path = os.path.join(self.axial_crop_data_path, str(study_id))
        n_eval = self.N_EVAL
            
        series_path = os.path.join(study_id_path, str(series_id))
        flag = False
        frame_labels = np.zeros((n_eval, 10))
        
        if study_id in self.label_coordinates.keys():
            if series_id in self.label_coordinates[study_id].keys():
                instance_number_list = set()
                instance_number_test_list = set()
                
                for condition_level in self.label_levels[study_id][series_id].keys():
                    l = list(self.label_levels[study_id][series_id][condition_level])
                    instance_number = random.sample(l, 1)[0]
                    instance_number_list.add(instance_number)
                    instance_number_test_list.add(l[0])
                instance_number_list = list(instance_number_list)
                instance_number_test_list = list(instance_number_test_list)
                for condition_level in self.label_levels[study_id][series_id].keys():
                    for ins in self.label_levels[study_id][series_id][condition_level]:
                        if ins not in instance_number_test_list:
                            instance_number_test_list.append(ins)
                
                total_instance = len(instance_number_list)

                if self.mode == "train":
                    instance_select_list = instance_number_list
                    
                else:
                    instance_select_list = instance_number_test_list[:n_eval]
                n_left = n_eval - len(instance_select_list)
                normal_select_list = []
                if n_left > 0:
                    n_slect = min(n_left, len(self.negative_frames[study_id][series_id]))
                    normal_select_list = random.sample(self.negative_frames[study_id][series_id], n_slect)
                    n_left -= n_slect
                
                select_list = instance_select_list + normal_select_list
                order_select_list = []
                for instance_number in select_list:
                    order_select_list.append((self.frame_orders[study_id][series_id][instance_number], instance_number))
                if self.mode == "train":
                    # if np.random.rand() < 0.5:
                    order_select_list = sorted(order_select_list, key = lambda x: x[0])
                    # else:
                    #     order_select_list = sorted(order_select_list, key = lambda x: x[0], reverse=True)
                else:
                    order_select_list = sorted(order_select_list, key = lambda x: x[0])
                    # select_list.sort()
                # print(order_select_list)
                flag = True
                for i, v in enumerate(order_select_list):
                    pos, instance_number = v
                    if instance_number in instance_select_list:
                        frame_label = self.label_coordinates[study_id][series_id][instance_number]
                    else:
                        frame_label = np.zeros((10,))
                    frame_labels[i] = frame_label
                    
                    # heat_map_labels[i] = self.heat_maps[study_id][series_id][instance_number]
        # if flag == False:
        #     select_list = random.sample(self.negative_frames[study_id][series_id], self.N_EVAL)
        #     if self.mode == "train":
        #         if np.random.rand() < 0.5:
        #             select_list.sort()
        #         else:
        #             select_list.sort(reverse=True)
        #     else:
        #         select_list.sort()
        #     for i, instance_number in enumerate(select_list):
        #         frame_labels[i] = np.zeros((10,))
        #         # heat_map_labels[i] = self.heat_maps[study_id][series_id][instance_number]

        image_paths = []
        for pos, instance_number in order_select_list:
            h, w = self.study_set[study_id][series_id][instance_number]
            image_path = os.path.join(series_path, str(instance_number) + ".jpg")
            # image_path = os.path.join(series_path, str(instance_number) + "_" + str(h) + "_" + str(w) + ".jpg")
            image_paths.append(image_path)

        total_features = len(image_paths)
        # print(total_images, step)
        series_stack = np.zeros((n_eval, self.image_sizes[0], self.image_sizes[1], 3)).astype("uint8")
        if self.mode == "train":
            is_flip = random.random() < 0.5
            for j in range(total_features):
                image_path = image_paths[j]
                instance_number = select_list[j]
                pil_img_1 = Image.open(image_path).convert('RGB')
                img = np.asarray(pil_img_1)
                
                if np.sum(frame_labels[j]) == 0:
                    img = self.train_normal_transforms(image=img)["image"]
                else:    
                    img = self.train_instance_transforms(image=img)["image"]
                if is_flip == True:
                    img = A.HorizontalFlip(p=1.0)(image=img)["image"]
                series_stack[j] = img
        else:
            
            for j in range(total_features):
                image_path = image_paths[j]
                instance_number = select_list[j]
                pil_img_1 = Image.open(image_path).convert('RGB')
                img = np.asarray(pil_img_1)
                img = self.test_transforms(image=img)["image"]
                series_stack[j] = img
            
        series_stacks[:, :, :, :] = series_stack
        multiview_frame_labels = frame_labels
        # cv2.imwrite("test.png", series_stacks.reshape(series_stack.shape[0]*series_stack.shape[1], series_stack.shape[2], series_stack.shape[3]))
        any_severe_spinal_label = 0
        
        return series_stacks, multiview_label, any_severe_spinal_label, multiview_frame_labels, study_id, series_id