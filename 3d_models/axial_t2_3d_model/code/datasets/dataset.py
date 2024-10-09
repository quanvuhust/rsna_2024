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
        self.data_path = "/ssd/kaggle/RSNA_2024/rsna_2024/raw_datasets/images/Axial T2"
        self.axial_crop_data_path = "/ssd/kaggle/RSNA_2024/rsna_2024/raw_datasets/images/Axial T2"
        
        self.study_set = {}
        self.dataset = dataset
        cond_list = ["Left Subarticular Stenosis", "Right Subarticular Stenosis"]
        level_list = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        level_map = {"L1/L2": 1, "L2/L3": 2, "L3/L4": 3, "L4/L5": 4, "L5/S1": 5}
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
                i_to_level[col_maps[cond][level]] = level
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
                    
                    self.study_set[study_id][series_id][instance_number] = (instance_number, instance_number)
        # print("HEHE: ", self.study_set.keys())
        self.axial_labels = {}
        for index, row in self.axial_coordinates_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            is_invert = row["is_invert"]
            instance_number = row["instance_number"]
            image_label = np.array([float(num) for num in row["image_label"].split(' ')])
            if study_id in self.study_set.keys() and series_id in self.study_set[study_id].keys():
                if study_id not in self.axial_labels.keys():
                    self.axial_labels[study_id] = {}
                if series_id not in self.axial_labels[study_id].keys():
                    self.axial_labels[study_id][series_id] = {}
                self.axial_labels[study_id][series_id][instance_number] = image_label
        self.label_coordinates = {}
        self.label_levels = {}
        self.label_conditions = {}
        self.positive_frames = {}
        self.n_patch = default_configs["image_size"] // 14
        for index, row in self.label_coordinates_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            instance_number = row["instance_number"]
            condition = row["condition"]
            level = row["level"]
            if "Subarticular Stenosis" in condition:
                if study_id in self.study_set.keys() and series_id in self.study_set[study_id].keys():
                    if study_id not in self.label_coordinates.keys():
                        self.label_coordinates[study_id] = {}
                        self.positive_frames[study_id] = {}
                        self.label_levels[study_id] = {}
                        self.label_conditions[study_id] = {}
                    self.label_conditions[study_id][instance_number] = col_maps[condition][level]
                    if series_id not in self.label_coordinates[study_id].keys():
                        self.label_coordinates[study_id][series_id] = {}
                        self.positive_frames[study_id][series_id] = set()
                        self.label_levels[study_id][series_id] = {}
                    self.positive_frames[study_id][series_id].add(instance_number)
                    if instance_number not in self.label_coordinates[study_id][series_id].keys():
                        self.label_coordinates[study_id][series_id][instance_number] = np.zeros((10,3))
                    if level not in self.label_levels[study_id][series_id].keys():
                        self.label_levels[study_id][series_id][level] = set()
                    self.label_levels[study_id][series_id][level].add(instance_number)
                    
                    label_id = col_maps[condition][level]

                    if row["x"] >= 6 and row["y"] >= 6:
                        self.label_coordinates[study_id][series_id][instance_number][label_id][0] = 1
                        self.label_coordinates[study_id][series_id][instance_number][label_id][1] = row["x"]
                        self.label_coordinates[study_id][series_id][instance_number][label_id][2] = row["y"]
     
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
        self.hflip_transform = A.Compose([
                A.HorizontalFlip(p=1),
            ])
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
                    level = i_to_level[i]
                    if level not in self.label_levels[study_id][series_id].keys():
                        if type(label[i]) == str:
                            new_label.append(np.array([-100, -100, -100]) )
                        else:
                            new_label.append(-100)
                        # flag = True
                    else:
                        if type(label[i]) == str:
                            new_label.append(np.array([float(num) for num in label[i].split(' ')]))
                        else:
                            # if -100 in label:
                            #     flag = True
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
        
        multiview_frame_labels = np.zeros((self.N_EVAL, 10, 3))
        
        multiview_label = self.labels[series_id]
        axis = "axial_t2"
        # print(study_id, injure_labels)
        study_id_path = os.path.join(self.axial_crop_data_path, str(study_id))
        n_eval = self.N_EVAL
            
        series_path = os.path.join(study_id_path, str(series_id))
        flag = False
        frame_labels = np.zeros((n_eval, 10, 3))
        severity_labels = np.full((n_eval, 2), -100)
        
        if study_id in self.label_coordinates.keys():
            if series_id in self.label_coordinates[study_id].keys():
                instance_number_list = set()
                instance_number_test_list = set()

                for level in self.label_levels[study_id][series_id].keys():
                    l = list(self.label_levels[study_id][series_id][level])
                    if random.random() < 0.75:
                        n_sam = 1
                    else:
                        n_sam = min(len(l), 2)
                    tmp = random.sample(l, n_sam)
                    for instance_number in tmp:
                        instance_number_list.add(instance_number)
                    # prob = random.random()
                    # if prob < 1/3 and (instance_number - 1) in self.study_set[study_id][series_id].keys():
                    #     instance_number_list.add(instance_number - 1)
                    # elif prob < 2/3 and (instance_number + 1) in self.study_set[study_id][series_id].keys():
                    #     instance_number_list.add(instance_number + 1)
                    # else:
                    
                    instance_number_test_list.add(l[0])
                instance_number_list = list(instance_number_list)
                instance_number_test_list = list(instance_number_test_list)
                for level in self.label_levels[study_id][series_id].keys():
                    for ins in self.label_levels[study_id][series_id][level]:
                        if ins not in instance_number_test_list:
                            instance_number_test_list.append(ins)

                if self.mode == "train":
                    instance_select_list = instance_number_list[:n_eval]
                else:
                    instance_select_list = instance_number_test_list[:n_eval]
                n_left = n_eval - len(instance_select_list)
                normal_select_list = []
                if n_left > 0:
                    n_slect = min(n_left, len(self.negative_frames[study_id][series_id]))
                    normal_select_list = random.sample(self.negative_frames[study_id][series_id], n_slect)
                    n_left -= n_slect
                # for i in range(n_left):
                #     normal_select_list += random.sample(self.negative_frames[study_id][series_id], 1)
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
                flag = True
                for i, v in enumerate(order_select_list):
                    pos, instance_number = v
                    if instance_number in instance_select_list:
                        # if instance_number in self.label_coordinates[study_id][series_id].keys():
                        frame_label = self.label_coordinates[study_id][series_id][instance_number]
                        severity_labels[i] = self.axial_labels[study_id][series_id][instance_number]
                        # elif instance_number + 1 in self.label_coordinates[study_id][series_id].keys():
                        #     frame_label = self.label_coordinates[study_id][series_id][instance_number + 1]
                        #     severity_labels[i] = self.axial_labels[study_id][series_id][instance_number + 1]
                        # elif instance_number - 1 in self.label_coordinates[study_id][series_id].keys():
                        #     frame_label = self.label_coordinates[study_id][series_id][instance_number - 1]
                        #     severity_labels[i] = self.axial_labels[study_id][series_id][instance_number - 1]
                    else:
                        frame_label = np.zeros((10,3))
                        severity_labels[i] = np.array([-100, -100])
                    frame_labels[i] = frame_label

        image_paths = []
        for pos, instance_number in order_select_list:
            h, w = self.study_set[study_id][series_id][instance_number]
            # image_path = os.path.join(series_path, str(instance_number) + "_" + str(h) + "_" + str(w) + ".jpg")
            image_path = os.path.join(series_path, str(instance_number) + ".jpg")
            image_paths.append(image_path)

        total_features = len(image_paths)
        # print(total_images, step)
        series_stack = np.zeros((n_eval, self.image_sizes[0], self.image_sizes[1], 3)).astype("uint8")
        metadata_stack = np.zeros((n_eval, 11))
        if self.mode == "train":
            # is_flip = random.random() < 0.5
            for j in range(total_features):
                image_path = image_paths[j]
                instance_number = select_list[j]
                pil_img = Image.open(image_path).convert('RGB')
                img = np.asarray(pil_img)
                h, w, _ = img.shape
                frame_labels[j][:,1] /= w
                frame_labels[j][:,2] /= h
                # print(j, frame_labels[j])
                if np.sum(frame_labels[j][:,0]) == 0:
                    img = self.train_normal_transforms(image=img)["image"]
                else:    
                    img = self.train_instance_transforms(image=img)["image"]
            #     if is_flip:
            #         img = self.hflip_transform(image=img)["image"]
            #         temp = frame_labels[j][:][0]
            #         frame_labels[j][:][0] = frame_labels[j][:][1]
            #         frame_labels[j][:][1] = temp
                series_stack[j] = img
                # metadata_stack[j] = self.meta_data[study_id][series_id][instance_number]
            # if is_flip:
            #     # print(multiview_label.shape, frame_labels.shape)
            #     temp = multiview_label[:5, :]
            #     multiview_label[:5, :] = multiview_label[5:, :]
            #     multiview_label[5:, :] = temp
        else:
            for j in range(total_features):
                image_path = image_paths[j]
                instance_number = select_list[j]
                pil_img = Image.open(image_path).convert('RGB')
                img = np.asarray(pil_img)
                h, w, _ = img.shape
                frame_labels[j][:,1] /= w
                frame_labels[j][:,2] /= h
                img = self.test_transforms(image=img)["image"]
                series_stack[j] = img
                # metadata_stack[j] = self.meta_data[study_id][series_id][instance_number]
            
        series_stacks[:, :, :, :] = series_stack
        multiview_frame_labels = frame_labels
        
        any_severe_spinal_label = 0
        # print(multiview_frame_labels)
        # print(multiview_label.shape, type(multiview_label), type(any_severe_spinal_label), type(multiview_frame_labels), type(study_id), type(series_stacks["axial_t2"]), type(series_stacks["sagittal_t2"]), type(series_stacks["sagittal_t1"]))
        return series_stacks, metadata_stack, multiview_label, any_severe_spinal_label, multiview_frame_labels, severity_labels, study_id, series_id