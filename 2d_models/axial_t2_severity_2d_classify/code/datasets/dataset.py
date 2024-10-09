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


class ImageFolder(data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, default_configs, mode):
        super().__init__()
        self.data_path = "/ssd/kaggle/RSNA_2024/rsna_2024/crop_data/axial_t2/images"
        self.df = df.reset_index(drop=True)
        self.filepaths = df["file_path"].values
        self.labels = df["label"].values
        self.mode = mode

        self.image_size = default_configs["image_size"]
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.8, 1.0), p=1),
            # A.Resize(height=self.image_size, width=self.image_size, p=1),
            A.HorizontalFlip(p=0.5),
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
            # A.Resize(height=self.image_size, width=self.image_size, p=1),
        ])

        self.test_transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1),
        ])

        self.mode = mode
        
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.filepaths[index])
        severity_label = self.labels[index] 


        pil_img = Image.open(image_path).convert('RGB')
        img = np.asarray(pil_img)
        if self.mode == 'train':
            img = self.train_transform(image=img)["image"]
        else:
            img = self.test_transform(image=img)["image"]
        
        img = img.transpose(2, 0, 1)
       
        return img, severity_label
