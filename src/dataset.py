# System imports
import os

# Library imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import pandas as pd
import glob

# Our imports
from configs import *


class AgeDataset(Dataset):

    def __init__(self, data_frame):
        
        self.data_frame = data_frame

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        
        f = self.data_frame.iloc[index].squeeze()
        file = os.path.join(PATH_TO_FOLDER, f.file)
        age = f.age
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)

        return img, age

    def preprocess_image(self, img):

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = torch.tensor(img)
        img = self.normalize(img/255)

        return img[None]

    def collate_fn(self, batch):

        imgs, ages = [], []

        for img, age in batch:
            img = self.preprocess_image(img)
            imgs.append(img)

            ages.append(float(int(age)/80)) 

        ages = [ torch.tensor(x).to(DEVICE) for x in ages ]
        imgs = torch.cat(imgs).to(DEVICE)

        return imgs, ages