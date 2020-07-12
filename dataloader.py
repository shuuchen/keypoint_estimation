import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms
#from image_augment_pairs import *

class SumitomoCADDS(Dataset):
    def __init__(self, file_path, train=True, augment=False, val=False,test=False):
        self.file_path = file_path
        self.augment = augment
        self.train = train
        self.val = val
        self.test = test
        
        with open(file_path) as f:
            self.list = f.readlines()
        f.close()
        
        self.list = [l[:-1] for l in self.list]
        if self.test:
            self.data_dir = Path('./test/21_masked_png')
        else:
            self.data_dir = Path('/mnt/data')

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):

        if self.test:
            image = Image.open(self.data_dir / f'{self.list[index]}.png')
            image = image.resize((256, 256))
            return F.to_tensor(image), self.list[index]
        else: 
            image = torch.load(self.data_dir / f'{self.list[index]}_im.pt')
            label = torch.load(self.data_dir / f'{self.list[index]}_lb.pt')

        if self.train and self.augment:
            
          # random rotations
          random_rotation(image, label)

          # random h-flips
          horizontal_flip(image, label)

          # random v-flips
          vertical_flip(image, label)

          # random crops
          #scale_augmentation(image, label)
        if self.val:
            return image, label, self.list[index]
        else:
            return image, label
