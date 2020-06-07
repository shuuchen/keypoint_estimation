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
    def __init__(self, file_path, train=True, augment=False):
        self.file_path = file_path
        self.augment = augment
        self.train = train
        
        with open(file_path) as f:
            self.list = f.readlines()
        f.close()
        
        self.list = [l[:-1] for l in self.list]
        
        self.img_dir = Path('../../data/sumitomo_cad/00_img')
        self.label_dir = Path('../../data/sumitomo_cad/40_spheres_center')

    def __len__(self):
        return len(self.list)

    # read img
    def _read_img(self, img_file):
        arr = np.load(img_file)
        arr = np.clip(arr, 0, 1) 
        im = Image.fromarray((arr * 255).astype(np.uint8))
        im = im.resize((1024, 1024))
        return im

    # label rendering
    def _json2label(self, json_file, output_size=1024):
        sphere = json.load(open(json_file, 'r'))
        hms = np.zeros((output_size, output_size), dtype=np.float64)
        sigma = output_size / 64
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        for p in sphere['keypoints'].values():
            if p['center_h_w'] == [None, None]:
                continue
            x, y = p['center_h_w']
            ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
            br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)
            c,d = max(0, -ul[0]), min(br[0], output_size) - ul[0]
            a,b = max(0, -ul[1]), min(br[1], output_size) - ul[1]
            cc,dd = max(0, ul[0]), min(br[0], output_size)
            aa,bb = max(0, ul[1]), min(br[1], output_size)
            hms[aa:bb,cc:dd] = np.maximum(hms[aa:bb,cc:dd], g[a:b,c:d])
        return Image.fromarray((hms * 255).astype(np.uint8))

    # random crop
    def _random_crop16(self, img, label):
        i = np.random.randint(0, 16)
        r, c = i // 4, i % 4
        crop_img = img.crop((c * 256, r * 256, (c+1) * 256, (r+1) * 256))
        crop_label = label.crop((c * 256, r * 256, (c+1) * 256, (r+1) * 256))
        return crop_img, crop_label

    def __getitem__(self, index):
        
        image = self._read_img(self.img_dir / f'{self.list[index]}.npy')
        label = self._json2label(self.label_dir / f'{self.list[index]}.json')

        image, label = self._random_crop16(image, label)

        if self.train and self.augment:
            
          # random rotations
          random_rotation(image, label)

          # random h-flips
          horizontal_flip(image, label)

          # random v-flips
          vertical_flip(image, label)

          # random crops
          #scale_augmentation(image, label)

        return F.to_tensor(image), F.to_tensor(label)
