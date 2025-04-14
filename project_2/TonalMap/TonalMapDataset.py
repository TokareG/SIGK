import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2
from numpy import ndarray
import numpy as np
import os

class TonalMapDataset(Dataset):
    def __init__(self, img_list, input_dir, transform=None):
        self.img_list = img_list
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_img = cv2.imread(filename=input_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        input_img = np.maximum(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), 0.0)

        #image = F.interpolate(image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        if self.transform:
            input_img = self.transform(input_img)
        input_img = torch.from_numpy(input_img)
        return input_img.permute(2, 0, 1)
