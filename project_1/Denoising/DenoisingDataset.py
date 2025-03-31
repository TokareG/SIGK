from torchvision.io import read_image
from torch.utils.data import Dataset
import os

class DenoisingDataset(Dataset):
    def __init__(self, img_list, target_dir, input_dir, transform=None, target_transform=None):
        self.img_list = img_list
        self.target_dir = target_dir
        self.input_dir = input_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        input_path = os.path.join(self.input_dir, img_path)
        input_img = read_image(input_path).float() / 255.

        target_img_path = os.path.join(self.target_dir, img_path)
        target_img = read_image(target_img_path).float() / 255.
        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            target_img = self.target_transform(target_img)
        return input_img, target_img


