from torchvision.io import read_image
from torch.utils.data import Dataset
import os

class InpaintingDataset(Dataset):
    def __init__(self, img_list, target_img_dir, input_img_dir, transform=None, target_transform=None):
        self.img_list = img_list
        self.target_img_dir = target_img_dir
        self.input_img_dir = input_img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        input_img_path = os.path.join(self.input_img_dir, img_name)
        input_img = read_image(input_img_path)
        input_img = input_img.float()
        mask = input_img[3:4, :, :]
        #mask = 1 - mask
        mask = mask.expand(3, -1, -1)
        input_img = input_img[:3, :, :]
        input_img = input_img / 255.0
        #image = F.interpolate(image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)

        target_img_path = os.path.join(self.target_img_dir, img_name)
        target_img = read_image(target_img_path)
        target_img = target_img.float() / 255.0
        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            target_img = self.target_transform(target_img)
        return input_img, mask, target_img
