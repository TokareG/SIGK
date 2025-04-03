import copy
from torchvision.io import read_image
from torch.utils.data import Dataset
import os

class DenoisingDataset(Dataset):
    def __init__(self, img_list, input_dir, transform=None, target_transform=None):
        self.img_list = img_list
        self.input_dir = input_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        input_path = os.path.join(self.input_dir, img_path)
        noised_img = read_image(input_path).float() / 255.
        real_image = copy.deepcopy(noised_img)
        if self.transform:
            input_img = self.transform(noised_img)

        return noised_img, real_image


