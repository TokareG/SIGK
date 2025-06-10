import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class Vimeo90KDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = root_dir  # e.g., 'data/vimeo_triplet'
        self.transform = transform
        list_path = os.path.join(root_dir, list_file)
        with open(list_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # e.g., sample = '00001/0266'
        img_dir = os.path.join(self.root_dir, 'sequences', sample)
        im1 = Image.open(os.path.join(img_dir, 'im1.png')).convert("RGB")
        im2 = Image.open(os.path.join(img_dir, 'im2.png')).convert("RGB")
        im3 = Image.open(os.path.join(img_dir, 'im3.png')).convert("RGB")

        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            im3 = self.transform(im3)

        return im1, im3, im2
