import os
import random

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from UNET_inpainting.InpaintingDataset import InpaintingDataset
import torchvision.transforms as transforms

class InpaintingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, target_dir, input_dir):
        super().__init__()
        self.batch_size = batch_size
        self.target_train_dir = os.path.join(target_dir, "train")
        self.target_val_dir = os.path.join(target_dir, "valid")
        self.target_test_dir = os.path.join(target_dir, "test")
        self.input_train_dir = os.path.join(input_dir, "train")
        self.input_val_dir = os.path.join(input_dir, "valid")
        self.input_test_dir = os.path.join(input_dir, "test")


    def prepare_data(self):

        self.train_list = [f for f in os.listdir(self.target_train_dir) if os.path.isfile(os.path.join(self.target_train_dir, f))]
        self.val_list = [f for f in os.listdir(self.target_val_dir) if os.path.isfile(os.path.join(self.target_val_dir, f))]
        self.test_list = [f for f in os.listdir(self.target_test_dir) if os.path.isfile(os.path.join(self.target_test_dir, f))]

        random.shuffle(self.train_list)
        #random.shuffle(self.val_list)
        #random.shuffle(self.test_list)

        print(f"Train examples:\t{len(self.train_list)}\n"
              f"Valid examples:\t{len(self.val_list)}\n"
              f"Test examples:\t{len(self.test_list)}")

    def setup(self, stage=None):
        #normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.train_dataset = InpaintingDataset(self.train_list, self.target_train_dir, self.input_train_dir)
        self.validate_dataset = InpaintingDataset(self.val_list, self.target_val_dir, self.input_val_dir)
        self.test_dataset = InpaintingDataset(self.test_list, self.target_test_dir, self.input_test_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers = 4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.validate_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers=1)
