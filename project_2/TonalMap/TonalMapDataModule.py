import os
import random

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from TonalMap.TonalMapDataset import TonalMapDataset
from TonalMap.HDRNormalize import HDRNormalize
import torchvision.transforms as transforms

class TonalMapDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, input_dir):
        super().__init__()
        self.batch_size = batch_size
        self.input_train_dir = os.path.join(input_dir, "train")
        self.input_val_dir = os.path.join(input_dir, "valid")
        self.input_test_dir = os.path.join(input_dir, "test")


    def prepare_data(self):

        self.train_list = [f for f in os.listdir(self.input_train_dir) if os.path.isfile(os.path.join(self.input_train_dir, f))]
        self.val_list = [f for f in os.listdir(self.input_val_dir) if os.path.isfile(os.path.join(self.input_val_dir, f))]
        self.test_list = [f for f in os.listdir(self.input_test_dir) if os.path.isfile(os.path.join(self.input_test_dir, f))]

        random.shuffle(self.train_list)
        #random.shuffle(self.val_list)
        #random.shuffle(self.test_list)

        print(f"Train examples:\t{len(self.train_list)}\n"
              f"Valid examples:\t{len(self.val_list)}\n"
              f"Test examples:\t{len(self.test_list)}")

    def setup(self, stage=None):
        #normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([HDRNormalize(scale=0.5)])
        self.train_dataset = TonalMapDataset(self.train_list, self.input_train_dir, transform=transform)
        self.validate_dataset = TonalMapDataset(self.val_list, self.input_val_dir, transform=transform)
        self.test_dataset = TonalMapDataset(self.test_list, self.input_test_dir, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers = 4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.validate_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers=1)
