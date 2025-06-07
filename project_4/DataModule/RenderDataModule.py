import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .RenderDataset import RenderDataset

class RenderDataModule(pl.LightningDataModule):
    def __init__(self, input_dir, batch_size, num_workers=4):
        super().__init__()
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        full_dataset = RenderDataset(self.input_dir)
        total_len = len(full_dataset)
        train_size = int(total_len * 0.8)
        val_size = int(total_len * 0.1)
        test_size = total_len - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          shuffle=False)