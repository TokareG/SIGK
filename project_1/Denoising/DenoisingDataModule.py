import os
import random
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from DenoisingDataset import DenoisingDataset


class DenoisingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, input_dir, sigma = 0.1):

        super().__init__()
        self.batch_size = batch_size
        self.sigma = sigma
        self.input_train_dir = os.path.join(input_dir, "train")
        self.input_valid_dir = os.path.join(input_dir, "valid")
        self.input_test_dir = os.path.join(input_dir, "test")

    def prepare_data(self):
        self.train_list = [f for f in os.listdir(self.input_train_dir) if
                           os.path.isfile(os.path.join(self.input_train_dir, f))]
        self.valid_list = [f for f in os.listdir(self.input_valid_dir) if
                         os.path.isfile(os.path.join(self.input_valid_dir, f))]
        self.test_list = [f for f in os.listdir(self.input_test_dir) if
                          os.path.isfile(os.path.join(self.input_test_dir, f))]

        random.shuffle(self.train_list)
        #random.shuffle(self.valid_list)
        #random.shuffle(self.test_list)

        print(f"Train examples:\t{len(self.train_list)}\n"
              f"Valid examples:\t{len(self.valid_list)}\n"
              f"Test examples:\t{len(self.test_list)}")

    def setup(self, stage=None):
        self.train_dataset = DenoisingDataset(self.train_list, self.input_train_dir, transform=v2.GaussianNoise(mean=0, sigma=self.sigma, clip=True))
        self.valid_dataset = DenoisingDataset(self.valid_list, self.input_valid_dir, transform=v2.GaussianNoise(mean=0, sigma=self.sigma, clip=True))
        self.test_dataset = DenoisingDataset(self.test_list, self.input_test_dir, transform=v2.GaussianNoise(mean=0, sigma=self.sigma, clip=True))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
