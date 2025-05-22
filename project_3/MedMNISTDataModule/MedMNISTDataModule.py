import pytorch_lightning as pl
import medmnist
from medmnist import INFO
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import os
import numpy as np

class ImageFolderWithReshapedLabels(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, torch.tensor([int(y)], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

class ConsistentLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]

        if isinstance(y, np.ndarray):
            y = int(y.item())
        elif isinstance(y, torch.Tensor) and y.ndim == 0:
            y = y.item()

        return x, torch.tensor([int(y)], dtype=torch.long)  # shape (1,)

    def __len__(self):
        return len(self.dataset)

class MedMNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 single_class = None,
                 dataset_name: str = 'dermamnist',
                 mixed_dataset: bool = False,
                 custom_data_dir: str = None,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 download: bool = True):
        super().__init__()

        if mixed_dataset and (custom_data_dir is None):
            raise ValueError("Mixed_dataset can not be true while no custom_data_dir has been provided!!!")

        self.single_class = single_class
        self.dataset_name = dataset_name
        self.mixed_dataset= mixed_dataset
        self.custom_data_dir = custom_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download



        # Load dataset info
        self.info = INFO[self.dataset_name]
        self.DataClass = getattr(medmnist, self.info['python_class'])
        self.n_classes = len(self.info['label'])

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    def setup(self, stage=None):

        if self.custom_data_dir is None or self.mixed_dataset:
            print("Reading MedMnist\n"
                  f"Dataset: \n\t{self.dataset_name}\n"
                  f"Number of classes:\n\t{self.n_classes}\n"
                  f"Classes:\n" + '\n'.join(f"\t- {key}: {label}" for key, label in self.info['label'].items()))
            self.train_dataset = self.DataClass(split='train', transform=self.transform, download=self.download, size=64, mmap_mode='r')
            self.val_dataset = self.DataClass(split='val', transform=self.transform, download=self.download, size=64, mmap_mode='r')
            self.test_dataset = self.DataClass(split='test', transform=self.transform, download=self.download, size=64, mmap_mode='r')

        if self.custom_data_dir is not None:
            print(f"Reading custom dataset from: {self.custom_data_dir}")

            train_path = os.path.join(self.custom_data_dir, "train")
            val_path = os.path.join(self.custom_data_dir, "val")
            test_path = os.path.join(self.custom_data_dir, "test")

            custom_train_dataset = ImageFolderWithReshapedLabels(ImageFolder(train_path, transform=self.transform))
            custom_val_dataset = ImageFolderWithReshapedLabels(ImageFolder(val_path, transform=self.transform))
            custom_test_dataset = ImageFolderWithReshapedLabels(ImageFolder(test_path, transform=self.transform))

        if self.custom_data_dir is not None:
            if not self.mixed_dataset:
                self.train_dataset = ConsistentLabelWrapper(custom_train_dataset)
                self.val_dataset = ConsistentLabelWrapper(custom_val_dataset)
                self.test_dataset = ConsistentLabelWrapper(custom_test_dataset)
            else:
                self.train_dataset = ConsistentLabelWrapper(ConcatDataset([self.train_dataset, custom_train_dataset]))
                self.val_dataset = ConsistentLabelWrapper(ConcatDataset([self.val_dataset, custom_val_dataset]))
                self.test_dataset = ConsistentLabelWrapper(ConcatDataset([self.test_dataset, custom_test_dataset]))


        if self.single_class is not None:
            indices = [i for i, (_, label) in enumerate(self.train_dataset) if label.item() == self.single_class]
            self.train_dataset = Subset(self.train_dataset, indices)
            indices = [i for i, (_, label) in enumerate(self.val_dataset) if label.item() == self.single_class]
            self.val_dataset = Subset(self.val_dataset, indices)
            indices = [i for i, (_, label) in enumerate(self.test_dataset) if label.item() == self.single_class]
            self.test_dataset = Subset(self.test_dataset, indices)

        print(f"train size:\n{len(self.train_dataset)}\n"
              f"val size:\n{len(self.val_dataset)}\n"
              f"test size:\n{len(self.test_dataset)}\n")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    @property
    def num_classes(self):
        return self.n_classes