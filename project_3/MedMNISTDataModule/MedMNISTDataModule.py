import pytorch_lightning as pl
import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader

class MedMNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_name: str = 'dermamnist',
                 batch_size: int = 64,
                 num_workers: int = 4,
                 download: bool = True):
        super().__init__()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = True

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
        # Assign train/val/test datasets for use in dataloaders
        self.train_dataset = self.DataClass(split='train', transform=self.transform, download=self.download, size=64, mmap_mode='r')
        self.val_dataset = self.DataClass(split='val', transform=self.transform, download=self.download, size=64, mmap_mode='r')
        self.test_dataset = self.DataClass(split='test', transform=self.transform, download=self.download, size=64, mmap_mode='r')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    @property
    def num_classes(self):
        return self.n_classes