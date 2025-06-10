import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from .Dataset import Vimeo90KDataset

class Vimeo90KDataModule(pl.LightningDataModule):
    def __init__(self, root_dir='data/vimeo_triplet', batch_size=8, num_workers=4, split_ratio=0.9):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio

        self.transform = T.Compose([
            T.ToTensor()
        ])

    def setup(self, stage=None):
        full_dataset = Vimeo90KDataset(
            root_dir=self.root_dir,
            list_file='tri_trainlist.txt',
            transform=self.transform
        )
        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
