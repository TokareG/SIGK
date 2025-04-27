import multiprocessing
import pytorch_lightning as pl
import torch

from MedMNISTDataModule.MedMNISTDataModule import MedMNISTDataModule
from MNISTClassifier.MedClassifier import MedClassifier
from MedDCGAN.DCGAN import DCGAN
from MNISTClassifier.Callbacks import get_early_stopping, get_checkpoint_callback

def main():
    dm = MedMNISTDataModule(dataset_name='bloodmnist', batch_size=64)
    model = DCGAN(dm.num_classes)
    trainer = pl.Trainer(
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        accelerator="auto",
        max_epochs=10,
    )

    trainer.fit(model=model, datamodule=dm)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Train the model
    model = main()