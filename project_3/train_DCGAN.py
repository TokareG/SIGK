import multiprocessing
import pytorch_lightning as pl
import torch

from MedMNISTDataModule.MedMNISTDataModule import MedMNISTDataModule
from MedDCGAN.DCGAN import DCGAN
from MedDCGAN.Callbacks import get_early_stopping, get_checkpoint_callback

CLASS_ID = None

def main():
    dm = MedMNISTDataModule(single_class=CLASS_ID, dataset_name='bloodmnist', batch_size=128)
    model = DCGAN()
    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        accelerator="auto",
        max_epochs=2000,
        callbacks=[get_checkpoint_callback(class_id=CLASS_ID), get_early_stopping()]
    )

    trainer.fit(model=model, datamodule=dm)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Train the model
    model = main()