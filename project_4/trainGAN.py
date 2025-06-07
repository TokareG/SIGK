import multiprocessing
import torch
torch.set_float32_matmul_precision('medium')

from DataModule.RenderDataModule import RenderDataModule
from GAN.GAN import GAN
import pytorch_lightning as pl

def main():
    dm = RenderDataModule(input_dir='./dataset', batch_size=128, num_workers=6)
    model = GAN(img_shape=(3,128,128), warmup_epochs=250, D_acc_threshold=0.8)
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=2000,
        accelerator='auto',
        log_every_n_steps=1,
    )

    trainer.fit(model=model, datamodule=dm)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = main()