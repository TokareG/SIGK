import multiprocessing
import torch
torch.set_float32_matmul_precision('medium')

from DataModule.RenderDataModule import RenderDataModule
from CNN.GeneratorModule import GeneratorModule
from CNN.Callbacks import get_early_stopping, get_checkpoint_callback
import pytorch_lightning as pl

def main():
    dm = RenderDataModule(input_dir='./dataset', batch_size=128, num_workers=6)
    model = GeneratorModule(img_shape=(3,128,128), c_dim=10, lr=0.0025)
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=2000,
        accelerator='auto',
        log_every_n_steps=1,
        callbacks=[get_checkpoint_callback(), get_early_stopping()]
    )

    trainer.fit(model=model, datamodule=dm)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = main()