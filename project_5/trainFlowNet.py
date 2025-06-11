import multiprocessing
import torch
torch.set_float32_matmul_precision('medium')

from DataModule_toflow.DataModule import Vimeo90KDataModule
from FlowNet.InterpolationModel import MyInterpolationModel

from FlowNet.Callbacks import get_early_stopping, get_checkpoint_callback
import pytorch_lightning as pl

def main():
    dm = Vimeo90KDataModule(root_dir='dataset_toflow/vimeo_triplet', batch_size=8, num_workers=4, split_ratio=0.9)
    model = MyInterpolationModel(lr=1e-5)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        max_epochs=2000,
        accelerator='auto',
        log_every_n_steps=1,
        callbacks=[get_checkpoint_callback(), get_early_stopping()]
    )

    trainer.fit(model=model, datamodule=dm)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = main()