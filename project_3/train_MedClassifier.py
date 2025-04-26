import multiprocessing
import pytorch_lightning as pl

from MedMNISTDataModule.MedMNISTDataModule import MedMNISTDataModule
from MNISTClassifier.MedClassifier import MedClassifier
from MNISTClassifier.MedModel import MedModel
from MNISTClassifier.Callbacks import get_early_stopping, get_checkpoint_callback

def main():
    dm = MedMNISTDataModule(dataset_name='bloodmnist', batch_size=128)

    model = MedModel(dm.num_classes)
    classifier = MedClassifier(model)

    trainer = pl.Trainer(check_val_every_n_epoch=1, num_sanity_val_steps=0, accelerator="auto", max_epochs=1000,
                         callbacks=[get_checkpoint_callback(), get_early_stopping()])

    trainer.fit(model=classifier, datamodule=dm)

    trainer.test(model=classifier, datamodule=dm)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()