import multiprocessing
import pytorch_lightning as pl

from MedMNISTDataModule.MedMNISTDataModule import MedMNISTDataModule
from MNISTClassifier.MedClassifier import MedClassifier
from MNISTClassifier.MedModel import MedModel
from MNISTClassifier.Callbacks import get_early_stopping, get_checkpoint_callback
from generate_DCGAN import CHECKPOINT_PATH

CUSTOM_DATASET_DIR = "generated_dataset"
#CUSTOM_DATASET_DIR = None

MIXED_DATASET = True
#MIXED_DATASET = False

MODEL_SUBFOLDER = "mixed"
CHECKPOINT_PATH = f"Classifier_model/{MODEL_SUBFOLDER}/model-epoch=04-val_loss=0.03.ckpt"

def main():
    dm = MedMNISTDataModule(dataset_name='bloodmnist', batch_size=128, custom_data_dir=CUSTOM_DATASET_DIR,
                            mixed_dataset=MIXED_DATASET)

    classifier = MedClassifier.load_from_checkpoint(CHECKPOINT_PATH, model=MedModel(dm.num_classes))



    trainer = pl.Trainer(check_val_every_n_epoch=1, num_sanity_val_steps=0, accelerator="auto", max_epochs=1000)


    trainer.test(model=classifier, datamodule=dm)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()