import multiprocessing
import pytorch_lightning as pl
import torch
import os
from MedDCGAN.DCGAN import DCGAN

CLASS_ID = 2
CHECKPOINT_PATH = "DCGAN_model/class_2/model-epoch=159-val_loss=0.27.ckpt"
OUTPUT_PATH = "generated_img/DCGAN"

def main():
    output_path = os.path.join(OUTPUT_PATH, f"class_{CLASS_ID}")
    model = DCGAN.load_from_checkpoint(CHECKPOINT_PATH)

    model.generate(output_path, 10)



if __name__ == "__main__":
    multiprocessing.freeze_support()
    model = main()