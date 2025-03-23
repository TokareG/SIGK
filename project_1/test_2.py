import multiprocessing

import cv2
import torch
import pytorch_lightning as pl
from torchvision.transforms.v2.functional import resize_bounding_boxes

from UNET_inpainting.InpaintingDataModule import InpaintingDataModule
# from UNET_inpainting.InpaintingClassifier import InpaintingClassifier
# from UNET_inpainting.UNetPartialConv import UNetPartialConv
import os
import torchvision.io as tvio
from torchvision.io import read_image

from utils.metrics import *
from utils.preprocessing import *

def main():
    print("GPU Available:", torch.cuda.is_available())
    target_dataset_dir = "TARGET_DATASET_PATH"
    input_dataset_dir = "INPUT_DATASET_PATH"

    dm = InpaintingDataModule(2, target_dataset_dir, input_dataset_dir)

    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    input, mask, target = batch

    print(mask.shape)
    print(input.shape)
    print(target.shape)
    print(mask.max())
    print(input.max())
    print(target.max())

    dir_path = "OUTPUT_PATH"
    test_save = 0

    # if target.dtype != torch.uint8:
    #     input = (target * 255).clamp(0, 255).to(torch.uint8)
    # input = input.cpu()
    # for i, image in enumerate(input):
    #     img_path = os.path.join(dir_path, f"image_{test_save}.png")
    #     test_save += 1
    #     tvio.write_png(image, img_path)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

    # target_dataset_dir = "TARGET_DATASET_PATH"
    # input_dataset_dir = "INPUT_DATASET_PATH"
    # dir_path = "OUTPUT_PATH"
    # dataset_hole_mask(target_dataset_dir, input_dataset_dir, True, [5,5], 3, [5,5])