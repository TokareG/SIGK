import multiprocessing
import torch
import pytorch_lightning as pl
from UNET_inpainting.InpaintingDataModule import InpaintingDataModule
# from UNET_inpainting.InpaintingClassifier import InpaintingClassifier
# from UNET_inpainting.UNetPartialConv import UNetPartialConv
import os
import torchvision.io as tvio


def main():
    print("GPU Available:", torch.cuda.is_available())
    target_dataset_dir = "TARGET_PATH"
    input_dataset_dir = "INPUT_PATH"

    dm = InpaintingDataModule(1, target_dataset_dir, input_dataset_dir)

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

    dir_path = "DIRECTORY_PATH"
    test_save = 0

    if target.dtype != torch.uint8:
        input = (target * 255).clamp(0, 255).to(torch.uint8)
    input = input.cpu()
    for i, image in enumerate(input):
        img_path = os.path.join(dir_path, f"image_{test_save}.png")
        test_save += 1
        tvio.write_png(image, img_path)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()