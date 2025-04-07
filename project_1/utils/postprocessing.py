import os
import cv2
import numpy as np
from skimage.restoration import denoise_bilateral

def bilateral(input_path, output_path):
    assert os.path.exists(input_path)
    assert os.path.exists(output_path)
    input_filenames = os.listdir(input_path)
    for input_filename in input_filenames:
        img = cv2.imread(os.path.join(input_path, input_filename), cv2.IMREAD_UNCHANGED) / 255.
        bilateral_img = denoise_bilateral(img, sigma_color=0.05, channel_axis=-1)
        bilateral_img = (bilateral_img * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, input_filename), bilateral_img)

def inpaint_telea(input_path, output_path):
    assert os.path.exists(input_path)
    assert os.path.exists(output_path)

    input_filenames = os.listdir(input_path)
    for filename in input_filenames:
        img = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_UNCHANGED)
        mask = img[:, :, 3]
        mask = 255 - mask
        img = img[:, :, 0:3]
        inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(output_path, filename), inpainted_img)
