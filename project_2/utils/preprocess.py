import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
import numpy as np
import random
import shutil
import torch

def random_split(input_dir, output_dir, test_val_count):
    if not (os.path.isdir(input_dir) and os.path.isdir(output_dir)):
        raise ValueError("Both input and output directories must be valid.")

    input_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if len(input_list) < (test_val_count[0] + test_val_count[1]):
        raise ValueError("Input dataset does not contain sufficient number of examples")

    train_list = random.sample(input_list, len(input_list) - test_val_count[0] - test_val_count[0])
    train_val_list = list(set(input_list) - set(train_list))
    test_list = random.sample(train_val_list, test_val_count[0])
    val_list = list(set(train_val_list) - set(test_list))

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    val_dir = os.path.join(output_dir, "valid")

    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(test_dir, exist_ok=False)
    os.makedirs(val_dir, exist_ok=False)

    sets = [
        (train_list, train_dir),
        (test_list, test_dir),
        (val_list, val_dir),
    ]

    for files, dir in sets:
        for file in files:
            img = rescale_img(os.path.join(input_dir, file), 0.5)
            cv2.imwrite(os.path.join(dir, file), img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

def rescale_img(path, scale):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img