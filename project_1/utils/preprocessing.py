import os
import cv2
import numpy as np
import random
import shutil


def resize_dataset(dst_size, dataset_directory, output_directory):
    if not (os.path.isdir(dataset_directory) and os.path.isdir(output_directory)):
        raise ValueError("Both dataset and output directories must be valid.")

    files_list = [f for f in os.listdir(dataset_directory) if os.path.isfile(os.path.join(dataset_directory, f))]
    if not files_list:
        raise ValueError("Dataset must contain at least one file.")

    for file in files_list:
        file_path = os.path.join(dataset_directory, file)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        min_dim = min(img.shape[:2])
        max_dim_idx = np.argmax(img.shape[:2])
        px_shift = (max(img.shape[:2]) - min_dim) // 2

        cropped_img = img[px_shift:px_shift+min_dim, :] if max_dim_idx == 0 else img[:, px_shift:px_shift+min_dim]
        resized_img = cv2.resize(cropped_img, (dst_size, dst_size))

        cv2.imwrite(os.path.join(output_directory, file), resized_img)

def random_test_split(downloaded_dir, output_train_dir, output_val_dir, test_val_count):
    if not (os.path.isdir(downloaded_dir) and os.path.isdir(output_val_dir) and os.path.isdir(output_train_dir)):
        raise ValueError("Both downloaded and output directories must be valid.")

    downloaded_train_list = [f for f in os.listdir(downloaded_dir) if os.path.isfile(os.path.join(downloaded_dir, f))]
    if len(downloaded_train_list) != (test_val_count[0] + test_val_count[1]):
        raise ValueError("Train dataset does not contain correct number of examples")

    train_list = random.sample(downloaded_train_list, test_val_count[0])
    val_list = list(set(downloaded_train_list) - set(train_list))

    for file in train_list:
        shutil.copy(os.path.join(downloaded_dir, file), os.path.join(output_train_dir, file))
    for file in val_list:
        shutil.copy(os.path.join(downloaded_dir, file), os.path.join(output_val_dir))


def copy_all_files(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)  # Create destination folder if it doesn't exist

    for file_name in os.listdir(src_folder):
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)

        if os.path.isfile(src_path):  # Copy only files, not subdirectories
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata




def create_folder_structure(parent_dir):
    if not os.path.isdir(parent_dir):
        raise ValueError("Given directory must be valid.")

    if os.listdir(parent_dir):
        raise ValueError("Given directory is not empty.")

    os.makedirs(os.path.join(parent_dir, "ori/train"))
    os.makedirs(os.path.join(parent_dir, "ori/valid"))
    os.makedirs(os.path.join(parent_dir, "ori/test"))

    os.makedirs(os.path.join(parent_dir, "rescale_256/train"))
    os.makedirs(os.path.join(parent_dir, "rescale_256/valid"))
    os.makedirs(os.path.join(parent_dir, "rescale_256/test"))

    os.makedirs(os.path.join(parent_dir, "hole_32/train"))
    os.makedirs(os.path.join(parent_dir, "hole_32/valid"))
    os.makedirs(os.path.join(parent_dir, "hole_32/test"))

    os.makedirs(os.path.join(parent_dir, "hole_3/train"))
    os.makedirs(os.path.join(parent_dir, "hole_3/valid"))
    os.makedirs(os.path.join(parent_dir, "hole_3/test"))


def create_hole_mask(mask_size, hole_size, hole_count, border_width):
    mask = np.ones(mask_size, dtype=int)
    for i in range(hole_count):
        x = random.randint(border_width[0], mask_size[0] - hole_size[0] - border_width[0])
        y = random.randint(border_width[1], mask_size[1] - hole_size[1] - border_width[1])
        mask[x:x+hole_size[0], y:y+hole_size[1]] = 0
    return mask

def png_hole_mask(input_path, output_path, keep_mask, hole_size, hole_count, border_width):
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[-1] != 3:
        raise ValueError("Input image must be a 3-channel (RGB) PNG.")

    mask = create_hole_mask(image.shape[:2], hole_size, hole_count, border_width)

    # Set RGB values to zero where the mask is 1
    image[mask == 0] = [0, 0, 0]
    if keep_mask:
        mask_normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = np.concatenate((image, np.expand_dims(mask_normalized, axis=-1)), axis=2)

    # Save the output image
    cv2.imwrite(output_path, image)

def dataset_hole_mask(dataset_directory, output_directory, keep_mask, hole_size, hole_count, border_width):
    if not (os.path.isdir(dataset_directory) and os.path.isdir(output_directory)):
        raise ValueError("Both dataset and output directories must be valid.")

    files_list = [f for f in os.listdir(dataset_directory) if os.path.isfile(os.path.join(dataset_directory, f))]
    if not files_list:
        raise ValueError("Dataset must contain at least one file.")

    for file in files_list:
        input_path = os.path.join(dataset_directory, file)
        output_path = os.path.join(output_directory, file)
        png_hole_mask(input_path, output_path, keep_mask, hole_size, hole_count, border_width)






