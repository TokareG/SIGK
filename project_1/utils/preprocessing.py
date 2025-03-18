import os
import cv2
import numpy as np

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


dataset_dir = 'C:/Users/jakub/Desktop/DIV2K_valid_HR'
output_dir = 'C:/Users/jakub/Desktop/DIV2K_output'
resize_dataset(100, dataset_directory=dataset_dir, output_directory=output_dir)