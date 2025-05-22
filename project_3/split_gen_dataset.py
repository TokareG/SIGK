from utils.dataset_split import split_and_copy_dataset

split_and_copy_dataset("generated_img/DCGAN", "generated_dataset", 0.7, 0.1, 0.2)