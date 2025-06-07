import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch


class RenderDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.img_dir = os.path.join(input_dir, 'img')
        self.img_list = os.listdir(self.img_dir)
        self.transform = transform

        self.scene_params_df = pd.read_csv(os.path.join(input_dir, 'scene_params.csv'), header=None)
        self.scene_params_df.columns = [
            "id",
            "model_x", "model_y", "model_z",
            "diffuse_r", "diffuse_g", "diffuse_b",
            "shininess",
            "light_x", "light_y", "light_z",
        ]
        self.scene_params_df['id'] = self.scene_params_df['id'].astype(int)
        self.scene_params_df = self.scene_params_df.set_index('id')
        self.normalize_scene_params()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        input_img = read_image(img_path).float() / 255.
        input_img = input_img * 2 - 1
        input_img = input_img[:3]
        scene_params = torch.tensor(self.scene_params_df.loc[idx].values.astype("float32"))
        if self.transform:
            input_img = self.transform(input_img)

        return input_img, scene_params

    def _min_max_normalize(self, val, min_val, max_val, eps=1e-8):
        return (val - min_val) / (max_val - min_val + eps)

    def normalize_scene_params(self):
        for col in ["model_x", "model_y"]:
            self.scene_params_df[col] = self._min_max_normalize(self.scene_params_df[col], -12., 8.)
        for col in ["light_x", "light_y"]:
            self.scene_params_df[col] = self._min_max_normalize(self.scene_params_df[col], -25., 15.)
        for col in ["model_z"]:
            self.scene_params_df[col] = self._min_max_normalize(self.scene_params_df[col], -22., -2.)
        for col in ["light_z"]:
            self.scene_params_df[col] =self._min_max_normalize(self.scene_params_df[col], -22., 0.)
        #for col in ["model_x", "model_y", "model_z","light_x", "light_y", "light_z"]:
        #    self.scene_params_df[col] = self._min_max_normalize(self.scene_params_df[col], -20., 20.)
        self.scene_params_df["shininess"] = self._min_max_normalize(self.scene_params_df["shininess"], 3., 20.)
