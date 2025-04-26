import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.img_dim = img_shape[0] * img_shape[1] * img_shape[2]  # <--- IMPORTANT

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, img):
        return self.model(img)