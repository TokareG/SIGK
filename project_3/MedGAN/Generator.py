import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        """
        Args:
            latent_dim (int): Size of random noise input
            img_shape (tuple): (channels, height, width)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.img_dim = img_shape[0] * img_shape[1] * img_shape[2]

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, self.img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), *self.img_shape)