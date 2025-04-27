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
            nn.ConvTranspose2d(in_channels=self.latent_dim, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=self.img_shape[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

        )

    def forward(self, z):
        return self.model(z)