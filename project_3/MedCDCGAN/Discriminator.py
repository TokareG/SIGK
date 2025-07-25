import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes, embedding_dim=64, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.img_dim = img_shape[0] * img_shape[1] * img_shape[2]  # <--- IMPORTANT

        self.embedd_labels = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim),
            nn.Linear(embedding_dim, 3 * 64 * 64))

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, img, label):
        embedded_labels = self.embedd_labels(label).view(-1, 3, 64, 64)
        inputs = torch.cat((img, embedded_labels), dim=1)
        return self.model(inputs)