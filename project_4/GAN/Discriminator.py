from torch import nn
import torch

class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=10, ndf=64):
        super().__init__()
        self.img_shape = img_shape
        self.c_dim = c_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.img_shape[0], out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
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

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()

            #nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            #nn.Flatten(),
            #nn.Sigmoid()
        )

        # Calculate flattened feature size after conv layers
        dummy_input = torch.zeros(1, *img_shape)
        conv_output_dim = self.model(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim + c_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, c):
        features = self.model(img)
        combined = torch.cat((features, c), dim=1)
        output = self.fc(combined)
        return output