from torch import nn


class DnCNN(nn.Module):

    def __init__(self, img_channels=3, kernel_size=3, n_channels=64, depth = 17):
        super(DnCNN, self).__init__()
        kernel_size = kernel_size
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=img_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=img_channels, kernel_size=kernel_size, padding=padding))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out