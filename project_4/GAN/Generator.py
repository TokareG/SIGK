from torch import nn


class Generator(nn.Module):
    def __init__(self, c_dim, img_channels=3, ngf=64):
        super().__init__()
        self.c_dim = c_dim
        self.img_channels = img_channels
        self.ngf = ngf

        self.fc = nn.Sequential(
            nn.Linear(c_dim, ngf * 8),
            nn.ReLU(True),

            nn.Linear(ngf * 8, ngf * 8),
            nn.ReLU(True),

            nn.Linear(ngf * 8, ngf * 8 * 8),
            nn.ReLU(True),
        )

        self.model = nn.Sequential(
            #512x8x8 → 256x16x16
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 256x16x16 → 128x32x32
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 128x32x32 → 64x64x64
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 64x64x64 → 4x128x128
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ngf, img_channels, 3, 1, 1, bias=False),
            nn.Tanh()

            # nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(inplace=True),
            #
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(inplace=True),
            #
            # nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(inplace=True),
            #
            # nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(inplace=True),
            #
            # nn.ConvTranspose2d(ngf, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.Tanh()
        )

    def forward(self, noice, c):
        x = self.fc(c)
        x = x.view(x.size(0), self.ngf, 8, 8)
        #return (self.model(x) + 1) * 0.5
        return self.model(x)