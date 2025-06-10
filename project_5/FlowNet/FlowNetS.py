import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic Conv -> LeakyReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.conv(x))

class FlowNetS(nn.Module):
    """
    A PyTorch implementation of FlowNetS for optical flow estimation.
    Input: 2 RGB frames concatenated along channel dimension (B, 6, H, W)
    Output: Optical flow map (B, 2, H, W)
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = ConvBlock(6, 64, 7, 2)     # (B, 64, H/2, W/2)
        self.conv2 = ConvBlock(64, 128, 5, 2)   # (B, 128, H/4, W/4)
        self.conv3 = ConvBlock(128, 256, 5, 2)  # (B, 256, H/8, W/8)
        self.conv3_1 = ConvBlock(256, 256)
        self.conv4 = ConvBlock(256, 512, 3, 2)  # (B, 512, H/16, W/16)
        self.conv4_1 = ConvBlock(512, 512)
        self.conv5 = ConvBlock(512, 512, 3, 2)  # (B, 512, H/32, W/32)
        self.conv5_1 = ConvBlock(512, 512)
        self.conv6 = ConvBlock(512, 1024, 3, 2) # (B, 1024, H/64, W/64)
        self.conv6_1 = ConvBlock(1024, 1024)

        # Flow prediction head
        self.predict_flow = nn.Conv2d(1024, 2, kernel_size=3, padding=1)

        # Optional: upsampling to input resolution
        self.upsample = nn.Upsample(scale_factor=64, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        x: (B, 6, H, W), where x = concat(frame1, frame2)
        returns: optical flow map (B, 2, H, W)
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv3_1(out)
        out = self.conv4(out)
        out = self.conv4_1(out)
        out = self.conv5(out)
        out = self.conv5_1(out)
        out = self.conv6(out)
        out = self.conv6_1(out)

        flow = self.predict_flow(out)          # (B, 2, H/64, W/64)
        flow_up = self.upsample(flow)          # (B, 2, H, W)
        return flow_up
