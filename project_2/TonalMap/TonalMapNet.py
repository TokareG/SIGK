import torch
import torch.nn as nn
from PIL import Image
import os


class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()

        self.ReLu = nn.ReLU()
        self.Conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.Conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.Conv3 = nn.Conv2d(32, 64, 3, 1, 1)

    def forward(self, input):
        x = self.Conv1(input)
        x = self.ReLu(x)
        x = self.Conv2(x)
        x = self.ReLu(x)
        x = self.Conv3(x)
        x = self.ReLu(x)
        return x


class TonalMapNet(nn.Module):
    def __init__(self):
        super(TonalMapNet, self).__init__()

        self.SharedEncoder = SharedEncoder()

        self.ReLu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        # Feature fusion
        self.Conv4 = nn.Conv2d(192, 192, 3, 1, 1)
        self.Conv5 = nn.Conv2d(192, 192, 1, 1, 0)

        # Decoder
        self.Conv6 = nn.Conv2d(192, 32, 3, 1, 1)
        self.Conv7 = nn.Conv2d(32, 16, 3, 1, 1)
        self.Conv8 = nn.Conv2d(16, 3, 3, 1, 1)

    def mul_exp(self, hdr_img):
        # x_p is a constant
        x_p = 1.21497

        # Compute max and median per image in the batch
        max_vals = torch.amax(hdr_img.view(hdr_img.size(0), -1), dim=1)  # shape: (B,)
        medians = hdr_img.view(hdr_img.size(0), -1).median(dim=1).values  # shape: (B,)

        log2 = torch.log(torch.tensor(2.0, device=hdr_img.device))

        c_start = torch.log2(x_p / max_vals) / log2  # shape: (B,)
        c_end = torch.log2(x_p / medians) / log2  # shape: (B,)
        c_middle = (c_start + c_end) / 2.0

        exp_values = [c_start, c_middle, c_end]
        output_list = []

        for exp in exp_values:
            sc = torch.pow(torch.sqrt(torch.tensor(2.0, device=hdr_img.device)), exp).view(-1, 1, 1,
                                                                                           1)  # reshape for broadcasting
            img_exp = torch.clip(hdr_img * sc, 0.0, 1.0)
            output_list.append(img_exp)

        return output_list[0], output_list[1], output_list[2]

    def forward(self, input):
        #print(input.shape)
        input1, input2, input3 = self.mul_exp(input)
        #print(input1.shape)

        x1 = self.SharedEncoder(input1)
        x2 = self.SharedEncoder(input2)
        x3 = self.SharedEncoder(input3)

        x = torch.cat((x1, x2, x3), dim=1)

        #print(x.shape)

        # Feature Fusion
        x = self.Conv4(x)
        x = self.ReLu(x)
        x = self.Conv5(x)
        x = self.ReLu(x)

        #print(x.shape)

        # Decoder
        x = self.Conv6(x)
        x = self.ReLu(x)
        x = self.Conv7(x)
        x = self.ReLu(x)
        x = self.Conv8(x)

        #print(x.shape)

        x = x + input1 + input2 + input3
        x = self.Sigmoid(x)

        return x