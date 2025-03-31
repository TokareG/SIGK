import torch
import torch.nn as nn
import torch.nn.functional as F


#class PartialConv2d(nn.Module):
#    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#        super(PartialConv2d, self).__init__()
#        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
#        self.weight_mask = torch.ones(1, 1, kernel_size, kernel_size)
#        self.slide_window_size = self.weight_mask.numel()
#
#    def forward(self, x, mask):
#        with torch.no_grad():
#            mask_sum = F.conv2d(mask, self.weight_mask.to(x.device), stride=self.conv.stride, padding=self.conv.padding)
#            mask_sum = torch.where(mask_sum == 0, torch.tensor(1.0, device=x.device),
#                                   mask_sum)  # Avoid division by zero
#
#        x = self.conv(x * mask)
#        x = x / mask_sum  # Normalize by the sum of valid input pixels
#        new_mask = F.conv2d(mask, self.weight_mask.to(x.device), stride=self.conv.stride, padding=self.conv.padding)
#        new_mask = torch.where(new_mask > 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
#
#        return x, new_mask



class PartialConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True):
        super(PartialConv2d, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # Initialize weights for mask convolution to be 1s (so it only normalizes)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.requires_grad_(False)  # No need to update mask weights

    def forward(self, input, mask):
        """
        input: Tensor of shape (N, 3, H, W) - the feature map
        mask: Tensor of shape (N, 3, H, W) - the binary mask (1 for valid, 0 for missing)
        """

        # Perform standard convolution
        input_masked = input * mask
        output = self.input_conv(input_masked)

        # Compute the sum of valid entries in the convolution window
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)  # Sum of mask elements in receptive field
            mask_sum = torch.clamp(mask_sum, min=1e-5)  # Avoid division by zero

        # Normalize output
        output = output / mask_sum

        # Update mask (new mask after convolution)
        new_mask = self.mask_conv(mask)
        new_mask = torch.clamp(new_mask, 0, 1)  # Ensure values remain between 0 and 1

        return output, new_mask