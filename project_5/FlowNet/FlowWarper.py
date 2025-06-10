import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowWarper(nn.Module):
    """
    Warps an image or feature map using a dense optical flow field.

    Inputs:
        img: Tensor of shape (B, C, H, W)
        flow: Tensor of shape (B, 2, H, W), where flow[:, 0, :, :] is dx and flow[:, 1, :, :] is dy
    Returns:
        Warped image of shape (B, C, H, W)
    """
    def __init__(self):
        super(FlowWarper, self).__init__()

    def forward(self, img, flow):
        B, C, H, W = img.size()
        # Create mesh grid
        y, x = torch.meshgrid(
            torch.arange(0, H, device=img.device),
            torch.arange(0, W, device=img.device),
            indexing='ij'
        )
        # Normalize grid to [-1, 1]
        x = x.float() / (W - 1) * 2 - 1
        y = y.float() / (H - 1) * 2 - 1

        grid = torch.stack((x, y), dim=0)  # (2, H, W)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)
        flow_x = flow[:, 0, :, :] / ((W - 1) / 2)
        flow_y = flow[:, 1, :, :] / ((H - 1) / 2)
        flow_norm = torch.stack((flow_x, flow_y), dim=1)
        warped_grid = (grid + flow_norm).permute(0, 2, 3, 1)  # (B, H, W, 2)

        # Sample the image with the warped grid
        warped_img = F.grid_sample(img, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_img
