import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from cv2.gapi.wip import GOutputs


class VGGFeatures(nn.Module):
    def __init__(self, layer=15, requires_grad=False):
        super(VGGFeatures, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True).features
        self.vgg16 = self.vgg16[:layer+1] # Use only till specific layer
        if not requires_grad:
            for param in self.vgg16.parameters():  # Corrected line
                param.requires_grad = False
        self.criterion = nn.L1Loss()  # You can also use nn.MSELoss()

    def forward(self, y):
        features = self.vgg16(y)
        return features

class InpaintingLoss(nn.Module):
    def __init__(self, vgg_layer=8):
        super(InpaintingLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.extract_features = VGGFeatures(layer=vgg_layer)

    def forward(self, y_pred, mask, y_true, device):
        # L1 Loss
        # L valid
        y_pred_valid = y_pred * mask
        y_true_valid = y_true * mask
        loss_valid = self.l1_loss(y_pred_valid, y_true_valid)

        # L hole
        y_pred_hole = y_pred * (1 - mask)
        y_true_hole = y_true * (1 - mask)
        loss_hole = self.l1_loss(y_pred_hole, y_true_hole)

        # Perceptual Loss
        y_comp = y_pred.clone()
        y_comp[mask == 0] = y_true[mask == 0]
        pred_features = self.extract_features(y_pred)
        comp_features = self.extract_features(y_comp)
        true_features = self.extract_features(y_true)
        perceptual_loss = self.l1_loss(pred_features, true_features) + self.l1_loss(comp_features, true_features)

        # Style Loss
        gram_pred_features = self.gram_matrix(pred_features)
        gram_comp_features = self.gram_matrix(comp_features)
        gram_true_features = self.gram_matrix(true_features)
        style_out_loss = self.l1_loss(gram_pred_features, gram_true_features)
        style_comp_loss = self.l1_loss(gram_comp_features, gram_true_features)

        # Smoothing Loss (Penalty)
        flipped_mask = 1 - mask
        #if flipped_mask.shape[1] > 1:
        #    flipped_mask = torch.any(flipped_mask > 0, dim=1, keepdim=True).float()
        # Dilate the mask to get the region R (1-pixel border)
        kernel = torch.ones((1, 3, 3, 3), device=device)
        dilated_mask = F.conv2d(flipped_mask.float(), kernel, padding=1)
        dilated_mask = (dilated_mask > 0).float()

        tv_h = torch.abs(y_comp[:, :, :, 1:] - y_comp[:, :, :, :-1])
        tv_v = torch.abs(y_comp[:, :, 1:, :] - y_comp[:, :, :-1, :])

        # Crop masks to match TV shape
        mask_h = dilated_mask[:, :, :, 1:] * dilated_mask[:, :, :, :-1]
        mask_v = dilated_mask[:, :, 1:, :] * dilated_mask[:, :, :-1, :]

        # Normalize by number of contributing pixels
        norm_h = torch.sum(mask_h)
        norm_v = torch.sum(mask_v)

        tv_loss = 0.0
        if norm_h > 0:
            tv_loss += torch.sum(tv_h * mask_h) / norm_h
        if norm_v > 0:
            tv_loss += torch.sum(tv_v * mask_v) / norm_v


        # Final Loss
        loss = loss_valid + 6 * loss_hole + 0.05 * perceptual_loss + 120 * (style_out_loss + style_comp_loss) + 0.1 * tv_loss
        return loss

    def gram_matrix(self, features):
        """
        Compute the Gram matrix (autocorrelation) of a feature map.
        features: shape (B, C, H, W)
        returns: shape (B, C, C)
        """
        B, C, H, W = features.size()
        features = features.view(B, C, H * W)  # Reshape to (B, C, N)
        gram = torch.bmm(features, features.transpose(1, 2))  # (B, C, C)
        gram = gram / (C * H * W)  # Normalize
        return gram