import torch
import torchmetrics

def SNE(output_img, target_img):
    assert output_img.shape == target_img.shape, "Images must have same shape"
    return torch.sum((output_img - target_img) ** 2)

def PSNR(output_img, target_img, max_value=1):
    assert output_img.shape == target_img.shape, "Images must have same shape"
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=255.0)
    return psnr_metric(output_img.float(), target_img.float())

def SSIM(output_img, target_img, max_value=1):
    assert output_img.shape == target_img.shape, "Images must have same shape"
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=max_value)
    return ssim_metric(output_img, target_img)


def LPIPS(output_img, target_img):
    assert output_img.shape == target_img.shape, "Images must have same shape"
    pass