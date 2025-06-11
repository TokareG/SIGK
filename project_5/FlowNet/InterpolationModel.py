import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .FlowWarper import FlowWarper
from .FlowNetS import FlowNetS

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchvision.utils import make_grid

class MyInterpolationModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.flownet = FlowNetS()         # Replace with your FlowNet or load pretrained
        self.warper = FlowWarper()
        self.loss_fn = nn.L1Loss()        # You can change to nn.MSELoss() or use both
        self.lr = lr

    def forward(self, frame0, frame1):
        # Estimate optical flow from frame0 to frame1
        flow = self.flownet(torch.cat([frame0, frame1], dim=1))  # (B, 2, H, W)

        # Warp both frames toward the center time
        warped0 = self.warper(frame0, 0.5 * flow)
        warped1 = self.warper(frame1, -0.5 * flow)

        # Blend the warped frames (simple average)
        interpolated = 0.5 * warped0 + 0.5 * warped1
        return interpolated

    def training_step(self, batch, batch_idx):
        frame0, frame1, target = batch  # (B, 3, H, W)
        output = self.forward(frame0, frame1)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frame0, frame1, target = batch
        output = self(frame0, frame1)

        loss = self.loss_fn(output, target)
        ssim_val = ssim(output, target)
        psnr_val = psnr(output, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_ssim", ssim_val, on_step=False, on_epoch=True)
        self.log("val_psnr", psnr_val, on_step=False, on_epoch=True)

        # Log images only for the first 5 batches (or just batch 0)
        if batch_idx < 1:
            self._log_images(frame0, frame1, output, target, stage='val')

        return loss

    def test_step(self, batch, batch_idx):
        frame0, frame1, target = batch
        output = self(frame0, frame1)

        loss = self.loss_fn(output, target)
        ssim_val = ssim(output, target)
        psnr_val = psnr(output, target)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_ssim", ssim_val, on_step=False, on_epoch=True)
        self.log("test_psnr", psnr_val, on_step=False, on_epoch=True)

        if batch_idx < 1:
            self._log_images(frame0, frame1, output, target, stage='test')

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def _log_images(self, frame0, frame1, output, target, stage='val'):
        # Take up to 5 samples from the batch
        num_samples = min(5, frame0.size(0))

        # Stack rows: [frame0 | frame1 | prediction | target]
        rows = []
        for i in range(num_samples):
            row = torch.stack([
                frame0[i], frame1[i], output[i], target[i]
            ])  # (4, 3, H, W)
            rows.append(row)

        grid = make_grid(torch.cat(rows, dim=0), nrow=4, normalize=True)
        self.logger.experiment.add_image(f"{stage}_examples", grid, self.current_epoch)
