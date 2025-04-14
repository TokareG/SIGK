import torch
import pytorch_lightning as pl
#from UNET_inpainting.Metrics import Metrics
import torch.nn as nn
import cv2
import numpy as np
import os
import torchvision.io as tvio
from TonalMap.TonalMapLoss import TonalMapLoss


class Classifier(pl.LightningModule):

    def __init__(self, model, lr, test_output_path):
        super().__init__()
        self.model = model
        self.lr = lr
        #self.metrics = Metrics().to(device=self.device)
        self.loss = TonalMapLoss()
        self.dir_path = test_output_path
        self.test_save = 0

    def lum(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute luminance from an image tensor using Rec. 709 luma coefficients.

        Args:
            img (torch.Tensor): Tensor of shape [H, W, 3] or [*, H, W, 3]

        Returns:
            torch.Tensor: Luminance tensor of shape [H, W] or [*, H, W]
        """
        r = img[..., 0, :, :]
        g = img[..., 1, :, :]
        b = img[..., 2, :, :]
        l = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return l

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, output, input):
        return self.loss(output, input, self.device)

    def common_step(self, batch, batch_idx):
        input = batch
        outputs = self(input)
        loss = self.compute_loss(outputs, input)
        return loss, outputs, input

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, input = self.common_step(batch, batch_idx)
        #self.metrics.update(preds, y)
        return loss, outputs, input

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        pass
        #results = self.metrics.compute()
        #self.log('val_acc', results['accuracy'], prog_bar=True)
        #self.log('PSNR',)
        #self.metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, outputs, input = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)

        for i in range(outputs.shape[0]):
            yy = input[i].squeeze()
            y_pred = outputs[i].squeeze()

            a = 0.6

            r = yy[0, :, :]
            g = yy[1, :, :]
            b = yy[2, :, :]

            y_gt_lum = self.lum(yy)
            y_pred_lum = self.lum(y_pred)

            img_out = torch.zeros_like(yy)
            img_out[0, :, :] = (r / y_gt_lum).pow(a) * y_pred_lum
            img_out[1, :, :] = (g / y_gt_lum).pow(a) * y_pred_lum
            img_out[2, :, :] = (b / y_gt_lum).pow(a) * y_pred_lum

            # Assuming img_out is a torch.Tensor with shape [3, H, W], values in [0, 1]
            img_out = img_out.cpu()

            # Convert to NumPy and change shape to [H, W, 3]
            img_out = img_out.permute(1, 2, 0).numpy()  # [H, W, 3], float32

            # Convert from RGB to BGR using OpenCV
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

            # Convert to uint8 if needed
            if img_out.dtype != np.uint8:
                img_out = (img_out * 255).clip(0, 255).astype(np.uint8)

            # Save image
            img_path = os.path.join(self.dir_path, f"image_{self.test_save}.png")
            self.test_save += 1
            cv2.imwrite(img_path, img_out)


    def on_test_epoch_end(self) -> None:
        pass
        #results = self.metrics.compute()
        #self.log('test_acc', results['accuracy'], prog_bar=True)
        #self.metrics.reset()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
        #return torch.optim.Adadelta(self.parameters(), lr=self.lr)


