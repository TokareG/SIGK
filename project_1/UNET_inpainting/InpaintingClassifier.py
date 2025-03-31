import torch
import pytorch_lightning as pl
from UNET_inpainting.Metrics import Metrics
import torch.nn as nn
import os
import torchvision.io as tvio
from UNET_inpainting.InpaintingLoss import InpaintingLoss


class InpaintingClassifier(pl.LightningModule):

    def __init__(self, model, lr, test_output_path):
        super().__init__()
        self.model = model
        self.lr = lr
        self.sigma = 3.3
        self.metrics = Metrics().to(device=self.device)
        self.loss = InpaintingLoss()
        self.dir_path = test_output_path
        self.test_save = 0

    def forward(self, x, mask):
        return self.model(x, mask)

    def compute_loss(self, output, mask, target):
        return self.loss(output, mask, target, self.device)
        #return  F.binary_cross_entropy(x, y)
        #return F.cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        image, mask, target = batch

        #mask = x[:, 3:4, :, :]
        #mask = 1-mask
        #mask = mask.expand(-1, 3, -1, -1)
        #image = x[:, :3, :, :]
        outputs = self(image, mask)

        loss = self.compute_loss(outputs, mask, target)
        return loss, outputs, target

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, target = self.common_step(batch, batch_idx)
        #self.metrics.update(preds, y)
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs, target = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, _ = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        results = self.metrics.compute()
        #self.log('val_acc', results['accuracy'], prog_bar=True)
        #self.log('PSNR',)
        #self.metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, outputs = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        if outputs.dtype != torch.uint8:
            outputs = (outputs * 255).clamp(0, 255).to(torch.uint8)
        outputs = outputs.cpu()
        for i, image in enumerate(outputs):
            img_path = os.path.join(self.dir_path, f"image_{self.test_save}.png")
            self.test_save +=1
            tvio.write_png(image, img_path)



    def on_test_epoch_end(self) -> None:
        results = self.metrics.compute()
        #self.log('test_acc', results['accuracy'], prog_bar=True)
        #self.metrics.reset()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
        #return torch.optim.Adadelta(self.parameters(), lr=self.lr)


