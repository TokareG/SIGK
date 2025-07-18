import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from .Generator import Generator
import flip_evaluator as flip

class GeneratorModule(pl.LightningModule):
    def __init__(self,
                 img_shape = (3, 128, 128),
                 c_dim = 10,
                 lr = 0.0025,
                 b1 = 0.5,
                 b2 = 0.98):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(c_dim=c_dim, img_channels=img_shape[0])


    def forward(self, c):
        return self.generator(c)

    def _shared_step(self, batch, stage: str):
        imgs, c = batch
        generated_images = self(c)
        loss = F.mse_loss(generated_images, imgs, reduce=True, reduction='mean')
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        imgs, c = batch
        generated_images = self(c)
        loss = F.mse_loss(generated_images, imgs, reduce=True, reduction='mean')
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        imgs, c = batch
        if batch_idx == 0:
            self.val_scene_data = c[:5]
            self.imgs_GT = imgs[:5]
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        sample_images = self(self.val_scene_data)
        gt_images = self.imgs_GT
        flip_images = []
        flip_scores = []
        for gt_img, gen_img in zip(gt_images, sample_images):

            gt_img = (gt_img + 1) / 2
            gt_img = gt_img.permute(1, 2, 0).cpu().numpy().clip(0, 1).astype(np.float32)

            gen_img = (gen_img + 1) / 2
            gen_img = gen_img.permute(1, 2, 0).cpu().numpy().clip(0, 1).astype(np.float32)

            flipErrorMap, meanFlipError, parameters = flip.evaluate(gt_img, gen_img, "LDR")
            flip_images.append(flipErrorMap)
            flip_scores.append(meanFlipError)

        flip_images_np = np.stack(flip_images)
        flip_scores_np = np.stack(flip_scores)
        totalMeanFlipError = np.mean(flip_scores)
        flip_images = torch.tensor(flip_images, device=self.device).permute(0, 3, 1, 2)
        all_images = torch.cat((sample_images, gt_images))
        all_images = (all_images + 1) / 2
        all_images = torch.cat((all_images, flip_images))
        grid_gen = torchvision.utils.make_grid(all_images, nrow=5, padding=5, pad_value=1)
        self.logger.experiment.add_image("val/generated_images", grid_gen, self.current_epoch)
        self.log(f"val/Mean_Flip", totalMeanFlipError, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        return torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))