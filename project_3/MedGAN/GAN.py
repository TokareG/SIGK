import torch
import torch.nn as nn
import pytorch_lightning as pl
from MedGAN.Generator import Generator
from MedGAN.Discriminator import Discriminator

class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, img_shape=(3, 28, 28), lr=2e-4):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=self.hparams.img_shape)
        print(self.hparams.img_shape)
        self.discriminator = Discriminator(img_shape=self.hparams.img_shape)

        self.criterion = nn.BCELoss()

        # THIS IS IMPORTANT
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.size(0)

        opt_g, opt_d = self.optimizers()

        # Train Discriminator
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z).detach()  # No gradient for generator

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        pred_real = self.discriminator(imgs)
        pred_fake = self.discriminator(generated_imgs)

        d_real_loss = self.criterion(pred_real, valid)
        d_fake_loss = self.criterion(pred_fake, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # Train Generator
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)

        pred_fake = self.discriminator(generated_imgs)
        g_loss = self.criterion(pred_fake, valid)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # Logging
        self.log('train/g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.size(0)

        # Generate images
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Discriminator predictions
        pred_real = self.discriminator(imgs)
        pred_fake = self.discriminator(generated_imgs)

        d_real_loss = self.criterion(pred_real, valid)
        d_fake_loss = self.criterion(pred_fake, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Generator loss
        pred_fake_for_g = self.discriminator(generated_imgs)
        g_loss = self.criterion(pred_fake_for_g, valid)

        # Logging
        self.log('val/g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/d_loss', d_loss, prog_bar=True, on_step=False, on_epoch=True)
        #val_loss = (g_loss + d_loss) / 2
        val_loss = g_loss
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.size(0)

        # Generate images
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Discriminator predictions
        pred_real = self.discriminator(imgs)
        pred_fake = self.discriminator(generated_imgs)

        d_real_loss = self.criterion(pred_real, valid)
        d_fake_loss = self.criterion(pred_fake, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Generator loss
        pred_fake_for_g = self.discriminator(generated_imgs)
        g_loss = self.criterion(pred_fake_for_g, valid)

        # Logging
        self.log('test/g_loss', g_loss, prog_bar=True)
        self.log('test/d_loss', d_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]
