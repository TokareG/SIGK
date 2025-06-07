import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


from .Generator import Generator
from .Discriminator import Discriminator


class GAN(pl.LightningModule):
    def __init__(self,
                 img_shape = (3,128, 128),
                 lr_G = 0.0025,
                 lr_D = 1e-5,
                 b1 = 0.5,
                 b2 = 0.98,
                 D_acc_threshold = 0.8,
                 warmup_epochs = 10):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(c_dim=10, img_channels=img_shape[0])
        self.discriminator = Discriminator(img_shape=img_shape)
        self.criterion = nn.BCELoss()
        #self.criterion = nn.MSELoss()
        self.last_d_accuracy = 0.0
        self.D_acc_threshold = D_acc_threshold
        self.val_scene_data = None
        self.latent_dim = 100
        self.noice_std = 0.1

    def forward(self, noice, z):
        return self.generator(noice, z)


    def training_step(self, batch, batch_idx):
        imgs, scene_data = batch
        #print(f"max value: {imgs.max()}")
        #print(f"min value: {imgs.min()}")
        #print(f"max value: {scene_data.max()}")
        #print(f"min value: {scene_data.min()}")
        #print(scene_data.shape)
        batch_size = imgs.size(0)

        opt_G, opt_D = self.optimizers()

        valid = torch.ones(batch_size, device=imgs.device, dtype=imgs.dtype) * 0.9
        fake = torch.ones(batch_size, device=imgs.device, dtype=imgs.dtype) * 0.1
        #fake = torch.zeros(batch_size, device=imgs.device, dtype=imgs.dtype)
        z = torch.randn(imgs.size(0), self.latent_dim, device=imgs.device)
        generated_images = self(z, scene_data)

        if self.current_epoch >= self.hparams.warmup_epochs:
            real_imgs_noisy = self.add_instance_noise(imgs)
            fake_imgs_noisy = self.add_instance_noise(generated_images.detach())
            R_predictions = self.discriminator(real_imgs_noisy, scene_data).view(-1)
            F_predictions = self.discriminator(fake_imgs_noisy, scene_data).view(-1)
            R_accuracy = (R_predictions > 0.5).float().mean()
            F_accuracy = (F_predictions < 0.5).float().mean()
            self.last_d_accuracy = ((R_accuracy + F_accuracy) / 2).item()
            self.log("train/D_accuracy", self.last_d_accuracy, on_step=True, prog_bar=True)

            #Discriminator training
            if self.last_d_accuracy <= self.D_acc_threshold:

                self.toggle_optimizer(opt_D)

                R_loss = self.criterion(R_predictions, valid)
                F_loss = self.criterion(F_predictions, fake)
                D_loss = (R_loss + F_loss) / 2
                self.log("train/D_loss", D_loss, on_step=True, on_epoch=False, prog_bar=True)

                self.manual_backward(D_loss)
                opt_D.step()
                opt_D.zero_grad()

                self.untoggle_optimizer(opt_D)
                self.log("train/D_skipped", False, prog_bar=True)
            else:
                self.log("train/D_skipped", True, prog_bar=True)

        #Generator training
        self.toggle_optimizer(opt_G)
        if self.current_epoch < self.hparams.warmup_epochs:
            G_loss = F.mse_loss(generated_images, imgs, reduce=True, reduction='mean')
        else:
            G_loss_mse = F.mse_loss(generated_images, imgs, reduce=True, reduction='mean')
            G_loss_diss = self.criterion(self.discriminator(generated_images, scene_data).view(-1), valid)
            G_loss = 0.9 * G_loss_mse + 0.1 * G_loss_diss
        self.manual_backward(G_loss)
        opt_G.step()
        opt_G.zero_grad()
        self.untoggle_optimizer(opt_G)
        self.log("train/G_loss", G_loss, on_step=True, on_epoch=False, prog_bar=True)



    def validation_step(self, batch, batch_idx):

        imgs, scene_data = batch
        batch_size = imgs.size(0)

        if batch_idx == 0:
            self.val_scene_data = scene_data[:5]
            self.imgs_GT = imgs[:5]

        if self.current_epoch >= self.hparams.warmup_epochs:
            valid = torch.ones(batch_size, device=imgs.device, dtype=imgs.dtype) * 0.9
            fake = torch.ones(batch_size, device=imgs.device, dtype=imgs.dtype) * 0.1
            z = torch.randn(imgs.size(0), self.latent_dim, device=imgs.device)

            generated_images = self(z, scene_data)

            G_predictions = self.discriminator(generated_images, scene_data).view(-1)
            G_loss = self.criterion(G_predictions, valid)

            R_predictions = self.discriminator(imgs, scene_data).view(-1)
            F_predictions = self.discriminator(generated_images.detach(), scene_data).view(-1)
            R_loss = self.criterion(R_predictions, valid)
            F_loss = self.criterion(F_predictions, fake)
            D_loss = (R_loss + F_loss) / 2

            self.log("val/G_loss", G_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/D_loss", D_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        z = torch.randn(self.val_scene_data.size(0), self.latent_dim, device=self.val_scene_data.device)
        sample_images = self(z, self.val_scene_data)
        gt_images = self.imgs_GT
        all_images = torch.cat((sample_images, gt_images))
    #     z = torch.randn(4, self.generator.latent_dim,
    #                     device=self.generator.model[0].weight.device,
    #                     dtype=self.generator.model[0].weight.dtype)
    #
    #     sample_images = self(z)
        all_images = (all_images + 1) / 2
        grid_gen = torchvision.utils.make_grid(all_images, nrow=5, padding=5, pad_value=1)
        self.logger.experiment.add_image("val/generated_images", grid_gen, self.current_epoch)

    def configure_optimizers(self):
        lr_D = self.hparams.lr_D
        lr_G = self.hparams.lr_G
        b1 = self.hparams.b1
        b2 = self.hparams.b2


        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_G, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D, betas=(b1, b2))

        return opt_g, opt_d

    def add_instance_noise(self, x):
        if self.noice_std > 0:
            noise = torch.randn_like(x) * self.noice_std
            x = x + noise
        return x.clamp(-1, 1)