import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from MedCDCGAN.Generator import Generator
from MedCDCGAN.Discriminator import Discriminator

class CDCGAN(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 latent_dim = 1000,
                 img_shape = (3,64,64),
                 lr = 2e-4,
                 b1 = 0.5,
                 b2 = 0.98):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(
            num_classes = self.hparams.num_classes,
            latent_dim = self.hparams.latent_dim,
            img_shape = self.hparams.img_shape)
        self.discriminator = Discriminator(
            num_classes = self.hparams.num_classes,
            img_shape = self.hparams.img_shape)
        self.criterion = nn.BCELoss()

        self.validation_z = torch.randn(16, self.hparams.latent_dim, 1, 1)
        self.validation_z_labels = torch.randint(low = 0, high = self.hparams.num_classes, size = (16,))

    def forward(self, z, z_labels):
        return self.generator(z, z_labels)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)

        labels = labels.view(-1)
        z_labels = torch.randint(low = 0, high = self.hparams.num_classes, size = (batch_size,), device=imgs.device)
        z = torch.randn(batch_size, self.hparams.latent_dim, 1, 1, device=imgs.device, dtype=imgs.dtype)

        opt_g, opt_d = self.optimizers()
        valid = torch.ones(batch_size, device=imgs.device, dtype=imgs.dtype)
        fake = torch.zeros(batch_size, device=imgs.device, dtype=imgs.dtype)

        #Train Generator
        self.toggle_optimizer(opt_g)
        generated_imgs = self(z, z_labels)
        g_loss = self.criterion(self.discriminator(generated_imgs, z_labels).view(-1), valid)
        self.log('train/g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        #Train Discriminator
        self.toggle_optimizer(opt_d)
        real_loss = self.criterion(self.discriminator(imgs, labels).view(-1), valid)
        fake_loss = self.criterion(self.discriminator(generated_imgs.detach(), z_labels).view(-1), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.log('train/d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)

        labels = labels.view(-1)
        z_labels = torch.randint(low=0, high=self.hparams.num_classes, size=(batch_size,), device=imgs.device)
        z = torch.randn(batch_size, self.hparams.latent_dim, 1, 1, device=self.device, dtype=self.dtype)
        generated_imgs = self(z, z_labels)

        valid = torch.ones(batch_size, device=self.device, dtype=self.dtype)
        fake = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

        # Strata Generatora — jak dobrze Generator oszukuje Dyskryminator
        g_pred = self.discriminator(generated_imgs, z_labels).view(-1)
        g_loss = self.criterion(g_pred, valid)

        # Strata Dyskryminatora — jak dobrze odróżnia prawdziwe i fałszywe
        d_real = self.discriminator(imgs, labels).view(-1)
        d_fake = self.discriminator(generated_imgs.detach(), z_labels).view(-1)
        real_loss = self.criterion(d_real, valid)
        fake_loss = self.criterion(d_fake, fake)
        d_loss = (real_loss + fake_loss) / 2

        # Logowanie strat
        self.log("val/g_loss", g_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/d_loss", d_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Obrazki prawdziwe i wygenerowane
        if batch_idx == 0:
            real_grid = torchvision.utils.make_grid(imgs[:16])
            fake_grid = torchvision.utils.make_grid(generated_imgs[:16])
            self.logger.experiment.add_image("validation/real_images", real_grid, self.current_epoch)
            self.logger.experiment.add_image("validation/generated_images", fake_grid, self.current_epoch)

    def on_validation_epoch_end(self):
        z = self.validation_z.to(dtype=self.generator.model[0].weight.dtype,
                                 device=self.generator.model[0].weight.device)
        z_labels = self.validation_z_labels.to(device=self.generator.model[0].weight.device)

        sample_imgs = self(z, z_labels)
        grid_gen = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("validation/generated_images", grid_gen, self.current_epoch)


    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d]