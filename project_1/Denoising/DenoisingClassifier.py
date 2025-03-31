import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torchvision.io as tvio
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class DenoisingClassifier(pl.LightningModule):

    def __init__(self, model, lr, test_x_output_path, test_output_path):
        super().__init__()
        self.model = model
        self.lr = lr
        self.l2_loss = nn.MSELoss()
        self.dir_path = test_output_path
        self.dir_x_path = test_x_output_path
        self.test_save = 0
        self.test_x_save = 0

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, noised_img, noise, target_img):
        return nn.functional.mse_loss(noise, noised_img - target_img, reduction="sum").div(2)
        #return self.l2_loss(denoised_img, target_img)

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        dn = x - outputs
        dn = torch.clamp(dn, 0.0, 1.0)
        #print(x.shape)
        #print(outputs.shape)
        #print(dn.shape)


        # to_pil = transforms.ToPILImage()
        # x_img = to_pil(x[0])
        # y_img = to_pil(y[0])
        # outputs_img = to_pil(outputs[0])
        # dn_img = to_pil(dn[0])
        # plt.imshow(x_img)
        # plt.axis("off")  # Ukrywa osie
        # plt.show()
        # plt.imshow(y_img)
        # plt.axis("off")  # Ukrywa osie
        # plt.show()
        # plt.imshow(outputs_img)
        # plt.axis("off")  # Ukrywa osie
        # plt.show()
        # plt.imshow(dn_img)
        # plt.axis("off")  # Ukrywa osie
        # plt.show()



        #a x_img.show()
        #y_img.show()
        #outputs_img.show()
        #dn_img.show()
        loss = self.compute_loss(x, outputs, y)
        return loss, x, dn, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, x, dn, outputs, y = self.common_step(batch, batch_idx)
        return loss, x, dn

    def training_step(self, batch, batch_idx):
        loss, _, dn, outputs, y = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

    # def on_validation_epoch_end(self) -> None:
    #     results = self.metrics.compute()
    #     self.log('val_acc', results['accuracy'], prog_bar=True)
    #     # self.metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, x, outputs = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        #print(x.shape)
        #print(outputs.shape)
        if x.dtype != torch.uint8:
            x = (x * 255).clamp(0, 255).to(torch.uint8)
        x = x.cpu()
        for i, image in enumerate(x):
            img_path = os.path.join(self.dir_x_path, f"image_{self.test_x_save}.png")
            self.test_x_save +=1
            tvio.write_png(image, img_path)

        if outputs.dtype != torch.uint8:
            outputs = (outputs * 255).clamp(0, 255).to(torch.uint8)
        outputs = outputs.cpu()
        for i, image in enumerate(outputs):
            img_path = os.path.join(self.dir_path, f"image_{self.test_save}.png")
            self.test_save +=1
            tvio.write_png(image, img_path)

    # def on_test_epoch_end(self) -> None:
    #     results = self.metrics.compute()
    #     self.log('test_acc', results['accuracy'], prog_bar=True)
    #     self.metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #return torch.optim.Adadelta(self.parameters(), lr=self.lr)