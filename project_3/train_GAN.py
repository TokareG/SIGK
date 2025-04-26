import multiprocessing
import pytorch_lightning as pl
import torch

from MedMNISTDataModule.MedMNISTDataModule import MedMNISTDataModule
from MNISTClassifier.MedClassifier import MedClassifier
from MedGAN.GAN import GAN
from MNISTClassifier.Callbacks import get_early_stopping, get_checkpoint_callback
import matplotlib.pyplot as plt


def plot_generated_images(model, num_images=16, device='cpu'):
    model.eval()
    model.to(device)

    # Generate random noise
    noise = torch.randn(num_images, model.latent_dim, device=device)

    # Generate fake images
    with torch.no_grad():
        fake_images = model.generator(noise)

    fake_images = fake_images.cpu()

    # Plot the images
    grid_size = int(num_images ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axes = axes.flatten()

    for img, ax in zip(fake_images, axes):
        if img.shape[0] == 1:  # If grayscale
            img = img.squeeze(0)
            ax.imshow(img, cmap='gray')
        else:  # If RGB
            img = img.permute(1, 2, 0)
            ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    dm = MedMNISTDataModule(dataset_name='bloodmnist', batch_size=128)

    model = GAN(dm.num_classes)

    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        accelerator="auto",
        max_epochs=1000,
        callbacks=[get_checkpoint_callback(), get_early_stopping()]
    )

    trainer.fit(model=model, datamodule=dm)

    # After training, plot some generated images
    plot_generated_images(model, num_images=16, device='cuda' if torch.cuda.is_available() else 'cpu')

    return model

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Train the model
    model = main()

    # After training, plot generated images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plot_generated_images(model, num_images=16, device=device)