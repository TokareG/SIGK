from typing import Any
import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from MNISTClassifier.MulticlassMetrics import MulticlassMetrics


class MedClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating a CNN model.

    Args:
        model: A PyTorch neural network model to be trained and evaluated.

    Attributes:
        model (nn.Module): The CNN model used for predictions.
        lr (float): The learning rate for the optimizer.
        out_features (int): Number of output features from the model.
        metrics (Metrics): A custom metrics object to compute evaluation metrics.
        loss_fn (nn.Module): Loss function used for training and evaluation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 0.001
        self.out_features = model.out_features
        self.metrics = MulticlassMetrics(self.out_features).to(device=self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model(x)

    def common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            tuple: Loss, logits, and true labels.
        """
        x, labels = batch
        labels = labels.squeeze(1).long()
        logits = self(x)
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Training step to calculate and log the loss.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        loss, _, _ = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def common_test_valid_step(self, batch, batch_idx):
        """
        Common step for validation and testing.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        loss, logits, labels = self.common_step(batch, batch_idx)
        #self.metrics = self.metrics.to(device=self.device)
        self.metrics.update(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """
        Validation step to calculate and log the validation loss.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log validation metrics at the end of the epoch.

        Returns:
            None
        """
        metrics = self.metrics.compute()
        self.log_dict({f'val_{k}': v for k, v in metrics.items()}, prog_bar=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx) -> None:
        """
        Test step to calculate and log the test loss.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """
        Compute and log test metrics at the end of the epoch.

        Returns:
            None
        """
        metrics = self.metrics.compute()
        self.log_dict({f'test_{k}': v for k, v in metrics.items()}, prog_bar=True)
        self.metrics.reset()

    def configure_optimizers(self):
        """
        Configure and return the optimizer.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
