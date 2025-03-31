from torchmetrics import Metric
import torch

class Metrics(Metric):
    """
    A custom metric class for calculating per-class metrics such as true positives, false positives,
    false negatives, true negatives, and total counts. It also computes mean accuracy across classes.

    Attributes:
        num_classes (int): The number of classes in the classification task.
        dist_sync_on_step (bool): Whether to synchronize metrics across distributed processes during each step.

    States:
        true_positives (torch.Tensor): Tensor tracking the count of true positives for each class.
        false_positives (torch.Tensor): Tensor tracking the count of false positives for each class.
        false_negatives (torch.Tensor): Tensor tracking the count of false negatives for each class.
        true_negatives (torch.Tensor): Tensor tracking the count of true negatives for each class.
        total (torch.Tensor): Tensor tracking the total count of instances per class.
    """
    def __init__(self, dist_sync_on_step=False):
        """
        Initialize the Metrics class.

        Args:
            num_classes (int): Number of classes in the classification task.
            dist_sync_on_step (bool, optional): Synchronize metric states across distributed processes. Defaults to False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)


    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric states based on predictions and targets.

        Args:
            preds (torch.Tensor): Predicted outputs (logits or probabilities) of shape (batch_size, num_classes).
            target (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).
        """
        # Convert predictions to one-hot encoding
        pass

    def compute(self):
        """
        Compute the mean accuracy across all classes.

        Returns:
            dict: A dictionary containing the mean accuracy with key `mean_accuracy`.
        """

        return {
            "mean_accuracy": 0
        }

    def reset(self):
        """
        Reset the metric states to their initial values.
        """
        pass
