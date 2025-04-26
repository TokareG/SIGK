import torch
import torchmetrics

class MulticlassMetrics(torchmetrics.Metric):
    def __init__(self, num_classes: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes

        # Define the sub-metrics
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        self.precision = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average="macro")
        self.recall = torchmetrics.classification.MulticlassRecall(num_classes=num_classes, average="macro")
        self.f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average="macro")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(preds, dim=1) if preds.ndim > 1 else preds
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1.update(preds, targets)

    def compute(self):
        return {
            "accuracy": self.accuracy.compute(),
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "f1": self.f1.compute(),
        }

    def reset(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
