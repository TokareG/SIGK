import numpy as np

class HDRNormalize:
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, image):
        """
        Args:
            image (Tensor): Tensor of shape [H, W, C] or [C, H, W]

        Returns:
            Tensor: normalized image
        """
        x_max = np.max(image)
        x_min = np.min(image)
        scale = x_max - x_min
        image_norm = (image - x_min) / scale

        image_norm = self.scale * image_norm / image_norm.mean()
        return image_norm