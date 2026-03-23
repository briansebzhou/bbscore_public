import numpy as np
import torch
from PIL import Image
from .dinov2 import DINOv2

class DINOv2Grayscale(DINOv2):
    """
    DINOv2 variant that forces grayscale input.
    Images are converted to grayscale, then replicated across 3 channels 
    to be compatible with the pretrained color weights.
    """

    def _to_numpy_rgb_uint8(self, x):
        """
        Overrides base method to force grayscale conversion before RGB replication.
        """
        if isinstance(x, np.ndarray):
            # Convert to PIL to use its robust grayscale conversion
            img = Image.fromarray(np.uint8(x))
        elif isinstance(x, Image.Image):
            img = x
        else:
            raise ValueError("Frame must be np.ndarray or PIL.Image")
        
        # 1. Force grayscale
        gray_img = img.convert("L")
        # 2. Convert back to RGB (replicates the single channel to 3)
        # This allows the model to process it while only seeing luminance info.
        rgb_img = gray_img.convert("RGB")
        
        return np.asarray(rgb_img)
