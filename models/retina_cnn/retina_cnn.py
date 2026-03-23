"""
BBScore wrapper for the primate retina CNN model.
Handles static TVSD images by replicating to 50-frame temporal window.
"""
import os
import numpy as np
import torch
from PIL import Image

from .cnn_model import CNN


class RetinaCNN:
    """BBScore wrapper for retina CNN (full model) and LN (linear-nonlinear) variants."""

    def __init__(self):
        self.static = True
        self.model = None
        self.model_config = None
        self._weights_dir = os.path.join(os.path.dirname(__file__), "weights")

    def get_model(self, identifier: str):
        """
        Load the retina model from checkpoint.
        identifier: "CNN" for full model, "LN" for linear-nonlinear variant.
        """
        if identifier == "CNN":
            ckpt_name = "cnn_out_best.tar"
        elif identifier == "LN":
            ckpt_name = "ln_out_best.tar"
        else:
            raise ValueError(
                f"Unknown identifier '{identifier}'. Use 'CNN' or 'LN'."
            )

        checkpoint_path = os.path.join(self._weights_dir, ckpt_name)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Place trained .tar checkpoint in models/retina_cnn/weights/."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.model_config = checkpoint["model_config"]
        # Ensure device is cpu for inference (BBScore handles device placement)
        self.model_config["device"] = "cpu"

        model = CNN(self.model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        self.model = model
        return model

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocess TVSD images for the retina CNN.
        - Converts to grayscale
        - Resizes to model's stimulus_dim
        - Normalizes (divide by 255, subtract mean)
        - Replicates to 50 frames (constant stimulus for static images)

        input_data: PIL Image or list of PIL Images
        Returns: torch.Tensor of shape (T, 50, H, W) with T=1 for single image
        """
        if self.model_config is None:
            raise RuntimeError(
                "Model not loaded. Call get_model() before preprocess_fn."
            )

        stimulus_dim = self.model_config["stimulus_dim"]
        history_frames = self.model_config["history_frames"]
        h, w = stimulus_dim

        # Collect raw frames
        if isinstance(input_data, Image.Image):
            frames = [input_data]
        elif isinstance(input_data, list):
            frames = list(input_data)
        else:
            raise ValueError(
                "input_data must be PIL.Image or list of PIL.Image"
            )

        processed = []
        for img in frames:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(np.uint8(img)).convert("L")
            elif isinstance(img, Image.Image):
                img = img.convert("L")
            else:
                raise ValueError("Each frame must be PIL.Image or numpy array")

            # Resize to stimulus_dim (H, W) before converting to float
            img = img.resize((w, h), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            # Match training normalization: /255 then mean subtract
            arr = arr - arr.mean()

            # Replicate to 50 frames: (50, H, W)
            arr_50 = np.tile(arr[np.newaxis, :, :], (history_frames, 1, 1))
            processed.append(arr_50)

        # Stack: (T, 50, H, W)
        out = np.stack(processed, axis=0).astype(np.float32)
        return torch.from_numpy(out)

    def postprocess_fn(self, features_np):
        """Flatten features to (batch_size, -1)."""
        if isinstance(features_np, torch.Tensor):
            features_np = features_np.cpu().numpy()
        batch_size = features_np.shape[0]
        return features_np.reshape(batch_size, -1)
