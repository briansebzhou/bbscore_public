import torchvision.transforms.v2 as T
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image


class PixelModel(nn.Module):
    def __init__(self):
        super(PixelModel, self).__init__()
        self.identity_layer = nn.Identity()

    def forward(self, x):
        # Returns the input as-is.
        return self.identity_layer(x)


class PIXEL:
    """Loads pre-trained Pixel (Masked Autoencoder) models."""

    def __init__(self, target_fps: int | None = None, image_size: tuple[int, int] | None = None):
        """Initializes the Pixel loader."""
        self.model_mappings = {
            "PIXEL": "",  # dummy mapping
        }

        # Optionally store a target fps for uniform sampling
        self.fps = target_fps

        # v2 transform pipeline (tensor backend, fast + batched-friendly)
        tfms = [
            T.ToImage(),  # PIL / ndarray -> (C, H, W) tensor, uint8
            T.Resize((256, 256), antialias=True),
            T.ToDtype(torch.float32, scale=True),  # [0, 1] float32
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]

        self.processor = T.Compose(tfms)

        # Static flag
        self.static = True

    def preprocess_fn(self, input_data, fps: int | None = None):
        """
        Preprocesses input data for Pixel.

        Args:
            input_data: Video file path (str), list of frames, or PIL.Image.
            fps (int, optional): Original fps of the video if known. Used for resampling.

        Returns:
            torch.Tensor: [T, C, H, W] preprocessed frames.

        Raises:
            ValueError: If the processor is not initialized or input is invalid.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first."
            )

        raw_frames = []

        # Handle different input types
        if isinstance(input_data, str) and os.path.isfile(input_data):
            # Load video from file path
            cap = cv2.VideoCapture(input_data)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR (OpenCV’s default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_frames.append(Image.fromarray(frame))

            cap.release()

        elif isinstance(input_data, list) or isinstance(input_data, Image.Image):
            if isinstance(input_data, Image.Image):
                input_data = [input_data]

            for frame in input_data:
                if isinstance(frame, np.ndarray):
                    # Assume HWC, uint8 / float [0,255]
                    frame = Image.fromarray(np.uint8(frame)).convert("RGB")
                elif isinstance(frame, Image.Image):
                    frame = frame.convert("RGB")
                else:
                    raise ValueError(
                        "Each frame must be a numpy array or PIL.Image.Image"
                    )
                raw_frames.append(frame)

        else:
            raise ValueError(
                "Input must be a video file path, list of frames, or PIL.Image.Image"
            )

        if len(raw_frames) == 0:
            raise ValueError("No frames were extracted from the input.")

        # --- FPS resampling BEFORE transforms (faster) ---
        if fps is not None:
            target_fps = getattr(self, "fps", None)
            if target_fps is not None and target_fps != fps and len(raw_frames) > 1:
                orig_count = len(raw_frames)
                duration_sec = orig_count / float(fps)
                new_count = int(round(duration_sec * float(target_fps)))
                new_count = max(new_count, 1)  # avoid empty

                indices = np.linspace(
                    0, orig_count - 1, num=new_count).astype(int)
                raw_frames = [raw_frames[i] for i in indices]

        # --- v2 transforms: applied to the whole structure at once ---
        processed = self.processor(raw_frames)
        # v2 keeps the same structure but converts elements, so for a list of images
        # we get a list of tensors. We stack them into [T, C, H, W].
        if isinstance(processed, list):
            frames_tensor = torch.stack(processed, dim=0)
        elif isinstance(processed, torch.Tensor):
            # In case future v2 versions return a batched tensor directly
            frames_tensor = processed
            if frames_tensor.ndim == 3:  # (C, H, W) -> (1, C, H, W)
                frames_tensor = frames_tensor.unsqueeze(0)
        else:
            raise TypeError(
                f"Unexpected type from processor: {type(processed)}"
            )

        return frames_tensor

    def get_model(self, identifier):
        """
        Loads a Pixel model based on the identifier.

        Args:
            identifier (str): Identifier for the Pixel variant.

        Returns:
            model: The loaded Pixel model.

        """
        self.model = PixelModel()
        return self.model

    def postprocess_fn(self, features_np):
        """Postprocesses Pixel model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from Pixel model
                as a numpy array. Expected shape:
                (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)
        return flattened_features


class GrayscalePixel:
    """
    Raw grayscale pixel luminance baseline (Condition A).
    No ImageNet normalization - just luminance values for V1 prediction baseline.
    """

    def __init__(self):
        self.static = True
        self.model = None
        # Consistent size for feature dimensionality (match common retina input)
        self.image_size = (80, 160)

    def get_model(self, identifier):
        """Returns identity model that passes through pixel values."""
        self.model = PixelModel()
        return self.model

    def preprocess_fn(self, input_data, fps=None):
        """
        Converts to grayscale, resizes, returns raw luminance (no ImageNet norm).
        Returns: (T, 1, H, W) for compatibility with static extractor.
        """
        h, w = self.image_size
        raw_frames = []

        if isinstance(input_data, Image.Image):
            input_data = [input_data]
        elif isinstance(input_data, list):
            pass
        else:
            raise ValueError(
                "input_data must be PIL.Image or list of PIL.Image"
            )

        for frame in input_data:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(np.uint8(frame)).convert("L")
            elif isinstance(frame, Image.Image):
                frame = frame.convert("L")
            else:
                raise ValueError("Each frame must be PIL.Image or numpy array")
            raw_frames.append(frame)

        # Resize, convert to tensor, raw [0,1] luminance (no normalization)
        tensors = []
        for img in raw_frames:
            img = img.resize((w, h), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).unsqueeze(0))  # (1, H, W)

        out = torch.stack(tensors, dim=0)  # (T, 1, H, W)
        return out

    def postprocess_fn(self, features_np):
        """Flatten to (batch_size, -1)."""
        if isinstance(features_np, torch.Tensor):
            features_np = features_np.cpu().numpy()
        batch_size = features_np.shape[0]
        return features_np.reshape(batch_size, -1)
