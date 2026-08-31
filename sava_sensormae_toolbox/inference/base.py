"""Abstract base class for all inference models.

This is a minimal template — concrete pipeline logic (predict, __call__,
unified ONNX inference) lives in the task-specific base classes
(e.g. ``SensorMAEObjectDetection``).
"""

from abc import ABC, abstractmethod
import os
import time

import cv2
import numpy as np


class Model(ABC):
    """Template for all RGB + modality-X inference models."""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, rgb_image, modality_x_image, **kwargs):
        """Full pipeline: preprocess → infer → postprocess."""
        preprocessed = self._preprocessing(rgb_image, modality_x_image, **kwargs)
        outputs = self._inference(preprocessed)
        results = self._postprocessing(outputs)
        return results

    @abstractmethod
    def _preprocessing(self, rgb_image, modality_x_image, **kwargs):
        """Return preprocessed data ready for the runtime."""

    @abstractmethod
    def _inference(self, preprocessed):
        """Run the runtime on preprocessed data and return raw outputs."""

    @abstractmethod
    def _postprocessing(self, outputs):
        """Convert raw outputs into structured results."""

    # ------------------------------------------------------------------
    # Shared image-processing utilities
    # ------------------------------------------------------------------
    @staticmethod
    def resize_and_pad(image: np.ndarray, size: int = 640, pad_value=0,
                       interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        """Resize longest side to *size* and zero-pad to a square."""
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        padded = cv2.copyMakeBorder(
            resized,
            top=0, bottom=size - new_h, left=0, right=size - new_w,
            borderType=cv2.BORDER_CONSTANT, value=pad_value,
        )
        return padded

    @staticmethod
    def normalize_imagenet(image: np.ndarray) -> np.ndarray:
        """Normalise an RGB image with ImageNet mean/std (expects uint8 or [0,255])."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (image.astype(np.float32) / 255.0 - mean) / std

    # ------------------------------------------------------------------
    # Shared visualisation / IO
    # ------------------------------------------------------------------
    @staticmethod
    def save_results(output_path: str, *images: np.ndarray) -> None:
        """Save a side-by-side panel of one or more images."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        h, w = images[0].shape[:2]

        def _to_bgr3(img):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            return img

        combined = np.hstack([_to_bgr3(img) for img in images])
        cv2.imwrite(output_path, combined)
