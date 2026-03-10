from abc import ABC, abstractmethod
import os
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Model(ABC):
    """Base class for all RGB + modality_x inference models.

    Every concrete model receives two images — an RGB image and a second
    modality image (thermal, depth, …).  Subclasses set ``MODALITY_X_NAME``
    to describe their secondary modality and implement the abstract pipeline
    methods (_preprocessing, _inference, _postprocessing).

    Shared image-processing utilities live here so that every leaf class
    can reuse them without duplication.
    """

    # Override in subclasses to name the secondary modality (e.g. "thermal", "depth")
    MODALITY_X_NAME: str = "modality_x"

    def __init__(self, runtime, **kwargs):
        self.session = runtime

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, rgb_image: np.ndarray, modality_x_image: np.ndarray, **kwargs):
        """Shortcut for :meth:`predict`."""
        return self.predict(rgb_image, modality_x_image, **kwargs)

    def predict(self, rgb_image: np.ndarray, modality_x_image: np.ndarray, **kwargs):
        """Run the full pipeline: preprocess → infer → postprocess.

        Subclasses may override this if the data-flow between stages
        requires custom wiring (e.g. passing orig_size to postprocessing).
        Extra ``**kwargs`` are accepted so that subclasses (e.g. 3D
        detection) can receive additional data such as calibration.
        """
        preprocessed = self._preprocessing(rgb_image, modality_x_image)
        outputs = self._inference(preprocessed)
        return self._postprocessing(outputs)

    # ------------------------------------------------------------------
    # Abstract pipeline methods — implement in every leaf class
    # ------------------------------------------------------------------
    @abstractmethod
    def _preprocessing(self, rgb_image: np.ndarray, modality_x_image: np.ndarray):
        """Return preprocessed tensor(s) ready for the runtime."""

    @abstractmethod
    def _inference(self, preprocessed):
        """Run the runtime on preprocessed data and return raw outputs."""

    @abstractmethod
    def _postprocessing(self, outputs):
        """Convert raw outputs into structured results."""

    # ------------------------------------------------------------------
    # Shared pre-processing utilities
    # ------------------------------------------------------------------
    @staticmethod
    def resize_and_pad(image: np.ndarray, size: int = 640, pad_value=0) -> np.ndarray:
        """Resize longest side to *size* and zero-pad to a square."""
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_bottom = size - new_h
        pad_right = size - new_w
        padded = cv2.copyMakeBorder(
            resized,
            top=0, bottom=pad_bottom, left=0, right=pad_right,
            borderType=cv2.BORDER_CONSTANT, value=pad_value,
        )
        return padded

    @staticmethod
    def normalize_imagenet(image: np.ndarray) -> np.ndarray:
        """Normalise an RGB image with ImageNet mean/std (expects uint8 or [0,255])."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = image.astype(np.float32) / 255.0
        return (image - mean) / std

    @staticmethod
    def apply_colormap(mask: np.ndarray, num_classes: int = 21) -> np.ndarray:
        """Convert a class-index mask to an RGB visualisation."""
        colormap = plt.cm.get_cmap("tab20", num_classes)
        colored = colormap(mask.astype(int))[:, :, :3]
        return (colored * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Shared visualisation / IO
    # ------------------------------------------------------------------
    @staticmethod
    def save_results(
        output_path: str,
        rgb_image: np.ndarray,
        modality_x_image: np.ndarray,
        result_image: np.ndarray,
    ) -> None:
        """Save a side-by-side panel: RGB | Modality_X | Result.

        All three images are resized / converted so they share the same
        height and are 3-channel BGR before horizontal stacking.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        h, w = rgb_image.shape[:2]

        def _to_bgr3(img: np.ndarray, target_hw=(h, w)) -> np.ndarray:
            """Ensure *img* is 3-channel and matches *target_hw*."""
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            if img.shape[:2] != target_hw:
                interp = cv2.INTER_LINEAR
                img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=interp)
            return img

        rgb_vis = _to_bgr3(rgb_image)
        modx_vis = _to_bgr3(modality_x_image)
        res_vis = _to_bgr3(result_image)

        combined = np.hstack((rgb_vis, modx_vis, res_vis))
        cv2.imwrite(output_path, combined)
