"""RGB + Thermal semantic-segmentation model (two-input ONNX)."""

from typing import List

import cv2
import numpy as np
import logging

from .segm_base import SensorMAESegmentation
from ..structures import DetectionListResult, DectObject

logger = logging.getLogger(__name__)


class SensorMAESegm_RGBThermal(SensorMAESegmentation):
    """SensorMAE segmentation with RGB + thermal inputs.

    The ONNX model expects **two separate inputs**: an RGB tensor
    ``[1, 3, H, W]`` and a single-channel thermal tensor ``[1, 1, H, W]``.
    The output is a per-pixel class-index map.
    """

    MODALITY_X_NAME = "thermal"

    def __init__(self, runtime, **kwargs):
        super().__init__(runtime, **kwargs)

    # ------------------------------------------------------------------
    # Modality-specific preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_thermal(image: np.ndarray) -> np.ndarray:
        """CLAHE → [0,1] normalise → (x-0.5)/0.28."""
        THERMAL_MEAN, THERMAL_STD = 0.5, 0.28
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
        image = clahe.apply(image.astype(np.uint16))
        image = cv2.normalize(image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        return (image - THERMAL_MEAN) / THERMAL_STD

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _preprocessing(self, rgb_image: np.ndarray, modality_x_image: np.ndarray):
        h, w = rgb_image.shape[:2]
        self._orig_hw = (h, w)

        # RGB
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb = self.normalize_imagenet(rgb)
        rgb = self.resize_and_pad(rgb)
        rgb = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32)

        # Thermal
        thermal = self._preprocess_thermal(modality_x_image)
        thermal = self.resize_and_pad(thermal, pad_value=0.5)
        thermal = np.expand_dims(np.expand_dims(thermal, 0), 0).astype(np.float32)

        logger.debug("RGB     shape=%s  min/max=(%.3f, %.3f)", rgb.shape, rgb.min(), rgb.max())
        logger.debug("Thermal shape=%s  min/max=(%.3f, %.3f)", thermal.shape, thermal.min(), thermal.max())
        return rgb, thermal

    def _inference(self, preprocessed):
        rgb_tensor, thermal_tensor = preprocessed
        input_names = [inp.name for inp in self.session.get_inputs()]
        return self.session.run(
            {
                input_names[0]: np.ascontiguousarray(rgb_tensor),
                input_names[1]: np.ascontiguousarray(thermal_tensor),
            },
            None,
        )

    def _postprocessing(self, outputs):
        results = DetectionListResult()
        for out in outputs[0]:
            h, w = self._orig_hw
            scale_h = h / max(self._orig_hw)
            scale_w = w / max(self._orig_hw)
            # Remove padding (640 is fixed for now)
            cropped = out[: int(scale_h * 640), : int(scale_w * 640)]
            results.append(DectObject(full_image_segm=cropped))
        return results