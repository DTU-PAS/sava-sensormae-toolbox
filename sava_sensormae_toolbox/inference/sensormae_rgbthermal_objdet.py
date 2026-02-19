"""RGB + Thermal object-detection model (two-input ONNX, softmax head)."""

from typing import List

import cv2
import numpy as np
import logging

from .objdet_base import SensorMAEObjectDetection
from ..structures import DetectionListResult, DectObject

logger = logging.getLogger(__name__)


class SensorMAEObjDet_RGBThermal(SensorMAEObjectDetection):
    """SensorMAE object detection with RGB + thermal inputs.

    The ONNX model expects **two separate inputs**: an RGB tensor
    ``[1, 3, H, W]`` and a single-channel thermal tensor ``[1, 1, H, W]``.
    Post-processing uses softmax over class logits.
    """

    MODALITY_X_NAME = "thermal"

    def __init__(self, runtime, *, num_classes: int = 20,
                 confidence_threshold: float = 0.0, **kwargs):
        super().__init__(runtime, num_classes=num_classes,
                         confidence_threshold=confidence_threshold, **kwargs)

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
        # RGB
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb = self.normalize_imagenet(rgb)
        rgb = self.resize_and_pad(rgb)
        rgb = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32)

        # Thermal
        thermal = self._preprocess_thermal(modality_x_image)
        thermal = self.resize_and_pad(thermal, pad_value=0.5)
        thermal = np.expand_dims(np.expand_dims(thermal, 0), 0).astype(np.float32)

        logger.debug("RGB  shape=%s  min/max=(%.3f, %.3f)", rgb.shape, rgb.min(), rgb.max())
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
        out_logits, out_bbox = outputs[0], outputs[1]
        prob = self.softmax(out_logits, -1)
        scores = np.max(prob, axis=-1)
        labels = np.argmax(prob, axis=-1)

        boxes = self.box_cxcywh_to_xyxy(out_bbox)

        results = DetectionListResult()
        for score, label, box in zip(scores, labels, boxes, strict=True):
            keep = (label != self.num_classes) & (score > self.confidence_threshold)
            results.append(DectObject(
                xywh=box[keep].tolist(),
                class_id=label[keep].tolist(),
                score=score[keep].tolist(),
            ))
        return results

