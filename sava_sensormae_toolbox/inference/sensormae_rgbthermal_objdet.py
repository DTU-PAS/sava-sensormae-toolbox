"""RGB + Thermal 2D object detection (two-input ONNX, softmax head)."""

import cv2
import numpy as np

from .objdet_base import SensorMAEObjectDetection
from ..structures import DetectionListResult, DectObject


class SensorMAEObjDet_RGBThermal(SensorMAEObjectDetection):
    """Two-input ONNX model: RGB ``[1,3,H,W]`` + Thermal ``[1,1,H,W]``."""

    def __init__(self, runtime, *, num_classes: int = 20,
                 confidence_threshold: float = 0.0, **kwargs):
        super().__init__(runtime, num_classes=num_classes,
                         confidence_threshold=confidence_threshold, **kwargs)


    def _preprocessing(self, rgb_image, modality_x_image, **kwargs):
        # RGB: BGR→RGB, ImageNet normalise, pad to square
        rgb = self._preprocess_rgb(rgb_image)
        rgb = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32)

        # Thermal: CLAHE → normalise → (x-0.5)/0.28, pad to square
        thermal = self._preprocess_thermal(modality_x_image)
        thermal = thermal[np.newaxis, np.newaxis, ...].astype(np.float32)

        input_names = [inp.name for inp in self.session.get_inputs()]
        return {input_names[0]: rgb, input_names[1]: thermal}


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

    def _preprocess_rgb(self, image: np.ndarray) -> np.ndarray:
        """BGR→RGB, ImageNet normalise, pad to square."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = self.normalize_imagenet(rgb)
        return self.resize_and_pad(rgb)

    def _preprocess_thermal(self, image: np.ndarray) -> np.ndarray:
        """CLAHE → [0,1] normalise → (x - 0.5) / 0.28."""
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
        image = clahe.apply(image.astype(np.uint16))
        image = cv2.normalize(image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        image = (image - 0.5) / 0.28
        return self.resize_and_pad(image, pad_value=0.5)
