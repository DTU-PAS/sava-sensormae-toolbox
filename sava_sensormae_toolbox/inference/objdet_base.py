"""Intermediate base class for SensorMAE object-detection models.

Collects detection-specific utilities (softmax, box conversion, drawing)
so that individual modality leaf classes only need to implement the
preprocessing, inference, and postprocessing pipeline.
"""

from typing import List, Optional

import cv2
import numpy as np

from .base import Model


class SensorMAEObjectDetection(Model):
    """Base for all SensorMAE object-detection models (any modality pair)."""

    def __init__(self, runtime, *, num_classes: int = 20,
                 confidence_threshold: float = 0.0, **kwargs):
        super().__init__(runtime, **kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Detection-specific static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def softmax(x: np.ndarray, axis=None) -> np.ndarray:
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def box_cxcywh_to_xyxy(x: np.ndarray) -> np.ndarray:
        """Convert ``[cx, cy, w, h]`` → ``[x_min, y_min, x_max, y_max]``."""
        x_c, y_c, w, h = np.moveaxis(x, -1, 0)
        w = np.clip(w, a_min=0.0, a_max=None)
        h = np.clip(h, a_min=0.0, a_max=None)
        return np.stack([x_c - 0.5 * w, y_c - 0.5 * h,
                         x_c + 0.5 * w, y_c + 0.5 * h], axis=-1)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def scale_draw_boxes(
        self,
        boxes: np.ndarray,
        image: np.ndarray,
        *,
        labels: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        scale_to_image: bool = False,
    ) -> np.ndarray:
        """Draw bounding boxes (with optional labels/scores) on an image.

        Args:
            boxes: ``(N, 4)`` array of ``[x_min, y_min, x_max, y_max]``.
                   If *scale_to_image* is ``True`` the coordinates are assumed
                   normalised to ``[0, 1]`` and will be scaled to the image size.
            image: RGB/BGR image ``(H, W, 3)`` to draw on (modified in-place).
            labels: Optional ``(N,)`` int array of class indices.
            scores: Optional ``(N,)`` float array of confidence scores.
            class_names: Optional list mapping class index → name.
            scale_to_image: If ``True``, multiply boxes by ``max(H, W)``
                            (useful for pad-and-resize models).

        Returns:
            The annotated image.
        """
        boxes = np.asarray(boxes, dtype=np.float64).copy()
        if scale_to_image:
            h, w = image.shape[:2]
            scale_up = max(h, w)
            boxes *= scale_up
        boxes = boxes.astype(np.int32)

        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.5, 2

        for i, box in enumerate(boxes):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
            if labels is not None or scores is not None:
                parts: list[str] = []
                if labels is not None:
                    cls_idx = int(labels[i])
                    name = (class_names[cls_idx]
                            if class_names and cls_idx < len(class_names)
                            else str(cls_idx))
                    parts.append(name)
                if scores is not None:
                    parts.append(f"{float(scores[i]):.2f}")
                text = " ".join(parts)
                tx, ty = box[0], max(box[1] - 4, 10)
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(image, (tx, ty - th - baseline),
                              (tx + tw, ty + baseline), color, cv2.FILLED)
                cv2.putText(image, text, (tx, ty), font, font_scale,
                            (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        return image
