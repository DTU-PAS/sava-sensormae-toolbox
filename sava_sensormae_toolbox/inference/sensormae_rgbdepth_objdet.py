"""RGB + Depth object-detection model (single concatenated ONNX input, sigmoid head).

All image loading uses cv2 (numpy arrays) for consistency with the rest of
the toolbox.  The ONNX model receives a single ``[1, 4, H, W]`` tensor
(3 RGB channels + 1 depth channel).  Post-processing uses sigmoid
activation with global top-K selection.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import logging

from .objdet_base import SensorMAEObjectDetection
from ..structures import DetectionListResult, DectObject

logger = logging.getLogger(__name__)


class SensorMAEObjDet_RGBDepth(SensorMAEObjectDetection):
    """SensorMAE object detection with RGB + depth inputs.

    The ONNX model expects a **single concatenated input** of shape
    ``[1, 4, H, W]`` (3 RGB + 1 depth).  Post-processing uses sigmoid
    activation with global top-K selection (focal-loss style).
    """

    MODALITY_X_NAME = "depth"

    def __init__(self, runtime, *, num_classes: int = 20,
                 confidence_threshold: float = 0.0,
                 input_size: Tuple[int, int] = (384, 384),
                 num_select: int = 300, **kwargs):
        super().__init__(runtime, num_classes=num_classes,
                         confidence_threshold=confidence_threshold, **kwargs)
        # input_size is (height, width)
        self.input_size = tuple(input_size)
        self.num_select = num_select
        # Stored by _preprocessing for use in _postprocessing
        self._orig_size: Tuple[int, int] | None = None  # (width, height)

    # ------------------------------------------------------------------
    # Modality-specific preprocessing (all cv2-based)
    # ------------------------------------------------------------------
    def _preprocess_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """BGR→RGB, resize to *input_size*, ImageNet normalise → ``[3, H, W]``.

        Also stores the original image dimensions on ``self._orig_size``.
        """
        h, w = rgb.shape[:2]
        self._orig_size = (w, h)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # input_size is (height, width); cv2.resize takes (width, height)
        rgb = cv2.resize(rgb, (self.input_size[1], self.input_size[0]),
                         interpolation=cv2.INTER_LINEAR)
        rgb = self.normalize_imagenet(rgb)
        return rgb.transpose(2, 0, 1)  # [3, H, W]

    @staticmethod
    def _preprocess_depth(depth: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        """Resize single-channel depth, normalise with mean=0.5 / std=0.5 → ``[1, H, W]``."""
        DEPTH_MEAN, DEPTH_STD = 0.5, 0.5
        depth = cv2.resize(depth, (target_hw[1], target_hw[0]),
                           interpolation=cv2.INTER_LINEAR)
        depth = depth.astype(np.float32) / 255.0
        depth = (depth - DEPTH_MEAN) / DEPTH_STD
        return depth[np.newaxis, ...]  # [1, H, W]

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _preprocessing(self, rgb_image: np.ndarray, modality_x_image: np.ndarray):
        rgb = self._preprocess_rgb(rgb_image)           # [3, H, W]
        depth = self._preprocess_depth(modality_x_image, self.input_size)  # [1, H, W]
        combined = np.concatenate([rgb, depth], axis=0)[np.newaxis, ...]   # [1, 4, H, W]

        logger.debug("RGB   shape=%s  min/max=(%.3f, %.3f)", rgb.shape, rgb.min(), rgb.max())
        logger.debug("Depth shape=%s  min/max=(%.3f, %.3f)", depth.shape, depth.min(), depth.max())
        return combined

    def _inference(self, preprocessed):
        input_name = self.session.get_inputs()[0].name
        output_names = [out.name for out in self.session.get_outputs()]
        return self.session.run({input_name: preprocessed}, output_names)

    def _postprocessing(self, outputs):
        """Sigmoid → global top-K → cxcywh→xyxy → scale to original → threshold."""
        out_bbox = outputs[0][0]     # [Q, 4] normalised cxcywh
        out_logits = outputs[1][0]   # [Q, C] raw logits
        num_queries, num_classes = out_logits.shape

        # 1. Sigmoid activation
        prob = self.sigmoid(out_logits)  # [Q, C]

        # 2. Flatten & global top-K
        flat_prob = prob.reshape(-1)
        k = min(self.num_select, flat_prob.size)
        topk_idx = np.argpartition(flat_prob, -k)[-k:]
        topk_idx = topk_idx[np.argsort(flat_prob[topk_idx])[::-1]]
        scores = flat_prob[topk_idx]

        # 3. Recover query / class
        query_indices = topk_idx // num_classes
        labels = topk_idx % num_classes

        # 4. Box conversion
        all_boxes_xyxy = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = all_boxes_xyxy[query_indices]

        # 5. Scale to original pixel coords
        orig_w, orig_h = self._orig_size
        scale = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
        boxes = boxes * scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # 6. Confidence filter
        keep = scores >= self.confidence_threshold
        scores, labels, boxes = scores[keep], labels[keep], boxes[keep]

        logger.info("Detections kept: %d", len(scores))

        det_results = DetectionListResult()
        for s, l, b in zip(scores, labels, boxes):
            det_results.append(DectObject(
                xywh=b.tolist(),
                class_id=int(l),
                score=float(s),
            ))
        return det_results

