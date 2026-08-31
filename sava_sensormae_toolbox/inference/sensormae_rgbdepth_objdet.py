"""RGB + Depth 2D object detection (concatenated ONNX input, sigmoid head)."""

from typing import Tuple

import cv2
import numpy as np

from ..structures import DectObject, DetectionListResult
from .objdet_base import SensorMAEObjectDetection


class SensorMAEObjDet_RGBDepth(SensorMAEObjectDetection):
    """Single-input ONNX model: concatenated ``[1, 4, H, W]`` (3 RGB + 1 depth)."""

    def __init__(
        self,
        runtime,
        *,
        num_classes: int = 20,
        confidence_threshold: float = 0.0,
        input_size: Tuple[int, int] = (640, 640),
        num_select: int = 300,
        **kwargs
    ):
        super().__init__(
            runtime,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        self.input_size = tuple(input_size)
        self.num_select = num_select
        self._orig_size: Tuple[int, int] | None = None  # (W, H)

    def _preprocessing(self, rgb_image, modality_x_image, **kwargs):
        h, w = rgb_image.shape[:2]
        self._orig_size = (w, h)

        rgb = self._preprocess_rgb(rgb_image)  # (H_pad, W_pad, 3)
        depth = self._preprocess_depth(modality_x_image)  # (H_pad, W_pad)

        combined = np.concatenate([rgb.transpose(2, 0, 1), depth[np.newaxis]], axis=0)[
            np.newaxis
        ].astype(np.float32)

        input_name = self.session.get_inputs()[0].name
        return {input_name: combined}

    def _postprocessing(self, outputs):
        out_bbox = outputs[0][0]  # [Q, 4] normalised cxcywh
        out_logits = outputs[1][0]  # [Q, C] raw logits
        num_queries, num_classes = out_logits.shape

        prob = self.sigmoid(out_logits)
        flat_prob = prob.reshape(-1)
        k = min(self.num_select, flat_prob.size)
        topk_idx = np.argpartition(flat_prob, -k)[-k:]
        topk_idx = topk_idx[np.argsort(flat_prob[topk_idx])[::-1]]
        scores = flat_prob[topk_idx]

        query_indices = topk_idx // num_classes
        labels = topk_idx % num_classes

        boxes = self.box_cxcywh_to_xyxy(out_bbox)[query_indices]
        orig_w, orig_h = self._orig_size
        scale = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
        boxes = boxes * scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        keep = scores >= self.confidence_threshold
        scores, labels, boxes = scores[keep], labels[keep], boxes[keep]

        det_results = DetectionListResult()
        for sc, lb, bx in zip(scores, labels, boxes):
            det_results.append(
                DectObject(
                    xywh=bx.tolist(),
                    class_id=int(lb),
                    score=float(sc),
                )
            )
        return det_results

    def _preprocess_rgb(self, image: np.ndarray) -> np.ndarray:
        """BGR→RGB, ImageNet normalise, pad to square."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = self.normalize_imagenet(rgb)
        return cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

    def _preprocess_depth(self, metric_depth: np.ndarray) -> np.ndarray:
        """Metric depth → visual depth [0,1] → (x - 0.5) / 0.28, resize to square."""
        depth_vis = self._metric_to_visual(metric_depth)
        depth_vis = (depth_vis - 0.5) / 0.28
        return cv2.resize(depth_vis, self.input_size, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _metric_to_visual(depth_map: np.ndarray) -> np.ndarray:
        """Metric depth [H, W] → visual depth [H, W] in [0, 1].

        Missing/invalid pixels (depth <= 0) are set to 0.5 (the SensorMAE
        depth mean), so they normalise to 0.0 — a neutral "no information"
        signal. Matches the 2D training pipeline (metric_depth_to_visual_pil).
        """
        valid = depth_map > 0
        if valid.sum() == 0:
            return np.full_like(depth_map, 0.5, dtype=np.float32)
        lo = np.percentile(depth_map[valid], 1)
        hi = np.percentile(depth_map[valid], 99)
        clipped = np.clip(depth_map, lo, hi)
        norm = 1.0 - (clipped - lo) / (hi - lo + 1e-6)
        norm[~valid] = 0.5
        return norm.astype(np.float32)
