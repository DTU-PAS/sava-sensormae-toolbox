"""RGB + Depth 3D object detection (BEV-based, CenterPoint / TransFusion head).

Post-processing decodes 3D bounding boxes in LiDAR frame, applies per-class
circle NMS, and can project results back onto the image as wireframes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .objdet_base import SensorMAEObjectDetection
from ..structures import DetectionListResult, DectObject

DEFAULT_NMS_RADII: Dict[str, float] = {
    "Car": 4.0, "Human": 0.175, "Cyclist": 0.85, "TwoWheeler": 0.85,
}

_WIREFRAME_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

_CLASS_COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 165, 255)}

class SensorMAEObjDet_RGBDepth3D(SensorMAEObjectDetection):
    """BEV-based 3D detection with RGB + metric depth + calibration.

    Accepts ``calib`` and ``lidar_points`` as kwargs through ``predict()``.
    Calibration can be a path to a KITTI ``.txt`` file or a pre-parsed dict.
    """

    def __init__(
        self, runtime, *,
        num_classes: int = 3,
        confidence_threshold: float = 0.3,
        input_size: Tuple[int, int] = (384, 384),
        head_type: str = "centerpoint",
        xbound: Tuple[float, float, float] = (0.0, 51.2, 0.4),
        ybound: Tuple[float, float, float] = (-20.0, 20.0, 0.4),
        nms_radii: Optional[Dict[str, float]] = None,
        score_threshold: float = 0.1,
        post_max_size: int = 83,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(runtime, num_classes=num_classes,
                         confidence_threshold=confidence_threshold, **kwargs)
        self.input_size = tuple(input_size)
        self.head_type = head_type
        self.xbound = tuple(xbound)
        self.ybound = tuple(ybound)
        self.nms_radii = nms_radii or DEFAULT_NMS_RADII
        self.score_threshold = score_threshold
        self.post_max_size = post_max_size
        self.class_names = class_names or ["Car", "Human", "Cyclist"]
        self.calib: Optional[Dict[str, np.ndarray]] = None
        self._orig_size: Optional[Tuple[int, int]] = None

    def _preprocess_rgb(self, image: np.ndarray) -> np.ndarray:
        """BGR→RGB, ImageNet normalise, resize to input_size."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = self.normalize_imagenet(rgb)
        return cv2.resize(rgb, (self.input_size[1], self.input_size[0]),
                          interpolation=cv2.INTER_LINEAR)

    def _preprocess_depth(self, metric_depth: np.ndarray) -> np.ndarray:
        """Metric depth → visual depth [0,1] → (x - 0.5) / 0.28, resize to input_size."""
        depth_vis = self._metric_to_visual(metric_depth)
        depth_vis = (depth_vis - 0.5) / 0.28
        return cv2.resize(depth_vis, (self.input_size[1], self.input_size[0]),
                          interpolation=cv2.INTER_LINEAR)

    def _preprocessing(self, rgb_image, metric_depth, *, lidar_points, calib, **kwargs):
        h, w = rgb_image.shape[:2]
        self._orig_size = (w, h)
        self.calib = calib
        target_h, target_w = self.input_size

        rgb = self._preprocess_rgb(rgb_image)             # (target_h, target_w, 3)
        depth_vis = self._preprocess_depth(metric_depth)  # (target_h, target_w)

        # Intrinsics scaling for direct resize
        sx = target_w / w
        sy = target_h / h
        K = calib["intrinsics"]
        K[0, :] *= sx
        K[1, :] *= sy

        rgb_tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32)
        depth_vis_tensor = depth_vis[np.newaxis, np.newaxis].astype(np.float32)

        if self.is_sparse:
            if lidar_points is None:
                raise ValueError("Sparse ONNX model requires lidar_points")
            lidar2cam = (calib["R0_rect_4x4"]
                         @ calib["Tr_velo_to_cam"]).astype(np.float32)
            return {
                "rgb":          rgb_tensor,
                "depth_vis":    depth_vis_tensor,
                "lidar_points": lidar_points.astype(np.float32),
                "intrinsics":   K[np.newaxis].astype(np.float32),
                "lidar2cam":    lidar2cam[np.newaxis],
            }

        # Dense model: metric depth resized to model resolution
        metric_resized = cv2.resize(
            metric_depth, (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return {
            "rgb":            rgb_tensor,
            "depth_vis":      depth_vis_tensor,
            "metric_depth":   metric_resized[np.newaxis, np.newaxis].astype(np.float32),
            "intrinsics_inv": np.linalg.inv(K).astype(np.float32)[np.newaxis],
            "cam2lidar":      calib["cam2lidar"].astype(np.float32)[np.newaxis],
        }

    def _postprocessing(self, outputs):
        
        boxes, scores, labels = self._decode(outputs)
        det_results = DetectionListResult()
        for i in range(len(scores)):
            det_results.append(DectObject(
                xyzwhd=boxes[i].tolist(),
                class_id=int(labels[i]),
                score=float(scores[i]),
            ))
        return det_results

    def _decode(self, outputs):
        heatmap = self.sigmoid(outputs[0][0])  # [C, Bx, By]
        offset = outputs[1][0]                 # [2, Bx, By]
        height = outputs[2][0]                 # [1, Bx, By]
        dim = outputs[3][0]                    # [3, Bx, By]
        rot = outputs[4][0]                    # [2, Bx, By]
        iou_raw = outputs[5][0] if len(outputs) > 5 else None

        C, _, By = heatmap.shape
        all_boxes, all_scores, all_labels = [], [], []

        for cls in range(C):
            hm = heatmap[cls].ravel()
            k = min(100, len(hm))
            idx = np.argpartition(hm, -k)[-k:]
            idx = idx[hm[idx] > self.score_threshold]
            if len(idx) == 0:
                continue

            scores_cls = hm[idx]
            gx, gy = idx // By, idx % By
            cx = gx.astype(np.float32) + offset[0].ravel()[idx]
            cy = gy.astype(np.float32) + offset[1].ravel()[idx]

            boxes_cls = np.stack([
                cx * self.xbound[2] + self.xbound[0],
                cy * self.ybound[2] + self.ybound[0],
                height[0].ravel()[idx],
                np.exp(dim[0].ravel()[idx]),
                np.exp(dim[1].ravel()[idx]),
                np.exp(dim[2].ravel()[idx]),
                np.arctan2(rot[0].ravel()[idx], rot[1].ravel()[idx]),
            ], axis=1)

            # if iou_raw is not None:
            #     scores_cls = scores_cls * self.sigmoid(iou_raw[0].ravel()[idx])

            cls_name = self.class_names[cls] if cls < len(self.class_names) else "Car"
            keep = self._circle_nms(boxes_cls[:, :2], scores_cls,
                               self.nms_radii.get(cls_name, 4.0), self.post_max_size)
            all_boxes.append(boxes_cls[keep])
            all_scores.append(scores_cls[keep])
            all_labels.append(np.full(len(keep), cls, dtype=np.int64))

        if all_boxes:
            boxes, scores, labels = (np.concatenate(all_boxes),
                                     np.concatenate(all_scores),
                                     np.concatenate(all_labels))
            order = scores.argsort()[::-1]
            mask = scores[order] >= self.confidence_threshold
            return boxes[order][mask], scores[order][mask], labels[order][mask]
        return np.zeros((0, 7), np.float32), np.zeros(0), np.zeros(0, dtype=np.int64)

    def compute_lidar2img(self, scale: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """Compute lidar-to-image projection [3, 4] from stored calib."""
        lidar2img = (self.calib["P2"]
                     @ self.calib["R0_rect_4x4"]
                     @ self.calib["Tr_velo_to_cam"])
        sx, sy = scale
        lidar2img[0, :] *= sx
        lidar2img[1, :] *= sy
        return lidar2img[:3, :]

    def draw_3d_boxes(self, image, results, lidar2img, class_names=None):
        """Draw projected 3D wireframe boxes on an image."""
        if len(results) == 0:
            return image

        H, W = image.shape[:2]
        boxes = np.array([d.xyzwhd for d in results])
        labels = np.array([d.class_id for d in results])
        scores = np.array([d.score for d in results])
        projected = self._project_boxes_to_image(boxes, lidar2img, (H, W))

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, uv in enumerate(projected):
            if uv is None:
                continue
            cls_idx = int(labels[i])
            color = _CLASS_COLORS.get(cls_idx, (0, 255, 0))
            pts = uv.astype(np.int32)
            for a, b in _WIREFRAME_EDGES:
                cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, 1)
            lbl = (class_names[cls_idx]
                   if class_names and cls_idx < len(class_names)
                   else str(cls_idx))
            text = f"{lbl} {scores[i]:.2f}"
            tx, ty = int(pts[0, 0]), int(pts[0, 1]) - 5
            (tw, th), baseline = cv2.getTextSize(text, font, 0.30, 1)
            cv2.rectangle(image, (tx, ty - th - baseline),
                          (tx + tw, ty + baseline), color, cv2.FILLED)
            cv2.putText(image, text, (tx, ty), font, 0.30,
                        (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        return image

    @staticmethod
    def _metric_to_visual(depth_map: np.ndarray) -> np.ndarray:
        """Metric depth [H, W] → visual depth [H, W] in [0, 1]."""
        valid = depth_map > 0
        if valid.sum() == 0:
            return np.zeros_like(depth_map, dtype=np.float32)
        lo = np.percentile(depth_map[valid], 1)
        hi = np.percentile(depth_map[valid], 99)
        clipped = np.clip(depth_map, lo, hi)
        norm = 1.0 - (clipped - lo) / (hi - lo + 1e-6)
        norm[~valid] = 0.0
        return norm.astype(np.float32)
    
    @staticmethod
    def _circle_nms(centers, scores, min_radius_sq, post_max_size=200):
        """Greedy circle NMS on BEV centres. Returns kept indices."""
        order = scores.argsort()[::-1]
        centers_sorted = centers[order]
        suppressed = np.zeros(len(order), dtype=bool)
        keep: List[int] = []
        for i in range(len(order)):
            if suppressed[i]:
                continue
            keep.append(order[i])
            if len(keep) >= post_max_size:
                break
            remaining = np.arange(i + 1, len(order))
            if len(remaining) == 0:
                break
            remaining = remaining[~suppressed[remaining]]
            dist_sq = ((centers_sorted[remaining] - centers_sorted[i]) ** 2).sum(axis=1)
            suppressed[remaining[dist_sq <= min_radius_sq]] = True
        return np.array(keep, dtype=np.int64)

    @staticmethod
    def _box3d_corners_lidar(boxes):
        """[N, 7] (x, y, z, l, w, h, heading) → [N, 8, 3] corners in LiDAR frame."""
        x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
        l, w, h = boxes[:, 3], boxes[:, 4], boxes[:, 5]
        cos_h, sin_h = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
        hl, hw, hh = l / 2, w / 2, h / 2

        dx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * hl[:, None]
        dy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * hw[:, None]
        dz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * hh[:, None]

        corners_x = x[:, None] + dx * cos_h[:, None] - dy * sin_h[:, None]
        corners_y = y[:, None] + dx * sin_h[:, None] + dy * cos_h[:, None]
        corners_z = z[:, None] + dz
        return np.stack([corners_x, corners_y, corners_z], axis=2)

    @staticmethod
    def _project_boxes_to_image(boxes, lidar2img, image_size):
        """Project 3D boxes to 2D pixel coords. Returns list of [8,2] or None."""
        H, W = image_size
        corners = SensorMAEObjDet_RGBDepth3D._box3d_corners_lidar(boxes)
        projected = []
        for i in range(len(boxes)):
            pts_hom = np.hstack([corners[i], np.ones((8, 1))])
            proj = (lidar2img @ pts_hom.T).T
            in_front = proj[:, 2] > 0.1
            if in_front.sum() < 2:
                projected.append(None)
                continue
            uv = proj[:, :2] / proj[:, 2:3].clip(min=0.1)
            visible = (
                (uv[:, 0] >= -W * 0.5) & (uv[:, 0] < W * 1.5)
                & (uv[:, 1] >= -H * 0.5) & (uv[:, 1] < H * 1.5)
                & in_front
            )
            projected.append(uv if visible.sum() >= 2 else None)
        return projected
    
    @property
    def is_sparse(self) -> bool:
        """Auto-detect sparse vs dense from ONNX model input names."""
        return "lidar_points" in [inp.name for inp in self.session.get_inputs()]
