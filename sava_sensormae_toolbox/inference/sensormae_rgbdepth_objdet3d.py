"""RGB + Depth 3D object-detection model (BEV-based, CenterPoint / TransFusion head).

Supports two BEV lifting modes (auto-detected from ONNX input names):

Dense depth model inputs:
    rgb            [1, 3, 384, 384]   ImageNet-normalised RGB
    depth_vis      [1, 1, 384, 384]   Visual depth (close=bright, far=dark)
    metric_depth   [1, 1, 384, 384]   Metric depth in metres
    intrinsics_inv [1, 3, 3]          Pre-computed camera intrinsic inverse
    cam2lidar      [1, 4, 4]          Camera-to-LiDAR extrinsics

Sparse LiDAR model inputs:
    rgb            [1, 3, 384, 384]   ImageNet-normalised RGB
    depth_vis      [1, 1, 384, 384]   Visual depth (close=bright, far=dark)
    lidar_points   [N, 3]             Raw LiDAR points (x, y, z) in LiDAR frame
    intrinsics     [1, 3, 3]          Camera intrinsics (scaled to model res)
    lidar2cam      [1, 4, 4]          Pre-computed LiDAR-to-camera extrinsics

Post-processing decodes 3D bounding boxes in LiDAR frame, applies per-class
circle NMS, and can project results back onto the image as wireframes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import logging

from .objdet_base import SensorMAEObjectDetection
from ..structures import DetectionListResult, DectObject

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# KITTI-format calibration parser (matches training repo)
# ──────────────────────────────────────────────────────────────────────────────

def parse_kitti_calib(calib_path: str) -> Dict[str, np.ndarray]:
    """Parse a KITTI calibration .txt file.

    Returns dict with:
        P2             [3, 4]
        R0_rect        [3, 3]
        Tr_velo_to_cam [4, 4]
        R0_rect_4x4    [4, 4]
        intrinsics     [3, 3]
        cam2lidar      [4, 4]
    """
    data: Dict[str, np.ndarray] = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            data[key.strip()] = np.array(vals.strip().split(), dtype=np.float32)

    P2 = data["P2"].reshape(3, 4)
    R0 = data["R0_rect"].reshape(3, 3)
    Tr = data["Tr_velo_to_cam"].reshape(3, 4)
    Tr_4x4 = np.eye(4, dtype=np.float32)
    Tr_4x4[:3, :] = Tr

    R0_4x4 = np.eye(4, dtype=np.float32)
    R0_4x4[:3, :3] = R0

    cam2lidar = np.linalg.inv(Tr_4x4) @ np.linalg.inv(R0_4x4)
    intrinsics = P2[:3, :3].copy()

    return {
        "P2": P2,
        "R0_rect": R0,
        "Tr_velo_to_cam": Tr_4x4,
        "R0_rect_4x4": R0_4x4,
        "intrinsics": intrinsics,
        "cam2lidar": cam2lidar.astype(np.float32),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pure-numpy circle NMS
# ──────────────────────────────────────────────────────────────────────────────

def _circle_nms(
    centers: np.ndarray,
    scores: np.ndarray,
    min_radius_sq: float,
    post_max_size: int = 200,
) -> np.ndarray:
    """Greedy circle NMS on BEV centres. Returns kept indices."""
    order = scores.argsort()[::-1]
    centers_sorted = centers[order]
    n = len(order)
    suppressed = np.zeros(n, dtype=bool)
    keep: List[int] = []

    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(order[i])
        if len(keep) >= post_max_size:
            break
        remaining = np.arange(i + 1, n)
        if len(remaining) == 0:
            break
        remaining = remaining[~suppressed[remaining]]
        diff = centers_sorted[remaining] - centers_sorted[i]
        dist_sq = (diff ** 2).sum(axis=1)
        suppressed[remaining[dist_sq <= min_radius_sq]] = True

    return np.array(keep, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# 3D box corner computation & image projection
# ──────────────────────────────────────────────────────────────────────────────

def _box3d_corners_lidar(boxes: np.ndarray) -> np.ndarray:
    """[N, 7] (x, y, z, l, w, h, heading) → [N, 8, 3] corners in LiDAR frame."""
    N = len(boxes)
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    l, w, h = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    heading = boxes[:, 6]
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hl, hw, hh = l / 2, w / 2, h / 2

    dx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * hl[:, None]
    dy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * hw[:, None]
    dz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * hh[:, None]

    corners_x = x[:, None] + dx * cos_h[:, None] - dy * sin_h[:, None]
    corners_y = y[:, None] + dx * sin_h[:, None] + dy * cos_h[:, None]
    corners_z = z[:, None] + dz
    return np.stack([corners_x, corners_y, corners_z], axis=2)


def _project_boxes_to_image(
    boxes: np.ndarray,
    lidar2img: np.ndarray,
    image_size: Tuple[int, int],
) -> List[Optional[np.ndarray]]:
    """Project 3D boxes to 2D pixel coords.  Returns list of [8,2] or None."""
    H, W = image_size
    corners = _box3d_corners_lidar(boxes)  # [N, 8, 3]
    projected: List[Optional[np.ndarray]] = []
    for i in range(len(boxes)):
        pts_hom = np.hstack([corners[i], np.ones((8, 1))])  # [8, 4]
        proj = (lidar2img @ pts_hom.T).T  # [8, 3]
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
        if visible.sum() < 2:
            projected.append(None)
            continue
        projected.append(uv)
    return projected


# ──────────────────────────────────────────────────────────────────────────────
# 3D detection model class
# ──────────────────────────────────────────────────────────────────────────────

# Default per-class NMS radii (squared, in metres)
DEFAULT_NMS_RADII: Dict[str, float] = {
    "Car": 4.0,
    "Human": 0.175,
    "Cyclist": 0.85,
    "TwoWheeler": 0.85,
}

# 3D wireframe edge list (pairs of corner indices)
_WIREFRAME_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # pillars
]

# Per-class colours (BGR) for visualization
_CLASS_COLORS = {
    0: (0, 255, 0),    # Car — green
    1: (0, 255, 255),  # Pedestrian — yellow
    2: (0, 165, 255),  # Cyclist — orange
}


class SensorMAEObjDet_RGBDepth3D(SensorMAEObjectDetection):
    """SensorMAE 3D object detection with RGB + metric depth + calibration.

    Supports both CenterPoint and TransFusion heads.  The ``head_type``
    must match the ONNX model that was exported.
    """

    MODALITY_X_NAME = "depth"

    def __init__(
        self,
        runtime,
        *,
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
        super().__init__(
            runtime,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )
        self.input_size = tuple(input_size)  # (H, W)
        self.head_type = head_type
        self.xbound = tuple(xbound)
        self.ybound = tuple(ybound)
        self.nms_radii = nms_radii or DEFAULT_NMS_RADII
        self.score_threshold = score_threshold
        self.post_max_size = post_max_size
        self.class_names = class_names or ["Car", "Human", "Cyclist"]

        # State set during predict()
        self._calib: Optional[Dict[str, np.ndarray]] = None
        self._orig_size: Optional[Tuple[int, int]] = None  # (width, height)

    # ------------------------------------------------------------------
    # Override predict to accept calibration
    # ------------------------------------------------------------------
    def predict(
        self,
        rgb_image: np.ndarray,
        metric_depth: np.ndarray,
        calib: Union[str, Path, Dict[str, np.ndarray], None] = None,
        lidar_points: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Run 3D inference.

        Args:
            rgb_image:    BGR numpy array from ``cv2.imread``.
            metric_depth: ``float32 [H, W]`` metric depth in metres
                          (loaded from ``.npy``).
            calib:        KITTI calib dict, or path to a ``.txt`` calib file.
            lidar_points: ``float32 [N, 3]`` raw LiDAR xyz points (required
                          for sparse-LiDAR models, ignored for dense-depth).
        """
        if calib is None:
            raise ValueError("3D detection requires calibration data (calib=...)")
        if isinstance(calib, (str, Path)):
            calib = parse_kitti_calib(str(calib))
        self._calib = calib

        preprocessed = self._preprocessing(rgb_image, metric_depth,
                                           lidar_points=lidar_points)
        outputs = self._inference(preprocessed)
        return self._postprocessing(outputs)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _metric_to_visual(depth_map: np.ndarray) -> np.ndarray:
        """Metric depth [H, W] → visual depth [H, W] in [0, 1].

        Matches ``metric_depth_to_visual`` from the training pipeline.
        """
        valid = depth_map > 0
        if valid.sum() == 0:
            return np.zeros_like(depth_map, dtype=np.float32)
        lo = np.percentile(depth_map[valid], 1)
        hi = np.percentile(depth_map[valid], 99)
        clipped = np.clip(depth_map, lo, hi)
        norm = (clipped - lo) / (hi - lo + 1e-6)
        norm = 1.0 - norm
        norm[~valid] = 0.0
        return norm.astype(np.float32)

    @property
    def is_sparse(self) -> bool:
        """Auto-detect sparse vs dense from ONNX model input names."""
        input_names = [inp.name for inp in self.session.get_inputs()]
        return "lidar_points" in input_names

    def _preprocessing(self, rgb_image: np.ndarray, metric_depth: np.ndarray,
                       lidar_points: Optional[np.ndarray] = None):
        h, w = rgb_image.shape[:2]
        self._orig_size = (w, h)
        H_model, W_model = self.input_size

        # RGB: BGR→RGB, resize, ImageNet normalise → [3, H, W]
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W_model, H_model), interpolation=cv2.INTER_LINEAR)
        rgb = self.normalize_imagenet(rgb).transpose(2, 0, 1)

        # Visual depth (percentile normalisation at original res, then resize)
        # Then standardise with mean=0.5, std=0.28 to match SensorMAE pretraining
        depth_vis = self._metric_to_visual(metric_depth)
        depth_vis = cv2.resize(depth_vis, (W_model, H_model),
                               interpolation=cv2.INTER_LINEAR)
        depth_vis = (depth_vis - 0.5) / 0.28
        depth_vis = depth_vis[np.newaxis, ...]

        # Intrinsics scaled to model resolution
        sx, sy = W_model / w, H_model / h
        K = self._calib["intrinsics"].copy()
        K[0, :] *= sx
        K[1, :] *= sy

        if self.is_sparse:
            # Sparse LiDAR model: raw points + intrinsics + lidar2cam
            if lidar_points is None:
                raise ValueError(
                    "Sparse LiDAR ONNX model requires lidar_points "
                    "(pass a .bin point cloud)"
                )
            # lidar2cam = R0_rect @ Tr_velo_to_cam (inverse of cam2lidar)
            lidar2cam = (
                self._calib["R0_rect_4x4"]
                @ self._calib["Tr_velo_to_cam"]
            ).astype(np.float32)

            return {
                "rgb":          rgb[np.newaxis].astype(np.float32),
                "depth_vis":    depth_vis[np.newaxis].astype(np.float32),
                "lidar_points": lidar_points.astype(np.float32),
                "intrinsics":   K[np.newaxis].astype(np.float32),
                "lidar2cam":    lidar2cam[np.newaxis],
            }
        else:
            # Dense depth model: metric depth + intrinsics_inv + cam2lidar
            metric_resized = cv2.resize(
                metric_depth, (W_model, H_model),
                interpolation=cv2.INTER_NEAREST,
            )[np.newaxis, ...]
            K_inv = np.linalg.inv(K).astype(np.float32)
            cam2lidar = self._calib["cam2lidar"].astype(np.float32)

            return {
                "rgb":            rgb[np.newaxis].astype(np.float32),
                "depth_vis":      depth_vis[np.newaxis].astype(np.float32),
                "metric_depth":   metric_resized[np.newaxis].astype(np.float32),
                "intrinsics_inv": K_inv[np.newaxis],
                "cam2lidar":      cam2lidar[np.newaxis],
            }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _inference(self, preprocessed):
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]
        # Map ONNX input names → preprocessed arrays
        # ONNX input names are: rgb, depth_vis, metric_depth, intrinsics_inv, cam2lidar
        feed = {name: preprocessed[name] for name in input_names}
        return self.session.run(feed, output_names)

    # ------------------------------------------------------------------
    # Post-processing: decode → NMS → results
    # ------------------------------------------------------------------
    def _postprocessing(self, outputs):
        if self.head_type == "centerpoint":
            boxes, scores, labels = self._decode_centerpoint(outputs)
        else:
            boxes, scores, labels = self._decode_transfusion(outputs)

        det_results = DetectionListResult()
        for i in range(len(scores)):
            det_results.append(DectObject(
                xyzwhd=boxes[i].tolist(),
                class_id=int(labels[i]),
                score=float(scores[i]),
            ))
        return det_results

    # ── CenterPoint decode ───────────────────────────────────────────
    def _decode_centerpoint(self, outputs):
        """Decode CenterPoint ONNX outputs → (boxes [N,7], scores [N], labels [N])."""
        # ONNX output order: heatmap, offset, height, dim, rot, iou
        heatmap_raw = outputs[0][0]  # [C, Bx, By]
        offset = outputs[1][0]       # [2, Bx, By]
        height = outputs[2][0]       # [1, Bx, By]
        dim = outputs[3][0]          # [3, Bx, By]
        rot = outputs[4][0]          # [2, Bx, By]
        iou_raw = outputs[5][0] if len(outputs) > 5 else None  # [1, Bx, By]

        heatmap = self.sigmoid(heatmap_raw)
        C, Bx, By = heatmap.shape
        topk = 100

        all_boxes, all_scores, all_labels = [], [], []

        for cls in range(C):
            hm = heatmap[cls].ravel()
            k = min(topk, len(hm))
            idx = np.argpartition(hm, -k)[-k:]
            valid = hm[idx] > self.score_threshold
            idx = idx[valid]
            if len(idx) == 0:
                continue

            scores_cls = hm[idx]
            gx = idx // By
            gy = idx % By

            cx = gx.astype(np.float32) + offset[0].ravel()[idx]
            cy = gy.astype(np.float32) + offset[1].ravel()[idx]

            x_lidar = cx * self.xbound[2] + self.xbound[0]
            y_lidar = cy * self.ybound[2] + self.ybound[0]
            z_lidar = height[0].ravel()[idx]

            l = np.exp(dim[0].ravel()[idx])
            w = np.exp(dim[1].ravel()[idx])
            h = np.exp(dim[2].ravel()[idx])

            sin_r = rot[0].ravel()[idx]
            cos_r = rot[1].ravel()[idx]
            heading = np.arctan2(sin_r, cos_r)

            boxes_cls = np.stack(
                [x_lidar, y_lidar, z_lidar, l, w, h, heading], axis=1
            )

            # IoU-aware scoring: multiply heatmap score by predicted IoU quality
            if iou_raw is not None:
                iou_pred = self.sigmoid(iou_raw[0].ravel()[idx])
                scores_cls = scores_cls * iou_pred

            # Circle NMS
            cls_name = self.class_names[cls] if cls < len(self.class_names) else "Car"
            radius_sq = self.nms_radii.get(cls_name, 4.0)
            keep = _circle_nms(
                boxes_cls[:, :2], scores_cls, radius_sq, self.post_max_size
            )

            all_boxes.append(boxes_cls[keep])
            all_scores.append(scores_cls[keep])
            all_labels.append(np.full(len(keep), cls, dtype=np.int64))

        if all_boxes:
            boxes = np.concatenate(all_boxes)
            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)
            order = scores.argsort()[::-1]
            # Apply confidence threshold
            mask = scores[order] >= self.confidence_threshold
            return boxes[order][mask], scores[order][mask], labels[order][mask]

        return np.zeros((0, 7), dtype=np.float32), np.zeros(0), np.zeros(0, dtype=np.int64)

    # ── TransFusion decode ───────────────────────────────────────────
    def _decode_transfusion(self, outputs):
        """Decode TransFusion ONNX outputs → (boxes [N,7], scores [N], labels [N])."""
        # ONNX output order: dense_heatmap, center, height, dim, rot,
        #                     heatmap, query_heatmap_score, query_labels
        center = outputs[1][0]               # [2, N]
        height_pred = outputs[2][0]          # [1, N]
        dim_pred = outputs[3][0]             # [3, N]
        rot_pred = outputs[4][0]             # [2, N]
        query_heatmap = outputs[5][0]        # [C, N]
        query_heatmap_score = outputs[6][0]  # [C, N]
        query_labels = outputs[7][0]         # [N]

        C, N = query_heatmap.shape

        # Score = sigmoid(query_heatmap) * initial_heatmap_score, masked per class
        query_scores = self.sigmoid(query_heatmap)  # [C, N]
        label_mask = np.zeros_like(query_scores)
        for n in range(N):
            cls = int(query_labels[n])
            if 0 <= cls < C:
                label_mask[cls, n] = 1.0
        final_scores_all = query_scores * query_heatmap_score * label_mask  # [C, N]
        scores_per_query = final_scores_all.max(axis=0)  # [N]
        labels_per_query = query_labels.astype(np.int64)  # [N]

        # Decode boxes to world coordinates
        x_world = center[0] * self.xbound[2] + self.xbound[0]
        y_world = center[1] * self.ybound[2] + self.ybound[0]
        z_world = height_pred[0]
        l = np.exp(dim_pred[0])
        w = np.exp(dim_pred[1])
        h = np.exp(dim_pred[2])
        heading = np.arctan2(rot_pred[0], rot_pred[1])

        boxes_all = np.stack([x_world, y_world, z_world, l, w, h, heading], axis=1)

        # Per-class circle NMS
        all_boxes, all_scores, all_labels = [], [], []
        for cls in range(C):
            cls_mask = labels_per_query == cls
            if cls_mask.sum() == 0:
                continue
            cls_scores = scores_per_query[cls_mask]
            cls_boxes = boxes_all[cls_mask]

            valid = cls_scores > self.score_threshold
            if valid.sum() == 0:
                continue
            cls_scores = cls_scores[valid]
            cls_boxes = cls_boxes[valid]

            cls_name = self.class_names[cls] if cls < len(self.class_names) else "Car"
            radius_sq = self.nms_radii.get(cls_name, 4.0)
            keep = _circle_nms(
                cls_boxes[:, :2], cls_scores, radius_sq, self.post_max_size
            )

            all_boxes.append(cls_boxes[keep])
            all_scores.append(cls_scores[keep])
            all_labels.append(np.full(len(keep), cls, dtype=np.int64))

        if all_boxes:
            boxes = np.concatenate(all_boxes)
            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)
            order = scores.argsort()[::-1]
            mask = scores[order] >= self.confidence_threshold
            return boxes[order][mask], scores[order][mask], labels[order][mask]

        return np.zeros((0, 7), dtype=np.float32), np.zeros(0), np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def compute_lidar2img(self, scale: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """Compute lidar-to-image projection matrix [3, 4] from stored calib.

        Args:
            scale: (sx, sy) to apply to the projection matrix rows
                   (e.g. if visualising on a resized image).
        """
        lidar2img = (
            self._calib["P2"]
            @ self._calib["R0_rect_4x4"]
            @ self._calib["Tr_velo_to_cam"]
        )
        sx, sy = scale
        lidar2img[0, :] *= sx
        lidar2img[1, :] *= sy
        return lidar2img[:3, :]

    def draw_3d_boxes(
        self,
        image: np.ndarray,
        results: DetectionListResult,
        lidar2img: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Draw projected 3D wireframe boxes on an image.

        Args:
            image:       BGR image to draw on (modified in-place).
            results:     Detection results from ``predict()``.
            lidar2img:   [3, 4] projection matrix.
            class_names: Optional list of class names for the label text.

        Returns:
            The annotated image.
        """
        if len(results) == 0:
            return image

        H, W = image.shape[:2]
        boxes = np.array([d.xyzwhd for d in results])  # [N, 7]
        labels = np.array([d.class_id for d in results])
        scores = np.array([d.score for d in results])

        projected = _project_boxes_to_image(boxes, lidar2img, (H, W))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.45, 1

        for i, uv in enumerate(projected):
            if uv is None:
                continue
            cls_idx = int(labels[i])
            color = _CLASS_COLORS.get(cls_idx, (0, 255, 0))
            pts = uv.astype(np.int32)

            for a, b in _WIREFRAME_EDGES:
                cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, 2)

            # Label text at top-front corner
            lbl = (class_names[cls_idx]
                   if class_names and cls_idx < len(class_names)
                   else str(cls_idx))
            text = f"{lbl} {scores[i]:.2f}"
            tx, ty = int(pts[0, 0]), int(pts[0, 1]) - 5
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(image, (tx, ty - th - baseline),
                          (tx + tw, ty + baseline), color, cv2.FILLED)
            cv2.putText(image, text, (tx, ty), font, font_scale,
                        (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        return image

    def save_results(
        self,
        output_path: str,
        rgb_image: np.ndarray,
        depth_vis: np.ndarray,
        result_image: np.ndarray,
    ) -> None:
        """Save side-by-side panel: RGB | Depth_vis | 3D result."""
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        h, w = rgb_image.shape[:2]

        def _to_bgr3(img, target_hw=(h, w)):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            if img.shape[:2] != target_hw:
                img = cv2.resize(img, (target_hw[1], target_hw[0]),
                                 interpolation=cv2.INTER_LINEAR)
            return img

        combined = np.hstack((
            _to_bgr3(rgb_image),
            _to_bgr3(depth_vis),
            _to_bgr3(result_image),
        ))
        cv2.imwrite(output_path, combined)
