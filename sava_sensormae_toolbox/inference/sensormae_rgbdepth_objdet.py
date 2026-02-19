from .base import Model
from typing import List, Optional, Tuple
import numpy as np
from ..utils.runtime import ONNXRuntime
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import logging
from sava_sensormae_toolbox.structures import DetectionListResult, DectObject
logger = logging.getLogger(__name__)
import os

class SensorMAEObjDet_RGBDepth(Model):

    def __init__(self, runtime: ONNXRuntime, num_classes: int = 20, confidence_threshold: float = 0.0, input_size: Tuple[int, int] = (384, 384), num_select: int = 300):
        self.session = runtime
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.num_select = num_select

    def __call__(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[np.ndarray]:
        return self.det_image(rgb_image, depth_image)
    
    @staticmethod
    def softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)
    
    @staticmethod
    def box_cxcywh_to_xyxy(x: np.ndarray) -> np.ndarray:
        """
        Convert [cx, cy, w, h] box format to [x_min, y_min, x_max, y_max].
        Args:
            x: numpy array of shape (..., 4)
        Returns:
            numpy array of shape (..., 4)
        """
        x_c, y_c, w, h = np.moveaxis(x, -1, 0)  # like torch.unbind(-1)

        w = np.clip(w, a_min=0.0, a_max=None)
        h = np.clip(h, a_min=0.0, a_max=None)

        b = [
            x_c - 0.5 * w,
            y_c - 0.5 * h,
            x_c + 0.5 * w,
            y_c + 0.5 * h,
        ]
        return np.stack(b, axis=-1)

    def det_image(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[np.ndarray]:
        combined_tensor, orig_size = self._preprocessing(rgb_image, depth_image) # [1, 3+1, H, W]
        outputs = self._inference(combined_tensor)
        results = self._postprocessing(outputs, orig_size)
        return results
    
    @staticmethod
    def _resize_and_pad(image, size=640, pad_value=0, pad_mask_value=0):
        h, w = image.shape[:2]

        # --- Step 1: Resize (LongestMaxSize) ---
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # --- Step 2: Pad (PadIfNeeded, position="top_left") ---
        pad_bottom = size - new_h
        pad_right = size - new_w

        padded = cv2.copyMakeBorder(
            resized,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,  # background value
        )

        return padded
    
    @staticmethod
    def _apply_colormap(mask: np.ndarray, num_classes: int = 21) -> np.ndarray:
        """Convert class indices in mask to RGB color using matplotlib colormap."""
        colormap = plt.cm.get_cmap("tab20", num_classes)
        colored_mask = colormap(mask.astype(int))[:, :, :3]  # Drop alpha channel
        return (colored_mask * 255).astype(np.uint8)
    
    def _preprocess_rgb(self, rgb: np.array, input_size: tuple) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load RGB image, resize, normalize → [3, H, W] float32.  Returns (array, original_size)."""
        RGB_MEAN = [0.485, 0.456, 0.406]
        RGB_STD  = [0.229, 0.224, 0.225]
    
        orig_w, orig_h = rgb.size
        img_resized = rgb.resize(input_size, Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = (arr - np.array(RGB_MEAN, dtype=np.float32)) / np.array(RGB_STD, dtype=np.float32)
        return arr.transpose(2, 0, 1), (orig_w, orig_h)

    def _preprocess_depth(self, depth: np.array, input_size: tuple) -> np.ndarray:
        """Load single-channel depth PNG, resize, normalize → [1, H, W] float32."""
        DEPTH_MEAN = [0.5]
        DEPTH_STD  = [0.5]

        img_resized = depth.resize(input_size, Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = (arr - np.array(DEPTH_MEAN, dtype=np.float32)) / np.array(DEPTH_STD, dtype=np.float32)
        return arr[np.newaxis, ...]

    def _inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        input_name = self.session.get_inputs()[0].name
        output_names = [out.name for out in self.session.get_outputs()]
        # Prepare input data for the ONNX Runtime session
        results = self.session.run({input_name: input_tensor}, output_names)
        return results


    def _preprocessing(self, rgb_image: np.ndarray, depth_image: np.ndarray):
        input_size = self.input_size
        rgb, orig_size = self._preprocess_rgb(rgb_image, input_size)
        depth = self._preprocess_depth(depth_image, input_size)
        combined = np.concatenate([rgb, depth], axis=0)[np.newaxis, ...]  # [1, 4, H, W]

        logger.debug("Input RGB shape: %s min/max: (%s, %s)", rgb.shape, rgb.min(), rgb.max())
        logger.debug("Input Depth shape: %s min/max: (%s, %s)", depth.shape, depth.min(), depth.max())

        return combined, orig_size

    def scale_draw_boxes(
        self,
        boxes: np.ndarray,
        image: np.ndarray,
        labels: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes (and optional labels/scores) on the image.

        Args:
            boxes: (N, 4) array of [x_min, y_min, x_max, y_max] in absolute pixel coords.
            image: RGB image (H, W, 3) to draw on.
            labels: Optional (N,) int array of class indices.
            scores: Optional (N,) float array of confidence scores.
            class_names: Optional list mapping class index → name string.

        Returns:
            Image with boxes (and labels) drawn.
        """
        boxes = np.array(boxes).astype(np.int32)
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        for i, box in enumerate(boxes):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
            if labels is not None or scores is not None:
                parts = []
                if labels is not None:
                    cls_idx = int(labels[i])
                    name = class_names[cls_idx] if (class_names and cls_idx < len(class_names)) else str(cls_idx)
                    parts.append(name)
                if scores is not None:
                    parts.append(f"{float(scores[i]):.2f}")
                text = " ".join(parts)
                # Place text just above the top-left corner of the box
                text_x = box[0]
                text_y = max(box[1] - 4, 10)
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(image, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), color, cv2.FILLED)
                cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        return image
    
    def _postprocessing(self, results: List[np.ndarray], orig_size: Tuple[int, int]) -> DetectionListResult:
        """Post-process raw ONNX outputs using the same top-K strategy as PyTorch RF-DETR.

        Steps (mirroring ``PostProcess`` in ``rfdetr/models/lwdetr.py``):
        1. Sigmoid on raw logits.
        2. Flatten queries × classes, select global top-K highest scores.
        3. Derive query index and class label from flattened indices.
        4. Convert boxes from normalised cxcywh → xyxy.
        5. Gather the top-K boxes and scale to original image dimensions.
        6. Filter by confidence threshold.
        """
        out_bbox   = results[0][0]   # [num_queries, 4]  — normalised cxcywh
        out_logits = results[1][0]   # [num_queries, num_classes] — raw logits

        num_queries, num_classes = out_logits.shape

        # 1. Sigmoid activation (focal-loss style, not softmax)
        prob = 1.0 / (1.0 + np.exp(-out_logits))  # [Q, C]

        # 2. Flatten and top-K selection across all (query, class) pairs
        flat_prob = prob.reshape(-1)                          # [Q * C]
        k = min(self.num_select, flat_prob.size)
        topk_indices = np.argpartition(flat_prob, -k)[-k:]    # unordered top-K
        topk_indices = topk_indices[np.argsort(flat_prob[topk_indices])[::-1]]  # sort descending
        scores = flat_prob[topk_indices]

        # 3. Recover query index and class label
        query_indices = topk_indices // num_classes
        labels = topk_indices % num_classes

        # 4. Convert all boxes cxcywh → xyxy, then gather top-K
        all_boxes_xyxy = self.box_cxcywh_to_xyxy(out_bbox)    # [Q, 4]
        boxes = all_boxes_xyxy[query_indices]                  # [K, 4]

        # 5. Scale from normalised [0, 1] to absolute pixel coords
        orig_w, orig_h = orig_size
        scale = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
        boxes = boxes * scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # 6. Confidence threshold filter
        keep = scores >= self.confidence_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes  = boxes[keep]

        print(f"Post-processing results: {len(scores)} detections kept after confidence thresholding.")
        print(f"Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Labels shape: {labels.shape}")
        print(f"Boxes sample (first 5): {boxes[:5]}")
        print(f"Labels sample (first 5): {labels[:5]}")
        print(f"Scores sample (first 5): {scores[:5]}")
        # Pack into DetectionListResult (one DectObject per detection)
        det_results = DetectionListResult()
        for score, label, box in zip(scores, labels, boxes):
            det_results.append(DectObject(
                xywh=box.tolist(),
                class_id=int(label),
                score=float(score),
            ))

        return det_results
    
    @staticmethod
    def save_results(output_path: str, rgb_image: np.ndarray, depth_image: np.ndarray, colored_mask: np.ndarray) -> None:
        """
        Save the inference results to disk as a side-by-side panel: RGB | Thermal | Segmentation.

        Args:
            output_path (str): Path to save the output image.
            rgb_image (np.ndarray): Original RGB image.
            depth_image (np.ndarray): Original depth image.
            colored_mask (np.ndarray): Colored segmentation mask (H, W, 3) or (H, W).
        """

        import cv2

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        h, w = rgb_image.shape[:2]

        # Prepare thermal visualization to match RGB size and 3 channels
        depth_vis = depth_image
        if depth_vis.shape[:2] != (h, w):
            depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_LINEAR)
        if depth_vis.ndim == 2:
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        elif depth_vis.ndim == 3 and depth_vis.shape[2] == 1:
            depth_vis = np.repeat(depth_vis, 3, axis=2)

        # Prepare segmentation visualization to match RGB size and 3 channels
        seg_vis = colored_mask
        if seg_vis.ndim == 2:
            # grayscale mask -> 3ch for visualization
            seg_vis = cv2.cvtColor(seg_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if seg_vis.shape[:2] != (h, w):
            seg_vis = cv2.resize(seg_vis, (w, h), interpolation=cv2.INTER_NEAREST)
        if seg_vis.ndim == 3 and seg_vis.shape[2] == 1:
            seg_vis = np.repeat(seg_vis, 3, axis=2)

        # Ensure RGB is 3-channel BGR for stacking
        rgb_vis = rgb_image
        if rgb_vis.ndim == 2:
            rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_GRAY2BGR)

        # Stack images horizontally: RGB | Thermal | Segmentation (no overlay)
        combined = np.hstack((rgb_vis, depth_vis, seg_vis))

        # Save the combined panel
        cv2.imwrite(output_path, combined)

