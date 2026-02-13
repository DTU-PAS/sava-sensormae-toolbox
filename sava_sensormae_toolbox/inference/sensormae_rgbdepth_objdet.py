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

class SensorMAEObjDet_RGBDepth(Model):

    def __init__(self, runtime: ONNXRuntime, num_classes: int = 20, confidence_threshold: float = 0.0):
        self.session = runtime
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

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
    
    def _preprocess_rgb(rgb_path: str, target_h: int, target_w: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load RGB image, resize, normalize → [3, H, W] float32.  Returns (array, original_size)."""
        RGB_MEAN = [0.485, 0.456, 0.406]
        RGB_STD  = [0.229, 0.224, 0.225]
        
        img = Image.open(rgb_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((target_w, target_h), Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = (arr - np.array(RGB_MEAN, dtype=np.float32)) / np.array(RGB_STD, dtype=np.float32)
        return arr.transpose(2, 0, 1), (orig_w, orig_h)

    def _preprocess_depth(depth_path: str, target_h: int, target_w: int) -> np.ndarray:
        """Load single-channel depth PNG, resize, normalize → [1, H, W] float32."""
        DEPTH_MEAN = [0.5]
        DEPTH_STD  = [0.5]

        img = Image.open(depth_path).convert("L")
        img_resized = img.resize((target_w, target_h), Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = (arr - np.array(DEPTH_MEAN, dtype=np.float32)) / np.array(DEPTH_STD, dtype=np.float32)
        return arr[np.newaxis, ...]

    def _inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        input_name = self.session.get_inputs()[0].name
        output_names = [out.name for out in self.session.get_outputs()]
        # Prepare input data for the ONNX Runtime session
        results = self.session.run(output_names, {input_name: input_tensor})
        return results


    def _preprocessing(self, rgb_image: np.ndarray, depth_image: np.ndarray):
        rgb, orig_size = self._preprocess_rgb(rgb_image)
        depth = self._preprocess_depth(depth_image)
        combined = np.concatenate([rgb, depth], axis=0)[np.newaxis, ...]  # [1, 4, H, W]

        logger.debug("Input RGB shape: %s min/max: (%s, %s)", rgb.shape, rgb.min(), rgb.max())
        logger.debug("Input Depth shape: %s min/max: (%s, %s)", depth.shape, depth.min(), depth.max())

        return combined, orig_size

    def scale_draw_boxes(self, boxes:np.ndarray, image:np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale_up = max(h, w) # This is because the image passed is 640 * 640 padded. If in the future we pass arbitrary image sizes, we need to change this.
        boxes = np.array(boxes)
        boxes[:, 0] *= scale_up  # x_min
        boxes[:, 1] *= scale_up  # y_min
        boxes[:, 2] *= scale_up  # x_max
        boxes[:, 3] *= scale_up  # y_max
        boxes = boxes.astype(np.int32)
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        return image
    
    def _postprocessing(self, results: List[np.ndarray], orig_size: Tuple[int, int]) -> List[np.ndarray]:
        out_bbox  = results[0][0]   # [num_queries, 4]
        out_logits = results[1][0]   # [num_queries, num_classes]
        # prob = self.softmax(out_logits, -1)
        prob = 1.0 / (1.0 + np.exp(-out_logits))  # sigmoid
        scores = np.max(prob, axis=-1)
        labels = np.argmax(prob, axis=-1)

        # Confidence filter
        keep = scores >= self.confidence_threshold
        if keep.sum() == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), []

        out_bbox = out_bbox[keep]
        scores = scores[keep]
        labels = labels[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes[:, [0, 2]] *= orig_size[0]  # orig_w
        boxes[:, [1, 3]] *= orig_size[1]  # orig_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_size[0])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_size[1])

        # filter detections by confidence threshold
        results = DetectionListResult()
        for score, label, box in zip(scores, labels, boxes, strict=True):
            no_class_filter = label != self.num_classes
            conf_filter = score > self.confidence_threshold
            keep = no_class_filter & conf_filter
            results.append(DectObject(
                xywh=box[keep].tolist(),
                class_id=label[keep].tolist(),
                score=score[keep].tolist()
            ))
        
        return results

