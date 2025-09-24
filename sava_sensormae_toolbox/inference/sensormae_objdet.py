
from .base import Model
from typing import List, Optional, Tuple
import numpy as np
from ..utils.runtime import ONNXRuntime
import cv2
import matplotlib.pyplot as plt
import logging
from sava_sensormae_toolbox.structures import DetectionListResult, DectObject
logger = logging.getLogger(__name__)

class SensorMAEObjDet(Model):

    def __init__(self, runtime: ONNXRuntime, num_classes: int = 20, confidence_threshold: float = 0.0):
        self.session = runtime
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

    def __call__(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> List[np.ndarray]:
        return self.det_image(rgb_image, thermal_image)
    
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

    def det_image(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> List[np.ndarray]:
        rgb_tensor, thermal_tensor = self._preprocessing(rgb_image, thermal_image)
        outputs = self._inference(rgb_tensor, thermal_tensor)
        results = self._postprocessing(outputs)
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
    
    @staticmethod
    def _normalize_rgb(image):
        """
        Normalize RGB image to ImageNet stats.
        Args:
            image: np.ndarray, shape (H, W, 3), dtype uint8 or float32
        Returns:
            np.ndarray, float32 normalized
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = image.astype(np.float32) / 255.0  # max_pixel_value=255
        image = (image - mean) / std
        return image


    @staticmethod
    def _normalize_thermal(image):
        """
        Normalize thermal image.
        Args:
            image: np.ndarray, shape (H, W) or (H, W, 1), dtype float32
                expected already in range [0, 1] since max_pixel_value=1
        Returns:
            np.ndarray, float32 normalized
        """
        mean = 0.5
        std  = 0.28

        image = (image - image.min())/(image.max() - image.min())  # normalize to [0,1]
        image = image.astype(np.float32)  # assume already scaled to [0,1]
        image = (image - mean) / std
        return image
    
    def _preprocess_rgb(self, rgb: np.ndarray, input_size=(640, 640)):
        """Read RGB image, resize, normalize, and convert to NCHW float32."""    
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = self._normalize_rgb(rgb)
        rgb = self._resize_and_pad(rgb)
        return rgb


    def _preprocess_thermal(self, thermal: np.ndarray, input_size=(640, 640)):
        """Read single-channel thermal image or return zeros if not given."""
        THERMAL_MEAN = 0.5
        THERMAL_STD = 0.28
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))

        thermal = clahe.apply(thermal.astype(np.uint16))
        thermal = cv2.normalize(thermal.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        thermal = (thermal - THERMAL_MEAN) / THERMAL_STD    
        thermal = self._resize_and_pad(thermal, pad_value=THERMAL_MEAN, pad_mask_value=0)
        return thermal

    def _inference(self, rgb_tensor: np.ndarray, thermal_tensor: np.ndarray) -> List[np.ndarray]:
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]
        # Prepare input data for the ONNX Runtime session
        outputs = self.session.run(
            {
                input_names[0]: np.ascontiguousarray(rgb_tensor),
                input_names[1]: np.ascontiguousarray(thermal_tensor),
            },
            None,
        )
        return outputs


    def _preprocessing(self, rgb_image: np.ndarray, thermal_image: np.ndarray):
        rgb = self._preprocess_rgb(rgb_image)
        thermal = self._preprocess_thermal(thermal_image)
        
        # add batch dimension to rgb
        rgb = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32)  # (1,3,H,W)
        thermal = np.expand_dims(np.expand_dims(thermal, axis=0), axis=0).astype(np.float32)  # (1,1,H,W)
        logger.debug("Input RGB shape: %s min/max: (%s, %s)", rgb.shape, rgb.min(), rgb.max())
        logger.debug(
            "Input Thermal shape: %s min/max: (%s, %s)",
            thermal.shape,
            thermal.min(),
            thermal.max(),
        )

        return rgb, thermal

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
    
    def _postprocessing(self, outputs: List[np.ndarray]) -> List[np.ndarray]:
        out_logits, out_bbox = outputs[0], outputs[1]
        prob = self.softmax(out_logits, -1)
        scores = np.max(prob, axis=-1)
        labels = np.argmax(prob, axis=-1)

        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        
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

