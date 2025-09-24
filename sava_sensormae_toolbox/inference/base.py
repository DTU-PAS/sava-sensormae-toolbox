
from abc import ABC, abstractmethod
import os
import numpy as np

class Model(ABC):
    """This class represents a generic model with different method that every subclass model
        is likely to use and reimplement

    Args:
        ABC (ABC): Abstract class
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self._inference(*args, **kwargs)

    @abstractmethod
    def _inference(self):
        pass

    @abstractmethod
    def _preprocessing(self):
        pass

    @abstractmethod
    def _postprocessing(self):
        pass
    
    @staticmethod
    def save_results(output_path: str, rgb_image: np.ndarray, thermal_image: np.ndarray, colored_mask: np.ndarray) -> None:
        """
        Save the inference results to disk as a side-by-side panel: RGB | Thermal | Segmentation.

        Args:
            output_path (str): Path to save the output image.
            rgb_image (np.ndarray): Original RGB image.
            thermal_image (np.ndarray): Original thermal image.
            colored_mask (np.ndarray): Colored segmentation mask (H, W, 3) or (H, W).
        """

        import cv2

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        h, w = rgb_image.shape[:2]

        # Prepare thermal visualization to match RGB size and 3 channels
        thermal_vis = thermal_image
        if thermal_vis.shape[:2] != (h, w):
            thermal_vis = cv2.resize(thermal_vis, (w, h), interpolation=cv2.INTER_LINEAR)
        if thermal_vis.ndim == 2:
            thermal_vis = cv2.cvtColor(thermal_vis, cv2.COLOR_GRAY2BGR)
        elif thermal_vis.ndim == 3 and thermal_vis.shape[2] == 1:
            thermal_vis = np.repeat(thermal_vis, 3, axis=2)

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
        combined = np.hstack((rgb_vis, thermal_vis, seg_vis))

        # Save the combined panel
        cv2.imwrite(output_path, combined)
