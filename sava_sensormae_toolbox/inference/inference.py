import os
from typing import Tuple

import numpy as np
import yaml

# Import the concrete model directly from its module to avoid circular import
from .sensormae_segm import SensorMAESegm

class InferenceEngine:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        # Optional classes mapping; many segmentation configs won't specify this
        classes = self.config.get("classes")
        if classes is None:
            self.category_mapping = None
        else:
            # Support list of single-key dicts or direct dict
            if isinstance(classes, dict):
                self.category_mapping = {str(k): v for k, v in classes.items()}
            else:
                self.category_mapping = {
                    str(list(d.keys())[0]): list(d.values())[0] for d in classes
                }
        # Instantiate the provider
        runtime = None
        if self.config["runtime"] == "onnxruntime":

            assert "model_path" in self.config, "model_path is required for onnxruntime"
            assert os.path.isfile(
                self.config["model_path"]
            ), "The model_path does not refer to a valid file."

            from ..utils.runtime import ONNXRuntime

            runtime = ONNXRuntime(
                path=self.config["model_path"],
                providers=self.config["providers"],
            )

        elif self.config["runtime"] == "tensorrt":
            raise NotImplementedError("TensorRT runtime is not implemented yet.")

        else:
            raise ValueError(
                f"Invalid runtime {self.config['runtime']} specified in config file."
            )

        self.model = SensorMAESegm(runtime=runtime)
    
    def predict(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> np.ndarray:
        """
        Perform inference on the input image and return the segmentation mask.

        Args:
            image (np.ndarray): Input image in HWC format.
        Returns:
            np.ndarray: Segmentation mask.
        """
        return self.model(rgb_image, thermal_image)

    
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

    
