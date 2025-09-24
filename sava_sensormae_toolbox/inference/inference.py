import os
from typing import Tuple

import numpy as np
import yaml

from .base import Model

class InferenceEngine:
    def __init__(self, config_path: str, model_class: Model) -> None:
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

        self.model = model_class(runtime=runtime)
    
    def predict(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> np.ndarray:
        """
        Perform inference on the input image and return the segmentation mask.

        Args:
            image (np.ndarray): Input image in HWC format.
        Returns:
            np.ndarray: Segmentation mask.
        """
        return self.model(rgb_image, thermal_image)
    

    


    
