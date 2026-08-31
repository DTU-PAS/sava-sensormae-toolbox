"""Inference engine with automatic model selection from YAML config.

The engine reads ``modalities.primary``, ``modalities.secondary`` and
``task`` from the YAML configuration and looks up the correct model class
in a registry.  Model-specific parameters (``num_classes``,
``confidence_threshold``, ``input_size``, …) are also forwarded from the
config.
"""

import os
from typing import Dict, Tuple, Type

import numpy as np
import yaml

from .base import Model

# ---------------------------------------------------------------------------
# Import all concrete model classes so the registry can reference them.
# ---------------------------------------------------------------------------
from .sensormae_rgbthermal_objdet import SensorMAEObjDet_RGBThermal
from .sensormae_rgbthermal_segm import SensorMAESegm_RGBThermal
from .sensormae_rgbdepth_objdet import SensorMAEObjDet_RGBDepth
from .sensormae_rgbdepth_objdet3d import SensorMAEObjDet_RGBDepth3D

# ---------------------------------------------------------------------------
# Registry:  (primary_modality, secondary_modality, task) -> Model subclass
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[Tuple[str, str, str], Type[Model]] = {
    ("rgb", "thermal", "detection"):      SensorMAEObjDet_RGBThermal,
    ("rgb", "thermal", "segmentation"):   SensorMAESegm_RGBThermal,
    ("rgb", "depth",   "detection"):      SensorMAEObjDet_RGBDepth,
    ("rgb", "depth",   "detection_3d"):   SensorMAEObjDet_RGBDepth3D,
}


def register_model(primary: str, secondary: str, task: str):
    """Decorator to add a new model class to the registry.

    Usage::

        @register_model("rgb", "lidar", "detection")
        class MyNewModel(SensorMAEObjectDetection):
            ...
    """
    def _decorator(cls: Type[Model]):
        MODEL_REGISTRY[(primary.lower(), secondary.lower(), task.lower())] = cls
        return cls
    return _decorator


class InferenceEngine:
    """High-level inference wrapper.

    Reads modality / task information from a YAML config, selects the right
    model class from :data:`MODEL_REGISTRY`, instantiates the runtime, and
    forwards model-specific hyper-parameters.

    Args:
        config_path: Path to YAML configuration file.  Must contain at least
            ``modalities.primary``, ``modalities.secondary``, ``task``,
            ``model_path``, ``runtime``, and ``providers``.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.config: dict = yaml.safe_load(f)

        # ---- Resolve model class from registry ----
        modalities = self.config["modalities"]
        primary = modalities["primary"].lower()
        secondary = modalities["secondary"].lower()
        task = self.config["task"].lower()

        key = (primary, secondary, task)
        if key not in MODEL_REGISTRY:
            available = ", ".join(f"{k}" for k in MODEL_REGISTRY)
            raise ValueError(
                f"No model registered for {key}. Available: {available}"
            )
        model_class = MODEL_REGISTRY[key]

        # ---- Optional category mapping ----
        classes = self.config.get("classes")
        if classes is None:
            self.category_mapping = None
        elif isinstance(classes, list) and all(isinstance(c, str) for c in classes):
            # Simple list of class name strings: ["car", "pedestrian", ...]
            self.category_mapping = {str(i): name for i, name in enumerate(classes)}
        elif isinstance(classes, dict):
            self.category_mapping = {str(k): v for k, v in classes.items()}
        else:
            # List of single-key dicts: [{0: "car"}, {1: "pedestrian"}, ...]
            self.category_mapping = {
                str(list(d.keys())[0]): list(d.values())[0] for d in classes
            }

        # ---- Instantiate runtime ----
        runtime = self._build_runtime()

        # ---- Collect model-specific kwargs from config ----
        model_kwargs: dict = {}
        if "num_classes" in self.config:
            model_kwargs["num_classes"] = self.config["num_classes"]
        elif "no_class" in self.config:
            model_kwargs["num_classes"] = self.config["no_class"]
        if "confidence_threshold" in self.config:
            model_kwargs["confidence_threshold"] = self.config["confidence_threshold"]
        if "input_size" in self.config:
            model_kwargs["input_size"] = tuple(self.config["input_size"])
        if "num_select" in self.config:
            model_kwargs["num_select"] = self.config["num_select"]

        # 3D detection-specific parameters
        for key in ("head_type", "xbound", "ybound", "nms_radii",
                     "score_threshold", "post_max_size", "class_names"):
            if key in self.config:
                model_kwargs[key] = self.config[key]

        self.model: Model = model_class(runtime=runtime, **model_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, rgb_image: np.ndarray, modality_x_image: np.ndarray, **kwargs):
        """Run inference on an RGB image and a secondary-modality image.

        Args:
            rgb_image: BGR numpy array (cv2 convention).
            modality_x_image: Secondary modality numpy array (grayscale or
                multi-channel depending on the modality).
            **kwargs: Additional data forwarded to the model (e.g. ``calib``
                for 3D detection).

        Returns:
            Model-specific result structure (e.g. ``DetectionListResult``).
        """
        return self.model(rgb_image, modality_x_image, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_runtime(self):
        rt_name = self.config["runtime"]

        if rt_name == "onnxruntime":
            assert "model_path" in self.config, "model_path is required for onnxruntime"
            assert os.path.isfile(self.config["model_path"]), (
                f"model_path does not refer to a valid file: {self.config['model_path']}"
            )
            from ..utils.runtime import ONNXRuntime
            return ONNXRuntime(
                path=self.config["model_path"],
                providers=self.config["providers"],
            )

        if rt_name == "tensorrt":
            raise NotImplementedError("TensorRT runtime is not implemented yet.")

        raise ValueError(f"Unknown runtime '{rt_name}' in config.")

