"""Test script for RGB + Thermal inference (object detection & segmentation).

Usage::

    python tests/test_inference_rgb_therm.py \\
        --config configs/sensormae_onnx_rgbthermal_det.yaml \\
        --rgb data/samples/LLVIP/Visible/010001.jpg \\
        --out data/samples/test_output_thermal.png

The infrared image is located automatically by replacing ``/Visible/``
with ``/Infrared/`` in the given RGB path.
"""

import os
import sys
import argparse

import cv2
import numpy as np

# Ensure repo root is on sys.path when running directly
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sava_sensormae_toolbox.inference import InferenceEngine


def find_infrared_path(visible_path: str) -> str:
    """Derive the infrared path from the visible path (``/Visible/`` → ``/Infrared/``)."""
    visible_path = os.path.abspath(visible_path)
    token = f"{os.sep}Visible{os.sep}"
    if token not in visible_path:
        raise FileNotFoundError(
            f"Expected '/Visible/' in path to locate the matching infrared image: {visible_path}"
        )
    infrared_path = visible_path.replace(token, f"{os.sep}Infrared{os.sep}")
    if not os.path.isfile(infrared_path):
        raise FileNotFoundError(f"Infrared image not found: {infrared_path}")
    return infrared_path


def run_inference(config_path: str, rgb_path: str, output_path: str) -> None:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    infrared_path = find_infrared_path(rgb_path)

    # Load images with cv2 (consistent with the rest of the toolbox)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    thermal = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE)

    # Engine auto-selects the model from config (modalities + task)
    engine = InferenceEngine(config_path)
    results = engine.predict(rgb, thermal)

    print(f"Results: {results}")

    # Visualise based on task
    task = engine.config.get("task", "").lower()
    if task == "segmentation":
        colored = engine.model.apply_colormap(results[0].full_image_segm)
        engine.model.save_results(output_path, rgb, thermal, colored)
        print("Segmentation mask shape:", results[0].full_image_segm.shape)

    elif task == "detection":
        class_names = engine.config.get("classes")
        annotated = engine.model.scale_draw_boxes(
            results[0].xywh, rgb.copy(), scale_to_image=True,
            class_names=class_names,
        )
        engine.model.save_results(output_path, rgb, thermal, annotated)

    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SensorMAE inference on RGB + Thermal images")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--rgb", "--visible", required=True, dest="rgb",
                        help="Path to visible (RGB) image")
    parser.add_argument("--out", default="data/samples/test_output_thermal.png",
                        help="Output image path")
    args = parser.parse_args()
    run_inference(args.config, args.rgb, args.out)