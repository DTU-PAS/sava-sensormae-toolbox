"""Test script for RGB + Depth inference (object detection & segmentation).

Usage::

    python tests/test_inference_rgb_depth.py \\
        --config configs/sensormae_onnx_rgbdepth_det.yaml \\
        --rgb data/samples/vod/RGB/07752.png \\
        --out data/samples/test_output_depth.png

The depth image is located automatically by replacing ``/RGB/`` with
``/Depth/`` in the given RGB path.
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


def find_depth_path(rgb_path: str) -> str:
    """Derive the depth image path from the RGB path (``/RGB/`` → ``/Depth/``)."""
    rgb_path = os.path.abspath(rgb_path)
    token = f"{os.sep}RGB{os.sep}"
    if token not in rgb_path:
        raise FileNotFoundError(
            f"Expected '/RGB/' in path to locate the matching depth image: {rgb_path}"
        )
    depth_path = rgb_path.replace(token, f"{os.sep}Depth{os.sep}")
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    return depth_path


def run_inference(config_path: str, rgb_path: str, output_path: str) -> None:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    depth_path = find_depth_path(rgb_path)

    # Load images with cv2 (consistent with the rest of the toolbox)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    # Engine auto-selects the model from config (modalities + task)
    engine = InferenceEngine(config_path)
    results = engine.predict(rgb, depth)

    print(f"Results: {results}")

    # Visualise based on task
    task = engine.config.get("task", "").lower()
    if task == "segmentation":
        colored = engine.model.apply_colormap(results[0].full_image_segm)
        engine.model.save_results(output_path, rgb, depth, colored)
        print("Segmentation mask shape:", results[0].full_image_segm.shape)

    elif task == "detection":
        class_names = engine.config.get("classes")
        all_boxes = np.array(results.getbboxes())
        all_labels = np.array([d.class_id for d in results])
        all_scores = np.array([d.score for d in results])
        annotated = engine.model.scale_draw_boxes(
            all_boxes, rgb.copy(),
            labels=all_labels, scores=all_scores, class_names=class_names,
        )
        engine.model.save_results(output_path, rgb, depth, annotated)

    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SensorMAE inference on RGB + Depth images")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--rgb", "--visible", required=True, dest="rgb",
                        help="Path to visible (RGB) image")
    parser.add_argument("--out", default="data/samples/test_output_depth.png",
                        help="Output image path")
    args = parser.parse_args()
    run_inference(args.config, args.rgb, args.out)