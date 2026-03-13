"""Test script for RGB + Depth 2D object detection.

Usage::

    python tests/test_inference_rgb_depth.py \
        --config configs/sensormae_onnx_rgbdepth_det.yaml \
        --rgb data/samples/KITTI/RGB/006525.png \
        --out data/samples/test_2d_006525.png

The metric depth ``.npy`` file is located automatically by replacing
``/RGB/`` with ``/Depth/`` in the given RGB path.
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
    """Derive the metric depth ``.npy`` path from the RGB path."""
    rgb_path = os.path.abspath(rgb_path)
    token = f"{os.sep}RGB{os.sep}"
    if token not in rgb_path:
        raise FileNotFoundError(
            f"Expected '/RGB/' in path to locate the matching depth file: {rgb_path}"
        )
    depth_path = rgb_path.replace(token, f"{os.sep}Depth{os.sep}")
    depth_path = os.path.splitext(depth_path)[0] + ".npy"
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"Metric depth .npy not found: {depth_path}")
    return depth_path


def run_inference(config_path: str, rgb_path: str, output_path: str) -> None:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    depth_path = find_depth_path(rgb_path)

    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    metric_depth = np.load(depth_path).astype(np.float32)

    engine = InferenceEngine(config_path)
    results = engine.predict(rgb, metric_depth)
    print(f"Results: {results}")

    class_names = engine.config.get("classes")
    all_boxes = np.array(results.getbboxes())
    all_labels = np.array([d.class_id for d in results])
    all_scores = np.array([d.score for d in results])

    # Draw boxes on RGB
    annotated_rgb = engine.model.scale_draw_boxes(
        all_boxes, rgb.copy(),
        labels=all_labels, scores=all_scores, class_names=class_names,
    )

    # Depth: metric → [0,255] → magma colormap → draw boxes
    depth_vis = engine.model._metric_to_visual(metric_depth)
    depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8),
                                      cv2.COLORMAP_MAGMA)
    annotated_depth = engine.model.scale_draw_boxes(
        all_boxes, depth_colored,
        labels=all_labels, scores=all_scores, class_names=class_names,
    )

    engine.model.save_results(output_path, annotated_rgb, annotated_depth)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SensorMAE 2D inference on RGB + metric depth")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--rgb", "--visible", required=True, dest="rgb",
                        help="Path to visible (RGB) image")
    parser.add_argument("--out", default="data/samples/test_output_depth.png",
                        help="Output image path")
    args = parser.parse_args()
    run_inference(args.config, args.rgb, args.out)
