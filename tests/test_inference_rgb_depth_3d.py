"""Test script for RGB + Depth **3D** inference (BEV-based object detection).

Usage::

    python tests/test_inference_rgb_depth_3d.py \\
        --config configs/sensormae_onnx_rgbdepth_det3d.yaml \\
        --rgb data/samples/KITTI/RGB/007454.png \\
        --out data/samples/test_output_depth_3d.png

Both the metric depth ``.npy`` and calibration ``.txt`` files are located
automatically from the RGB path:

- Depth: ``/RGB/`` → ``/Depth/``, extension → ``.npy``
- Calib: ``/RGB/`` → ``/Calib/``, extension → ``.txt``

Output is a side-by-side panel:  RGB | Depth visualisation | 3D wireframes.
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


def find_depth_npy_path(rgb_path: str) -> str:
    """Derive the metric depth ``.npy`` path from the RGB path.

    Replaces ``/RGB/`` with ``/Depth/`` and changes the extension to ``.npy``.
    """
    rgb_path = os.path.abspath(rgb_path)
    token = f"{os.sep}RGB{os.sep}"
    if token not in rgb_path:
        raise FileNotFoundError(
            f"Expected '/RGB/' in path to locate the matching depth file: {rgb_path}"
        )
    depth_path = rgb_path.replace(token, f"{os.sep}Depth{os.sep}")
    # Replace image extension with .npy
    depth_path = os.path.splitext(depth_path)[0] + ".npy"
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"Metric depth .npy not found: {depth_path}")
    return depth_path


def find_calib_path(rgb_path: str) -> str:
    """Derive the calibration ``.txt`` path from the RGB path.

    Replaces ``/RGB/`` with ``/Calib/`` and changes the extension to ``.txt``.
    """
    rgb_path = os.path.abspath(rgb_path)
    token = f"{os.sep}RGB{os.sep}"
    if token not in rgb_path:
        raise FileNotFoundError(
            f"Expected '/RGB/' in path to locate the matching calib file: {rgb_path}"
        )
    calib_path = rgb_path.replace(token, f"{os.sep}Calib{os.sep}")
    calib_path = os.path.splitext(calib_path)[0] + ".txt"
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calib file not found: {calib_path}")
    return calib_path


def find_velodyne_path(rgb_path: str) -> str | None:
    """Derive the LiDAR ``.bin`` path from the RGB path.

    Replaces ``/RGB/`` with ``/Velodyne/`` and changes the extension to ``.bin``.
    Returns None if the file does not exist (optional for dense models).
    """
    rgb_path = os.path.abspath(rgb_path)
    token = f"{os.sep}RGB{os.sep}"
    if token not in rgb_path:
        return None
    bin_path = rgb_path.replace(token, f"{os.sep}Velodyne{os.sep}")
    bin_path = os.path.splitext(bin_path)[0] + ".bin"
    return bin_path if os.path.isfile(bin_path) else None


def run_inference(
    config_path: str,
    rgb_path: str,
    output_path: str,
) -> None:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    depth_npy_path = find_depth_npy_path(rgb_path)
    calib_path = find_calib_path(rgb_path)
    velodyne_path = find_velodyne_path(rgb_path)

    # Load inputs
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    metric_depth = np.load(depth_npy_path).astype(np.float32)

    # Load raw LiDAR points if available (required for sparse models)
    lidar_points = None
    if velodyne_path is not None:
        pcd = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        lidar_points = pcd[:, :3].copy()  # xyz only
        print(f"Loaded {lidar_points.shape[0]} LiDAR points from {velodyne_path}")

    # Engine auto-selects the 3D model from config (task = detection_3d)
    engine = InferenceEngine(config_path)
    import time
    start_time = time.time()
    results = engine.predict(rgb, metric_depth, calib=calib_path,
                             lidar_points=lidar_points)
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")
    print(f"3D detections: {len(results)}")
    class_names = engine.config.get("class_names")
    for i, det in enumerate(results):
        cls_name = (class_names[det.class_id]
                    if class_names and det.class_id < len(class_names)
                    else str(det.class_id))
        box = det.xyzwhd
        print(f"  [{i}] {cls_name}  score={det.score:.3f}  "
              f"xyz=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f})  "
              f"lwh=({box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f})  "
              f"heading={box[6]:.2f}")

    # Visualise: project 3D boxes onto original image
    model = engine.model
    lidar2img = model.compute_lidar2img()
    annotated = model.draw_3d_boxes(
        rgb.copy(), results, lidar2img, class_names=class_names,
    )

    # Depth visualisation (for the side-by-side panel)
    depth_vis = model._metric_to_visual(metric_depth)  # [H, W] float32 0..1
    depth_vis_uint8 = (depth_vis * 255).astype(np.uint8)

    model.save_results(output_path, rgb, depth_vis_uint8, annotated)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SensorMAE 3D inference on RGB + metric depth"
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    parser.add_argument("--rgb", "--visible", required=True, dest="rgb",
                        help="Path to visible (RGB) image")
    parser.add_argument("--out", default="data/samples/test_output_depth_3d.png",
                        help="Output image path")
    args = parser.parse_args()
    run_inference(args.config, args.rgb, args.out)
