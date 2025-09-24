import os
import sys
import argparse
import numpy as np
import cv2
from functools import partial
import yaml

# Ensure the repository root is on sys.path when running this file directly
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sava_sensormae_toolbox.inference import InferenceEngine, SensorMAESegm, SensorMAEObjDet

def find_infrared_path(visible_path: str) -> str:
    """Return the infrared path by replacing 'Visible' with 'Infrared'.

    Raises FileNotFoundError if the resulting path does not exist.
    """
    visible_path = os.path.abspath(visible_path)
    vis_token = f"{os.sep}Visible{os.sep}"
    if vis_token not in visible_path:
        raise FileNotFoundError(
            f"Expected 'Visible' in the path to locate the matching infrared image: {visible_path}"
        )
    infrared_path = visible_path.replace(vis_token, f"{os.sep}Infrared{os.sep}")
    if not os.path.isfile(infrared_path):
        raise FileNotFoundError(
            f"Infrared image not found at: {infrared_path} (derived from {visible_path})"
        )
    return infrared_path


def run_inference(config_path: str, visible_path: str, output_path: str) -> None:
    # Validate inputs
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(visible_path):
        raise FileNotFoundError(f"Visible image not found: {visible_path}")

    # Resolve infrared path
    infrared_path = find_infrared_path(visible_path)

    # Load images
    rgb = cv2.imread(visible_path, cv2.IMREAD_UNCHANGED)
    thermal = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE)

    model_class = None

    # Load config
    with open(args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    if "segm" in config_path.lower():
        model_class = partial(SensorMAESegm)
    elif "det" in config_path.lower():
        with open(config_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        model_class = partial(SensorMAEObjDet, num_classes=config.get("no_class", 20), confidence_threshold=config.get("confidence_threshold", 0.0))
    else:
        raise ValueError("Config file name must indicate 'segm' or 'det' to select the model class.")
    
    # Create Inference Engine
    inference_engine = InferenceEngine(config_path, model_class)

    # Perform inference
    results = inference_engine.predict(rgb, thermal)

    # Save side-by-side panel
    if "segm" in config_path.lower():
        inference_engine.model.save_results(output_path, rgb, thermal, inference_engine.model.apply_colormap(results[0].full_image_segm))
        print("Segmentation mask shape:", results[0].full_image_segm.shape)
    elif "det" in config_path.lower():
        inference_engine.model.save_results(output_path, rgb, thermal, inference_engine.model.scale_draw_boxes(results[0].xywh, rgb.copy()))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SensorMAE segmentation on RGB + IR images")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("visible", help="Path to Visible (RGB) image")
    parser.add_argument(
        "--out",
        default="data/samples/test_output.png",
        help="Output image path (side-by-side panel)",
    )
    args = parser.parse_args()
    run_inference(args.config, args.visible, args.out)