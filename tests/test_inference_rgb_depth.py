import os
import sys
import argparse
import numpy as np
from PIL import Image
from functools import partial
import yaml

CLASSES = [
    # "animals",              # 0
    "car",             # 1
    "large_vehicle",       # 2
    "two_wheeler", # 3
    "pedestrian",          # 4
    # "miscellaneous",        # 5
]

# Ensure the repository root is on sys.path when running this file directly
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sava_sensormae_toolbox.inference import InferenceEngine, SensorMAEObjDet_RGBDepth

def find_depth_path(rgb_path: str) -> str:
    """Return the infrared path by replacing 'Visible' with 'Infrared'.

    Raises FileNotFoundError if the resulting path does not exist.
    """
    rgb_path = os.path.abspath(rgb_path)
    rgb_token = f"{os.sep}RGB{os.sep}"
    if rgb_token not in rgb_path:
        raise FileNotFoundError(
            f"Expected 'RGB' in the path to locate the matching depth image: {rgb_path}"
        )
    depth_path = rgb_path.replace(rgb_token, f"{os.sep}Depth{os.sep}")
    
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(
            f"Infrared image not found at: {depth_path} (derived from {rgb_path})"
        )
            
    return depth_path


def run_inference(config_path: str, visible_path: str, output_path: str) -> None:
    # Validate inputs
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(visible_path):
        raise FileNotFoundError(f"Visible image not found: {visible_path}")

    # Resolve infrared path
    depth_path = find_depth_path(visible_path)

    # Load images
    rgb = Image.open(visible_path).convert("RGB")
    depth = Image.open(depth_path).convert("L")
   
    model_class = None

    # Load config
    with open(args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    if "segm" in config_path.lower():
        model_class = partial(SensorMAESegm)
    elif "det" in config_path.lower():
        with open(config_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        model_class = partial(SensorMAEObjDet_RGBDepth, num_classes=config.get("no_class", 20), confidence_threshold=config.get("confidence_threshold", 0.0), input_size=config.get("input_size", [384, 384]))
    else:
        raise ValueError("Config file name must indicate 'segm' or 'det' to select the model class.")
    
    # Create Inference Engine
    inference_engine = InferenceEngine(config_path, model_class)

    # Perform inference
    results = inference_engine.predict(rgb, depth)
    rgb = np.array(rgb)
    depth = np.array(depth)
    print(f"results: {results}")
    # Save side-by-side panel
    if "segm" in config_path.lower():
        inference_engine.model.save_results(output_path, rgb, depth, inference_engine.model.apply_colormap(results[0].full_image_segm))
        print("Segmentation mask shape:", results[0].full_image_segm.shape)
    elif "det" in config_path.lower():
        all_boxes = np.array(results.getbboxes())  # (N, 4) array of all detection boxes
        all_labels = np.array([det.class_id for det in results])
        all_scores = np.array([det.score for det in results])
        annotated = inference_engine.model.scale_draw_boxes(all_boxes, rgb.copy(), labels=all_labels, scores=all_scores, class_names=CLASSES)
        inference_engine.model.save_results(output_path, rgb, depth, annotated)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SensorMAE segmentation on RGB + IR images")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--visible", required=True, help="Path to Visible (RGB) image")
    parser.add_argument(
        "--out",
        default="data/samples/test_output.png",
        help="Output image path (side-by-side panel)",
    )
    args = parser.parse_args()
    run_inference(args.config, args.visible, args.out)