import os
import sys
import numpy as np
import cv2

# Ensure the repository root is on sys.path when running this file directly
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sava_sensormae_toolbox.inference import InferenceEngine

# Specify the path to your YAML file (must exist)
CONFIG_FILE = "configs/sensormae_onnx.yaml"


def test_inference_engine():
    # Load the sample image
    sample_rgb_path = (
        f"data/samples/Visible/00040.png"
    )
    sample_thermal_path = (
        f"data/samples/Infrared/00040.png"
    )

    rgb = cv2.imread(str(sample_rgb_path), cv2.IMREAD_UNCHANGED)
    thermal = cv2.imread(str(sample_thermal_path), cv2.IMREAD_GRAYSCALE)

    # Create Inference Engine
    inference_engine = InferenceEngine(CONFIG_FILE)

    # Perform inference
    results = inference_engine.predict(rgb, thermal)

    inference_engine.save_results("data/samples/test_output.png", rgb, thermal, results)
    print("Segmentation mask shape:", results.shape)


if __name__ == "__main__":
    test_inference_engine()