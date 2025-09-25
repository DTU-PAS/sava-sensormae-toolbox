# SAVA SensorMAE Toolbox

Multi‑sensor inference toolkit (RGB + Thermal) for SensorMAE family models supporting **semantic segmentation** and **object detection** for SAVA project. The library provides a basic usage structure with preprocessing (resize + pad + normalization), ONNX Runtime execution, and post‑processing into simple, configurable Python & CLI interfaces.


---
## 🛠 Installation

```bash
git clone https://github.com/DTU-PAS/sava-sensormae-toolbox.git
cd sava-sensormae-toolbox
python3 -m venv .env
source .env/bin/activate   # Windows: .env\Scripts\activate
pip install -e .
```

> Tip: If you only have CPU (e.g. macOS), adjust provider list to `[CPUExecutionProvider]` to silence CUDA warnings.

---
## 📁 Repository Layout (excerpt)
```
configs/                # YAML configuration files
data/samples/Visible/   # Example RGB images
data/samples/Infrared/  # Matching IR images
sava_sensormae_toolbox/
  inference/
    inference.py        # InferenceEngine wrapper
    sensormae_segm.py   # Segmentation model wrapper
    sensormae_objdet.py # Object detection model wrapper
  structures/
    savaio.py           # DectObject + DetectionListResult
  utils/runtime.py      # ONNXRuntime session helper
tests/
  test_inference_rgb_therm.py  # CLI / example script
```

---
## ⚙️ Configuration (YAML)
Minimal example (segmentation):
```yaml
runtime: onnxruntime
model_path: data/20250925_123721_rgb-thermal_vit-medium_ConvNextHead_FMB.onnx
providers:
  - CPUExecutionProvider  # or CUDAExecutionProvider if available
batch_size: 1
input_size: [640, 640]  # height, width
# classes:               # (optional) mapping or palette info
# confidence_threshold: 0.25   # (object detection only)
```
If the configuration file has `obj` in the name of the file, the object detection pipeline will be executed. `segm` in the name will execute the segmentation pipeline.

Key fields:
* `runtime`: currently `onnxruntime` (TensorRT not yet implemented).
* `providers`: ordered preference list forwarded to ONNXRuntime.
* `input_size`: square model canvas (preprocessing scales longer side then pads).
* `confidence_threshold`: used in detection post‑processing (ignored for segmentation if absent).
* `no_class`: number of classes.

---
## 🧪 CLI Usage (Dual‑Modality)
Script: `tests/test_inference_rgb_therm.py`

The script expects:
1. Config path
2. Visible (RGB) image path (must contain `/Visible/` directory segment)

It auto‑derives the infrared path by replacing `Visible` with `Infrared`.

Examples:
```bash
# Segmentation
python tests/test_inference_rgb_therm.py \
  configs/sensormae_onnx_segm.yaml\
  data/samples/FMB/Visible/00040.png \
  --out data/samples/test_output_segm.png

# Object Detection
python tests/test_inference_rgb_therm.py \
  configs/sensormae_onnx_det.yaml\
  data/samples/LLVIP/Visible/180154.jpg \
  --out data/samples/test_output_objdet.png
```

Arguments:
* `config`: YAML config file.
* `visible`: RGB image path (infrared inferred automatically).
* `--out`: Example output path path (RGB | IR | Segm) – optional.

> For object detection models, the same script runs — the returned structure contains detection objects. (If you want a dedicated detection visualization helper, add one under `inference/`.)

---
## 🧩 Python API (Core)
```python
from sava_sensormae_toolbox.inference import InferenceEngine
import cv2

engine = InferenceEngine("configs/sensormae_onnx.yaml")
rgb = cv2.imread("data/samples/Visible/00001.png", cv2.IMREAD_UNCHANGED)
thermal = cv2.imread("data/samples/Infrared/00001.png", cv2.IMREAD_GRAYSCALE)
result = engine.predict(rgb, thermal)

# Segmentation: result is an (H, W) mask (or structure containing full image segm objects)
# Detection: result is DetectionListResult with DectObject entries.
engine.save_results("out/panel.png", rgb, thermal, result)
```

---
## 🔄 Pre & Post Processing Summary
1. Resize longest side to `input_size[0]` (square assumption).
2. Pad bottom / right (top-left origin kept) to square.
3. Normalize RGB (ImageNet mean/std) & Thermal (custom mean/std + CLAHE + min–max scaling).
4. Model inference (ONNXRuntime session.run).
5. Segmentation: crop padded area, resize back to original H×W.
6. Detection: boxes converted from center format, filtered by confidence.

---
## ⚠️ Notes & Limitations
* Batch size fixed at 1.
* Square model canvas required (padding logic assumes this).
* CUDA provider warning on macOS is normal (set providers to CPU to silence).
* TensorRT runtime not yet implemented.
* Color palette for segmentation not finalized (uses Matplotlib if colorization step applied elsewhere).
* Object detection visualization (drawing boxes) presently done inside model helper (`scale_draw_boxes`) — not exposed as a dedicated high-level API.

---
## 🧭 Roadmap / Todo
| Status | Item |
|--------|------|
| ⏳ | Dynamic batch size |
| ⏳ | Non-square arbitrary input & removal of fixed padding assumption |
| ⏳ | TensorRT runtime backend |

Legend: ⏳ planned / in-progress, ✅ done (update as features land)


