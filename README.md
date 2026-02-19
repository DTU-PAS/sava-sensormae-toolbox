# SAVA SensorMAE Toolbox

Multi-sensor inference toolkit for SensorMAE-family models supporting **semantic segmentation** and **object detection** across arbitrary modality pairs (RGB + Thermal, RGB + Depth, …) for the SAVA project. The library provides preprocessing (resize + pad + normalisation), ONNX Runtime execution, and post-processing behind simple, config-driven Python & CLI interfaces.

**Python ≥ 3.12** — tested with Python 3.12.3 and the pinned dependencies in `requirements.txt`.

---
## 🛠 Installation

```bash
git clone https://github.com/DTU-PAS/sava-sensormae-toolbox.git
cd sava-sensormae-toolbox
python3 -m venv .env
source .env/bin/activate   # Windows: .env\Scripts\activate
pip install -e .
```

> **Tip:** If you only have CPU (e.g. macOS), set `providers: [CPUExecutionProvider]` in your config to silence CUDA warnings.

---
## 📁 Repository Layout
```
sava-sensormae-toolbox/
├── configs/                                 # YAML configuration files
│   ├── sensormae_onnx_rgbdepth_det.yaml     #   → RGB + Depth detection
│   ├── sensormae_onnx_rgbdepth_seg.yaml     #   → RGB + Depth segmentation
│   ├── sensormae_onnx_rgbthermal_det.yaml   #   → RGB + Thermal detection
│   └── sensormae_onnx_rgbthermal_segm.yaml  #   → RGB + Thermal segmentation
├── data/
│   ├── *.onnx                               # ONNX model weights
│   └── samples/                             # Example paired images
│       ├── FMB/{RGB,Thermal}/
│       ├── LLVIP/{RGB,Thermal}/
│       ├── kitti/{RGB,Depth}/
│       └── vod/{RGB,Depth}/
├── sava_sensormae_toolbox/                  # Main Python package
│   ├── inference/
│   │   ├── base.py                          # Abstract Model (generic modality_x)
│   │   ├── objdet_base.py                   # Object detection intermediate base
│   │   ├── segm_base.py                     # Segmentation intermediate base
│   │   ├── inference.py                     # InferenceEngine + MODEL_REGISTRY
│   │   ├── sensormae_rgbthermal_objdet.py   # RGB + Thermal detection
│   │   ├── sensormae_rgbthermal_segm.py     # RGB + Thermal segmentation
│   │   ├── sensormae_rgbdepth_objdet.py     # RGB + Depth detection
│   │   └── sensormae_rgbdepth_segm.py       # RGB + Depth segmentation (stub)
│   ├── structures/
│   │   └── savaio.py                        # DectObject + DetectionListResult
│   └── utils/
│       └── runtime.py                       # ONNXRuntime session wrapper
├── tests/
│   ├── test_inference_rgb_depth.py          # CLI example (RGB + Depth)
│   └── test_inference_rgb_therm.py          # CLI example (RGB + Thermal)
├── requirements.txt
├── setup.py
└── README.md
```

---
## 🏗 Architecture

```
Model (ABC)                         ← base.py — generic "modality_x" naming, shared utilities
├── SensorMAEObjectDetection        ← objdet_base.py — box conversion, drawing helpers
│   ├── SensorMAEObjDet_RGBThermal  ← 2-input ONNX, softmax head
│   └── SensorMAEObjDet_RGBDepth    ← 1-input (concatenated) ONNX, sigmoid/top-K head
└── SensorMAESegmentation           ← segm_base.py
    ├── SensorMAESegm_RGBThermal    ← 2-input ONNX
    └── SensorMAESegm_RGBDepth      ← stub (not yet implemented)
```

The **`InferenceEngine`** reads `modalities` and `task` from the YAML config and automatically selects the correct model class from a central **`MODEL_REGISTRY`**. No manual class passing is needed — just point it at a config file.

---
## ⚙️ Configuration (YAML)

Every config must declare three key sections: **modalities**, **task**, and **runtime/model**. The engine reads these to auto-select the right model class and forward hyper-parameters.

### RGB + Depth — Object Detection
```yaml
# configs/sensormae_onnx_rgbdepth_det.yaml
modalities:
  primary: rgb
  secondary: depth
task: detection

runtime: onnxruntime
model_path: data/model_final.onnx
providers:
  - CUDAExecutionProvider
  - CPUExecutionProvider
batch_size: 1

input_size: [576, 576]
confidence_threshold: 0.7
num_classes: 4
classes:
  - car
  - large_vehicle
  - two_wheeler
  - pedestrian
```

### RGB + Thermal — Object Detection
```yaml
# configs/sensormae_onnx_rgbthermal_det.yaml
modalities:
  primary: rgb
  secondary: thermal
task: detection

runtime: onnxruntime
model_path: data/20250923_130344_rgb-thermal_vit-medium_RFDETRHead_LLVIP.onnx
providers:
  - CUDAExecutionProvider
  - CPUExecutionProvider
batch_size: 1

input_size: [640, 640]
confidence_threshold: 0.7
num_classes: 1
classes:
  - person
```

### RGB + Thermal — Segmentation
```yaml
# configs/sensormae_onnx_rgbthermal_segm.yaml
modalities:
  primary: rgb
  secondary: thermal
task: segmentation

runtime: onnxruntime
model_path: data/20250925_123721_rgb-thermal_vit-medium_ConvNextHead_FMB.onnx
providers:
  - CUDAExecutionProvider
  - CPUExecutionProvider
batch_size: 1

input_size: [640, 640]
```

### Key fields

| Field | Description |
|-------|-------------|
| `modalities.primary` | Always `rgb` |
| `modalities.secondary` | `thermal`, `depth`, or any future modality |
| `task` | `detection` or `segmentation` |
| `runtime` | `onnxruntime` (TensorRT planned) |
| `providers` | Ordered preference list forwarded to ONNX Runtime |
| `input_size` | `[height, width]` — preprocessing canvas size |
| `num_classes` | Number of object classes |
| `classes` | List of class names (used for visualisation labels) |
| `confidence_threshold` | Detection confidence filter (detection only) |

---
## 🧪 CLI Usage

### RGB + Depth inference

```bash
# Object detection
python tests/test_inference_rgb_depth.py \
  --config configs/sensormae_onnx_rgbdepth_det.yaml \
  --rgb data/samples/vod/RGB/07752.png \
  --out data/samples/test_output_depth_det.png
```

The script auto-derives the depth image path by replacing `/RGB/` with `/Depth/` in the given path.

### RGB + Thermal inference

```bash
# Object detection
python tests/test_inference_rgb_therm.py \
  --config configs/sensormae_onnx_rgbthermal_det.yaml \
  --rgb data/samples/LLVIP/RGB/180154.jpg \
  --out data/samples/test_output_thermal_det.png

# Segmentation
python tests/test_inference_rgb_therm.py \
  --config configs/sensormae_onnx_rgbthermal_segm.yaml \
  --rgb data/samples/FMB/RGB/00040.png \
  --out data/samples/test_output_thermal_segm.png
```

The script auto-derives the thermal image path by replacing `/RGB/` with `/Thermal/`.

### Arguments (both scripts)

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config file |
| `--rgb` / `--visible` | Path to the RGB image |
| `--out` | Output image path (default: `data/samples/test_output_*.png`) |

---
## 🧩 Python API

```python
import cv2
from sava_sensormae_toolbox.inference import InferenceEngine

# The engine auto-selects the model from the config
engine = InferenceEngine("configs/sensormae_onnx_rgbdepth_det.yaml")

rgb   = cv2.imread("data/samples/vod/RGB/07752.png", cv2.IMREAD_UNCHANGED)
depth = cv2.imread("data/samples/vod/Depth/07752.png", cv2.IMREAD_GRAYSCALE)

results = engine.predict(rgb, depth)

# Detection results
for det in results:
    print(det.xywh, det.class_id, det.score)

# Draw boxes with class names from config
import numpy as np
class_names = engine.config.get("classes")
boxes  = np.array(results.getbboxes())
labels = np.array([d.class_id for d in results])
scores = np.array([d.score for d in results])
annotated = engine.model.scale_draw_boxes(
    boxes, rgb.copy(), labels=labels, scores=scores, class_names=class_names,
)

# Save side-by-side panel: RGB | Depth | Annotated
engine.model.save_results("output.png", rgb, depth, annotated)
```

For thermal models the usage is identical — just swap the config and images:

```python
engine = InferenceEngine("configs/sensormae_onnx_rgbthermal_segm.yaml")

rgb     = cv2.imread("data/samples/FMB/RGB/00040.png", cv2.IMREAD_UNCHANGED)
thermal = cv2.imread("data/samples/FMB/Thermal/00040.png", cv2.IMREAD_GRAYSCALE)

results = engine.predict(rgb, thermal)

# Segmentation result
mask = results[0].full_image_segm
colored = engine.model.apply_colormap(mask)
engine.model.save_results("output_segm.png", rgb, thermal, colored)
```

### Extending with a new modality

1. Create a leaf class inheriting `SensorMAEObjectDetection` or `SensorMAESegmentation`
2. Set `MODALITY_X_NAME = "your_modality"`
3. Implement `_preprocessing`, `_inference`, `_postprocessing`
4. Register it in the `MODEL_REGISTRY` or use the `@register_model` decorator:
   ```python
   from sava_sensormae_toolbox.inference import register_model, SensorMAEObjectDetection

   @register_model("rgb", "lidar", "detection")
   class SensorMAEObjDet_RGBLidar(SensorMAEObjectDetection):
       MODALITY_X_NAME = "lidar"
       ...
   ```
5. Create a YAML config with `modalities: {primary: rgb, secondary: lidar}` and `task: detection`

---
## 🔄 Pre- & Post-Processing Summary

1. **Resize** longest side to `input_size[0]` (square canvas).
2. **Pad** bottom/right to square (top-left origin preserved).
3. **Normalise** RGB (ImageNet mean/std) and secondary modality (modality-specific normalisation).
4. **Inference** via ONNX Runtime `session.run`.
5. **Segmentation:** crop padded area, map to original H×W.
6. **Detection:** decode boxes (cxcywh → xyxy), filter by confidence, scale to original coordinates.

---
## ⚠️ Notes & Limitations

- Batch size fixed at 1.
- Square model canvas required (padding logic assumes this).
- CUDA provider warning on CPU-only machines is normal — set `providers: [CPUExecutionProvider]`.
- TensorRT runtime not yet implemented.
- RGB + Depth segmentation model is a stub (not yet implemented).
- All image loading uses cv2 (BGR convention).

---
## 🧭 Roadmap

| Status | Item |
|--------|------|
| ⏳ | Dynamic batch size |
| ⏳ | Non-square arbitrary input & removal of fixed padding assumption |
| ⏳ | TensorRT runtime backend |
| ⏳ | RGB + Depth segmentation model |

Legend: ⏳ planned, ✅ done


