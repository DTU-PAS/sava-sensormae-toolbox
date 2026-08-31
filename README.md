# SAVA SensorMAE Toolbox

Multi-sensor inference toolkit for SensorMAE-family models supporting **semantic segmentation**, **2D object detection**, and **3D object detection** (BEV-based) across arbitrary modality pairs (RGB + Thermal, RGB + Depth, …) for the SAVA project. The library provides preprocessing, ONNX Runtime execution, and post-processing behind simple, config-driven Python & CLI interfaces.

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
│   ├── sensormae_onnx_rgbdepth_det.yaml     #   → RGB + Depth 2D detection
│   ├── sensormae_onnx_rgbdepth_det3d.yaml   #   → RGB + Depth 3D detection (BEV)
│   ├── sensormae_onnx_rgbthermal_det.yaml   #   → RGB + Thermal 2D detection
│   └── sensormae_onnx_rgbthermal_segm.yaml  #   → RGB + Thermal 2D segmentation
├── data/
│   ├── *.onnx                               # ONNX model weights
│   └── samples/                             # Example paired images
│       ├── FMB/{RGB,Thermal}/
│       ├── LLVIP/{RGB,Thermal}/
│       ├── KITTI/{RGB,Depth,Calib,Velodyne}/
│       └── VoD/{RGB,Depth,Calib,Velodyne}/
├── sava_sensormae_toolbox/                  # Main Python package
│   ├── inference/
│   │   ├── base.py                          # Abstract Model (generic modality_x)
│   │   ├── objdet_base.py                   # Object detection intermediate base
│   │   ├── segm_base.py                     # Segmentation intermediate base
│   │   ├── inference.py                     # InferenceEngine + MODEL_REGISTRY
│   │   ├── sensormae_rgbthermal_objdet.py   # RGB + Thermal detection
│   │   ├── sensormae_rgbthermal_segm.py     # RGB + Thermal segmentation
│   │   ├── sensormae_rgbdepth_objdet.py     # RGB + Depth 2D detection
│   │   └── sensormae_rgbdepth_objdet3d.py   # RGB + Depth 3D detection (BEV)
│   ├── structures/
│   │   └── savaio.py                        # DectObject + DetectionListResult
│   └── utils/
│       └── runtime.py                       # ONNXRuntime session wrapper
├── tests/
│   ├── test_inference_rgb_depth.py          # CLI example (RGB + Depth 2D)
│   ├── test_inference_rgb_depth_3d.py       # CLI example (RGB + Depth 3D)
│   └── test_inference_rgb_therm.py          # CLI example (RGB + Thermal)
├── requirements.txt
├── setup.py
└── README.md
```

---
## 🏗 Architecture

```
Model (ABC)                            ← base.py — predict pipeline, shared utilities
├── SensorMAEObjectDetection           ← objdet_base.py — ONNX inference, box helpers
│   ├── SensorMAEObjDet_RGBThermal     ← 2-input ONNX, softmax head
│   ├── SensorMAEObjDet_RGBDepth       ← 2-input (concat) ONNX, sigmoid/top-K head
│   └── SensorMAEObjDet_RGBDepth3D     ← BEV 3D detection, CenterPoint head
└── SensorMAESegmentation              ← segm_base.py
    └── SensorMAESegm_RGBThermal       ← 2-input ONNX
```

The **`InferenceEngine`** reads `modalities` and `task` from the YAML config and automatically selects the correct model class from a central **`MODEL_REGISTRY`**. No manual class passing is needed — just point it at a config file.

---
## ⚙️ Configuration (YAML)

Every config must declare three key sections: **modalities**, **task**, and **runtime/model**. The engine reads these to auto-select the right model class and forward hyper-parameters.

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

### RGB + Depth — Object Detection
```yaml
# configs/sensormae_onnx_rgbdepth_det.yaml
modalities:
  primary: rgb
  secondary: depth
task: detection

model_path: data/20260315_095251_rgb-depth_vit-medium_RFDETRHead_2D_KITTI_VoD.onnx
runtime: onnxruntime
providers:
  - CUDAExecutionProvider
  - CPUExecutionProvider
batch_size: 1

input_size: [576, 576]
confidence_threshold: 0.7
classes:
  - Car
  - Large Vehicle
  - Two Wheeler
  - Pedestrian
 
```

### RGB + Depth — 3D Object Detection (BEV)
```yaml
# configs/sensormae_onnx_rgbdepth_det3d.yaml
modalities:
  primary: rgb
  secondary: depth
task: detection_3d

model_path: data/20260315_095415_rgb-depth_vit-medium_CenterpointHead_3D_KITTI_VoD.onnx
runtime: onnxruntime
providers:
  - CUDAExecutionProvider
  - CPUExecutionProvider
batch_size: 1

input_size: [384, 768]
num_classes: 3
confidence_threshold: 0.4

class_names:
  - Car
  - Human
  - Two Wheeler

xbound: [0.0, 50.0, 0.25]
ybound: [-20.0, 20.0, 0.25]

# Post-processing
score_threshold: 0.1
post_max_size: 83

nms_radii:
  Car: 4.0  
  Human: 0.175 
  Cyclist: 0.85
  TwoWheeler: 0.85
```

### Key fields

| Field | Description |
|-------|-------------|
| `modalities.primary` | Always `rgb` |
| `modalities.secondary` | `thermal`, `depth`, or any future modality |
| `task` | `detection`, `detection_3d`, or `segmentation` |
| `runtime` | `onnxruntime` (TensorRT planned) |
| `providers` | Ordered preference list forwarded to ONNX Runtime |
| `input_size` | `[height, width]` — must match ONNX model export resolution |
| `num_classes` | Number of object classes |
| `classes` / `class_names` | List of class names (used for visualisation labels) |
| `confidence_threshold` | Detection confidence filter (detection only) |

#### 3D-specific fields

| Field | Description |
|-------|-------------|
| `xbound` / `ybound` | BEV grid `[min, max, cell_size]` — must match training |
| `score_threshold` | Pre-NMS score filter per class |
| `post_max_size` | Max detections per class after NMS |
| `nms_radii` | Per-class circle NMS radii (squared, in metres) |

---
## 🧪 CLI Usage

### RGB + Thermal inference

```bash
# Object detection
python tests/test_inference_rgb_therm.py \
  --config configs/sensormae_onnx_rgbthermal_det.yaml \
  --rgb data/samples/LLVIP/RGB/180154.jpg \
  --out data/test_output_rgb_therm_2d_det.png

# Segmentation
python tests/test_inference_rgb_therm.py \
  --config configs/sensormae_onnx_rgbthermal_segm.yaml \
  --rgb data/samples/FMB/RGB/00040.png \
  --out data/test_output_rgb_therm_2d_segm.png
```

The script auto-derives the thermal image path by replacing `/RGB/` with `/Thermal/`.

### RGB + Depth inference

```bash
# 2D object detection
python tests/test_inference_rgb_depth.py \
  --config configs/sensormae_onnx_rgbdepth_det.yaml \
  --rgb data/samples/KITTI/RGB/006810.png \
  --out data/test_output_rgb_depth_2d_det.png

# 3D object detection (BEV)
python tests/test_inference_rgb_depth_3d.py \
  --config configs/sensormae_onnx_rgbdepth_det3d.yaml \
  --rgb /home/bdasz/Desktop/SAVA/code/sava-sensormae-toolbox/data/samples/VoD/RGB/00500.jpg \
  --out data/test_output_rgb_depth_3d_det.png
```

The 2D script auto-derives the depth path by replacing `/RGB/` with `/Depth/`.
The 3D script additionally derives `/Calib/` (`.txt`), `/Velodyne/` (`.bin`) and `/Depth/` (`.npy`) paths from the RGB path.

### Arguments (both scripts)

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config file |
| `--rgb` | Path to the RGB image |
| `--out` | Output image path (default: `data/samples/test_output_*.png`) |

The 3D test script expects a KITTI-style directory layout:
```
dataset/
├── RGB/          # .png or .jpg images
├── Depth/        # .npy metric depth arrays (float32, metres)
├── Calib/        # KITTI-format .txt calibration files (P2, R0_rect, Tr_velo_to_cam)
└── Velodyne/     # .bin LiDAR point clouds (float32, [N, 4] — xyz + reflectance)
```

---
## 🗺 Generating Metric Depth Maps

The RGB + Depth pipelines require metric depth as a float32 `.npy` array (metres). If you only have raw LiDAR point clouds, use the included depth generation utility:

```bash
python -m sava_sensormae_toolbox.utils.generate_depthmap \
  --lidar  data/samples/KITTI/Velodyne/006810.bin \
  --calib  data/samples/KITTI/Calib/006810.txt \
  --image  data/samples/KITTI/RGB/006810.png \
  --out    data/samples/KITTI/Depth/006810.npy
```

This projects LiDAR points onto the image plane using KITTI calibration, then fills gaps via Delaunay triangulation with automatic rejection of overly large or depth-discontinuous triangles.

Or from Python:

```python
from sava_sensormae_toolbox.utils.generate_depthmap import (
    parse_kitti_calib, generate_metric_depth, guided_filter_depth,
)
import cv2

calib = parse_kitti_calib("data/samples/KITTI/Calib/006810.txt")
img = cv2.imread("data/samples/KITTI/RGB/006810.png")
h, w = img.shape[:2]

depth = generate_metric_depth("data/samples/KITTI/Velodyne/006810.bin", calib, (w, h))
```

| Argument | Description |
|----------|-------------|
| `--lidar` | Path to `.bin` LiDAR file (float32, `[N, 4]`) |
| `--calib` | Path to KITTI-format `.txt` calibration file |
| `--image` | Path to RGB image (used only for dimensions) |
| `--out` | Output `.npy` path |
| `--channels` | LiDAR channels per point (default: 4) |
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

### 3D detection (BEV)

The 3D pipeline requires metric depth (`.npy`), KITTI-format calibration, and LiDAR points:

```python
import cv2, numpy as np
from sava_sensormae_toolbox.inference import InferenceEngine

engine = InferenceEngine("configs/sensormae_onnx_rgbdepth_det3d.yaml")

rgb = cv2.imread("data/samples/KITTI/RGB/006810.png", cv2.IMREAD_UNCHANGED)
metric_depth = np.load("data/samples/KITTI/Depth/006810.npy").astype(np.float32)
pcd = np.fromfile("data/samples/KITTI/Velodyne/006810.bin", dtype=np.float32).reshape(-1, 4)
lidar_points = pcd[:, :3].copy()

# Parse calibration (KITTI format)
from sava_sensormae_toolbox.utils.generate_depthmap import parse_kitti_calib
calib = parse_kitti_calib("data/samples/KITTI/Calib/006810.txt")

results = engine.predict(rgb, metric_depth, lidar_points=lidar_points, calib=calib)

# 3D results: xyz position, lwh dimensions, heading angle
for det in results:
    print(det.xyzwhd, det.class_id, det.score)

# Visualise: project 3D wireframes onto image
model = engine.model
class_names = engine.config.get("class_names")
lidar2img = model.compute_lidar2img()
annotated = model.draw_3d_boxes(rgb.copy(), results, lidar2img, class_names=class_names)
```

### Thermal models

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
---
## 🔄 Pre- & Post-Processing Summary

### RGB + Thermal (detection & segmentation)
1. **Resize & pad** longest side to `input_size[0]`, zero-pad to square.
2. **Normalise** RGB (ImageNet mean/std), thermal (CLAHE → (x-0.5)/0.28).
3. **Inference** via ONNX Runtime.
4. **Detection:** softmax → filter background class → scale boxes to original coords.
5. **Segmentation:** crop padded area back to original H×W.

### RGB + Depth 2D detection
1. **Resize** directly to `input_size` (no padding).
2. **Normalise** RGB (ImageNet mean/std), depth (percentile normalisation → (x-0.5)/0.28).
3. **Concatenate** `[1, 4, H, W]` (3 RGB + 1 depth) as single ONNX input.
4. **Inference** via ONNX Runtime.
5. **Decode:** sigmoid → global top-K → xyxy boxes scaled to original coords.

### RGB + Depth 3D detection (BEV)
1. **Resize** directly to `input_size` `[H, W]` (supports rectangular, e.g. 384×768).
2. **Normalise** RGB (ImageNet), depth (percentile → (x-0.5)/0.28).
3. **Scale intrinsics** to match resized image (separate sx, sy for non-square).
4. **Inference** with rgb, depth_vis, lidar_points, intrinsics, lidar2cam inputs.
5. **Decode:** CenterPoint head → per-class circle NMS → 3D boxes `(x, y, z, l, w, h, heading)` in LiDAR frame.

---
## ⚠️ Notes & Limitations

- Batch size fixed at 1.
- RGB + Thermal models use square resize & pad; RGB + Depth models use direct resize to `input_size`.
- 3D detection requires KITTI-format calibration files and LiDAR point clouds.
- CUDA provider warning on CPU-only machines is normal — set `providers: [CPUExecutionProvider]`.
- TensorRT runtime not yet implemented.
- All image loading uses cv2 (BGR convention).

---
## 🧭 Roadmap

| Status | Item |
|--------|------|
| ✅ | RGB + Depth 3D detection (BEV, CenterPoint) |
| ✅ | Rectangular input support (non-square `input_size`) |
| ⏳ | Dynamic batch size |
| ⏳ | TensorRT runtime backend |
| ⏳ | RGB + Depth segmentation model |

Legend: ⏳ planned, ✅ done


