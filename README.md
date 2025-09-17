# SAVA SENSORMAE TOOLBOX

SAVA SensorMAE Toolbox is a Python library for Machine Learning tasks. It provides a collection of tools and utilities to simplify the process of deploying SensorMAE family of models for SAVA project.

## Installation

### Clone the repository

```bash
git clone https://github.com/username/sava-sensormae-toolbox.git
```
And navigate into the folder
```bash
cd sava-sensormae-toolbox
```

### Create environment (Optional)

Create a virtual environment in python with `venv`:
```bash
python3 -m venv .env
``` 
And activate it on Windows:
```bash 
.env\Scripts\activate 
```
And activate it on Linux/Mac:
```bash 
source .env/bin/activate 
```
### Install the library
Navigate to the root directory of the project (where the `setup.py` file is located), and run:
```bash 
pip install -e .
```
## Usage
Pick one of the configurations from the `config` folder or create one in YAML format, with the following template:

```yaml
# Runtime configuration
model_path: path/to/model
runtime: onnxruntime
providers:
  - CUDAExecutionProvider
  - CPUExecutionProvider
batch_size: 1 # For now, 1 is the only accepted value

# Model configuration
input_size: [640, 640] # [height, width]


```

Then you can use the Inference Engine as follow:
```python
from sava_ml_toolbox.inference import InferenceEngine
# ...

# Load the sample image
sample_rgb_path = (
    f"data/samples/Visible/00001.png"
)
sample_thermal_path = (
    f"data/samples/Infrared/00001.png"
)

rgb = cv2.imread(str(sample_rgb_path), cv2.IMREAD_UNCHANGED)
thermal = cv2.imread(str(sample_thermal_path), cv2.IMREAD_GRAYSCALE)

# Create Inference Engine
inference_engine = InferenceEngine(CONFIG_FILE)

# Perform inference
results = inference_engine.predict(rgb, thermal)

inference_engine.save_results("data/samples/test_output.png", rgb, thermal, results)

```

The output of the `predict` function is an image of size 640x640.

## License

