from pathlib import Path

# Fetch `notebook_utils` module
import requests

# r = requests.get(
#     url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
# )

# open("notebook_utils.py", "w").write(r.text)

from notebook_utils import download_file, VideoPlayer

# Download a test sample
# IMAGE_PATH = Path("./data/coco_bike.jpg")
IMAGE_PATH = Path("./dog.jpg")

# download_file(
#     url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg",
#     filename=IMAGE_PATH.name,
#     directory=IMAGE_PATH.parent,
# )

models_dir = Path("./models")
models_dir.mkdir(exist_ok=True)

from PIL import Image
from ultralytics import YOLO

DET_MODEL_NAME = "yolov8n"

# det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")
det_model = YOLO(f"{DET_MODEL_NAME}.pt")
label_map = det_model.model.names

print("Usando YOLO model")
res = det_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:, :, ::-1])

# object detection model
det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
# det_model_path =  "{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=True)



import ipywidgets as widgets
import openvino as ov

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)

device    

# IMAGE_PATH = Path("./bus.jpg")

import torch

core = ov.Core()

det_ov_model = core.read_model(det_model_path)

ov_config = {}
if device.value != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
det_compiled_model = core.compile_model(det_ov_model, device.value, ov_config)


def infer(*args):
    result = det_compiled_model(args)
    return torch.from_numpy(result[0])


det_model.predictor.inference = infer
det_model.predictor.model.pt = False

print("Usando openVINO model")
res = det_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:, :, ::-1])