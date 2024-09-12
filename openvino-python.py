import numpy as np
from openvino.runtime import Core
import cv2

# Initialize the inference engine
core = Core()

# Load the network
model_path = 'models/yolov8n_openvino_model/yolov8n.xml'
# weights_path = 'models/yolov8n_openvino_model/yolov8n.bin'
# net = ie.read_network(model=model_path, weights=weights_path)
model = core.read_model(model=model_path)

# Load the model to the CPU (or other available devices like GPU, MYRIAD, etc.)
compiled_model = core.compile_model(model=model, device_name='CPU')

# Get input and output tensors
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Read an image and preprocess
image = cv2.imread('dog.jpg')
image_resized = cv2.resize(image, (640, 640))  # Resize as needed for YOLOv8
image_transposed = np.transpose(image_resized, (2, 0, 1))  # Change to CHW format
image_normalized = image_transposed / 255.0
input_data = np.expand_dims(image_normalized, axis=0)  # Add batch dimension

# Perform inference
result = compiled_model([input_data])[0]

# # Post-process and visualize results (this will depend on your specific use case)
print("Detection results:", result)
