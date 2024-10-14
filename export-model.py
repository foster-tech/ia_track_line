from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # or use your own trained model

# Export the model to ONNX format
model.export(format='onnx')