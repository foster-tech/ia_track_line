from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("models/yolov8n_openvino_model/")

# Run inference
results = ov_model("bus.jpg")

# results_original = model("bus.jpg")
count_person = 0

for r in results:
        
        boxes = r.boxes
        for box in boxes:
            
            c = box.cls
            obj = model.names[int(c)]
            if obj == 'person':
                  count_person = count_person + 1

print(count_person)