from ultralytics import YOLO
import cv2, os
import numpy as np

# Load the YOLO model for object detection
model = YOLO('yolov8n.pt')  # Replace with your model

# # Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")
# ov_model = YOLO("yolov8n-seg_openvino_model/")

# rtsp_link = 'http://camera.buffalotrace.com/mjpg/video.mjpg'
rtsp_link = os.getenv("RTSP_LINK")

# # Define two areas of interest (AOI)
# area_of_interest_1 = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
# area_of_interest_2 = np.array([[400, 100], [600, 100], [600, 300], [400, 300]])
# Define the polygonal areas of interest (AOI)
area_1 = np.array([[84, 62], [727, 66], [732, 217], [84, 195]])
area_2 = np.array([[84, 338], [715, 349], [720, 526], [92, 498]])

# Replace 'rtsp_link_here' with your RTSP stream link
# rtsp_link = 'rtsp://username:password@ip_address:port/path'  # Replace with your RTSP link

cap = cv2.VideoCapture(rtsp_link)  # Initialize video capture with RTSP link

if not cap.isOpened():
    print("Error: Unable to open the RTSP stream.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection on the frame
    results = ov_model(frame)

    # Draw bounding boxes and check for entry and exit in both areas
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0]  # Bounding box coordinates
            obj_label = detection.cls  # Object class label
            label = model.names[int(obj_label)]

            # Draw bounding box around detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Define the center point of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)

            # Check if the center point is inside any of the areas of interest
            inside_area1 = cv2.pointPolygonTest(area_1, center_point, False) >= 0
            inside_area2 = cv2.pointPolygonTest(area_2, center_point, False) >= 0

            print()
            # Print alert if detected object is inside both areas
            if inside_area1 and inside_area2:
                print(f"Object '{label}' detected in both areas of interest.")

    # Draw the areas of interest on the frame
    cv2.polylines(frame, [area_1], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [area_2], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
