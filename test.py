from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model with tracking enabled
# model = YOLO('yolov8n.pt')  # Replace with your model
model = YOLO("yolov8n-seg.pt", task="segment")  # Load an official Segment model

# # Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
# ov_model = YOLO("yolov8n_openvino_model/")
ov_model = YOLO("yolov8n-seg_openvino_model/")
rtsp_link = 'http://camera.buffalotrace.com/mjpg/video.mjpg'

tracker = ov_model.track(rtsp_link, show=True)  # Initialize the tracker

# Define two areas of interest (AOI)
area_of_interest_1 = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
area_of_interest_2 = np.array([[400, 100], [600, 100], [600, 300], [400, 300]])

# Track the state and positions of the objects
object_states = {}  # Stores the state of each object
object_tracks = {}  # Stores the tracking points of each object

def is_point_inside_area(point, area):
    return cv2.pointPolygonTest(area, point, False) >= 0

def check_entry_exit(object_id, center_point, area1, area2):
    if object_id not in object_states:
        object_states[object_id] = {"inside_area1": False, "inside_area2": False, "in_both": False}

    # Check if the object is inside each area
    inside_area1_now = is_point_inside_area(center_point, area1)
    inside_area2_now = is_point_inside_area(center_point, area2)

    was_inside_area1 = object_states[object_id]["inside_area1"]
    was_inside_area2 = object_states[object_id]["inside_area2"]

    # Update the state for each area
    object_states[object_id]["inside_area1"] = inside_area1_now
    object_states[object_id]["inside_area2"] = inside_area2_now

    # Check if the object has entered and left both areas
    if was_inside_area1 and not inside_area1_now and was_inside_area2 and not inside_area2_now:
        if not object_states[object_id]["in_both"]:
            print(f"Object {object_id} entered and left both areas.")
            object_states[object_id]["in_both"] = True

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

    # Run the frame through YOLO with tracking enabled
    results = tracker.update(frame)

    # Process the results
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0]  # Bounding box coordinates
            object_id = detection.id  # Unique object ID
            label = model.names[detection.cls]

            # Define the center point of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)

            # Initialize tracking list if not already done
            if object_id not in object_tracks:
                object_tracks[object_id] = []

            # Append the current center point to the tracking list
            object_tracks[object_id].append(center_point)

            # Check for entry and exit in both areas
            check_entry_exit(object_id, center_point, area_of_interest_1, area_of_interest_2)

            # Draw the tracking line
            for i in range(1, len(object_tracks[object_id])):
                cv2.line(frame, object_tracks[object_id][i-1], object_tracks[object_id][i], (0, 255, 255), 2)

    # Draw the areas of interest on the frame
    cv2.polylines(frame, [area_of_interest_1], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(frame, [area_of_interest_2], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
