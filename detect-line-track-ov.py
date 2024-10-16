import cv2, time, os, datetime
from ultralytics import YOLO
from send_email import EmailNotifier
import numpy as np
from termcolor import colored
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors
from openvino.runtime import Core

# Define a dictionary to store object tracking paths
tracking_paths = defaultdict(list)

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to check if any object has a tracking line longer than 50 pixels
def check_tracking_line_length(tracking_paths, min_length=50):
    any_obj = False
    for object_id, path in tracking_paths.items():
        # Calculate the length of the tracking line if there are at least two points
        if len(path) > 1:
            total_length = 0
            for i in range(1, len(path)):
                total_length += euclidean_distance(path[i-1], path[i])
            if total_length >= min_length:
                print(f"Object ID {object_id} has a tracking line longer than {min_length}px: {total_length:.2f}px")
                any_obj = True
    if (any_obj):
        return True

def capture_screenshot(frame, filename='screenshot.png'):
    # Save the screenshot in a file and return the name
    cv2.imwrite(filename, frame)
    print(f'Screenshot saved as {filename}')
    return filename 

IP = os.getenv("RTSP_IP")
PORT = os.getenv("RTSP_PORT")
USER = os.getenv("RTSP_USER")
PASS = os.getenv("RTSP_PASS")

# RTSP link of the video stream
rtsp_link = f'rtsp://{USER}:{PASS}@{IP}:{PORT}'

# # Load the YOLOv8 model
# model = YOLO('yolov8n.pt')  # Replace with the path to your YOLOv8 model

# # # Export the model
# model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Initialize the inference engine
core = Core()

# Load the network
model_path = 'yolov8n_openvino/yolov8n.xml'
# weights_path = 'models/yolov8n_openvino_model/yolov8n.bin'
# net = core.read_network(model=model_path, weights=weights_path)
model = core.read_model(model=model_path)

# Load the model to the CPU (or other available devices like GPU, MYRIAD, etc.)
ov_model = core.compile_model(model=model, device_name='CPU')

# # # Export the model
# model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# # Load the exported OpenVINO model
# ov_model = YOLO("yolov8n_openvino_model/")
# # ov_model = YOLO("yolov8n-seg_openvino_model/")

notifier = EmailNotifier()    

vid_frame_count = 0
allow_send_mail = False
count_person = 0

# Open a video capture object
cap = cv2.VideoCapture(rtsp_link)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(rtsp_link)
        continue
    # vid_frame_count += 1
    # Run the frame through YOLO with tracking enabled
    # results = tracker.update(frame)
    # results = ov_model(frame)
    # Run object tracking on the frame
    # results = tracker.update(frame)
    results = ov_model.track(frame, persist=True)

   # Extract classes names
    names = model.model.names

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        # Visualize the results on the frame
        # frame = results[0].plot()

        annotator = Annotator(frame, line_width=2, example=str(names))

        for box, track_id, cls in zip(boxes, track_ids, clss):
            label = str(names[cls]) + ' - ' + str(track_id)
            annotator.box_label(box, label, color=colors(cls, True))
            b1 = (box[0] + box[2]) / 2
            b2 = (box[1] + box[3]) / 2
            # pt1 = tuple([int(point[0] ), int( point[1] ) ])
            # pt2 = tuple([int(point[0] ), int( point[1] ) ])
            bbox_center =  tuple([int( b1 ), int( b2 ) ])  # Bbox center

             # Update the tracking path for the current object
            tracking_paths[track_id].append(bbox_center)

            # Draw the tracking line for the object
            if len(tracking_paths[track_id]) > 1:
                for j in range(1, len(tracking_paths[track_id])):
                    cv2.line(frame, tracking_paths[track_id][j-1], tracking_paths[track_id][j], (0, 255, 255), 2)

    # Check if any object has a tracking line longer than 50 pixels
    allow_send_mail = check_tracking_line_length(tracking_paths, min_length=100)
    # check_tracking_line_length(tracking_paths, min_length=50)
    
    current_time = time.time()
    current_dt = datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
    # print(current_dt)
    print(colored(current_dt, 'yellow'))

    # # # Display the number of people detected
    # cv2.putText(frame, f'Objetos que se moveram:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame, f'{object_ids}', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Stream', frame)

    if ( allow_send_mail ):
        print(current_time - notifier.last_sent_time)
        print(notifier.interval_seconds)

        # Check if the interval has passed since the last notification
        if current_time - notifier.last_sent_time >= notifier.interval_seconds:
            # Save screen shot in a file 
            attachment_path = capture_screenshot(frame)
            print(attachment_path)
            # Send email with attachment
            subject = f'ALERTA! Camera {notifier.camera_name}'
            text = f"{label} foi detectado se movendo no local."
            result = notifier.send_email(attachment_path, "", "", subject, text)

        else:
            # Check the remaining time and print in the log 
            remaining_time = notifier.interval_seconds - (current_time - notifier.last_sent_time)
            print(f"Notification skipped. Try again in {int(remaining_time)} seconds.")   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


