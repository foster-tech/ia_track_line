import cv2, time, os, datetime
from ultralytics import YOLO
from send_email import EmailNotifier
import numpy as np
from termcolor import colored
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

# Define the polygonal areas of interest (AOI)
area_1 = np.array([[418, 459], [971, 441], [1048, 657], [431, 653]])  #Mato
area_2 = np.array([[414, 297], [786, 297], [823, 421], [423, 419]])  #Mato
# area_1 = np.array([[122, 482], [450, 698], [789, 703], [258, 405]]) #dutra
# area_2 = np.array([[337, 374], [890, 697], [1209, 698], [471, 310]]) #dutra
# area_1 = np.array([[352, 279], [513, 275], [580, 310], [358, 326]]) #Japan
# area_2 = np.array([[366, 333], [640, 290], [641, 486], [361, 485]]) #Japan

def capture_screenshot(frame, filename='screenshot.png'):
    # Save the screenshot in a file and return the name
    cv2.imwrite(filename, frame)
    print(f'Screenshot saved as {filename}')
    return filename 

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the path to your YOLOv8 model

IP = os.getenv("RTSP_IP")
PORT = os.getenv("RTSP_PORT")
USER = os.getenv("RTSP_USER")
PASS = os.getenv("RTSP_PASS")

# RTSP link of the video stream
# rtsp_link = 'http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30'
# rtsp_link = 'http://camera.buffalotrace.com/mjpg/video.mjpg'
# rtsp_link = 'http://61.211.241.239/nphMotionJpeg?Resolution=640x480&Quality=Standard'
# rtsp_link = '../ultralytics/files/dutra.mp4' # Dutra Saida SP
rtsp_link = f'rtsp://{USER}:{PASS}@{IP}:{PORT}'

# Track the state of the objects
object_states = {}  # Stores the state of each object

object_ids = {}

# # Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")
# ov_model = YOLO("yolov8n-seg_openvino_model/")


notifier = EmailNotifier()    

vid_frame_count = 0
allow_send_mail = False
count_person = 0
# def mouse_callback(event, x, y, flags, param):
#     """
#     Handles mouse events for region manipulation.

#     Parameters:
#         event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN).
#         x (int): The x-coordinate of the mouse pointer.
#         y (int): The y-coordinate of the mouse pointer.
#         flags (int): Additional flags passed by OpenCV.
#         param: Additional parameters passed to the callback (not used in this function).

#     Global Variables:
#         current_region (dict): A dictionary representing the current selected region.

#     Mouse Events:
#         - LBUTTONDOWN: Initiates dragging for the region containing the clicked point.
#         - MOUSEMOVE: Moves the selected region if dragging is active.
#         - LBUTTONUP: Ends dragging for the selected region.

#     Notes:
#         - This function is intended to be used as a callback for OpenCV mouse events.
#         - Requires the existence of the 'counting_regions' list and the 'Polygon' class.

#     Example:
#         >>> cv2.setMouseCallback(window_name, mouse_callback)
#     """
#     global current_region

#     # Mouse left button down event
#     if event == cv2.EVENT_LBUTTONDOWN:
#         for region in counting_regions:
#             if region["polygon"].contains(Point((x, y))):
#                 current_region = region
#                 current_region["dragging"] = True
#                 current_region["offset_x"] = x
#                 current_region["offset_y"] = y

#     # Mouse move event
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if current_region is not None and current_region["dragging"]:
#             dx = x - current_region["offset_x"]
#             dy = y - current_region["offset_y"]
#             current_region["polygon"] = Polygon(
#                 [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
#             )
#             current_region["offset_x"] = x
#             current_region["offset_y"] = y

#     # Mouse left button up event
#     elif event == cv2.EVENT_LBUTTONUP:
#         if current_region is not None and current_region["dragging"]:
#             current_region["dragging"] = False


def is_point_inside_area(point, area):
    pt = tuple([int(point[0] ), int( point[1] ) ])
    return cv2.pointPolygonTest(area, pt, False) >= 0

def check_entry_exit(object_id, center_point, area_1, area_2):
    if object_id not in object_states:
        object_states[object_id] = {"inside_area1": False, "inside_area2": False, "in_both": False}
    
    # print(object_id)
    # print(object_states)

    # Check if the object is inside each area
    inside_area1_now = is_point_inside_area(center_point, area_1)
    inside_area2_now = is_point_inside_area(center_point, area_2)

    was_inside_area1 = object_states[object_id]["inside_area1"]
    was_inside_area2 = object_states[object_id]["inside_area2"]

    # Update the state for each area
    if object_states[object_id]["inside_area1"] == False:
        object_states[object_id]["inside_area1"] = inside_area1_now
    if object_states[object_id]["inside_area2"] == False:
        object_states[object_id]["inside_area2"] = inside_area2_now

    # Check if the object has entered and left both areas
    # if was_inside_area1 and not inside_area1_now and was_inside_area2 and not inside_area2_now:
    if was_inside_area1 and was_inside_area2:
        if not object_states[object_id]["in_both"]:
            # print(colored(f"Object {object_id} entered and left both areas.",'green'))
            print(colored(f"Object {object_id} entered both areas.",'green'))
            object_states[object_id]["in_both"] = True
            object_ids[object_id] = str(object_id)
            return True

    return False    

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
            label = str(track_id) + ' - ' + str(names[cls])
            annotator.box_label(box, label, color=colors(cls, True))
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

            # Check for entry and exit in both areas
            allow_send_mail = check_entry_exit(track_id, bbox_center, area_1, area_2)
            # print(track_id,' - ',str(names[cls]))
            track = track_history[track_id]  # Tracking Lines plot
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=2)


    # Draw the areas of interest on the frame
    # cv2.polylines(frame, [area_1], isClosed=True, color=(0, 255, 0), thickness=2)
    # cv2.polylines(frame, [area_2], isClosed=True, color=(255, 0, 0), thickness=2)
    
    current_time = time.time()
    current_dt = datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
    # print(current_dt)
    print(colored(current_dt, 'yellow'))

    # # Display the number of people detected
    cv2.putText(frame, f'Objetos que entraram nas duas areas:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'{object_ids}', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # if vid_frame_count == 1:
    #     cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
    #     cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
    # cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)
    # Display the Video Stream
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
            subject = f'ALERTA! Uma pessoa passou nas duas areas marcadas.'
            text = f"{label} passou nas duas areas marcadas."
            result = notifier.send_email(attachment_path, "", "", subject, text)

        else:
            # Check the remaining time and print in the log 
            remaining_time = notifier.interval_seconds - (current_time - notifier.last_sent_time)
            print(f"Notification skipped. Try again in {int(remaining_time)} seconds.")   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


