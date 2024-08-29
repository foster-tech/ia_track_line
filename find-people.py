################################################################################
# 
#  find-people.py - Detect if there are same people inside defined area
# 
################################################################################

import cv2, time, os, datetime
from ultralytics import YOLO
from send_email import EmailNotifier
import numpy as np
from termcolor import colored

# Define the polygonal area of interest (AOI)
area_of_interest = np.array([[411, 89], [610, 101], [1118, 719], [449, 720]])

def capture_screenshot(frame, filename='screenshot.png'):
    # Save the screenshot in a file and return the name
    cv2.imwrite(filename, frame)
    print(f'Screenshot saved as {filename}')
    return filename 

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the path to your YOLOv8 model

# RTSP link of the video stream
# rtsp_link = 'http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30'
rtsp_link = os.getenv("RTSP_LINK")
# rtsp_link = 'http://camera.buffalotrace.com/mjpg/video.mjpg'
# rtsp_link = 'http://61.211.241.239/nphMotionJpeg?Resolution=640x640&Quality=Standard'


# Open a video capture object
cap = cv2.VideoCapture(rtsp_link)

# # Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

notifier = EmailNotifier()    

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    if not ret:
        # print("Error: Failed to retrieve frame. Try to get cap again")
        print(colored('Warning: Failed to retrieve frame. Try to get cap again', 'yellow'))
        # print(f"{bcolors.WARNING}Warning: Failed to retrieve frame. Try to get cap again{bcolors.ENDC}")
        # Open a video capture object
        cap = cv2.VideoCapture(rtsp_link)
        continue

    # Perform inference
    results = ov_model(frame)

    if not results or len(results) == 0:
        print("nada")
    
    allow_send_mail = False
    count_person = 0

    for r in results:
        # print(r.boxes)  # print the Boxes object containing the detection bounding boxes   
        boxes = r.boxes
        for box in boxes:
            # print(box.xyxy)
            c = box.cls
            obj = model.names[int(c)]
            # if (obj == 'person' or obj == 'car' or obj == 'truck'):
            if (obj == 'person'):
                count_person = count_person + 1

                for b in box.xyxy:  # xyxy format
                    x1, y1, x2, y2 = b
                    label = obj  # Get class name
                    color = (0, 255, 0)  # Bounding box color (green)

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Define the center point of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    center_point = (center_x, center_y)

                    # Check if the center point is inside the area of interest
                    if cv2.pointPolygonTest(area_of_interest, center_point, False) >= 0:
                        allow_send_mail = True
                        print(f"{label} entered the area")
                    
                    # Draw label
                    # label_text = f'{label} {conf:.2f}'
                    label_text = f'{label}'
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(count_person)
    current_time = time.time()
    current_dt = datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
    # print(current_dt)
    print(colored(current_dt, 'yellow'))
    # print(time.localtime(current_time))

    # Draw the area of interest on the frame
    cv2.polylines(frame, [area_of_interest], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # Display the number of people detected
    cv2.putText(frame, f'Pessoas encontradas: {count_person}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Video Stream', frame)

    # if ( count_person >= 1 ):
    if ( allow_send_mail ):
        

        print(current_time - notifier.last_sent_time)
        print(notifier.interval_seconds)

        # Check if the interval has passed since the last notification
        if current_time - notifier.last_sent_time >= notifier.interval_seconds:
            # Save screen shot in a file 
            attachment_path = capture_screenshot(frame)
            print(attachment_path)
            # Send email with attachment
            subject = f'ALERTA! Pessoa encontrada no local. Quantidade: {count_person} '
            text = f"{label} entrou na area marcada"
            result = notifier.send_email(attachment_path, "", "", subject, text)

        else:
            # Check the remaining time and print in the log 
            remaining_time = notifier.interval_seconds - (current_time - notifier.last_sent_time)
            print(f"Notification skipped. Try again in {int(remaining_time)} seconds.")   
            
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
