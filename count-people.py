import cv2, time, pandas
import numpy as np
from ultralytics import YOLO
from send_email import EmailNotifier

def capture_screenshot(frame, filename='screenshot.png'):
    # Save the screenshot in a file and return the name
    cv2.imwrite(filename, frame)
    print(f'Screenshot saved as {filename}')
    return filename 

# def draw_results(img, results):
#     """
#     Draw bounding boxes and labels on the image.

#     :param img: Input image (as a NumPy array).
#     :param results: Detection results from YOLO model.
#     :return: Image with drawn bounding boxes.
#     """
#     # Extract detections from results
#     detections = results.pandas().xyxy[0]  # Get results as pandas DataFrame
    
#     for _, row in detections.iterrows():
#         x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
#         label = model.names[int(cls)]  # Get class name
#         color = (0, 255, 0)  # Bounding box color (green)

#         # Draw bounding box
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
#         # Draw label
#         label_text = f'{label} {conf:.2f}'
#         cv2.putText(img, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     return img

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the path to your YOLOv8 model

# RTSP link of the video stream
# rtsp_link = 'http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30'
rtsp_link = 'rtsp://tijuco:T2024@189.91.55.194:3554/cam/realmonitor?channel=1&subtype=0'
# rtsp_link = 'rtsp://46.151.101.134:8082/?action=stream'

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
        print("Error: Failed to retrieve frame.")
        break

    # Perform inference
    # results = model(frame)
    results = ov_model(frame)

    if not results or len(results) == 0:
        print("nada")
    
    count_person = 0

    for r in results:
        # print(r.boxes)  # print the Boxes object containing the detection bounding boxes   
        boxes = r.boxes
        for box in boxes:
            # print(box.xyxy)
            c = box.cls
            obj = model.names[int(c)]
            if obj == 'person':
                count_person = count_person + 1

                for b in box.xyxy:  # xyxy format
                    x1, y1, x2, y2 = b
                    label = obj  # Get class name
                    color = (0, 255, 0)  # Bounding box color (green)

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    # label_text = f'{label} {conf:.2f}'
                    label_text = f'{label}'
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            # # Draw bounding boxes and labels on the frame
            # for index, row in count_person:
            #     x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.putText(frame, f'Person {index+1}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    

    print(count_person)
    # # Process results
    # detections = results.xyxy[0]  # Get detections in pandas dataframe format
    
    # # Filter out the 'person' class (typically class 0 for COCO dataset)
    # people_detections = detections[detections['name'] == 'person']
    
    # # Count the number of people detected
    # num_people = len(people_detections)
    
    # # Draw bounding boxes and labels on the frame
    # for index, row in num_people:
    #     x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, f'Person {index+1}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the number of people detected
    cv2.putText(frame, f'Pessoas encontradas: {count_person}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw results on frame
    # frame_with_boxes = draw_results(frame, results)
    
    # Show the frame
    cv2.imshow('Video Stream', frame)
    # cv2.imshow('Video Stream', frame_with_boxes)

    if ( count_person >= 1 ):
        current_time = time.time()

        print(current_time - notifier.last_sent_time)
        print(notifier.interval_seconds)

        # Check if the interval has passed since the last notification
        if current_time - notifier.last_sent_time >= notifier.interval_seconds:
            # Save screen shot in a file 
            attachment_path = capture_screenshot(frame)
            print(attachment_path)
            # Send email with attachment
            subject = f'ALERTA! Pessoa encontrada no local. Quantidade: {count_person} '
            result = notifier.send_email(attachment_path, "renatofk@gmail.com", "Renato", subject, subject)
            # # Update the last sent time
            # notifier.last_sent_time = current_time

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
