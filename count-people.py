import cv2, time, os
from ultralytics import YOLO
from send_email import EmailNotifier

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
        print("Error: Failed to retrieve frame.")
        break

    # Perform inference
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
            if (obj == 'person'):
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

    print(count_person)
    
    # Display the number of people detected
    cv2.putText(frame, f'Pessoas encontradas: {count_person}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Video Stream', frame)

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
            result = notifier.send_email(attachment_path, "", "", subject, subject)

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
