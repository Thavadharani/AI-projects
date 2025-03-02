import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (Pretrained on COCO dataset)
model = YOLO("yolov8n.pt")

# Open webcam or video feed
cap = cv2.VideoCapture(0)  # Change '0' to video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 for people detection
    results = model(frame)
    
    people_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            conf = box.conf.item()
            
            # Only detect people (Class ID for person in COCO is 0)
            if class_id == 0 and conf > 0.5:
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Define Crowd Levels
    if people_count == 0:
        status = "No Crowd"
        color = (255, 255, 255)
    elif people_count < 5:
        status = "Less Crowd"
        color = (0, 255, 0)
    else:
        status = "Crowd Alert!"
        color = (0, 0, 255)

    # Display crowd status
    cv2.putText(frame, f'People Count: {people_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Crowd Management", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# Placeholder function for misbehavior detection
def detect_misbehavior(frame):
    # Your action recognition model logic here
    # Example: If a fight is detected, return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 for people detection
    results = model(frame)
    
    people_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id == 0:  # Detecting only people
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Check for misbehavior
    if detect_misbehavior(frame):
        status = "Misbehavior Detected!"
        color = (0, 0, 255)
        cv2.putText(frame, "Security Alert: Misbehavior!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame
    cv2.imshow("Crowd Management", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
