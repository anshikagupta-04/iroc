# Import necessary libraries
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture(0)

# Load YOLO model  
model = YOLO("yolov8n.pt")
classNames = [
    "person", "bicycle", "car", "motorbike", "bus", "train", "truck", "boat", "traffic light", "fire hydrant","stop signal", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear","zebra", "giraffe", "backpack", "umbrella", "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","potted plant","bed","dining table","toilet","tv monitor","laptop","mouse","remort","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair driver","toothbrush"]

#tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits=[443,297,673,297]
totalCOunt=0

# Main loop for capturing and processing frames
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Run YOLO on the frame to detect objects
    results = model(img, stream=True)
    detections = np.empty((0, 5))

    # Loop through the detected objects
    for r in results:
        boxes = r.boxes
        # Loop through the bounding boxes of detected objects
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the class index is within the range of classNames
            if 0 <= box.cls[0] < len(classNames):
                conf = math.ceil((box.conf[0] * 100)) / 100
                # class name
                cls = int(box.cls[0])
                currentclass = classNames[cls]
                
                # Check if the current detection is a person
                if currentclass == "person" and conf > 0.3:
                    # Draw a green rectangle around the detected person
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    w, h = x2 - x1, y2 - y1

                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, currentclass, org, font, fontScale, color, thickness)

                    # Add the detection to the detections array
                    currentArray = np.array([x1, y1, x2, y2, conf]) 
                    detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    for results in resultsTracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        w, h = x2 - x1, y2 - y1
        # cvzone.cornerRect(img, (x1, 1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f' {Id}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=3)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] - 20:
            totalCOunt += 1

    # cvzone.putTextRect(img, f' {totalCOunt}', (50, 50), scale=0.7, thickness=1, offset=3)

    # Display the frame with detected persons
    cv2.imshow("Image Region", img)

    # Wait for a key press
    cv2.waitKey(2)


