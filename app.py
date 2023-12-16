# This is the main app runner

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

# Load YOLOv8 model
# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)


# Open webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

# Annotation
box_annotator = sv.BoxAnnotator(thickness=2,
                                text_scale=1,
                                text_thickness=2)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Perform object detection on an image using the model
    results = model(frame)[0]

    # Convert results to supervisions format and display
    detections = sv.Detections.from_ultralytics(results)
    print(detections)
    labels = [
        f"{model.model.names[class_id]} {confidence:.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]
    frame = box_annotator.annotate(frame, detections, labels)

    cv2.imshow('Output Window For Debugging', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
