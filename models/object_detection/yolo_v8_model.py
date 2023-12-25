# models/object_detection/yolov8_object_detection.py

import torch
# from torchvision.models.detection import yolo_v3
from ultralytics import YOLO
import supervision as sv


class YOLOv8ObjectDetectionModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_model(self):

        # Load a pretrained YOLO model (recommended for training)
        model = YOLO(self.model_path)

        # # Train the model using the 'coco128.yaml' dataset for 3 epochs
        # results = model.train(data='coco128.yaml', epochs=3)
        return model

    def preprocess_image(self, image):
        # Placeholder for preprocessing image
        # Replace this with actual image preprocessing logic
        return image

    def predict(self, frame, annotator):
        results = self.model(frame)[0]

        # Convert results to supervisions format and display
        detections = sv.Detections.from_ultralytics(results)
        # print(detections)
        labels = [
            f"{self.model.model.names[class_id]} {confidence:.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
        frame = annotator.annotate(frame, detections, labels)
        return frame


# # Example Usage:
# if __name__ == "__main__":
#     model_path = "path/to/your/pretrained_model.pth"
#     yolo_model = YOLOv8ObjectDetectionModel(model_path)

#     # Example: Load an image (replace with actual image loading code)
#     input_image = torch.rand((1, 3, 416, 416)).to(yolo_model.device)

#     # Example: Make predictions
#     predictions = yolo_model.predict(input_image)

#     print("YOLOv8 Object Detection Result:", predictions)
