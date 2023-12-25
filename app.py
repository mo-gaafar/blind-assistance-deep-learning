# This is the main app runner

import cv2
import torch
from pathlib import Path
import supervision as sv
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import threading
# from gtts import gTTS
import os
# Open webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

# Annotation
box_annotator = sv.BoxAnnotator(thickness=2,
                                text_scale=1,
                                text_thickness=2)


# Flag to control the loop
running = True
show_annotations = False
scene_to_speech = True


def toggle_running():
    global running
    running = not running


def toggle_annotations():
    global show_annotations
    show_annotations = not show_annotations


def toggle_scene_to_speech():
    global scene_to_speech
    scene_to_speech = not scene_to_speech


def tts(text):
    # text = "Hello, this is a test message."
    # tts = gTTS(text=text, lang='en')
    # tts.save("output.mp3")
    # os.system("mpg321 output.mp3")

    pass

# Create an overlay button


from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Webcam Feed")
        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        self.label = QLabel()
        self.label_caption = QLabel()
        self.button_tts = QPushButton("Run Scene-to-Speech")
        self.button_tts.setStyleSheet("background-color: blue; color: white;")
        self.button_pause = QPushButton("Pause")
        self.button_pause.setStyleSheet("background-color: red; color: white;")
        self.button_annotations = QPushButton("Toggle Annotations")
        self.button_annotations.setStyleSheet(
            "background-color: green; color: white;")

        self.grid_layout.addWidget(self.button_tts, 0, 0)
        self.grid_layout.addWidget(self.button_pause, 0, 1)
        self.grid_layout.addWidget(self.button_annotations, 0, 2)
        self.grid_layout.addWidget(self.label_caption, 1, 0, 1, 3)  # Add caption label

        self.layout.addWidget(self.label)
        self.layout.addLayout(self.grid_layout)
        self.setLayout(self.layout)

        self.button_tts.clicked.connect(toggle_scene_to_speech)
        self.button_pause.clicked.connect(toggle_running)
        self.button_annotations.clicked.connect(toggle_annotations)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def update_frame(self):
        # Read frame from webcam
        ret, frame = cap.read()

        # Check if the loop should continue
        if not running:
            return
        self.frame = frame  # save in memory


        # Update caption label width based on inner text
        self.label_caption.adjustSize()
        # self.label_caption.setMinimumWidth(self.label_caption.sizeHint().width())
        self.label_caption.setMaximumHeight(50)

        if scene_to_speech:
            # Display loading message
            self.label_caption.setText("Loading...")

            # Perform captioning asynchronously
            def perform_captioning():
                # Caption the frame
                from models.object_detection.vit_gpt2_caption import vit_gpt2_captioning
                self.caption = vit_gpt2_captioning(self.frame)[0]

                # Update the caption label
                self.label_caption.setText(self.caption)

            # Start the captioning process in a separate thread
            threading.Thread(target=perform_captioning).start()
            # Toggle scene-to-speech if enabled
            if scene_to_speech:
                toggle_scene_to_speech()

        if show_annotations:
            # annotations model
            from models.object_detection.yolo_v8_model import YOLOv8ObjectDetectionModel
            model = YOLOv8ObjectDetectionModel('data/yolov8n.pt')
            # overlay to frame
            frame = model.predict(frame=frame, annotator=box_annotator)

        # Convert frame to QImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_image = QImage(frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)

        # Add captioning model text output

        # caption = "No scene description"
        # self.label_caption.setText(caption)

        self.label_caption.setStyleSheet(
            "background-color: yellow; color: black;")

        # Scale the image to fit the label while preserving aspect ratio
        scaled_image = q_image.scaled(
            self.label.size(), Qt.AspectRatioMode.KeepAspectRatio)

        # Display the scaled image in the label
        self.label.setPixmap(QPixmap.fromImage(scaled_image))

    def resizeEvent(self, event):
        # Set the minimum size of the window
        self.setMinimumSize(400, 300)

        # Call the base class resizeEvent
        super().resizeEvent(event)


app = QApplication([])
window = MainWindow()
window.show()
app.exec_()

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
