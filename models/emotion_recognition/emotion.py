import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear',
          3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            face_image = gray[q:q+s, p:p+r]
            cv2.rectangle(image, (p, q), (p+r, q+s), (255, 0, 0), 2)
            face_image = cv2.resize(face_image, (48, 48))
            feature = np.array(face_image)
            feature = feature.reshape(1, 48, 48, 1)
            feature = feature/255.0
            pred = model.predict(feature)
            prediction_label = labels[pred.argmax()]
            cv2.putText(image, '% s' % (prediction_label), (p-10, q-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
    except cv2.error:
        pass

    return image
