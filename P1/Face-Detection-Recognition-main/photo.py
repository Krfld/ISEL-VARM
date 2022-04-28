import matplotlib.pyplot as plt
import numpy as np
import cv2
from mtcnn import MTCNN

# pip install opencv-python
# pip install mtcnn
# pip install tensorflow

image_raw = 'friends.jpg'


def classifier(classifier, image):
    if (classifier == 'haar_cascade'):
        # Find faces:
        faces = cv2.CascadeClassifier(
            'cascades/data/haarcascade_eye.xml').detectMultiScale(image)
        # For each face found:
        for result in faces:
            x, y, w, h = result
            x1, y1 = x + w, y + h
            # Draw rectangle:
            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
    elif (classifier == 'mtcnn'):
        # Find faces:
        faces = MTCNN().detect_faces(image)
        # For each face found:
        for result in faces:
            x, y, w, h = result['box']
            x1, y1 = x + w, y + h
            # Draw rectangle:
            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)


image = cv2.imread(image_raw)
classifier(classifier='haar_cascade', image=image)

cv2.imshow('Image', image)
while cv2.waitKey(20) & 0xFF != ord('q'):
    pass
