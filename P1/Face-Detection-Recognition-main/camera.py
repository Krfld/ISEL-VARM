import matplotlib.pyplot as plt
import numpy as np
import cv2
from mtcnn import MTCNN

# pip install opencv-python
# pip install mtcnn
# pip install tensorflow


def classifier(classifier, image):
    if (classifier == 'haar_cascade'):
        # Find faces:
        faces = cv2.CascadeClassifier(
            'cascades/data/haarcascade_frontalface_alt2.xml').detectMultiScale(image)
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


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # Try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    classifier(classifier='haar_cascade', image=frame)
    cv2.imshow("preview", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):  # Exit with 'q'
        break
