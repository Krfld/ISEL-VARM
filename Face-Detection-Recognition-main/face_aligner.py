import numpy as np
import cv2


class FaceAligner:
    def __init__(self, desired_left_eye_position=(16, 24), desired_right_eye_position=(31, 24),
                 desired_face_width=46, desired_face_height=56, expected_eye_portion=0.1, eye_scale_factor=1.2, eye_min_neighbors=3):
        self.desired_left_eye_position = desired_left_eye_position
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        self.face_cascade = cv2.CascadeClassifier(
            'cascades/data/haarcascade_frontalface_alt2.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            'cascades/data/haarcascade_eye.xml')
        self.desired_left_eye_position = desired_left_eye_position
        self.desired_right_eye_position = desired_right_eye_position
        self.expected_eye_portion = expected_eye_portion
        self.eye_scale_factor = eye_scale_factor
        self.eye_min_neighbors = eye_min_neighbors

    def findFaces(self, image):
        minSize = int((image.shape[0] + image.shape[1]) / 2 * 0.1)
        faces = self.face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=3, minSize=[minSize, minSize])

        return ([(image[y:y+h, x:x+w], x, y) for (x, y, w, h) in faces])

    def findEyes(self, image):
        minSize = int((image.shape[0] + image.shape[1]) / 2
                      * self.expected_eye_portion)
        return self.eye_cascade.detectMultiScale(image, scaleFactor=self.eye_scale_factor, minNeighbors=self.eye_min_neighbors, minSize=[minSize, minSize])

    def align(self, image):
        faces = self.findFaces(image)
        if (len(faces) < 1):
            return None
        face, x, y = faces[0]
        eyes = self.findEyes(face)
        if (len(eyes) < 2):
            return None
        x1, y1, w1, h1 = eyes[0]  # Left eye area
        x1 += x
        y1 += y
        x2, y2, w2, h2 = eyes[1]  # Right eye area
        x2 += x
        y2 += y
        left_eye = (x1 + (w1 / 2), y1 + (h1 / 2))  # Left eye center
        right_eye = (x2 + (w2 / 2), y2 + (h2 / 2))  # Right eye center

        if (left_eye[0] > right_eye[0]):  # Check if eyes are switched
            left_eye, right_eye = right_eye, left_eye

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        # Angle of line that connects both eyes
        angle = np.degrees(np.arctan2(dY, dX))

        # Knowing the distance between eyes, and based on the desired distance between eyes,
        # scale factor can be obtained
        actual_distance = np.sqrt((dX ** 2) + (dY ** 2))
        desired_distance = self.desired_right_eye_position[0] - \
            self.desired_left_eye_position[0]
        scale_factor = desired_distance / actual_distance

        # Find eyes center
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)

        M = cv2.getRotationMatrix2D(
            eyes_center, angle, scale_factor)  # Rotation matrix

        # Update the translation component of the rotation matrix
        tX = self.desired_face_width * 0.5
        tY = self.desired_left_eye_position[1]

        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # Apply the affine transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        return output
