from time import sleep
import numpy as np
import cv2
from cv2 import aruco
from a import load_coefficients


def main():
    virtual_objects = [
        cv2.imread('./images/1.png'),
        cv2.imread('./images/2.png'),
        cv2.imread('./images/3.png'),
        cv2.imread('./images/4.png'),
        cv2.imread('./images/5.png'),
        cv2.imread('./images/6.png'),
        cv2.imread('./images/7.png'),
        cv2.imread('./images/8.png'),
        cv2.imread('./images/9.png'),
        cv2.imread('./images/10.png'),
        cv2.imread('./images/11.png'),
        cv2.imread('./images/12.png'),
    ]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cameraMatrix, distCoeffs = load_coefficients('calibration_chessboard.yml')
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_markers = frame.copy()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        if ids is not None:
            for i in range(len(ids)):
                bbox = corners[i]
                tl = bbox[0][0][0], bbox[0][0][1]
                tr = bbox[0][1][0], bbox[0][1][1]
                br = bbox[0][2][0], bbox[0][2][1]
                bl = bbox[0][3][0], bbox[0][3][1]
                obj = virtual_objects[ids[i][0] - 1]
                h, w, c = obj.shape
                pts1 = np.array([tl, tr, br, bl])
                pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                matrix, _ = cv2.findHomography(pts2, pts1)
                imgOut = cv2.warpPerspective(
                    obj, matrix, (frame_markers.shape[1], frame_markers.shape[0]))

                # Keep real background:
                cv2.fillConvexPoly(frame_markers, pts1.astype(int), (0, 0, 0))
                frame_markers = frame_markers + imgOut

        # Display the resulting frame
        cv2.imshow('frame', frame_markers)
        # cv2.imshow('frame', gray)

        if cv2.waitKey(1) == ord('q'):
            break

        # sleep(1/5)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
