from time import sleep
import cv2
from cv2 import aruco
import numpy as np
from a2 import load_coefficients


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cameraMatrix, distCoeffs = load_coefficients(
        'calibration_chessboard.yml')
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        frame_markers = frame.copy()
        if ids != None:
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, 5, cameraMatrix, distCoeffs)

            # for i in ids:
            frame_markers = aruco.drawDetectedMarkers(
                frame_markers, corners, ids)
            # frame_markers = aruco.drawAxis(
            # frame.copy(), cameraMatrix, distCoeffs, rvecs, tvecs, 5)

        # Display the resulting frame
        cv2.imshow('frame', frame_markers)
        # cv2.imshow('frame', gray)

        if cv2.waitKey(1) == ord('q'):
            break

        sleep(1/5)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
