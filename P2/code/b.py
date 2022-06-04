from time import sleep
import numpy as np
import cv2
from cv2 import aruco
from a import load_coefficients


def main():
    ARUCO_MARKER_SIZE = 5.5  # cm

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

        frame_markers = aruco.drawDetectedMarkers(
            frame_markers, corners, ids)

        if ids is not None:
            # Estimate pose for markers
            rvecs, tvecs, objPoints = aruco.estimatePoseSingleMarkers(
                corners, ARUCO_MARKER_SIZE, cameraMatrix, distCoeffs)

            # Draw axis for each marker
            for i in range(len(ids)):
                frame_markers = aruco.drawAxis(
                    frame_markers, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], ARUCO_MARKER_SIZE/2)

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
