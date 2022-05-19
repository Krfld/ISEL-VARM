from time import sleep
from tkinter import N
import cv2
import numpy as np
import pathlib


def calibrate_chessboard(dir_path, image_format, square_size, width, height):
    '''Calibrate a camera using chessboard images.'''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = pathlib.Path(dir_path).glob(f'*.{image_format}')

    # Iterate through all images
    for fname in images:
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(10000)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)

    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def getDistortion():
    # Parameters
    IMAGES_DIR = 'images'
    IMAGES_FORMAT = 'jpg'
    SQUARE_SIZE = 1.5
    WIDTH = 6
    HEIGHT = 9

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(
        IMAGES_DIR,
        IMAGES_FORMAT,
        SQUARE_SIZE,
        WIDTH,
        HEIGHT
    )

    # Save coefficients into a file
    save_coefficients(mtx, dist, "calibration_chessboard.yml")


def undistort():
    # Load coefficients
    mtx, dist = load_coefficients('calibration_chessboard.yml')
    original = cv2.imread('images/distorted.jpg')

    newcameramtx = None
    # h,  w = original.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    #     mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(original, mtx, dist, None, newcameramtx)
    cv2.imwrite('undist.jpg', dst)


def main():
    SQUARE_SIZE = 1.5
    WIDTH = 6
    HEIGHT = 9

    '''Calibrate a camera using chessboard images.'''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((HEIGHT*WIDTH, 3), np.float32)
    objp[:, :2] = np.mgrid[0:WIDTH, 0:HEIGHT].T.reshape(-1, 2)

    objp = objp * SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (WIDTH, HEIGHT), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(10000)

            # Calibrate camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

            # Save coefficients into a file
            save_coefficients(mtx, dist, "calibration_chessboard.yml")

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break

        sleep(5)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
