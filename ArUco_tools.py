import os
import math
import cv2 as cv
import numpy as np
from cv2 import aruco
from scipy.spatial.transform import Rotation as R


class ArUcoTools:
    # The different ArUco dictionaries built into the OpenCV library.
    ARUCO_DICT = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_50": aruco.DICT_7X7_50,
        "DICT_7X7_100": aruco.DICT_7X7_100,
        "DICT_7X7_250": aruco.DICT_7X7_250,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL
    }

    _mtx = None
    _dist = None

    def __init__(self):
        print('ArUco tools has been started')

    @staticmethod
    def take_images(n_samples: int, save_local: bool = False) -> np.ndarray:
        """
            Open the camera and take some photos
        :param save_local: Enable save images on local machine
        :param n_samples: Number of photos desired
        :return: photos
        """

        images = []

        # The code below is to open camera and process the image
        camera = cv.VideoCapture(0)
        if not camera.isOpened():
            print('Cannot open camera')
            exit()

        while len(images) < n_samples:
            ret, frame = camera.read()

            if not ret:
                print("Can't receive frame\nExiting ...")
                exit()

            cv.imshow(f'Image from camera {camera.getBackendName()}', frame)
            if cv.waitKey(1) != -1:
                images.append(frame)
                print('Photo was taken')
                if save_local:
                    cv.imwrite(f'Photo_{len(images) - 1}_{camera.getBackendName()}.png', frame)
                    print('The photo was saved')

        camera.release()
        cv.destroyAllWindows()

        return np.array(images)

    @staticmethod
    def take_local_images(file_path: str, file_extension: str = 'png') -> np.ndarray:
        """
            Check files in local directory
        :param file_path:
        :param file_extension:
        :return:
        """

        if not os.path.isdir(file_path):
            print('The path provided does not have a directory')
            exit()

        files = [file for file in os.listdir(file_path) if str(file).endswith(f'.{file_extension}')]

        local_images = [cv.imread(file) for file in files]

        return np.array(local_images)

    @staticmethod
    def camera_calibration(images: np.ndarray, checkerboard_size: tuple = (6, 9)) -> tuple:
        """
            Find intrinsic camera matrix and lens distortion coefficients.
        :return:
        """

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Store 3D points for each checkerboard image
        objpoints = []
        # Store 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[0, :, : 2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

        img_example = None

        for image in images:

            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(image, checkerboard_size,
                                                    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK
                                                    + cv.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                objpoints.append(objp)
                corners2d = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2d)

            if img_example is None:
                img_example = image

        _, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, img_example.shape[::-1], None, None)

        return mtx, dist

    def generate_single_marker(self, marker_id: int, aruco_dict: str = 'DICT_4X4_50', marker_size: int = 400,
                               path_save: str = None, file_ext: str = 'png') -> None:
        """

        :param marker_id:
        :param aruco_dict:
        :param marker_size:
        :param path_save:
        :param file_ext:
        :return:
        """
        aruco_dict = aruco.getPredefinedDictionary(self.ARUCO_DICT[aruco_dict])

        marker_img = aruco.generateImageMarker(aruco_dict, marker_size, marker_id)

        if not os.path.isdir(path_save):
            print('The path provided does not have a directory')
            path_save = None

        cv.imwrite(f'{path_save}ArUco_{marker_size}_{marker_id}.{file_ext}', marker_img)

    def marker_identifier_on_image(self, image: np.ndarray, desired_aruco_dict: str = 'DICT_4X4_50') -> tuple:
        """

        :param image:
        :param desired_aruco_dict:
        :return:
        """

        aruco_dict = aruco.getPredefinedDictionary(self.ARUCO_DICT[desired_aruco_dict])
        aruco_param = aruco.DetectorParameters()

        aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_param)

        corners, ids, rejects = aruco_detector.detectMarkers(image)

        return corners, ids, rejects

    def marker_identifier_real_time(self, desired_aruco_dict: str = 'DICT_4X4_50') -> tuple:
        """

        :param desired_aruco_dict:
        :return:
        """

        aruco_dict = aruco.getPredefinedDictionary(self.ARUCO_DICT[desired_aruco_dict])
        aruco_param = aruco.DetectorParameters()

        aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_param)

        camera = cv.VideoCapture(0)

        if not camera.isOpened():
            print('Cannot open camera')
            exit()

        while True:
            ret, frame = camera.read()

            if not ret:
                print("Can't receive frame\nExiting ...")
                exit()

            corners_, ids_, rejects_ = aruco_detector.detectMarkers(frame)

            image_with_detections = aruco.drawDetectedMarkers(frame, corners_, ids_)

            cv.imshow(f'Camera image', image_with_detections)

            if cv.waitKey(1) != -1:
                camera.release()
                cv.destroyAllWindows()

                return corners_, ids_, rejects_

    @staticmethod
    def euler_from_quaternion(x, y, z, w) -> tuple:
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)

        :param x:
        :param y:
        :param z:
        :param w:
        :return
        """
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def marker_estimate_position(self, corners: np.ndarray, marker_size: float, mtx: np.ndarray = _mtx,
                                 dist: np.ndarray = _dist) -> tuple:
        """
            This will estimate the rvec and tvec for each of the marker corners detected by:
               corners, ids, rejectedImgPoints = detector.detectMarkers(image)
                corners - is an array of detected corners for each detected marker in the image
                marker_size - is the size of the detected markers
                mtx - is the camera matrix
                distortion - is the camera distortion matrix
                RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        """

        if mtx is None or dist is None:
            mtx, dist = self.camera_calibration(self.take_images(4))

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, -marker_size / 2, 0],
                                  [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        # trash = []
        rvecs = []
        tvecs = []

        for c in corners:
            _, R, t = cv.solvePnP(marker_points, c, mtx, dist, False, cv.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            # trash.append(nada)
        return rvecs, tvecs


if __name__ == "__main__":
    print('Code has been started')

    blabla = ArUcoTools()

    image_ = blabla.take_images(1)

    # input(f'{image_.shape}')

    c_, i_, r_ = blabla.marker_identifier_on_image(image_[0])

    print(f'{c_=}\n{i_=}\n{r_=}')
