import os
import math
import cv2 as cv
import numpy as np
from cv2 import aruco
from scipy.spatial.transform import Rotation as R

"""
    This is a personal project to explore OpenCV ArUco.
    
    Gustavo Freitas
"""


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

    _last_x_value = None
    _last_y_value = None
    _last_z_value = None

    _calibration_param_path = ""
    _calibration_images = None

    # In meters
    _aruco_marker_side_length_standard = 0.0785

    def __init__(self, local_path: str = os.getcwd()):

        if not os.path.isdir(local_path):
            local_path = os.getcwd()

        self._calibration_param_path = local_path

        print('ArUco tools has been started')

        self._mtx, self._dist = self.find_calibrate_parameters(file_path=self._calibration_param_path)

        if self._mtx is None or self._dist is None:
            print(f'Taking calibration images from {local_path}')
            self._calibration_images = self.take_local_images(file_path=local_path)
            if not self._calibration_images.any():
                print('Any pictures was found. You need take 4 different photos with Chess board')
                self._calibration_images = self.take_images(n_samples=4, save_local=True)
                self.camera_calibration(images=self._calibration_images)
        print('Calibrated camera')

    @staticmethod
    def find_calibrate_parameters(file_path: str) -> tuple:
        """
            Find camera matrix and distortion coefficients
        :param file_path: Directory with yaml file
        :return: Camera matrix, Distortion coefficients
        """

        calibrate_param_file_name = None

        for root, dirs, files in os.walk(file_path):
            calibrate_param_file_name = [file for file in files if str(file).endswith('.yaml')]
            if calibrate_param_file_name:
                break

        if not calibrate_param_file_name:
            return None, None

        open_file = cv.FileStorage(os.path.join(file_path, calibrate_param_file_name[0]), cv.FILE_STORAGE_READ)
        param_mtx = open_file.getNode('K').mat()
        param_dist = open_file.getNode('D').mat()
        open_file.release()

        return param_mtx, param_dist

    @staticmethod
    def take_images(n_samples: int, save_local: bool = False) -> np.ndarray:
        """
            Open the camera and take some photos
        :param n_samples: Number of photos desired
        :param save_local: Enable save images on local machine
        :return: Photos
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
    def take_local_images(file_path: str, file_extension: str = 'png', show_images: bool = False) -> np.ndarray:
        """
            Get photos with chessboard from local directory
        :param file_path: Photo directory
        :param file_extension: Image extension, like png, jpeg, etc.
        :param show_images: Show the photos taken
        :return: Images with chess board
        """

        if not os.path.isdir(file_path):
            print('The path provided does not have a directory')
            exit()

        files = [file for file in os.listdir(file_path) if str(file).endswith(f'.{file_extension}')]

        local_images = [cv.imread(file) for file in files]

        if show_images and len(local_images):
            for image in local_images:
                while cv.waitKey(1) == -1:
                    cv.imshow('Local image - Press any key to continue', image)
            cv.destroyAllWindows()

        return np.array(local_images)

    def camera_calibration(self, images: np.ndarray, checkerboard_size: tuple = (6, 9)) -> tuple:
        """
            Find intrinsic camera matrix and lens distortion coefficients.
        :param images: Images with Chess board
        :param checkerboard_size: Number of inner corners per a chessboard row and column
        :return: Intrinsic camera matrix, len distortion coefficients
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

        self._mtx = mtx
        self._dist = dist

        open_file = cv.FileStorage(os.path.join(self._calibration_param_path, 'camera_calibration_parameters.yaml'),
                                   cv.FILE_STORAGE_WRITE)
        open_file.write('K', self._mtx)
        open_file.write('D', self._dist)
        open_file.release()

        return mtx, dist

    def generate_single_marker(self, marker_id: int, aruco_dict: str = 'DICT_4X4_50', marker_size: int = 400,
                               path_save: str = None, file_ext: str = 'png') -> np.ndarray:
        """
            Generates an ArUco model
        :param marker_id: ArUco marker ID
        :param aruco_dict: ArUco dictionary wanted
        :param marker_size: Size of the image in pixels (1px = 0.0264583333 cm)
        :param path_save: Directory path to save the image
        :param file_ext: File extension, like png
        :return: ArUco image
        """
        aruco_dict = aruco.getPredefinedDictionary(self.ARUCO_DICT[aruco_dict])

        marker_img = aruco.generateImageMarker(aruco_dict, marker_size, marker_id)

        if not os.path.isdir(path_save):
            print('The path provided does not have a directory')
            path_save = None

        cv.imwrite(f'{path_save}ArUco_{marker_size}_{marker_id}.{file_ext}', marker_img)

        return marker_img

    def marker_identifier_on_image(self, image: np.ndarray, desired_aruco_dict: str = 'DICT_4X4_50') -> tuple:
        """
            Identifies ArUco markers
        :param image: Image shown
        :param desired_aruco_dict: ArUco dictionary wanted
        :return: ArUco corners, ArUco IDs and reject points
        """

        aruco_dict = aruco.getPredefinedDictionary(self.ARUCO_DICT[desired_aruco_dict])
        aruco_param = aruco.DetectorParameters()

        aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_param)

        corners, ids, rejects = aruco_detector.detectMarkers(image)

        return corners, ids, rejects

    def marker_identifier_real_time(self, desired_aruco_dict: str = 'DICT_4X4_50') -> tuple:
        """
            Identifies ArUco markers and display them
        :param desired_aruco_dict: ArUco dictionary wanted
        :return: ArUco corners, ArUco IDs and reject points
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

    def markers_translation_movement(self, tvecs: np.ndarray) -> tuple:
        """
            Calculate translation movement
        :param tvecs: Translation vectors estimated
        :return: X movement, Y movement, Z movement
        """

        if self._last_x_value is None:
            self._last_x_value = tvecs[0][0]
            self._last_y_value = tvecs[1][0]
            self._last_z_value = tvecs[2][0]

            return 0, 0, 0

        dif_x = tvecs[0][0] - self._last_x_value
        dif_y = tvecs[1][0] - self._last_y_value
        dif_z = tvecs[2][0] - self._last_z_value

        return dif_x, dif_y, dif_z

    @staticmethod
    def markers_rotation_movement(rvecs: np.ndarray) -> tuple:
        """
            Calculate rotation movement
        :param rvecs: Rotation vectors estimated
        :return: Roll, Pitch, Yaw
        """

        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv.Rodrigues(np.array(rvecs))[0]

        r = R.from_matrix(rotation_matrix[0:3, 0:3])
        quaternion = r.as_quat()

        # Quaternion format
        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = math.degrees(math.atan2(t0, t1))

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.degrees(math.asin(t2))

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.degrees(math.atan2(t3, t4))

        return roll_x, pitch_y, yaw_z

    def marker_estimate_position(self, corners: np.ndarray,
                                 marker_size: float = _aruco_marker_side_length_standard, mtx: np.ndarray = _mtx,
                                 dist: np.ndarray = _dist) -> tuple:
        """
            Estimates translation and rotation markers position
        :param corners: Vector of detected marker corners
        :param marker_size: ArUco marker side length, in meters
        :param mtx: Intrinsic camera matrix
        :param dist: Lens distortion coefficient
        :return:
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

    def draw_markers_pose(self, desired_aruco_dict: str = 'DICT_4X4_50', verbose_: bool = False) -> None:
        """
            Identifies ArUco markers and draw axis over them
        :param desired_aruco_dict: ArUco dictionary wanted
        :param verbose_: Shows markers translation and rotation movement
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

            frame = aruco.drawDetectedMarkers(frame, corners_, ids_)

            if ids_ is not None:
                rvecs_, tvecs_ = self.marker_estimate_position(corners=corners_,
                                                               marker_size=self._aruco_marker_side_length_standard,
                                                               mtx=self._mtx,
                                                               dist=self._dist)

                for i, pose in enumerate(ids_):
                    frame = cv.drawFrameAxes(frame, self._mtx, self._dist, np.array(rvecs_[i]), np.array(tvecs_[i]),
                                             self._aruco_marker_side_length_standard, 2)

                    if verbose_:
                        x, y, z = self.markers_translation_movement(tvecs=tvecs_[i])
                        print(f'Translation movement for ID {pose}:\n\tX = {x}\n\tY = {y}\n\tZ = {z}')

                        roll_, pitch_, yaw_ = self.markers_rotation_movement(rvecs=np.array(rvecs_[i]))
                        print(f'Rotation movement for ID {pose}:\n\troll = {roll_}\n\tpitch = {pitch_}\n\tyaw = {yaw_}')

            cv.imshow(f'Camera image with markers and axis', frame)

            if cv.waitKey(1) != -1:
                camera.release()
                cv.destroyAllWindows()


if __name__ == "__main__":
    print('Code has been started')

    _aruco = ArUcoTools()

    # Check ArUcos and estimate their poses
    _aruco.draw_markers_pose()
