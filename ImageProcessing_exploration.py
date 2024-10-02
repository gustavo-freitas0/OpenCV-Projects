from typing import Optional

import cv2 as cv
import numpy as np


def test_function() -> None:
    """
        Just for test the class
    """
    _imageprocessing = ImageProcessing()

    _image = _imageprocessing.take_picture()

    # Apply image processing

    cv.imshow("Final image", _image)
    if cv.waitKey(0) != -1:
        cv.destroyAllWindows()


class ImageProcessing:
    _camera_id = None

    def __init__(self, camera_id: int = 0):
        print('class Image Processing')

        self._camera_id = camera_id

    @staticmethod
    def remove_gaussian_noise(_image: np.ndarray) -> np.ndarray:
        """
            To remove Gaussian noise, low-pass filter
        :param _image: Image to be processed
        :return: Image processing
        """

        # input image can have any number of channels
        # ksize, width and height can differ but they both must be positive and odd
        # If sigmaX == sigmaY == 0 they are compute from ksize.width and ksize.height

        return cv.GaussianBlur(src=_image, ksize=(5, 5), sigmaX=0, sigmaY=0, borderType=cv.BORDER_CONSTANT)

    @staticmethod
    def remove_salt_and_pepper_noise(_image: np.ndarray) -> np.ndarray:
        """
            To remove salt and pepper noise
        :param _image: Image to be processed
        :return: Image processing
        """

        # input image can be 1, 3 or 4 channel image
        # ksize must be odd and greater than 1

        return cv.medianBlur(src=_image, ksize=5)

    @staticmethod
    def remove_noise_keeping_edges(_image: np.ndarray) -> np.ndarray:
        """
            Reduce noise and keep edges sharp
        :param _image: Image to be processed
        :return: Image processing
        """

        # input image with 1 or 3 channel
        # d - if it is non-positive, it is computed from SigmaSpace
        # SigmaColor - high values result in larger areas of semi-equal color
        # SigmaSpace - larger value of the parameter means that farther pixels will influence each other
        #   as long as their colors are close enough

        return cv.bilateralFilter(src=_image, d=9, sigmaColor=75, sigmaSpace=75, borderType=cv.BORDER_CONSTANT)

    @staticmethod
    def adjust_gamma(_image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
            Gamma correction
        :param _image: Image to be processed
        :param gamma: Gamma value - [0, 1.0]
        :return: Image processing
        """

        invgamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invgamma) * 255 for i in np.arange(0, 256)]).astype('uint8')

        return cv.LUT(_image, table)

    @staticmethod
    def color_filter_for_tracking(_image: np.ndarray, _color: np.ndarray, _threshold: int = 10) -> np.ndarray:
        """
            Color filter to help object tracking.
        :param _image: Image to be processed
        :param _color: Color desired in HSV format
        :param _threshold: Color threshold
        :return: Image processing
        """

        # Color convertion
        image_hsv = cv.cvtColor(_image, cv.COLOR_BGR2HSV)

        # range color
        lower_color = upper_color = _color
        lower_color[0] -= _threshold
        upper_color[0] += _threshold

        # Color mask
        mask = cv.inRange(image_hsv, lower_color, upper_color)

        return cv.bitwise_and(src1=_image, src2=_image, mask=mask)

    @staticmethod
    def zooming(_image: np.ndarray, _scale: tuple = (2, 2)) -> np.ndarray:
        """
            Zoom scaling
        :param _image: Image to be processed
        :param _scale: Zoom rate
        :return: Larger image
        """

        height, width = _image.shape[:2] if len(_image) > 2 else _image.shape

        # For zoom resizing it is recommended to use cubic interpolation
        return cv.resize(src=_image, dsize=(2 * width, 2 * height), interpolation=cv.INTER_CUBIC)

    def take_picture(self) -> Optional[np.ndarray, None]:
        """
           Take the photo
        :return: Image or None (if the camera has a problem)
        """
        camera = cv.VideoCapture(self._camera_id)
        frame = None

        if camera.isOpened():
            print('Enter any button to take the photo')
            while cv.waitKey(1) == -1:
                ret, frame = camera.read()
                if ret:
                    cv.imshow(f'Image from camera {self._camera_id} : {camera.getBackendName()}', frame)

        camera.release()
        cv.destroyAllWindows()
        return frame


if __name__ == "__main__":
    print('Code has been started')

    test_function()
