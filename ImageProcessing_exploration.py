import cv2 as cv
import numpy as np


def test_function() -> None:
    """
        Just for test changes in the class
    """
    _imageprocessing = ImageProcessing()

    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        print('Camera not opened')
        exit()

    while cv.waitKey(1) == -1:
        ret, frame = camera.read()

        if ret:
            # Here you apply your image processing
            frame = _imageprocessing.remove_noise(frame)

            cv.imshow('Final image', frame)


class ImageProcessing:
    _camera_id = None

    def __init__(self, camera_id: int = 0):
        print('class Image Processing')

        self._camera_id = camera_id

        # self.live_camera()

    def filter(self, image: np.ndarray, filter_: str) -> np.ndarray:
        """

        :param image:
        :param filter_:
        :return:
        """

        # filter_functions = {
        #     'blur': cv.blur,
        #     'gaussianblur': cv.GaussianBlur,
        #
        #
        # }

        match filter_:
            case 'bilateral':
                # highly effective in noise removal while keeping edges sharp. But the operation is slower compared
                # to other filters.
                cv.bilateralFilter(image, 9, 75, 75)

            case 'blur':
                cv.blur(image, (5, 5))

        return image

    @staticmethod
    def remove_gaussian_noise(_image: np.ndarray) -> np.ndarray:
        """
            To remove noise you need a low-pass filter

            - Mean filter - affect borders

            - Gaussian blurring is highly effective in removing Gaussian noise - low pass filter

            - Median blurring is highly effective against salt-and-pepper noise



        :param _image: Image to be processed
        :return: Image processing
        """

        # If sigmaX == sigmaY == 0 they are compute from ksize.width and ksize.height

        return cv.GaussianBlur(src=_image, ksize=(5, 5), sigmaX=0, sigmaY=0, borderType=cv.BORDER_CONSTANT)

    def live_camera(self) -> None:
        camera = cv.VideoCapture(self._camera_id)

        if camera.isOpened():
            while cv.waitKey(1) == -1:
                ret, frame = camera.read()
                if ret:
                    # apply here your image processing

                    cv.imshow(f'Camera {camera.getBackendName()} image', frame)


if __name__ == "__main__":
    print('Code has been started')

    test_function()
