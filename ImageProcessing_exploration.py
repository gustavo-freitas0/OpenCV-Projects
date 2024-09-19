import cv2 as cv
import numpy as np


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

        match filter_:
            case 'bilateral':
                # highly effective in noise removal while keeping edges sharp. But the operation is slower compared
                # to other filters.
                cv.bilateralFilter(image, 9, 75, 75)

            case 'blur':
                cv.blur(image, (5, 5))

        return image

    def live_camera(self) -> None:
        camera = cv.VideoCapture(self._camera_id)

        if camera.isOpened():
            while cv.waitKey(1) == -1:
                ret, frame = camera.read()
                if ret:
                    # apply here your image processing

                    cv.imshow(f'Camera {camera.getBackendName()} image', frame)


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

            cv.imshow('Final image', frame)


if __name__ == "__main__":
    print('Code has been started')

    test_function()
