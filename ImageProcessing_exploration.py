import cv2 as cv
import numpy as np


class ImageProcessing:

    def __init__(self):
        print('class Image Processing')

        self.live_camera()

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
        camera = cv.VideoCapture(0)

        if camera.isOpened():
            while cv.waitKey(1) == -1:
                ret, frame = camera.read()
                if ret:
                    # apply here your image processing

                    cv.imshow(f'Camera {camera.getBackendName()} image', frame)


if __name__ == "__main__":
    print('Code has been started')

    blaiblaidi = ImageProcessing()
