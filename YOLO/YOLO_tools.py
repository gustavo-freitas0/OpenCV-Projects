import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO


class YoloUtralytics:
    _model = None

    _cam = None
    _cam_id = None

    _yolo_models = {
        'v11n': 'yolo11n.pt',
        'v11s': 'yolo11s.pt',
        'v11m': 'yolo11m.pt',
        'v11l': 'yolo11l.pt',
        'v11x': 'yolo11x.pt',
        'v8n': 'yolov8n.pt'
    }

    _coco_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                     "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                     "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                     "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                     "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                     "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                     "teddy bear", "hair drier", "toothbrush"
                     ]

    _patience = 3

    def __init__(self, model: str = 'v11n', camera_id: int = 0) -> None:

        self._model = YOLO(self._yolo_models[model])

        self._cam_id = camera_id

        while not self.start_video_stream():
            if self._patience > 2:
                print(f"Can't open camera {self._cam_id} - {self._cam.getBackendName()}")
                exit()
            self._patience += 1
        print('YOLO and Camera have been started')

    @staticmethod
    def get_colours(cls_num: int) -> tuple:
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [base_colors[color_index][i] + increments[color_index][i] *
                 (cls_num // len(base_colors)) % 256 for i in range(3)]
        return tuple(color)

    def start_video_stream(self) -> bool:
        self._cam = cv.VideoCapture(self._cam_id, cv.CAP_DSHOW)
        return self._cam.isOpened()

    def __del__(self):
        self._cam.release()
        cv.destroyAllWindows()

    def object_detection(self, _image: np.ndarray, stream: bool = True) -> list:
        """

        :param _image:
        :param stream:
        :return:
        """

        obj_list = []

        results_obj_ = self._model(_image, stream=stream)

        for result in results_obj_:
            for box in result.boxes:
                obj_list.append((self._coco_classes[int(box.cls[0])], box.conf[0]))

        return obj_list

    def track(self, stream: bool = True, verbose: bool = False) -> None:
        while stream:
            ret, frame = self._cam.read()
            if not ret:
                continue
            results = self._model.track(frame, stream=stream)

            for result in results:
                # get the classes names
                classes_names = result.names

                if verbose: print(result.names)

                # iterate over each box
                for box in result.boxes:
                    # check if confidence is greater than 40 percent
                    if box.conf[0] > 0.4:
                        # get coordinates
                        [x1, y1, x2, y2] = box.xyxy[0]
                        # convert to int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # get the class
                        cls = int(box.cls[0])

                        # get the class name
                        class_name = classes_names[cls]

                        # get the respective colour
                        colour = self.get_colours(cls)

                        # draw the rectangle
                        cv.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                        # put the class name and confidence on the image
                        cv.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

            # show the image
            cv.imshow('frame', frame)

            # break the loop if press any button
            if cv.waitKey(1) != -1:
                break

        cv.destroyAllWindows()


if __name__ == "__main__":
    print('Code has been started')

    # yolo = YoloUtralytics()
    #
    # yolo.track(verbose=True)

    # Just learning

    model = YOLO('yolo11n.pt')

    # print(model.info())

    cam = cv.VideoCapture(0)

    while cv.waitKey() == -1:
        _, frame = cam.read()
        results = model(frame, stream=True)
        for res in results:
            print(res.boxes)
