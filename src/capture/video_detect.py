import collections
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

from src.definitions import root_dir
from src.utils.common_logger import get_logger

log = get_logger()


class VideoDetector(threading.Thread):
    def __init__(self, is_stopped: threading.Event, capture_buf: collections.deque, capture_spec):
        super().__init__()

        log.info("Loading video detect component...")
        self.is_stopped = is_stopped
        self.capture_buf = capture_buf
        self.capture_spec = capture_spec

        self.processing_time = []
        self.fps = 0

        w, h, fps = self.capture_spec
        self.processing_frame = np.zeros((h, w, 3), np.uint8)

        log.info("Loading YOLO detection model [stub]...")
        self.model = YOLO('yolov8n.pt')

    def run(self):
        frameW, frameH, fps = self.capture_spec

        while not self.is_stopped.is_set():
            if len(self.capture_buf) <= 0:
                continue

            start = time.time()

            ts, frame = self.capture_buf[-1]
            results = self.model(frame, verbose=False)

            annotated_frame = frame

            cv2.line(annotated_frame, (0, int(frameH / 2)), (int(frameW), int(frameH / 2)), (255, 255, 255), 1)
            cv2.line(annotated_frame, (int(frameW / 2), 0), (int(frameW / 2), int(frameH)), (255, 255, 255), 1)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    x, y, w, h = box.xywh[0].tolist()
                    xn, yn, wn, hn = box.xywhn[0].tolist()
                    class_id = box.cls
                    class_name = self.model.names[int(class_id)]

                    if class_name != "cell phone":
                        continue

                    sensitivity = 10
                    droneX, droneY = 100, 100
                    interX, interY = droneX + (xn - .5) * sensitivity, droneY + (yn - .5) * sensitivity

                    annotator = Annotator(frame)
                    annotator.box_label(b, f"{interX:.2f}, {interY:.2f}")
                    annotated_frame = annotator.result()

                    cv2.line(annotated_frame, (int(frameW / 2), int(frameH / 2)), (int(x), int(y)), (255, 255, 255), 1)

            self.processing_frame = annotated_frame

            end = time.time()
            processing_time = end - start
            self.processing_time.append(processing_time)
            if len(self.processing_time) > 20:
                self.processing_time.pop(0)
            self.fps = 1 / np.mean(self.processing_time)
