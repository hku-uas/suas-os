import collections
import threading
import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

from src.definitions import root_dir
from src.utils.common_logger import get_logger

log = get_logger()


class VideoInference(threading.Thread):
    def __init__(self, is_stopped: threading.Event, capture_buf: collections.deque, capture_spec):
        super().__init__()

        log.info("Loading video detect component...")
        self.is_stopped = is_stopped
        self.capture_buf = capture_buf
        self.capture_spec = capture_spec

        self.processing_time = []
        self.fps = 0

        w, h, fps = self.capture_spec
        self.annotated_frame = np.zeros((h, w, 3), np.uint8)

        log.info("Loading YOLO detection models...")

        self.model_locate = YOLO(list(sorted(root_dir.rglob("dataset_locate.pt")))[-1])
        self.model_identify_letters = YOLO(list(sorted(root_dir.rglob("dataset_identify_letters.pt")))[-1])

        self.testtt = None

    def run(self):
        frameW, frameH, fps = self.capture_spec

        while not self.is_stopped.is_set():
            if len(self.capture_buf) <= 0:
                continue

            start = time.time()

            ts, frame = self.capture_buf[-1]
            results = self.model_locate(frame, verbose=False)

            annotated_frame = frame.copy()

            cv2.line(annotated_frame, (0, int(frameH / 2)), (int(frameW), int(frameH / 2)), (255, 255, 255), 1)
            cv2.line(annotated_frame, (int(frameW / 2), 0), (int(frameW / 2), int(frameH)), (255, 255, 255), 1)

            annotator = Annotator(annotated_frame)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    # t, l, b, r = b.tolist()
                    x, y, w, h = box.xywh[0].tolist()
                    xn, yn, wn, hn = box.xywhn[0].tolist()
                    class_id = box.cls
                    class_name = self.model_locate.names[int(class_id)]

                    sensitivity = 10
                    droneX, droneY = 100, 100
                    interX, interY = droneX + (xn - .5) * sensitivity, droneY + (yn - .5) * sensitivity

                    frame_cropped = frame[int(y - (h / 2)):int(y + (h / 2)), int(x - (w / 2)):int(x + (w / 2))]
                    # frame_cropped = np.mean(frame_cropped, axis=2, keepdims=True).astype(np.uint8)
                    # frame_cropped = np.concatenate([frame_cropped] * 3, axis=2)
                    self.testtt = frame_cropped

                    predicted_letter = self.inference_letter(frame_cropped)
                    # predicted_letter = "?"
                    label = f"C/{predicted_letter if predicted_letter else '?'} {interX:.2f}, {interY:.2f}"
                    annotator.box_label(b, label)

            annotated_frame = annotator.result()
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x, y, w, h = box.xywh[0].tolist()
                    cv2.line(annotated_frame, (int(frameW / 2), int(frameH / 2)), (int(x), int(y)), (255, 255, 255), 1)

            self.annotated_frame = annotated_frame

            end = time.time()
            processing_time = end - start
            self.processing_time.append(processing_time)
            if len(self.processing_time) > 20:
                self.processing_time.pop(0)
            self.fps = 1 / np.mean(self.processing_time)

    def inference_letter(self, cropped_frame) -> Optional[str]:
        results = self.model_identify_letters(cropped_frame, verbose=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = box.cls
                class_name = self.model_identify_letters.names[int(class_id)]
                return class_name
        return None
