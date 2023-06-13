import collections
import threading
import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

from src.capture.found_entry import FoundEntry
from src.definitions import root_dir
from src.utils.common_logger import get_logger

log = get_logger()


class VideoInference(threading.Thread):
    def __init__(
            self,
            is_stopped: threading.Event,
            capture_buf: collections.deque,
            capture_spec,
            entry_buf: collections.deque
    ):
        super().__init__()

        log.info("Loading video detect component...")
        self.is_stopped = is_stopped
        self.capture_buf = capture_buf
        self.capture_spec = capture_spec

        self.entry_buf = entry_buf
        self.entry_id = 0

        self.processing_time = []
        self.fps = 0

        w, h, fps = self.capture_spec
        self.annotated_frame = np.zeros((h, w, 3), np.uint8)

        log.info("Loading YOLO detection models...")

        def choose_latest(path, exp):
            versions = path.rglob(exp)
            versions = list(sorted(versions, key=lambda o: o.parent.stem))
            chosen = versions[-1]
            log.info(f"Choosing {exp} version {chosen.parent.stem}")
            return chosen

        self.model_locate = YOLO(choose_latest(root_dir, "dataset_locate.pt"))
        self.model_identify_letters = YOLO(choose_latest(root_dir, "dataset_identify_letters.pt"))
        self.model_identify_shapes = YOLO(choose_latest(root_dir, "dataset_identify_shapes.pt"))

    def run(self):
        frameW, frameH, fps = self.capture_spec

        while not self.is_stopped.is_set():
            time.sleep(.1)
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
                    x, y, w, h = box.xywh[0].tolist()
                    confidence = box.conf.tolist()[0]
                    if confidence < 0.5:
                        continue

                    xn, yn, wn, hn = box.xywhn[0].tolist()
                    sensitivity = 1
                    interX, interY = (xn - .5) * sensitivity, (yn - .5) * sensitivity

                    frame_cropped = frame[int(y - (h / 2)):int(y + (h / 2)), int(x - (w / 2)):int(x + (w / 2))]
                    # frame_cropped = np.mean(frame_cropped, axis=2, keepdims=True).astype(np.uint8)
                    # frame_cropped = np.concatenate([frame_cropped] * 3, axis=2)

                    time.sleep(0.2)
                    predicted_letter = self.inference_cropped(self.model_identify_letters, frame_cropped)
                    time.sleep(0.2)
                    predicted_shape = self.inference_cropped(self.model_identify_shapes, frame_cropped)
                    
                    # predicted_letter = None
                    # predicted_shape = None

                    label = f"C-" \
                            f"{predicted_letter if predicted_letter else '?'}/" \
                            f"{predicted_shape if predicted_shape else '?'}/" \
                            f"?/" \
                            f"? " \
                            f"[{confidence:.2f}] " \
                            f"{interX:.2f}, {interY:.2f}"
                    annotator.box_label(b, label)
                    self.entry_buf.append(
                        (
                            self.entry_id, time.time(),
                            "C", predicted_letter, predicted_shape, None, None,
                            confidence, interX, interY, frame_cropped
                        )
                    )
                    self.entry_id += 1

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

    def inference_cropped(self, model, cropped_frame) -> Optional[str]:
        results = model(cropped_frame, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = box.cls
                class_name = model.names[int(class_id)]
                return class_name
        return None
