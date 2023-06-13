import collections
import platform
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from src.capture.list_capture_devices import auto_select_device
from src.utils.common_logger import get_logger

log = get_logger()


class VideoCapture(threading.Thread):
    def __init__(self, is_stopped: threading.Event, capture_buf: collections.deque, capture_spec,
                 path_mimic_video: Optional[Path]):
        super().__init__()

        log.info("Loading video capture...")
        self.is_stopped = is_stopped
        self.capture_buf = capture_buf
        self.capture_spec = capture_spec
        self.path_mimic_video = path_mimic_video

    def run(self):
        # fps = 30
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width, height, fps = self.capture_spec

        mimic_webcam = self.path_mimic_video is not None

        last_update = time.time()
        while not self.is_stopped.is_set():

            if not mimic_webcam:
                #tmp
                if platform.system() == "Linux":
                    dev = 0
                else:
                    selected_device = auto_select_device()
                    if selected_device is None:
                        time.sleep(1)
                        continue
                    log.info(f"Starting capture of {selected_device['model']}...")
                    dev = selected_device["streams"][0]["opencv_capture_idx"]
            else:
                dev = str(self.path_mimic_video.resolve())

            try:
                cap = cv2.VideoCapture(dev)
                while not self.is_stopped.is_set():
                    now = time.time()
                    if now - last_update >= 1 / fps:
                        last_update = now

                        ret, frame = cap.read()
                        if not ret:
                            if mimic_webcam:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            break

                        frame = cv2.resize(frame, (width, height))
                        self.capture_buf.append((now, frame))

                cap.release()
            except Exception as ex:
                log.info(ex)
                log.info(f"Stopping capture of f{selected_device['model']}...")
