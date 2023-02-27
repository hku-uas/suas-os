import collections
import threading
import time
from typing import Optional

import cv2

from src.capture.list_capture_devices import auto_select_device, list_capture_streams
from src.utils.common_logger import get_logger

log = get_logger()


def video_capture_thread(is_stopped: threading.Event, capture_buf: collections.deque, capture_spec):
    # fps = 30
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height, fps = capture_spec

    last_update = time.time()
    while not is_stopped.is_set():
        selected_device = auto_select_device()
        if selected_device is None:
            time.sleep(1)
            continue
        log.info(f"Starting capture of {selected_device['model']}...")
        device_idx = selected_device["streams"][0]["opencv_capture_idx"]
        try:
            cap = cv2.VideoCapture(device_idx)
            while not is_stopped.is_set():
                now = time.time()
                if now - last_update >= 1 / fps:
                    last_update = now
                    ret, frame = cap.read()
                    capture_buf.append((now, frame))

            cap.release()
        except Exception as ex:
            print(ex)
            log.info(f"Stopping capture of f{selected_device['model']}...")
