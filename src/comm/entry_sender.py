import base64
import collections
import time
import threading

import cv2
from flask_socketio import SocketIO
from src.utils.common_logger import get_logger

log = get_logger()


class EntrySender(threading.Thread):
    def __init__(self, is_stopped: threading.Event, entry_buf: collections.deque, sio: SocketIO):
        super().__init__()
        self.is_stopped = is_stopped
        self.entry_buf = entry_buf
        self.sio = sio

    def run(self):
        while not self.is_stopped.is_set():
            time.sleep(0.1)
            if len(self.entry_buf) <= 0:
                time.sleep(0.1)
                continue

            entry = self.entry_buf.popleft()
            entry_id, t, obj_type, letter, shape, bg_c, fg_c, confidence, interX, interY, frame_cropped = entry
            log.info(f"Found object: {obj_type}-{letter}/{shape}/{bg_c}/{fg_c} [{confidence:.2f}] {interX}, {interY}")
            ret, img_buf = cv2.imencode('.jpg', frame_cropped, [cv2.IMWRITE_JPEG_QUALITY, 30])
            img_base64 = base64.b64encode(img_buf)
            self.sio.emit("entry", [entry_id, t, obj_type, letter, shape, bg_c, fg_c, confidence, interX, interY,
                                    img_base64.decode('utf-8')])
