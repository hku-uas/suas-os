import collections
import threading
import time

import cv2

from src.capture.video_capture import video_capture_thread
from src.capture.video_save import video_save_thread
from src.utils.common_logger import init_logger, get_logger

if __name__ == '__main__':
    init_logger()
    log = get_logger()
    log.info("Initializing...")

    capture_buf = collections.deque(maxlen=10000)

    capture_spec = [1280, 720, 30]
    log.info(f"Capture spec: {capture_spec} (width, height, fps)")

    is_stopped = threading.Event()
    t_capture = threading.Thread(target=video_capture_thread, args=(is_stopped, capture_buf, capture_spec))
    t_save = threading.Thread(target=video_save_thread, args=(is_stopped, capture_buf, capture_spec))
    t_capture.start()
    t_save.start()

    try:
        while not is_stopped.is_set():
            time.sleep(.1)
            # if len(capture_buf) > 0:
            #     ts, frame = capture_buf[-1]
            #     cv2.imshow("frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     is_stopped.set()
    except KeyboardInterrupt:
        pass

    log.info("Exiting...")
    cv2.destroyAllWindows()

    is_stopped.set()
    t_capture.join()
    t_save.join()
