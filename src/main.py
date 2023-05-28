import collections
import threading
import time

import cv2
import numpy as np

from src.capture.video_capture import video_capture_thread
from src.capture.video_detect import VideoDetector
from src.utils.common_logger import init_logger, get_logger

if __name__ == '__main__':
    init_logger()
    log = get_logger()
    log.info("Initializing...")

    capture_buf = collections.deque(maxlen=10000)

    capture_spec = [1280, 720, 30]
    log.info(f"Forcing frame size to be {capture_spec} (width, height, fps)")

    is_stopped = threading.Event()

    t_capture = threading.Thread(target=video_capture_thread, args=(is_stopped, capture_buf, capture_spec))
    # t_save = threading.Thread(target=video_save_thread, args=(is_stopped, capture_buf, capture_spec))
    t_detect = VideoDetector(is_stopped, capture_buf, capture_spec)

    t_capture.start()
    # t_save.start()
    t_detect.start()

    font = cv2.FONT_HERSHEY_DUPLEX
    try:
        while not is_stopped.is_set():
            time.sleep(.01)
            if len(capture_buf) > 0:
                ts, frame = capture_buf[-1]
                render_frame = np.hstack((frame, t_detect.processing_frame))
                h, w = render_frame.shape[:2]
                cv2.putText(render_frame, f'Realtime', (10, h - 10), font, .8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(render_frame, f'Processed [{t_detect.fps:.2f}fps]', (int(w / 2) + 10, h - 10), font, .8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("frame", render_frame)
            if cv2.waitKey(1) == 27:
                is_stopped.set()
    except KeyboardInterrupt:
        pass

    log.info("Exiting...")
    cv2.destroyAllWindows()

    is_stopped.set()
    t_capture.join()
    # t_save.join()
