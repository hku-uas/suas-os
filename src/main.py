import collections
import threading

import cv2
import numpy as np

from src.capture.video_capture import VideoCapture
from src.capture.video_inference import VideoInference
from src.definitions import root_dir
from src.utils.common_logger import init_logger, get_logger

if __name__ == '__main__':
    init_logger()
    log = get_logger()
    log.info("Initializing...")

    capture_buf = collections.deque(maxlen=10000)

    capture_spec = [1280, 720, 30]
    width, height, fps = capture_spec
    log.info(f"Forcing frame size to be {capture_spec} (width, height, fps)")

    is_stopped = threading.Event()

    t_capture = VideoCapture(is_stopped, capture_buf, capture_spec,
                             root_dir / ".." / "footage_sped_up_lossy.mp4",
                             # None
                             )
    t_inference = VideoInference(is_stopped, capture_buf, capture_spec)
    # t_save = threading.Thread(target=video_save_thread, args=(is_stopped, capture_buf, capture_spec))

    t_capture.start()
    t_inference.start()
    # t_save.start()

    font = cv2.FONT_HERSHEY_DUPLEX
    wind_title = "HKU SUAS 2023 Drone Onboard System ODCL Preview"

    try:
        while not is_stopped.is_set():
            if len(capture_buf) > 0:
                ts, realtime_frame = capture_buf[-1]
                render_frame = np.hstack((realtime_frame, t_inference.annotated_frame))
                wind_height, wind_width = render_frame.shape[:2]
                cv2.putText(render_frame, f'Realtime',
                            (10, wind_height - 10),
                            font, .8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(render_frame, f'Inferenced Annotated [{t_inference.fps:.2f}fps]',
                            (wind_width // 2 + 10, wind_height - 10),
                            font, .8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(wind_title, render_frame)

            if t_inference.testtt is not None:
                cv2.imshow("testtt", t_inference.testtt)

            if cv2.waitKey(1) == 27:
                is_stopped.set()
    except KeyboardInterrupt:
        pass

    log.info("Exiting...")
    cv2.destroyAllWindows()

    is_stopped.set()
    t_capture.join()
    # t_save.join()
