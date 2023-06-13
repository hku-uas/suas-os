import argparse
import collections
import threading
import time

import cv2
import numpy as np
from flask import Flask, send_from_directory
from flask_socketio import SocketIO

from src.capture.video_capture import VideoCapture
from src.capture.video_inference import VideoInference
from src.comm.entry_sender import EntrySender
from src.utils.common_logger import init_logger, get_logger

if __name__ == '__main__':
    t1 = time.time()

    init_logger()
    log = get_logger()
    log.info("Initializing...")

    parser = argparse.ArgumentParser(
        description='Main HKU UAS Onboard System')
    parser.add_argument('--force-device', "-d", action="store", type=int,
                        help='Force OpenCV capture device')
    args = parser.parse_args()

    capture_buf = collections.deque(maxlen=10000)
    entry_buf = collections.deque(maxlen=1000)

    capture_spec = [1280, 720, 30]
    width, height, fps = capture_spec
    log.info(f"Forcing frame size to be {capture_spec} (width, height, fps)")

    is_stopped = threading.Event()

    t_capture = VideoCapture(
        is_stopped, capture_buf, capture_spec,
        # root_dir / ".." / "1.mp4",
        None,
        args.force_device
    )
    t_capture.start()

    t_inference = VideoInference(is_stopped, capture_buf, capture_spec, entry_buf)
    t_inference.start()

    # t_save = threading.Thread(target=video_save_thread, args=(is_stopped, capture_buf, capture_spec))
    # t_save.start()

    # logging.getLogger('werkzeug').setLevel(logging.INFO)
    app = Flask(__name__)
    sio = SocketIO(app, cors_allowed_origins="*")


    @app.route("/", defaults={'file': 'index.html'})
    @app.route("/<path:file>")
    def serve_static(file):
        return send_from_directory("../public", file)


    t_flask = threading.Thread(target=lambda: sio.run(app, host="0.0.0.0", port=8080, debug=True, use_reloader=False))
    t_flask.daemon = True
    t_flask.start()

    t_entry_handler = EntrySender(is_stopped, entry_buf, sio)
    t_entry_handler.start()

    t2 = time.time()
    log.info(f"Ready ({t2 - t1:.2f}s). Please press ESC to exit.")

    font = cv2.FONT_HERSHEY_DUPLEX
    wind_title = "HKU SUAS OBS Preview [ESC to exit]"
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

            # if t_inference.testtt is not None:
            #     cv2.imshow(f"Cropped {randint(1, 4)}", t_inference.testtt)

            if cv2.waitKey(1) == 27:
                is_stopped.set()
    except KeyboardInterrupt:
        pass

    log.info("Exiting...")
    cv2.destroyAllWindows()

    is_stopped.set()
    for t in [
        t_capture,
        t_inference,
        t_entry_handler
    ]:
        t.join()

    log.info("All exited. Have a good day :)")
