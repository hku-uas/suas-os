import collections
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


def video_save_thread(stopped: threading.Event, capture_buf: collections.deque, capture_spec):
    output_dir = Path(".") / "video_output"
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    width, height, fps, device_idx = capture_spec

    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    frame_id = 0
    video_writer: Optional[cv2.VideoWriter] = None
    while not stopped.is_set():
        if video_writer is None or frame_id >= fps * 10:
            frame_id = 0
            if video_writer is not None:
                video_writer.release()
            video_file = output_dir / (datetime.now().strftime('%Y-%m-%dT%H:%M:%S') + '.mp4')
            video_writer = cv2.VideoWriter(str(video_file.resolve()), video_codec, fps, (width, height))

        try:
            ts, frame = capture_buf.popleft()
            video_writer.write(frame)
            frame_id += 1
        except IndexError:
            pass


def video_capture_thread(stopped: threading.Event, capture_buf: collections.deque, capture_spec):
    # fps = 30
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height, fps, device_idx = capture_spec

    cap = cv2.VideoCapture(device_idx)

    last_update = time.time()
    try:
        while not stopped.is_set():
            now = time.time()
            if now - last_update >= 1 / fps:
                now = time.perf_counter()
                ret, frame = cap.read()
                capture_buf.append((now, frame))
                # cv2.imshow("frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     halt = True
                #     break
    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_buf = collections.deque(maxlen=1000)

    capture_spec = [1280, 720, 30, 2]

    is_stopped = threading.Event()
    t_capture = threading.Thread(target=video_capture_thread, args=(is_stopped, capture_buf, capture_spec))
    t_save = threading.Thread(target=video_save_thread, args=(is_stopped, capture_buf, capture_spec))
    t_capture.start()
    t_save.start()

    try:
        while True:
            time.sleep(.1)
    except KeyboardInterrupt:
        pass

    print("Exiting...")

    is_stopped.set()
    t_capture.join()
    t_save.join()
