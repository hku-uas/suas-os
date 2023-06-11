import collections
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from src.definitions import root_dir
from src.utils.common_logger import get_logger

log = get_logger()


def get_folder_size(folder: Path):
    size = 0
    for file in folder.iterdir():
        if file.is_file():
            size += file.stat().st_size
        elif file.is_dir():
            size += get_folder_size(file)
    return size


def video_save_thread(is_stopped: threading.Event, capture_buf: collections.deque, capture_spec):
    output_dir = root_dir / "video_output"
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    max_dir_size = 16 * 1024 * 1024 * 1024  # 16GB

    width, height, fps = capture_spec
    log.info(capture_spec)

    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    frame_id = 0
    video_writer: Optional[cv2.VideoWriter] = None
    video_file_path = None
    while not is_stopped.is_set():
        cur_folder_size = get_folder_size(output_dir)
        if cur_folder_size > max_dir_size:
            log.warning(f"Recording directory is full ({cur_folder_size} bytes), stopping recording...")
            time.sleep(10)
            continue

        if video_writer is None or frame_id >= fps * 5:
            frame_id = 0
            if video_writer is not None:
                log.info(f"Saving {video_file_path}...")
                video_writer.release()
            video_file_path = output_dir / (datetime.now().strftime('%Y-%m-%dT%H:%M:%S') + '.mp4')
            video_writer = cv2.VideoWriter(str(video_file_path.resolve()), video_codec, fps, (width, height))

        try:
            ts, frame = capture_buf.popleft()
            # log.info(f"Writing frame #{frame_id}...")
            video_writer.write(frame)
            frame_id += 1
        except IndexError:
            pass
