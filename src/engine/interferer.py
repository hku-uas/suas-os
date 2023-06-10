import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

from src.definitions import root_dir
from src.utils.common_logger import get_logger

log = get_logger()


class Interferer:
    def __init__(self):
        super().__init__()

        log.info("Loading YOLO detection model [locate]...")
        self.model = YOLO(root_dir / 'best.pt')

    def interfere(self, frame: np.array):
        results = self.model(frame, verbose=False)

        annotated_frame = frame

        # cv2.line(annotated_frame, (0, int(frameH / 2)), (int(frameW), int(frameH / 2)), (255, 255, 255), 1)
        # cv2.line(annotated_frame, (int(frameW / 2), 0), (int(frameW / 2), int(frameH)), (255, 255, 255), 1)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                x, y, w, h = box.xywh[0].tolist()
                xn, yn, wn, hn = box.xywhn[0].tolist()
                class_id = box.cls
                class_name = self.model.names[int(class_id)]

                sensitivity = 10
                droneX, droneY = 100, 100
                interX, interY = droneX + (xn - .5) * sensitivity, droneY + (yn - .5) * sensitivity

                annotator = Annotator(frame)
                annotator.box_label(b, f"{interX:.2f}, {interY:.2f}")
                annotated_frame = annotator.result()

                # cv2.line(annotated_frame, (int(frameW / 2), int(frameH / 2)), (int(x), int(y)), (255, 255, 255), 1)

        return annotated_frame
