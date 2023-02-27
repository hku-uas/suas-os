import json
import platform
import re
import subprocess
from typing import List, Optional, Any

from src.utils.common_logger import get_logger
from src.utils.scorer import highest_score

log = get_logger()


def list_capture_streams() -> Optional[Any]:
    platform_name = platform.system()
    devices = []
    log.debug("Fetching capture devices...")
    if platform_name == "Darwin":
        import objc
        import AVFoundation
        for device in AVFoundation.AVCaptureDevice.devicesWithMediaType_("vide"):
            devices.append({
                "uniqueID": device.uniqueID(),
                "vendorID": None,
                "vendor": device.manufacturer(),
                "modelID": device.modelID(),
                "model": device.localizedName(),
                "streams": [
                    {
                        "opencv_capture_idx": len(devices),
                        "path": None,
                    }
                ]
            })
    elif platform_name == "Windows":
        return None
    elif platform_name == "Linux":
        try:
            output = subprocess.check_output(["v4l2-ctl", "--list-devices"], stderr=subprocess.DEVNULL).decode('utf-8')
            output = [o.strip() for o in output.split('\n')]
        except subprocess.SubprocessError:
            output = []

        for line in output:
            grps = re.findall(r"(.+) \((.*)\):", line)
            if grps is not None:
                devices.append({
                    "uniqueID": None,
                    "vendorID": None,
                    "vendor": None,
                    "modelID": None,
                    "model": grps[0][0],
                    "streams": []
                })
                continue

            grps = re.findall(r"(/dev/video(\d+))", line)
            if grps is not None:
                devices[-1]["streams"].append({
                    "path": grps[0][0],
                    "opencv_capture_idx": int(grps[0][1]),
                })
                continue
    log.debug(devices)
    return devices


def auto_select_device() -> Optional[Any]:
    devices = list_capture_streams()
    if devices is None:
        return 0
    if len(devices) == 0:
        return None

    def scorer_func(o):
        if platform.system() == "Linux":
            if "RealSense" in o["model"]:
                return 2
        elif platform.system() == "Darwin":
            if "FaceTime HD Camera" in o["model"]:
                return 1
        return 0

    highest_score_device = highest_score(devices, scorer_func)
    return highest_score_device

# print(json.dumps(list_capture_streams(), indent=4))
