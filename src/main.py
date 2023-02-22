import logging
from pathlib import Path

import cv2
from flask import Flask, Response

from utils.common_logger import init_logger, get_logger

if __name__ == "__main__":
    init_logger()
    log = get_logger()

    app = Flask(__name__, static_url_path='', static_folder='../video_output')


    @app.route("/status", methods=["GET"])
    def get_status():
        return {
            "status": "ok",
        }


    app.run(host="0.0.0.0", port=8080)
