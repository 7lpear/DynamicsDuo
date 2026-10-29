"""Shared calibration loading and image rectification for live/offline detection."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_calibration(path=None):
    path = Path(path) if path else ROOT / "calibration/board.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(config["homography"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Calibration homography must be a finite, invertible 3x3 matrix")
    for key in ("image_size", "warp_size"):
        if len(config[key]) != 2 or any(not isinstance(v, int) or v <= 0 for v in config[key]):
            raise ValueError(f"Calibration {key} must contain two positive integers")
    return config


def rectify_frame(frame, config):
    actual_size = [frame.shape[1], frame.shape[0]]
    if actual_size != config["image_size"]:
        raise ValueError(f"Image size {actual_size} differs from calibration {config['image_size']}; recalibrate")
    return cv2.warpPerspective(frame, np.asarray(config["homography"]), tuple(config["warp_size"]))
