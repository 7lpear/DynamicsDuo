"""Platform-aware camera selection and Unicode-safe image files."""
import sys
from pathlib import Path

import cv2
import numpy as np

BACKENDS = ("auto", "dshow", "msmf", "avfoundation", "v4l2")


def require_gui():
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:") and line.split(":", 1)[1].strip().upper() == "NONE":
            raise RuntimeError(
                f"OpenCV has no GUI support in this Python: {sys.executable}\n"
                "Activate the so-arm-101 Conda environment (conda activate so-arm-101).\n"
                "This camera preview requires opencv-python, not opencv-python-headless.")


def resolve_backend(name="auto", platform=None):
    platform = sys.platform if platform is None else platform
    defaults = {"win32": "dshow", "darwin": "avfoundation", "linux": "v4l2"}
    if name == "auto":
        name = defaults.get(platform, "auto")
    supported = {"dshow": "win32", "msmf": "win32", "avfoundation": "darwin", "v4l2": "linux"}
    if name not in BACKENDS:
        raise ValueError(f"Unknown camera backend: {name}")
    if name in supported and supported[name] != platform:
        raise ValueError(f"--backend {name} is not supported on {platform}; use --backend auto")
    api = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
           "avfoundation": cv2.CAP_AVFOUNDATION, "v4l2": cv2.CAP_V4L2}[name]
    return name, api


def open_camera(index, backend, width=1920, height=1080):
    name, api = resolve_backend(backend)
    print(f"Opening camera {index} with {name} backend...", flush=True)
    cap = cv2.VideoCapture(index, api)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index} ({name}). Close other camera apps and check --camera. "
                               "On macOS allow camera access for Terminal/VS Code in System Settings.")
        print(f"Camera opened. Requesting {width}x{height}...", flush=True)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print("Waiting for camera frames...", flush=True)
        return cap
    except BaseException:
        cap.release()
        raise


def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def write_image(path, image):
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(str(path))
    return True
