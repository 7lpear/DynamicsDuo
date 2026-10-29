"""Preview a Windows webcam and save original frames for board calibration."""
import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=1, help="Camera index (try 0 or 2 if needed)")
    parser.add_argument("--backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "captures")
    args = parser.parse_args()

    import cv2

    backend = cv2.CAP_DSHOW if args.backend == "dshow" else cv2.CAP_MSMF
    cap = cv2.VideoCapture(args.index, backend)
    window = "Board camera - S: save / Q: quit"
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera. Close other camera apps; try --index 0 or --index 2.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print(f"Camera index: {args.index}; backend: {args.backend}")
        print(f"Save folder: {args.output.resolve()}")
        print("Focus the preview window. S: save original frame; Q or Esc: quit.")
        reported = False
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Cannot read frame. Try --backend msmf or a different camera index.")
            height, width = frame.shape[:2]
            if not reported:
                print(f"Actual frame: {width} x {height}")
                reported = True
            scale = min(960 / width, 540 / height, 1)
            preview = cv2.resize(frame, (round(width * scale), round(height * scale)))
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                args.output.mkdir(parents=True, exist_ok=True)
                path = args.output / datetime.now().strftime("board_%Y%m%d_%H%M%S_%f.png")
                # imencode + tofile also supports Windows paths containing Korean text.
                encoded_ok, encoded = cv2.imencode(".png", frame)
                if not encoded_ok:
                    raise RuntimeError("PNG encoding failed.")
                encoded.tofile(str(path))
                print(f"Saved: {path} ({width} x {height})")
            elif key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
