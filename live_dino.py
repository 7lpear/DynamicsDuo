"""Continuous camera preview with asynchronous DINO boxes overlaid on video."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
from detection_utils import ROOT
from board_calibration import load_calibration, rectify_frame


def worker(jobs, results, ready, stop, args):
    try:
        from PIL import Image
        from dino_detector import DinoDetector
        detector = DinoDetector(args.device, args.prompt, args.threshold, args.text_threshold)
        ready.set()
        print("Model ready. Automatic analysis started.", flush=True)
        while not stop.is_set():
            try:
                job = jobs.get(timeout=.1)
            except queue.Empty:
                continue
            if stop.is_set():
                break
            frame, frame_id, captured, captured_at = job
            items, elapsed = detector.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            results.put(dict(frame=frame, frame_id=frame_id, captured=captured,
                             captured_at=captured_at, detections=items, seconds=elapsed,
                             counts=detector.last_counts.copy()))
            ready.set()
    except Exception as exc:
        results.put({"error": f"{type(exc).__name__}: {exc}"})


def draw_detections(frame, detections):
    canvas = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = [round(v) for v in item["box"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(canvas, f'{item["score"]:.2f}', (max(0, x1), max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    return canvas


def draw_live_overlay(frame, latest, inverse, status):
    """Map rectified detections back onto the current original camera frame."""
    canvas = frame.copy()
    if latest:
        for item in latest["detections"]:
            x1, y1, x2, y2 = item["box"]
            corners = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
            polygon = np.round(cv2.perspectiveTransform(corners, inverse)[0]).astype(np.int32)
            cv2.polylines(canvas, [polygon], True, (0, 255, 0), 3)
            x, y = polygon[0]
            cv2.putText(canvas, f'{item["score"]:.2f}', (max(0, int(x)), max(20, int(y)-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    scale = min(1200 / canvas.shape[1], 720 / canvas.shape[0], 1)
    canvas = cv2.resize(canvas, (round(canvas.shape[1]*scale), round(canvas.shape[0]*scale)))
    display = np.zeros((canvas.shape[0]+80, canvas.shape[1], 3), dtype=np.uint8)
    display[80:] = canvas
    age = max(0, time.monotonic()-latest["captured"]) if latest else 0
    text = f'LIVE | {status}'
    if latest:
        text += f' | objects {len(latest["detections"])} | detection age {age:.1f}s | inference {latest["seconds"]:.1f}s'
    else:
        text += ' | waiting for first detection'
    cv2.putText(display, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
    cv2.putText(display, 'Boxes update after each analysis; video remains live. P pause | S save | Q quit',
                (10, 57), cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230), 1)
    return display


def save_snapshot(current, latest, args, display=None):
    folder = ROOT / "live_dino_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=True)
    images = {"live_board.png": current}
    if display is not None:
        images["live_overlay.png"] = display
    data = {"prompt": args.prompt, "threshold": args.threshold, "text_threshold": args.text_threshold,
            "coordinate_space": "rectified image", "status": "no analysis yet"}
    if latest:
        images["analyzed_input.png"] = latest["frame"]
        images["detections.png"] = draw_detections(latest["frame"], latest["detections"])
        data.update(status="candidate detections, not confirmed board", frame_id=latest["frame_id"],
                    captured_at=latest["captured_at"], result_age_seconds=time.monotonic()-latest["captured"],
                    processing_seconds=latest["seconds"], detections=latest["detections"])
        data["counts"] = latest.get("counts", {})
    for name, img in images.items():
        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError(f"Cannot encode {name}")
        encoded.tofile(str(folder / name))
    (folder / "detections.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved: {folder}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prompt", default="bottle cap.")
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--text-threshold", type=float, default=.15)
    parser.add_argument("--replay", type=Path, help="Test using a still image without opening USB camera")
    parser.add_argument("--duration", type=float, default=0, help="Stop after N seconds; 0 means unlimited")
    parser.add_argument("--no-window", action="store_true", help="Headless replay verification")
    parser.add_argument("--stop-after-results", type=int, default=0, help="Stop after N results, saving the last")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.text_threshold <= 1 or not args.prompt.strip():
        parser.error("Thresholds must be 0..1 and prompt must not be empty")
    if args.duration < 0 or args.stop_after_results < 0:
        parser.error("Duration and result count must be nonnegative")
    if args.no_window and not (args.duration or args.stop_after_results):
        parser.error("Headless mode requires --duration or --stop-after-results")
    config = load_calibration()
    if config["grid_spacing"] != 100:
        parser.error("Stone size filter requires calibration with 100px grid spacing")
    matrix = np.array(config["homography"])
    inverse = np.linalg.inv(matrix)
    camera = None
    source = None
    if args.replay:
        source = cv2.imdecode(np.fromfile(str(args.replay), dtype=np.uint8), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"Cannot read {args.replay}")
    else:
        camera = cv2.VideoCapture(args.index, cv2.CAP_DSHOW if args.backend == "dshow" else cv2.CAP_MSMF)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError("Cannot open camera. Close camera_test.py and other camera apps.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config["image_size"][0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config["image_size"][1])
    jobs, results = queue.Queue(maxsize=1), queue.Queue()
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(target=worker, args=(jobs, results, ready, stop, args))
    window = "Live DINO detection | P pause | S save | Q quit"
    started, frame_id, count = time.monotonic(), 0, 0
    latest, busy, paused = None, False, False
    thread.start()
    print("Loading DINO in background. Keep camera/board fixed.", flush=True)
    print("Live camera video with detection boxes. Boxes refresh after each analysis; age shows delay.", flush=True)
    print("No fixed object-count limit. All detections passing confidence/size/duplicate filters are shown.", flush=True)
    print("P: pause/resume analysis; S: save; Q/Esc: quit. Candidates only; no board confirmation.", flush=True)
    try:
        while True:
            captured = time.monotonic()
            if args.duration and captured-started >= args.duration:
                break
            if camera:
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera frame read failed")
            else:
                frame = source
            board = rectify_frame(frame, config)
            frame_id += 1
            try:
                result = results.get_nowait()
            except queue.Empty:
                result = None
            if result:
                busy = False
                if "error" in result:
                    raise RuntimeError(result["error"])
                latest = result
                count += 1
                print(f"Result {count}: {len(result['detections'])} candidates; {result['seconds']:.2f}s; frame {result['frame_id']}", flush=True)
                print(f"  Detection stages: {result.get('counts', {})}", flush=True)
                if args.stop_after_results and count >= args.stop_after_results:
                    save_snapshot(board, latest, args, draw_live_overlay(frame, latest, inverse, "ANALYZING"))
                    break
            if ready.is_set() and not busy and not paused:
                ready.clear()
                busy = True
                jobs.put_nowait((board.copy(), frame_id, captured, datetime.now().isoformat()))
            if not args.no_window:
                status = "PAUSED" if paused else "ANALYZING" if busy else "LOADING MODEL"
                canvas = draw_live_overlay(frame, latest, inverse, status)
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("p"), ord("P")):
                    paused = not paused
                if key in (ord("s"), ord("S")):
                    save_snapshot(board, latest, args, canvas)
            elif args.replay:
                time.sleep(.02)
    finally:
        stop.set()
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        if thread.is_alive():
            print("Closing; waiting for current model operation to finish...", flush=True)
        thread.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as exc:
        print(f"STOP: {exc}")
        raise SystemExit(1)
