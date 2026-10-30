"""Interactive calibration of a physical Gomoku board from a USB camera."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from board_detection import detect_board
from stone_tracking import StoneTracker

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "calibration.json"
SOURCE = "Camera - automatic board detection"
BOARD = "Board - corrected grid"
SIDE, MARGIN = 800, 48
SIZES = (7, 8, 9, 15, 19)


def destination():
    lo, hi = MARGIN, SIDE - MARGIN - 1
    return np.float32([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])


def valid_corners(points):
    contour = np.asarray(points, dtype=np.float32)
    if contour.shape != (4, 2) or not np.isfinite(contour).all():
        return False
    # In image coordinates TL -> TR -> BR -> BL has positive signed area.
    return (cv2.isContourConvex(contour)
            and cv2.contourArea(contour, oriented=True) > 1000)


def transform(points):
    return cv2.getPerspectiveTransform(np.float32(points), destination())


def grid_points(size):
    axis = np.linspace(MARGIN, SIDE - MARGIN - 1, size)
    return np.float32([[(x, y) for x in axis] for y in axis])


def annotate_grid(frame, grid):
    result = frame.copy()
    for row in grid:
        cv2.line(result, tuple(np.rint(row[0]).astype(int)),
                 tuple(np.rint(row[-1]).astype(int)), (70, 200, 70), 1, cv2.LINE_AA)
    for col in grid.transpose(1, 0, 2):
        cv2.line(result, tuple(np.rint(col[0]).astype(int)),
                 tuple(np.rint(col[-1]).astype(int)), (70, 200, 70), 1, cv2.LINE_AA)
    for row in grid:
        for point in row:
            cv2.circle(result, tuple(np.rint(point).astype(int)), 2, (0, 180, 255), -1)
    return result


def label(frame, message, y=26):
    cv2.putText(frame, message, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, message, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--size", type=int, choices=SIZES, default=None,
                        help="Intersection count per side: an 8x8-cell board uses --size 9")
    parser.add_argument("--reset", action="store_true", help="Ignore saved calibration")
    parser.add_argument("--capture-empty", action="store_true",
                        help="Capture empty-board baseline after grid detection; remove ALL stones first")
    parser.add_argument("--resume-baseline", action="store_true",
                        help="Reuse diagnostic baseline + calibration ONLY if board/camera have not moved")
    args = parser.parse_args()
    size = args.size or 9
    points = []
    saved = None
    resume_reference = None
    config_path = ROOT / "diagnostics/latest_calibration.json" if args.resume_baseline else CONFIG
    if args.resume_baseline and (args.reset or args.capture_empty):
        parser.error("--resume-baseline cannot be combined with --reset or --capture-empty")
    if args.resume_baseline and not config_path.exists():
        parser.error("No saved diagnostic calibration; capture an empty board first")
    if config_path.exists() and not args.reset:
        try:
            saved = json.loads(config_path.read_text())
            normalized = np.asarray(saved["corners"], dtype=np.float32)
            if (normalized.shape != (4, 2) or not np.isfinite(normalized).all()
                    or np.any(normalized < 0) or np.any(normalized > 1)
                    or saved["size"] not in SIZES):
                raise ValueError("Invalid calibration")
            if saved["camera"] != args.camera or (args.size is not None and saved["size"] != args.size):
                saved = None
            elif args.size is None:
                size = saved["size"]
        except (ValueError, KeyError, TypeError, OSError):
            print("Saved calibration is invalid; please calibrate again.")
            saved = None
    if args.resume_baseline:
        if saved is None:
            parser.error("Saved baseline calibration does not match camera or grid size")
        resume_reference = cv2.imread(str(ROOT / "diagnostics/latest_empty_reference.png"), cv2.IMREAD_GRAYSCALE)
        if resume_reference is None or resume_reference.shape != (400, 400):
            parser.error("Saved empty reference is missing or invalid")

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap.release()
        print("Cannot open camera. Try --camera 1 or --camera 2.\n"
              "On macOS, allow camera access for the launching app in System Settings.\n"
              "Close QuickTime or other apps using the camera.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    status = "Looking for an empty board. A: retry detection, M: manual corners"
    last_shape = None
    auto_pending = True
    frame_count = 0
    allowed_sizes = (args.size,) if args.size else (9, 15, 19)
    tracker = None
    tracker_geometry = None
    previous_event = None

    def click(event, x, y, flags, param):
        nonlocal status
        if (event == cv2.EVENT_LBUTTONDOWN and not auto_pending and len(points) < 4
                and last_shape is not None and 0 <= y < last_shape[0] and 0 <= x < last_shape[1]):
            points.append((x, y))
            if len(points) == 4:
                if valid_corners(points):
                    status = "Grid ready. Check alignment; S saves calibration + image."
                else:
                    points.clear()
                    status = "Invalid corners. Click in order: TL, TR, BR, BL."

    try:
        cv2.namedWindow(SOURCE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(SOURCE, click)
        print("Automatic detection: show an empty board, with the entire grid visible.")
        print("Sizes count intersections, not cells: 8x8 cells = 9x9 intersections.")
        print("Keys: A/R auto detect | M manual | S save | D diagnostics | U undo | N next camera | G grid size | Q quit")
        print("After grid alignment: remove ALL stones, press B, wait for empty-board capture.")
        print(f"Camera index: {args.camera}. Verify the preview shows the Logitech camera.")
        failures = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 30:
                    print("No camera frames received. Check permissions and camera index.", file=sys.stderr)
                    return 1
                if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                    break
                continue
            failures = 0
            height, width = frame.shape[:2]
            scale = min(1.0, 1100 / width, 650 / height)
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            h, w = frame.shape[:2]
            if last_shape is not None and last_shape != (h, w):
                points.clear()
                auto_pending = True
                status = "Video dimensions changed. Please recalibrate."
            if saved is not None:
                if saved.get("aspect") and abs(width / height - saved["aspect"]) > 0.01:
                    resume_reference = None
                    status = "Video aspect ratio changed. Please recalibrate."
                else:
                    restored = (np.asarray(saved["corners"]) * [w - 1, h - 1]).tolist()
                    if valid_corners(restored):
                        points[:] = restored
                        auto_pending = False
                        status = "Loaded calibration. If camera/board moved, press R."
                    else:
                        resume_reference = None
                saved = None
            last_shape = (h, w)
            frame_count += 1
            if auto_pending and frame_count % 15 == 0:
                detection = detect_board(frame, allowed_sizes)
                if detection is not None:
                    corners, size, score = detection
                    points[:] = corners
                    auto_pending = False
                    status = f"Auto detected {size}x{size}. Check overlay; S saves, A retries."
                    print(f"Detected {size}x{size} grid; alignment score {score:.2f}")
                else:
                    status = "No grid found. Show full empty board from above. A:retry M:manual"
            preview = frame.copy()
            corrected = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
            warped = None
            if len(points) == 4:
                matrix = transform(points)
                warped = cv2.warpPerspective(frame, matrix, (SIDE, SIDE))
                grid = grid_points(size)
                geometry = (size, args.camera, tuple(np.asarray(points).ravel()))
                if tracker is None or geometry != tracker_geometry:
                    tracker = StoneTracker(grid)
                    tracker_geometry = geometry
                    previous_event = None
                    if resume_reference is not None:
                        tracker.reference = resume_reference.astype(np.float32)
                        resume_reference = None
                        print("Restored empty reference; keep board/camera fixed. A then B if moved.", flush=True)
                    if args.capture_empty:
                        tracker.start_capture()
                        args.capture_empty = False
                was_ready = tracker.reference is not None
                tracker.update(warped, time.monotonic())
                if not was_ready and tracker.reference is not None:
                    print("Empty-board baseline ready. Place ONE stone and remove your hand.", flush=True)
                    folder = ROOT / "diagnostics"
                    try:
                        folder.mkdir(exist_ok=True)
                        if not cv2.imwrite(str(folder / "latest_empty_reference.png"),
                                           np.clip(tracker.reference, 0, 255).astype(np.uint8)):
                            raise OSError("Could not write empty-board reference")
                        (folder / "latest_calibration.json").write_text(json.dumps({
                            "size": size, "camera": args.camera, "aspect": width / height,
                            "corners": (np.asarray(points) / [w - 1, h - 1]).tolist()}, indent=2) + "\n")
                    except OSError as error:
                        print(f"Could not save diagnostic reference: {error}", file=sys.stderr)
                corrected = annotate_grid(warped, grid)
                tracker.draw(corrected, grid)
                label(corrected, tracker.message, SIDE - 12)
                if tracker.events and tracker.events[-1] != previous_event:
                    previous_event = tracker.events[-1].copy()
                    print(tracker.message)
                if tracker.events:
                    event = tracker.events[-1]
                    label(corrected, f"Last: {event['action']} {event['coordinate']} {event['color']}", 24)
                inverse = np.linalg.inv(matrix)
                source_grid = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), inverse)
                projected_grid = source_grid.reshape(size, size, 2)
                preview = annotate_grid(preview, projected_grid)
                tracker.draw(preview, projected_grid)
                for i in range(size):
                    xy = tuple(np.rint(grid[i, 0] + [-30, 5]).astype(int))
                    cv2.putText(corrected, str(i + 1), xy, cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 180, 255), 1, cv2.LINE_AA)
                    xy = tuple(np.rint(grid[0, i] + [-5, -15]).astype(int))
                    cv2.putText(corrected, chr(65 + i), xy, cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 180, 255), 1, cv2.LINE_AA)
            else:
                tracker = None
                tracker_geometry = None
                label(corrected, "Looking for grid... M: manual, A: automatic detection"
                      if auto_pending else "Click outer grid corners: TL, TR, BR, BL.")
            for i, point in enumerate(points):
                xy = tuple(np.rint(point).astype(int))
                cv2.circle(preview, xy, 6, (0, 0, 255), -1)
                cv2.putText(preview, str(i + 1), (xy[0] + 8, xy[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            label(preview, f"Camera {args.camera} | {size}x{size} intersections / {size-1}x{size-1} cells | {width}x{height}")
            label(preview, status, 52)
            footer = np.full((94, w, 3), 24, dtype=np.uint8)
            if tracker is not None:
                coordinates = tracker.descriptions()
                mode = "PAUSED" if tracker.paused else "LIVE"
                summary = ", ".join(coordinates) or "none"
                label(footer, f"{mode} | Detected: {len(coordinates)} | {summary}", 24)
                label(footer, tracker.message, 49)
            else:
                label(footer, "Waiting for board detection...", 24)
            label(footer, "B:baseline A:auto M:manual S:save D:debug N:camera G:size Q:quit", 79)
            preview = np.vstack((preview, footer))
            cv2.imshow(SOURCE, preview)
            cv2.imshow(BOARD, corrected)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("a")):
                points.clear()
                auto_pending = True
                status = "Looking for a full empty grid..."
            elif key == ord("b"):
                if tracker is not None:
                    tracker.start_capture()
                    previous_event = None
                    status = "Empty baseline requested. Remove ALL stones and hands."
                else:
                    status = "Detect the board first, then remove ALL stones and press B."
            elif key == ord("m"):
                points.clear()
                auto_pending = False
                status = "Click TL, TR, BR, BL of the OUTERMOST GRID."
            elif key == ord("u") and points:
                points.pop()
                auto_pending = False
                status = "Last point removed. Continue selecting corners."
            elif key == ord("g"):
                size = SIZES[(SIZES.index(size) + 1) % len(SIZES)]
                allowed_sizes = (size,)
                points.clear()
                auto_pending = True
                status = f"Looking for a {size}x{size} grid..."
            elif key == ord("n"):
                cap.release()
                args.camera = (args.camera + 1) % 4
                points.clear()
                auto_pending = True
                cap = cv2.VideoCapture(args.camera, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                status = f"Camera {args.camera}. Looking for grid..."
                print(f"Switched to camera {args.camera}")
            elif key == ord("d"):
                folder = ROOT / "diagnostics"
                try:
                    folder.mkdir(exist_ok=True)
                    pictures = {"latest_camera": frame, "latest_overlay": corrected}
                    if warped is not None:
                        pictures["latest_board"] = warped
                    for name, picture in pictures.items():
                        if not cv2.imwrite(str(folder / f"{name}.png"), picture):
                            raise OSError(f"Could not save {name}")
                    if tracker is not None:
                        (folder / "latest_state.json").write_text(json.dumps(tracker.state(), indent=2) + "\n")
                    status = "Saved diagnostic images and tracking state in diagnostics/."
                    print(status)
                except OSError as error:
                    print(f"Diagnostic save failed: {error}", file=sys.stderr)
            elif key == ord("s"):
                if warped is None:
                    status = "Select all four corners before saving."
                    continue
                config = {"size": size, "camera": args.camera, "aspect": width / height,
                          "corners": (np.asarray(points) / [w - 1, h - 1]).tolist()}
                folder = ROOT / "captures"
                folder.mkdir(exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                try:
                    CONFIG.write_text(json.dumps(config, indent=2) + "\n")
                    for name, picture in (("camera", frame), ("board", warped), ("grid", corrected)):
                        if not cv2.imwrite(str(folder / f"{stamp}_{name}.png"), picture):
                            raise OSError(f"Could not save {name} image")
                    if tracker is not None:
                        (folder / f"{stamp}_state.json").write_text(
                            json.dumps(tracker.state(), indent=2) + "\n")
                    status = "Saved calibration and camera/board/grid PNG images."
                    print(f"{status}\n{folder}")
                except OSError as error:
                    status = f"Save failed: {error}"
                    print(status, file=sys.stderr)
            if (cv2.getWindowProperty(SOURCE, cv2.WND_PROP_VISIBLE) < 1
                    or cv2.getWindowProperty(BOARD, cv2.WND_PROP_VISIBLE) < 1):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
