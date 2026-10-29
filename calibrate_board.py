"""Click outer grid intersections to calibrate a 9x9 board."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read: {args.image}")
    height, width = frame.shape[:2]
    scale = min(1200 / width, 800 / height, 1)
    display_size = (round(width * scale), round(height * scale))
    points = []
    window = "Click TL, TR, BR, BL | R: reset | S: save | Q: quit"
    preview_window = "Rectified 9x9 - check all 81 dots"
    destination = np.float32([[50, 50], [850, 50], [850, 850], [50, 850]])

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x * width / display_size[0], y * height / display_size[1]])

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    print("Click outer GRID intersections: top-left, top-right, bottom-right, bottom-left.")
    print("R: reset. S: save after checking all dots. Q/Esc: quit.")
    try:
        while True:
            overlay = frame.copy()
            matrix = warped = rectified_overlay = None
            if len(points) == 4:
                source = np.float32(points)
                if not cv2.isContourConvex(source.astype(np.int32)) or cv2.contourArea(source) < width * height * .01:
                    print("Invalid corners. Resetting; click TL, TR, BR, BL.")
                    points.clear()
                    continue
                matrix = cv2.getPerspectiveTransform(source, destination)
                warped = cv2.warpPerspective(frame, matrix, (901, 901))
                rectified_overlay = warped.copy()
                grid = np.float32([[[50 + c * 100, 50 + r * 100] for c in range(9)] for r in range(9)])
                original_grid = cv2.perspectiveTransform(grid.reshape(1, -1, 2), np.linalg.inv(matrix))[0]
                for x, y in original_grid:
                    cv2.circle(overlay, (round(float(x)), round(float(y))), 4, (0, 0, 255), -1)
                for x, y in grid.reshape(-1, 2):
                    cv2.circle(rectified_overlay, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.imshow(preview_window, cv2.resize(rectified_overlay, (720, 720)))
            for i, (x, y) in enumerate(points):
                position = (round(x), round(y))
                cv2.circle(overlay, position, 9, (0, 255, 0), 2)
                cv2.putText(overlay, str(i + 1), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(window, cv2.resize(overlay, display_size))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                points.clear()
                if cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow(preview_window)
            if key == ord("s") and matrix is not None:
                output = Path(__file__).resolve().parent / "calibration"
                output.mkdir(exist_ok=True)
                data = {"board_size": 9, "source_image": str(args.image.resolve()),
                        "image_size": [width, height], "source_points": points,
                        "homography": matrix.tolist(), "warp_size": [901, 901],
                        "grid_margin": 50, "grid_spacing": 100,
                        "origin": "top-left", "axes": "row down, column right"}
                (output / "board.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
                for name, img in (("empty_board.png", warped), ("grid_preview.png", rectified_overlay), ("source_grid.png", overlay)):
                    ok, encoded = cv2.imencode(".png", img)
                    if not ok:
                        raise RuntimeError(f"Cannot encode {name}")
                    encoded.tofile(str(output / name))
                print(f"Saved calibration and previews: {output}")
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
