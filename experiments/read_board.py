"""Experimental empty/white board reader using a calibrated empty reference."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detection_utils import ROOT


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--threshold", type=float, default=8.0)
    args = parser.parse_args()
    config = json.loads((ROOT / "calibration/board.json").read_text(encoding="utf-8"))
    current = read_image(args.image)
    reference_path = Path(config["source_image"])
    if not reference_path.is_absolute():
        reference_path = ROOT / reference_path
    reference = read_image(reference_path)
    if current.shape != reference.shape:
        raise RuntimeError("Image size differs from calibration; recalibrate.")
    matrix = np.array(config["homography"])
    size = tuple(config["warp_size"])
    current = cv2.warpPerspective(current, matrix, size)
    reference = cv2.warpPerspective(reference, matrix, size)
    gray = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float)
    empty = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float)
    delta = gray - empty
    board = np.zeros((9, 9), dtype=int)
    scores = np.zeros((9, 9))
    yy, xx = np.mgrid[-45:46, -45:46]
    radius = np.hypot(xx, yy)
    inner, ring = radius <= 29, (radius >= 38) & (radius <= 45)
    overlay = current.copy()
    for row in range(9):
        for col in range(9):
            x = config["grid_margin"] + col * config["grid_spacing"]
            y = config["grid_margin"] + row * config["grid_spacing"]
            patch = delta[y-45:y+46, x-45:x+46]
            # Local background subtraction reduces exposure and smooth lighting changes.
            residual = patch - np.median(patch[ring])
            score = float(np.percentile(residual[inner], 90))
            scores[row, col] = score
            # Only empty/white is supported: dark objects and hands are not classified.
            board[row, col] = 2 if score >= args.threshold else 0
            color = (0, 180, 0) if board[row, col] == 2 else (180, 100, 0)
            cv2.circle(overlay, (x, y), 30 if board[row, col] else 4, color, 2)
            cv2.putText(overlay, f"{row},{col}:{score:.0f}", (x-35, y-33), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1)
    output = ROOT / "vision_results" / args.image.stem
    output.mkdir(parents=True, exist_ok=True)
    result = {"source_image": str(args.image.resolve()), "board": board.tolist(),
              "scores": scores.round(2).tolist(), "threshold": args.threshold,
              "labels": {"0": "empty", "2": "white candidate"},
              "note": "Experimental; assumes only white stones, fixed camera and board. Not connected to robot."}
    (output / "board.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Cannot encode preview")
    encoded.tofile(str(output / "preview.png"))
    print("0=empty, 2=white candidate; row/column start at 0 in top-left")
    print(board)
    print("Candidates:", np.argwhere(board == 2).tolist())
    print("Scores:\n", np.round(scores, 1))
    print("Saved:", output)
    if not args.no_window:
        cv2.imshow("White candidates - press any key", cv2.resize(overlay, (810, 810)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
