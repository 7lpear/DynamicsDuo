"""Shared image preparation and output for offline detector comparisons."""
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent


def load_photo(path, rectify=False):
    from PIL import Image
    photo = Image.open(path).convert("RGB")
    if rectify:
        import numpy as np
        from board_calibration import load_calibration, rectify_frame
        photo = Image.fromarray(rectify_frame(np.asarray(photo), load_calibration()))
    return photo


def save_result(photo, detections, args, model, elapsed, backend):
    from PIL import ImageDraw
    output = ROOT / f"{backend}_results" / args.image.stem / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    photo.save(output / "input.png")
    draw = ImageDraw.Draw(photo)
    for item in detections:
        box = item["box"]
        draw.rectangle(box, outline="lime", width=3)
        draw.text((max(0, box[0]), max(0, box[1]-14)), f'{item["label"]} {item["score"]:.2f}', fill="red")
    photo.save(output / "detections.png")
    data = {"model": model, "source": str(args.image.resolve()), "rectified": args.rectify,
            "coordinate_space": "rectified image" if args.rectify else "original image",
            "prompt": args.prompt, "threshold": args.threshold, "inference_seconds": elapsed,
            "detections": detections}
    if hasattr(args, "text_threshold"):
        data["text_threshold"] = args.text_threshold
    for key in ("stone_filter", "autocontrast", "imgsz"):
        if hasattr(args, key):
            data[key] = getattr(args, key)
    (output / "detections.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Objects: {len(detections)}; inference: {elapsed:.2f}s")
    for item in detections:
        print(f'{item["label"]}: {item["score"]:.3f}, box={item["box"]}')
    print(f"Saved: {output}")


def filter_stone_boxes(detections):
    """Geometry prior for rectified board: pitch=100px, stone diameter ~65px.

    This rejects board-sized boxes; it is not a learned stone classification.
    """
    return [d for d in detections
            if 25 <= d["box"][2]-d["box"][0] <= 120
            and 25 <= d["box"][3]-d["box"][1] <= 120
            and .5 <= (d["box"][2]-d["box"][0]) / max(d["box"][3]-d["box"][1], 1) <= 2]


def suppress_duplicate_boxes(detections, threshold=.45):
    kept = []
    for item in sorted(detections, key=lambda d: d["score"], reverse=True):
        a = item["box"]
        duplicate = False
        for other in kept:
            b = other["box"]
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
            if intersection / max(union, 1e-6) > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def validate_args(parser, args):
    import torch
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be 0..1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    if not args.prompt.strip():
        parser.error("Prompt must not be empty")
