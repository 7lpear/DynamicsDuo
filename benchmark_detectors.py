"""Compare pretrained detectors on measured photo annotations, without training.

Generated inputs are deterministic crops/contrast variants of the real photo.
Scores are tuning results on ONE image, not an independent accuracy estimate.
"""
import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from detection_utils import ROOT, load_photo

PROMPTS = ["white cylindrical game piece", "white stone", "white plastic cylinder",
           "bottle cap", "white round button", "white checker piece"]
THRESHOLDS = [.02, .05, .1, .15, .25, .4, .6]
# Manual visual annotations in the calibrated 901x901 image, enclosing stone tops.
GT = [[621, 114, 691, 183], [418, 418, 482, 484], [817, 417, 890, 487]]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(union, 1e-6)


def nms(items):
    kept = []
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        if not any(iou(item["box"], other["box"]) > .45 for other in kept):
            kept.append(item)
    return kept


def evaluate(items, truth):
    remaining = set(range(len(truth)))
    tp = 0
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        if remaining:
            index = max(remaining, key=lambda j: iou(item["box"], truth[j]))
            if iou(item["box"], truth[index]) >= .5:
                remaining.remove(index)
                tp += 1
    fp, fn = len(items)-tp, len(truth)-tp
    f1 = 2*tp / max(2*tp+fp+fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, f1=f1)


def views(photo, tiled):
    if not tiled:
        return [(photo, 0, 0)]
    # Fixed four overlapping crops, independent of stone locations.
    return [(photo.crop((x, y, x+550, y+550)), x, y) for y in (0, 351) for x in (0, 351)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["yolo", "dino"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    photo = load_photo(ROOT / "captures/board_20261029_123855_712442.png", True)
    output = ROOT / "detector_benchmark" / args.backend
    output.mkdir(parents=True, exist_ok=True)
    variants = {"rectified": photo, "autocontrast": ImageOps.autocontrast(photo, cutoff=1)}
    # Save actual generated inputs and manual truth for visual audit.
    for name, img in variants.items():
        img.save(output / f"input_{name}.png")
        for j, (tile, _, _) in enumerate(views(img, True)):
            tile.save(output / f"input_{name}_tile{j}.png")
    annotated = photo.copy()
    draw = ImageDraw.Draw(annotated)
    for box in GT:
        draw.rectangle(box, outline="red", width=3)
    annotated.save(output / "manual_ground_truth.png")
    if args.backend == "yolo":
        from ultralytics import YOLOWorld
        model = YOLOWorld(str(ROOT / "models/yolov8s-worldv2.pt"))
    else:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True).eval()
    all_rows = []
    for prompt in PROMPTS:
        if args.backend == "yolo":
            model.set_classes([prompt])
        for name, img in variants.items():
            for tiled in (False, True):
                started = time.perf_counter()
                predictions = {t: [] for t in ([.15, .25] if args.backend == "dino" else [None])}
                for tile, x, y in views(img, tiled):
                    if args.backend == "yolo":
                        result = model.predict(tile, conf=.01, imgsz=960, device="cpu", verbose=False)[0]
                        predictions[None].extend({"box": [float(b[0])+x, float(b[1])+y, float(b[2])+x, float(b[3])+y],
                                                  "score": float(s)} for b, s in zip(result.boxes.xyxy, result.boxes.conf))
                    else:
                        inputs = processor(images=tile, text=prompt+".", return_tensors="pt")
                        with torch.inference_mode():
                            raw = model(**inputs)
                        method = processor.post_process_grounded_object_detection
                        key = "threshold" if "threshold" in inspect.signature(method).parameters else "box_threshold"
                        for text_t in predictions:
                            result = method(raw, input_ids=inputs.input_ids, text_threshold=text_t,
                                            target_sizes=[(tile.height, tile.width)], **{key: .01})[0]
                            predictions[text_t].extend({"box": [float(b[0])+x, float(b[1])+y, float(b[2])+x, float(b[3])+y],
                                                        "score": float(s)} for b, s in zip(result["boxes"], result["scores"]))
                elapsed = time.perf_counter()-started
                local_rows = []
                for text_t, items in predictions.items():
                    items = nms(items)
                    for threshold in THRESHOLDS:
                        detections = [d for d in items if d["score"] >= threshold]
                        row = dict(prompt=prompt, variant=name, tiled=tiled, threshold=threshold,
                                   text_threshold=text_t, seconds=elapsed, detections=detections,
                                   **evaluate(detections, GT))
                        local_rows.append(row)
                all_rows.extend(local_rows)
                best = max(local_rows, key=lambda r: (r["f1"], -r["fp"]))
                print(f'{args.backend} | {prompt} | {name} tiled={tiled} | TP={best["tp"]} FP={best["fp"]} FN={best["fn"]} | {elapsed:.1f}s', flush=True)
                (output / "results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    all_rows.sort(key=lambda r: (r["f1"], -r["fp"], r["threshold"]), reverse=True)
    (output / "ranked.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    for index, row in enumerate(all_rows[:5]):
        preview = photo.copy()
        draw = ImageDraw.Draw(preview)
        for box in GT:
            draw.rectangle(box, outline="red", width=2)
        for item in row["detections"]:
            draw.rectangle(item["box"], outline="lime", width=3)
            draw.text(tuple(item["box"][:2]), f'{item["score"]:.2f}', fill="blue")
        preview.save(output / f"best_{index+1}.png")
    print("BEST", json.dumps({k:v for k,v in all_rows[0].items() if k != "detections"}), flush=True)


if __name__ == "__main__":
    main()
