"""Stress-test selected DINO configuration on deterministic photo variants."""
import json
import argparse
import time
from PIL import ImageDraw, ImageEnhance, Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from detection_utils import ROOT, load_photo, filter_stone_boxes, suppress_duplicate_boxes
from benchmark_detectors import evaluate, GT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="bottle cap.")
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--stone-filter", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    tag = "bottle_cap" if args.prompt == "bottle cap." else "custom"
    output = ROOT / "detector_benchmark/validation" / tag
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True).eval()
    rows = []
    for kind, filename, truth in [("stones", "board_20261029_123855_712442.png", GT),
                                  ("empty", "board_20261029_122834_782438.png", [])]:
        base = load_photo(ROOT / "captures" / filename, True)
        rng = np.random.default_rng(42)
        variants = {"original": base, "dim_075": ImageEnhance.Brightness(base).enhance(.75),
                    "bright_125": ImageEnhance.Brightness(base).enhance(1.25),
                    "contrast_070": ImageEnhance.Contrast(base).enhance(.7),
                    "noise_sigma3": Image.fromarray(np.clip(np.array(base).astype(float)+rng.normal(0, 3, (base.height, base.width, 3)), 0, 255).astype(np.uint8))}
        for name, photo in variants.items():
            stem = f"{kind}_{name}"
            photo.save(output / f"{stem}_input.png")
            inputs = processor(images=photo, text=args.prompt, return_tensors="pt")
            started = time.perf_counter()
            with torch.inference_mode():
                raw = model(**inputs)
            result = processor.post_process_grounded_object_detection(raw, input_ids=inputs.input_ids, threshold=args.threshold,
                              text_threshold=.15, target_sizes=[(photo.height, photo.width)])[0]
            items = [{"box": b.tolist(), "score": float(s)} for b, s in zip(result["boxes"], result["scores"])]
            if args.stone_filter:
                items = filter_stone_boxes(items)
            items = suppress_duplicate_boxes(items)
            metrics = evaluate(items, truth)
            row = dict(kind=kind, variant=name, prompt=args.prompt, threshold=args.threshold,
                       stone_filter=args.stone_filter, seconds=time.perf_counter()-started, detections=items, **metrics)
            rows.append(row)
            draw = ImageDraw.Draw(photo)
            for box in truth:
                draw.rectangle(box, outline="red", width=2)
            for item in items:
                draw.rectangle(item["box"], outline="lime", width=3)
                draw.text(tuple(item["box"][:2]), f'{item["score"]:.2f}', fill="blue")
            photo.save(output / f"{stem}_result.png")
            (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(stem, metrics, flush=True)


if __name__ == "__main__":
    main()
