"""Detect prompted objects with pretrained Grounding DINO; no training."""
import argparse
import inspect
from pathlib import Path
import time
from detection_utils import load_photo, save_result, validate_args, filter_stone_boxes, suppress_duplicate_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--prompt", default="bottle cap.", help="Separate descriptions with periods; default tested on current cylindrical pieces")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rectify", action="store_true")
    parser.add_argument("--stone-filter", action="store_true", help="Reject boxes outside plausible stone size; requires --rectify")
    parser.add_argument("--autocontrast", action="store_true")
    args = parser.parse_args()
    validate_args(parser, args)
    if args.stone_filter and not args.rectify:
        parser.error("--stone-filter requires --rectify (100px grid pitch)")
    if not 0 <= args.text_threshold <= 1:
        parser.error("--text-threshold must be 0..1")
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = "IDEA-Research/grounding-dino-tiny"
    photo = load_photo(args.image, args.rectify)
    if args.autocontrast:
        from PIL import ImageOps
        photo = ImageOps.autocontrast(photo, cutoff=1)
    print(f"Loading {model_id}; first run downloads model files.", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(args.device).eval()
    prompt = args.prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."
    inputs = processor(images=photo, text=prompt, return_tensors="pt").to(args.device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    method = processor.post_process_grounded_object_detection
    # Transformers versions use either threshold or box_threshold.
    threshold_key = "threshold" if "threshold" in inspect.signature(method).parameters else "box_threshold"
    result = method(outputs, input_ids=inputs.input_ids,
                    text_threshold=args.text_threshold, target_sizes=[(photo.height, photo.width)],
                    **{threshold_key: args.threshold})[0]
    labels = result["text_labels"] if "text_labels" in result else result["labels"]
    detections = [{"label": str(label), "score": float(score), "box": box.detach().cpu().tolist()}
                  for label, score, box in zip(labels, result["scores"], result["boxes"])]
    if args.stone_filter:
        before = len(detections)
        detections = filter_stone_boxes(detections)
        print(f"Geometry filter: {before} -> {len(detections)} boxes")
    detections = suppress_duplicate_boxes(detections)
    save_result(photo, detections, args, model_id, elapsed, "grounding_dino")


if __name__ == "__main__":
    main()
