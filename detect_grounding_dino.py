"""Detect prompted objects with pretrained Grounding DINO; no training."""
import argparse
from pathlib import Path
from detection_utils import load_photo, save_result, validate_args
from dino_detector import DinoDetector


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
    photo = load_photo(args.image, args.rectify)
    if args.autocontrast:
        from PIL import ImageOps
        photo = ImageOps.autocontrast(photo, cutoff=1)
    print("Loading Grounding DINO; first run downloads model files.", flush=True)
    detector = DinoDetector(args.device, args.prompt, args.threshold, args.text_threshold)
    detections, elapsed = detector.detect(photo, stone_filter=args.stone_filter)
    print(f"Detection stages: {detector.last_counts}")
    save_result(photo, detections, args, detector.model_id, elapsed, "grounding_dino")


if __name__ == "__main__":
    main()
