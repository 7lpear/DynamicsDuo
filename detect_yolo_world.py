"""Detect prompted objects with pretrained YOLO-World; no training."""
import argparse
from pathlib import Path
import time
from detection_utils import load_photo, save_result, validate_args, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--prompt", default="white cylindrical game piece", help="Separate classes with semicolons")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rectify", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference input size, e.g. 640 or 960")
    parser.add_argument("--autocontrast", action="store_true")
    args = parser.parse_args()
    validate_args(parser, args)
    from ultralytics import YOLOWorld
    classes = [s.strip() for s in args.prompt.split(";") if s.strip()]
    photo = load_photo(args.image, args.rectify)
    if args.autocontrast:
        from PIL import ImageOps
        photo = ImageOps.autocontrast(photo, cutoff=1)
    model_path = ROOT / "models" / "yolov8s-worldv2.pt"
    model_path.parent.mkdir(exist_ok=True)
    print("Loading YOLO-World; first run downloads detector and text encoder.", flush=True)
    model = YOLOWorld(str(model_path))
    model.set_classes(classes)
    started = time.perf_counter()
    result = model.predict(source=photo, conf=args.threshold, imgsz=args.imgsz, device="0" if args.device == "cuda" else "cpu", verbose=False)[0]
    elapsed = time.perf_counter() - started
    detections = [{"label": result.names[int(box.cls.item())], "score": float(box.conf.item()),
                   "box": box.xyxy[0].cpu().tolist()} for box in result.boxes]
    save_result(photo, detections, args, "yolov8s-worldv2", elapsed, "yolo_world")


if __name__ == "__main__":
    main()
