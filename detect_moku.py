"""Run pretrained Moku on a photo; no training or robot connection."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rectify", action="store_true", help="Use saved board calibration before inference")
    args = parser.parse_args()
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    model_id = "kaya-go/moku-v3"
    print(f"Loading {model_id}; first run downloads model files.", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id).to(args.device).eval()
    photo = Image.open(args.image).convert("RGB")
    if args.rectify:
        import cv2
        import numpy as np
        config = json.loads((Path(__file__).resolve().parent / "calibration/board.json").read_text(encoding="utf-8"))
        if list(photo.size) != config["image_size"]:
            raise RuntimeError("Image size differs from calibration")
        photo = Image.fromarray(cv2.warpPerspective(np.array(photo), np.array(config["homography"]), tuple(config["warp_size"])))
    inputs = processor(images=photo, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        predictions = model(**inputs)
    result = processor.post_process_object_detection(
        predictions, target_sizes=torch.tensor([[photo.height, photo.width]]),
        threshold=args.threshold)[0]
    draw = ImageDraw.Draw(photo)
    detections = []
    colors = {"black_stone": "red", "white_stone": "lime", "board_corner": "cyan"}
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        name = model.config.id2label[int(label)]
        coords = box.detach().cpu().tolist()
        confidence = float(score)
        detections.append({"label": name, "score": confidence, "box": coords})
        color = colors.get(name, "yellow")
        draw.rectangle(coords, outline=color, width=4)
        draw.text((max(0, coords[0]), max(0, coords[1]-15)), f"{name} {confidence:.2f}", fill=color)
        print(f"{name}: {confidence:.3f}, box={[round(v, 1) for v in coords]}")
    suffix = "_rectified" if args.rectify else ""
    if args.threshold != 0.3:
        suffix += f"_threshold_{args.threshold:g}"
    output = Path(__file__).resolve().parent / "moku_results" / (args.image.stem + suffix)
    output.mkdir(parents=True, exist_ok=True)
    photo.save(output / "detections.png")
    (output / "detections.json").write_text(json.dumps({
        "model": model_id, "source": str(args.image.resolve()),
        "threshold": args.threshold, "detections": detections,
        "rectified": args.rectify,
    }, indent=2), encoding="utf-8")
    print(f"Detected {len(detections)} objects. Saved: {output}")


if __name__ == "__main__":
    main()
