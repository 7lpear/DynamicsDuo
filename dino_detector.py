"""Reusable Grounding DINO engine; load once, infer many images."""
import inspect
import time
from detection_utils import filter_stone_boxes, suppress_duplicate_boxes


class DinoDetector:
    def __init__(self, device="cpu", prompt="bottle cap.", threshold=.25, text_threshold=.15):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu")
        if device == "cpu":
            torch.set_num_threads(4)
        self.torch = torch
        self.device = device
        self.prompt = prompt.strip().lower().rstrip(".") + "."
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.last_counts = {}
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(device).eval()

    def detect(self, photo):
        started = time.perf_counter()
        inputs = self.processor(images=photo, text=self.prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        method = self.processor.post_process_grounded_object_detection
        key = "threshold" if "threshold" in inspect.signature(method).parameters else "box_threshold"
        result = method(outputs, input_ids=inputs.input_ids, text_threshold=self.text_threshold,
                        target_sizes=[(photo.height, photo.width)], **{key: self.threshold})[0]
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        raw = [{"label": str(label), "score": float(score), "box": box.detach().cpu().tolist()}
               for label, score, box in zip(labels, result["scores"], result["boxes"])]
        # No fixed object count or top-N truncation: retain every valid detection.
        sized = filter_stone_boxes(raw)
        items = suppress_duplicate_boxes(sized)
        self.last_counts = {"above_threshold": len(raw), "after_size_filter": len(sized),
                            "after_duplicate_removal": len(items)}
        return items, time.perf_counter()-started
