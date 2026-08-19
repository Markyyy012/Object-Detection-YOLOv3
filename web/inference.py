import base64
import threading

import cv2
import numpy as np
import torch

from src.utils import draw_detections, infer_frame, load_model

CFG_PATH = "weights/yolov3.cfg"
WEIGHTS_PATH = "weights/yolov3.weights"

_load_lock = threading.Lock()
_infer_lock = threading.Lock()

_model = None
_device = None
_half = False


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), False
    return torch.device("cpu"), False


def get_model():
    """Return the shared (model, device, half) tuple, loading on first use."""
    global _model, _device, _half
    if _model is None:
        with _load_lock:
            if _model is None:
                device, half = select_device()
                model = load_model(CFG_PATH, WEIGHTS_PATH, device, half)
                _model, _device, _half = model, device, half
    return _model, _device, _half


def is_loaded():
    return _model is not None


def device_info():
    device, _ = select_device()
    return {
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": is_loaded(),
    }


def detect_image(data, conf=0.5, iou=0.45, img_size=416):
    """Decode an image, run inference, and return (detections, b64_jpeg)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode the uploaded image")

    model, device, half = get_model()
    with _infer_lock:
        detections = infer_frame(model, img, img_size, conf, iou, device, half)
    annotated = draw_detections(img.copy(), detections)

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("failed to encode annotated image")

    det_list = [
        {
            "label": d.label,
            "score": round(float(d.score), 4),
            "box": [round(float(v), 1) for v in d.box],
        }
        for d in detections
    ]
    return det_list, base64.b64encode(buf.tobytes()).decode()


def detect_frame_bytes(data, params):
    """Decode a JPEG byte frame and run inference; returns (detections, jpeg_bytes)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode frame")
    return detect_frame(
        img,
        conf=params.get("conf", 0.5),
        iou=params.get("iou", 0.45),
        img_size=params.get("img_size", 416),
    )


def detect_frame(frame, conf=0.5, iou=0.45, img_size=416):
    """Run inference on an already-decoded BGR frame; returns (detections, jpeg_bytes)."""
    model, device, half = get_model()
    with _infer_lock:
        detections = infer_frame(model, frame, img_size, conf, iou, device, half)
    annotated = draw_detections(frame.copy(), detections)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return detections, None
    return detections, buf.tobytes()
