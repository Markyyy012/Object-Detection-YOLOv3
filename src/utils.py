import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from src.model import Darknet

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

PALETTE = (
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (192, 0, 0), (0, 192, 0), (0, 0, 192),
    (255, 165, 0), (128, 128, 255), (255, 128, 128), (128, 255, 128), (0, 128, 255),
)


@dataclass
class Detection:
    box: np.ndarray  # (x1, y1, x2, y2) in original image coords
    score: float
    class_id: int
    label: str


def load_model(cfg_path, weights_path, device="cpu", half=False):
    model = Darknet(cfg_path)
    model.load_darknet_weights(weights_path)
    model.to(device).eval()
    if half:
        model.half()
    return model


def letterbox(img, new_shape=416, color=(114, 114, 114)):
    """Resize while keeping aspect ratio, padding to a square."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (new_shape - new_w) / 2, (new_shape - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, (r, left, top)


def preprocess(padded_bgr, device="cpu", half=False):
    """BGR letterboxed image -> normalized (1,3,H,W) float tensor."""
    rgb = padded_bgr[:, :, ::-1]
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    if half:
        tensor = tensor.half()
    return tensor


def decode_scale(pred, anchors, num_classes, img_dim):
    """Decode one scale's raw tensor into center/wh boxes and class scores."""
    batch, _attr, g, _g2 = pred.shape
    stride = img_dim // g
    num_anchors = len(anchors)
    pred = pred.view(batch, num_anchors, 5 + num_classes, g, g).permute(0, 1, 3, 4, 2)
    pred = pred.contiguous()

    grid_y, grid_x = torch.meshgrid(
        torch.arange(g, device=pred.device),
        torch.arange(g, device=pred.device),
        indexing="ij",
    )
    anchor_w = torch.tensor([a[0] for a in anchors], device=pred.device).view(1, num_anchors, 1, 1)
    anchor_h = torch.tensor([a[1] for a in anchors], device=pred.device).view(1, num_anchors, 1, 1)

    cx = (torch.sigmoid(pred[..., 0]) + grid_x) * stride
    cy = (torch.sigmoid(pred[..., 1]) + grid_y) * stride
    w = torch.exp(pred[..., 2]) * anchor_w
    h = torch.exp(pred[..., 3]) * anchor_h
    obj = torch.sigmoid(pred[..., 4])
    cls = torch.sigmoid(pred[..., 5:])

    scores = obj.unsqueeze(-1) * cls
    boxes_xy = torch.stack((cx, cy), dim=-1).view(batch, -1, 2)
    boxes_wh = torch.stack((w, h), dim=-1).view(batch, -1, 2)
    scores = scores.view(batch, -1, num_classes)
    return boxes_xy, boxes_wh, scores


def box_iou(a, b):
    x1 = torch.maximum(a[0], b[:, 0])
    y1 = torch.maximum(a[1], b[:, 1])
    x2 = torch.minimum(a[2], b[:, 2])
    y2 = torch.minimum(a[3], b[:, 3])
    inter = ((x2 - x1).clamp(min=0)) * ((y2 - y1).clamp(min=0))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-6)


def nms_torch(boxes, scores, iou_thres):
    """Greedy per-class NMS. Returns kept indices."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        keep_mask = box_iou(boxes[i], boxes[rest]) <= iou_thres
        order = rest[keep_mask]
    return keep


def postprocess(preds, anchors_list, num_classes, img_dim, conf_thres, iou_thres):
    """Full decode + confidence filter + per-class NMS."""
    boxes_xy, boxes_wh, scores = [], [], []
    for pred, anchors in zip(preds, anchors_list):
        xy, wh, sc = decode_scale(pred, anchors, num_classes, img_dim)
        boxes_xy.append(xy)
        boxes_wh.append(wh)
        scores.append(sc)

    xy = torch.cat(boxes_xy, dim=1)
    wh = torch.cat(boxes_wh, dim=1)
    sc = torch.cat(scores, dim=1)

    x1 = xy[..., 0] - wh[..., 0] / 2
    y1 = xy[..., 1] - wh[..., 1] / 2
    x2 = xy[..., 0] + wh[..., 0] / 2
    y2 = xy[..., 1] + wh[..., 1] / 2
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    conf, cls_id = sc.max(dim=-1)

    batch_results = []
    for b in range(boxes.shape[0]):
        keep = conf[b] > conf_thres
        bbox, score, cid = boxes[b][keep], conf[b][keep], cls_id[b][keep]
        dets = []
        for c in range(num_classes):
            sel = cid == c
            if not sel.any():
                continue
            bbox_c, score_c = bbox[sel], score[sel]
            kept = nms_torch(bbox_c, score_c, iou_thres)
            for k in kept:
                dets.append((bbox_c[k].cpu().numpy(), float(score_c[k].cpu()), int(c)))
        batch_results.append(dets)
    return batch_results


def rescale_boxes(box, r, left, top, orig_w, orig_h):
    """Convert a box from letterboxed pixel space back to original image coords."""
    x1, y1, x2, y2 = box
    x1 = (x1 - left) / r
    y1 = (y1 - top) / r
    x2 = (x2 - left) / r
    y2 = (y2 - top) / r
    return np.array(
        [
            max(0.0, min(x1, orig_w - 1)),
            max(0.0, min(y1, orig_h - 1)),
            max(0.0, min(x2, orig_w - 1)),
            max(0.0, min(y2, orig_h - 1)),
        ]
    )


def infer_frame(model, frame_bgr, img_dim, conf_thres, iou_thres, device="cpu", half=False):
    """Run one BGR frame through the model; returns list[Detection] in original coords."""
    orig_h, orig_w = frame_bgr.shape[:2]
    padded, (r, left, top) = letterbox(frame_bgr, img_dim)
    tensor = preprocess(padded, device, half)

    with torch.no_grad():
        preds = model(tensor)

    anchors_list = model.masked_anchors()
    results = postprocess(
        preds, anchors_list, model.num_classes, img_dim, conf_thres, iou_thres
    )

    detections = []
    for box, score, cls in results[0]:
        box_orig = rescale_boxes(box, r, left, top, orig_w, orig_h)
        detections.append(
            Detection(
                box=box_orig,
                score=score,
                class_id=cls,
                label=COCO_NAMES[cls] if cls < len(COCO_NAMES) else str(cls),
            )
        )
    return detections


def draw_detections(frame, detections, show_conf=True):
    """Render boxes + labels onto a BGR frame in place and return it."""
    for d in detections:
        color = PALETTE[d.class_id % len(PALETTE)]
        x1, y1, x2, y2 = [int(v) for v in d.box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if show_conf:
            label = f"{d.label} {d.score * 100:.0f}%"
        else:
            label = d.label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def make_video_writer(path, fps, size):
    """Open a VideoWriter; fall back to MJPG/AVI if the primary codec is missing."""
    candidates = [
        (path, "mp4v", ".mp4"),
        (path.rsplit(".", 1)[0] + ".avi", "MJPG", ".avi"),
    ]
    for p, codec, ext in candidates:
        writer = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, p
    raise RuntimeError("Could not open a video writer with mp4v or MJPG codecs")


def safe_output_path(directory, name, ext):
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)
