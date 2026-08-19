import argparse
import os
import sys
import time

import cv2
import torch

from src.utils import infer_frame, draw_detections, make_video_writer, load_model


def select_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def process_image(args, model, device, half):
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"ERROR: could not read image: {args.image}")

    detections = infer_frame(
        model, img, args.img_size, args.conf, args.iou, device, half
    )
    annotated = draw_detections(img.copy(), detections)

    name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.save_dir, f"{name}_detected.jpg")
    os.makedirs(args.save_dir, exist_ok=True)
    cv2.imwrite(out_path, annotated)

    print(f"Detected {len(detections)} object(s) in {args.image}:")
    for d in detections:
        print(
            f"  {d.label:<20} {d.score * 100:5.1f}%  "
            f"box=({d.box[0]:.0f},{d.box[1]:.0f})-({d.box[2]:.0f},{d.box[3]:.0f})"
        )
    print(f"Saved annotated image -> {out_path}")


def process_video(args, model, device, half):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    name = os.path.splitext(os.path.basename(args.video))[0]
    out_path = os.path.join(args.save_dir, f"{name}_detected.mp4")
    os.makedirs(args.save_dir, exist_ok=True)
    writer, out_path = make_video_writer(out_path, fps, (width, height))

    frame_idx = 0
    processed = 0
    start = time.time()
    print(f"Processing {args.video} ({width}x{height}, {fps:.1f} fps, {total} frames)...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.fps_skip != 0:
            frame_idx += 1
            continue
        detections = infer_frame(
            model, frame, args.img_size, args.conf, args.iou, device, half
        )
        annotated = draw_detections(frame, detections)
        writer.write(annotated)
        processed += 1
        frame_idx += 1
        if processed % 10 == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(f"  frame {frame_idx}/{total}  ({processed / elapsed:.1f} fps)")

    cap.release()
    writer.release()
    print(f"Saved annotated video -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv3 object detection (WSL)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an input image")
    src.add_argument("--video", help="path to an input video file")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--img-size", type=int, default=416, help="YOLO input size")
    parser.add_argument("--fps-skip", type=int, default=1, help="process every Nth video frame")
    parser.add_argument("--weights", default="weights/yolov3.weights")
    parser.add_argument("--cfg", default="weights/yolov3.cfg")
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--half", action="store_true", help="use FP16 on GPU")
    args = parser.parse_args()

    device = select_device(not args.cpu)
    print(f"Device: {device}{' (CUDA)' if device.type == 'cuda' else ''}")

    model = load_model(args.cfg, args.weights, device, args.half)
    print("Model loaded.")

    if args.image:
        process_image(args, model, device, args.half)
    else:
        process_video(args, model, device, args.half)


if __name__ == "__main__":
    main()
