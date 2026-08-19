import argparse
import os
import queue
import sys
import threading
import time

import cv2
import torch

from src.utils import infer_frame, draw_detections, load_model

DEFAULT_PORT = 4747
WINDOW_NAME = "YOLOv3 - DroidCam/Video"


def select_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class StreamReader:
    """Reads frames from a VideoCapture source, optionally on a background thread.

    Threading decouples network latency (DroidCam MJPEG) from the inference loop.
    On read failure the capture is re-opened so phone disconnects auto-reconnect.
    """

    def __init__(self, src, threaded):
        self.src = src
        self.threaded = threaded
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {src}")
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.5)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.src)
                    continue
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put(frame)
            except Exception:
                time.sleep(0.5)
                try:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.src)
                except Exception:
                    pass

    def read(self):
        if self.threaded:
            try:
                return self._queue.get(timeout=5)
            except queue.Empty:
                return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


def build_url(args):
    if args.url:
        return args.url
    base = f"http://{args.ip}:{args.port}/video"
    if args.res:
        return f"{base}/force/{args.res}"
    return base


def has_display():
    return bool(os.environ.get("DISPLAY"))


def main():
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv3 detection: DroidCam phone camera or video file"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ip", help="DroidCam phone WiFi IP (e.g. 192.168.1.50)")
    src.add_argument("--url", help="full stream URL, e.g. http://192.168.1.50:4747/video")
    src.add_argument("--video", help="play a local video file instead of a camera")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="DroidCam port (default 4747)")
    parser.add_argument("--res", default=None, help="requested stream size, e.g. 640x480")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--img-size", type=int, default=416, help="YOLO input size")
    parser.add_argument("--weights", default="weights/yolov3.weights")
    parser.add_argument("--cfg", default="weights/yolov3.cfg")
    parser.add_argument("--no-display", action="store_true", help="disable window even if DISPLAY is set")
    parser.add_argument("--save-every", type=int, default=0, help="save every Nth annotated frame to outputs/ (0 = off)")
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--half", action="store_true", help="use FP16 on GPU")
    args = parser.parse_args()

    device = select_device(not args.cpu)
    print(f"Device: {device}{' (CUDA)' if device.type == 'cuda' else ''}")

    model = load_model(args.cfg, args.weights, device, args.half)
    print("Model loaded.")

    if args.video:
        source = args.video
        is_network = False
    else:
        source = build_url(args)
        is_network = True

    reader = StreamReader(source, threaded=is_network)
    print(f"Source: {source}")

    show_window = not args.no_display and has_display()
    if show_window:
        try:
            cv2.namedWindow("probe")
            cv2.destroyWindow("probe")
        except cv2.error:
            print("  GUI unavailable (headless OpenCV build); running headless.")
            show_window = False
        else:
            print("  Live window enabled (press 'q' to quit).")
    os.makedirs(args.save_dir, exist_ok=True)

    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 640)

    frame_idx = 0
    fps = 0.0
    start_time = time.time()
    try:
        while True:
            frame = reader.read()
            if frame is None:
                if is_network:
                    print("  waiting for frames...")
                    time.sleep(0.5)
                    continue
                break

            t0 = time.time()
            detections = infer_frame(
                model, frame, args.img_size, args.conf, args.iou, device, args.half
            )
            annotated = draw_detections(frame.copy(), detections)
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-6))
            frame_idx += 1

            if args.save_every and frame_idx % args.save_every == 0:
                save_path = os.path.join(args.save_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(save_path, annotated)

            if detections:
                summary = ", ".join(
                    f"{d.label}:{d.score * 100:.0f}%" for d in detections[:6]
                )
                print(f"[{frame_idx}] {len(detections)} obj  {summary}  ({fps:.1f} fps)")

            if show_window:
                if detections:
                    tag = f"{len(detections)} obj  {fps:.1f} fps  q=quit"
                else:
                    tag = f"{fps:.1f} fps  q=quit"
                cv2.putText(
                    annotated,
                    tag,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = time.time() - start_time
    print(f"Stopped after {frame_idx} frames in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
