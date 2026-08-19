import asyncio
import json
import queue
import threading
import time

import cv2

from fastapi import WebSocket, WebSocketDisconnect

from . import inference


def _json_detections(detections):
    return [
        {
            "label": d.label,
            "score": round(float(d.score), 4),
            "box": [round(float(v), 1) for v in d.box],
        }
        for d in detections
    ]


def _droidcam_worker(url, out_q, stop_event, params):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        out_q.put(("error", f"could not open stream: {url}"))
        return
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(url)
            continue
        try:
            detections, jpeg = inference.detect_frame(
                frame,
                conf=params.get("conf", 0.5),
                iou=params.get("iou", 0.45),
                img_size=params.get("img_size", 416),
            )
        except Exception as exc:
            out_q.put(("error", str(exc)))
            continue
        if jpeg is None:
            continue
        item = ("frame", (detections, jpeg))
        try:
            out_q.put_nowait(item)
        except queue.Full:
            try:
                out_q.get_nowait()
            except queue.Empty:
                pass
            try:
                out_q.put_nowait(item)
            except queue.Full:
                pass
    cap.release()


async def live_endpoint(ws: WebSocket):
    await ws.accept()
    params = {"conf": 0.5, "iou": 0.45, "img_size": 416}
    mode = "idle"
    stop_event = threading.Event()
    out_q = queue.Queue(maxsize=2)
    worker = None

    async def sender():
        while True:
            item = await asyncio.to_thread(out_q.get)
            if item is None:
                break
            kind, payload = item
            if kind == "error":
                await ws.send_text(
                    json.dumps({"type": "error", "message": payload})
                )
                continue
            detections, jpeg = payload
            meta = {
                "type": "detections",
                "count": len(detections),
                "items": _json_detections(detections[:10]),
            }
            await ws.send_text(json.dumps(meta))
            await ws.send_bytes(jpeg)

    sender_task = asyncio.create_task(sender())

    def start_droidcam(data):
        nonlocal worker, stop_event, mode
        if worker is not None:
            stop_event.set()
            worker.join(timeout=1.0)
        url = data.get("url")
        if not url:
            ip = data.get("ip")
            port = int(data.get("port", 4747))
            url = f"http://{ip}:{port}/video"
        stop_event = threading.Event()
        mode = "droidcam"
        worker = threading.Thread(
            target=_droidcam_worker,
            args=(url, out_q, stop_event, params),
            daemon=True,
        )
        worker.start()

    def stop_droidcam():
        nonlocal worker
        if worker is not None:
            stop_event.set()
            worker.join(timeout=1.0)
            worker = None

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is not None:
                if mode == "webcam":
                    try:
                        detections, jpeg = await asyncio.to_thread(
                            inference.detect_frame_bytes, data, params
                        )
                    except Exception as exc:
                        await ws.send_text(
                            json.dumps({"type": "error", "message": str(exc)})
                        )
                        continue
                    if jpeg is not None:
                        meta = {
                            "type": "detections",
                            "count": len(detections),
                            "items": _json_detections(detections[:10]),
                        }
                        await ws.send_text(json.dumps(meta))
                        await ws.send_bytes(jpeg)
                continue

            text = msg.get("text")
            if text is None:
                continue
            try:
                control = json.loads(text)
            except (TypeError, ValueError):
                continue

            t = control.get("type")
            if t == "webcam":
                stop_droidcam()
                mode = "webcam"
                await ws.send_text(json.dumps({"type": "mode", "mode": mode}))
            elif t == "droidcam":
                start_droidcam(control)
                await ws.send_text(json.dumps({"type": "mode", "mode": mode}))
            elif t == "params":
                if "conf" in control:
                    params["conf"] = float(control["conf"])
                if "iou" in control:
                    params["iou"] = float(control["iou"])
                if "img_size" in control:
                    params["img_size"] = int(control["img_size"])
            elif t == "stop":
                stop_droidcam()
                mode = "idle"
                await ws.send_text(json.dumps({"type": "mode", "mode": mode}))
    except WebSocketDisconnect:
        pass
    finally:
        stop_droidcam()
        out_q.put(None)
        sender_task.cancel()
        try:
            await sender_task
        except (asyncio.CancelledError, Exception):
            pass
