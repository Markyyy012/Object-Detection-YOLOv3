from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from . import inference

router = APIRouter()


@router.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0.01, le=1.0),
    iou: float = Query(0.45, ge=0.01, le=1.0),
    img_size: int = Query(416, ge=320, le=608),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        detections, b64 = await run_in_threadpool(
            inference.detect_image, data, conf, iou, img_size
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "count": len(detections),
        "detections": detections,
        "image": f"data:image/jpeg;base64,{b64}",
    }


@router.get("/api/health")
async def health():
    return inference.device_info()
