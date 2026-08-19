import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import inference
from .live import live_endpoint
from .routes import router

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=inference.get_model, daemon=True).start()
    yield


app = FastAPI(title="YOLOv3 Object Detection", lifespan=lifespan)
app.include_router(router)
app.add_api_websocket_route("/ws/live", live_endpoint)


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
