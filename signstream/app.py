import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .data import fixture
from .features import LABELS, TokenBuffer
from .model import ROOT, Recognizer


@asynccontextmanager
async def lifespan(app):
    app.state.recognizer = Recognizer()
    app.state.inference_lock = asyncio.Lock()
    yield


app = FastAPI(title="SignStream", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class Window(BaseModel):
    frames: list = Field(min_length=8, max_length=96)


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/api/info")
def info():
    return {
        "vocabulary": LABELS,
        "language": "None: procedural motion fixtures, not ASL",
        "evaluation": json.loads((ROOT / "models/evaluation.json").read_text()),
    }


@app.get("/api/fixture/{label}")
def get_fixture(label: str, seed: int = 9000):
    if label not in LABELS or not 0 <= seed <= 1000000:
        raise HTTPException(400, "Invalid fixture label or seed")
    return {"frames": fixture(label, seed).tolist(), "source": "procedural synthetic fixture"}


async def predict(frames):
    async with app.state.inference_lock:
        return await asyncio.to_thread(app.state.recognizer.predict, frames)


@app.post("/api/predict")
async def infer(window: Window):
    try:
        return await predict(window.frames)
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error


@app.websocket("/ws")
async def stream(ws: WebSocket):
    origin = ws.headers.get("origin")
    if origin and origin.split("://")[-1] != ws.headers.get("host"):
        await ws.close(code=1008)
        return
    await ws.accept()
    buffer = TokenBuffer()
    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > 2500000:
                await ws.close(code=1009)
                return
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object")
                if payload.get("reset"):
                    buffer = TokenBuffer()
                    await ws.send_json({"tokens": []})
                    continue
                result = await predict(payload.get("frames", []))
                result["tokens"] = buffer.accept(result["label"])
                await ws.send_json(result)
            except (ValueError, TypeError) as error:
                buffer.accept(None)
                await ws.send_json({"error": str(error)})
    except WebSocketDisconnect:
        pass
