import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import socket
import tomllib
import tomli_w
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import speaker as spk

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_PATH = Path("config.toml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    spk.start()
    yield
    spk.stop_playback()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    text: str


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    spk.chat_history.append({"role": "user", "text": req.text})
    spk.start_timer()
    state = spk.query_llm(spk.ctx.llm_client, spk.ctx.system, req.text)
    return {"response": f"[{state.value}]"}


@app.get("/history")
async def history():
    return spk.chat_history


@app.get("/audio-devices")
async def audio_devices():
    return spk.enumerate_audio_devices()


@app.get("/settings")
async def get_settings():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


@app.post("/settings")
async def update_settings(settings: dict[str, Any] = Body(...)):
    with open(CONFIG_PATH, "wb") as f:
        f.write(tomli_w.dumps(settings).encode())
    return {"ok": True}


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


@app.post("/stop")
async def stop():
    spk.stop_playback()
    return {"ok": True}


@app.get("/status")
async def status():
    return {
        "state": spk.speaker_state.value,
        "playing": spk._player_active,
        "ip": _local_ip(),
    }
