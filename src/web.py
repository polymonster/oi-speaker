import asyncio
import io
import struct
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import socket
import tomllib
import tomli_w
import httpx
import numpy as np
import scipy.io.wavfile as wavfile
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

import speaker as spk

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_PATH = Path("config.toml")

_MDNS_TYPE = "_oi-speaker._tcp.local."


class SpeakRequest(BaseModel):
    text: str


class ResolveRequest(BaseModel):
    url: str


class SyncConfigRequest(BaseModel):
    peer_name: str


_peers_lock = threading.Lock()
_peers: dict[str, dict] = {}  # keyed by room name
_zeroconf: AsyncZeroconf | None = None
_browser: AsyncServiceBrowser | None = None


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


class _PeerListener:
    """AsyncServiceBrowser callback handler."""

    def add_service(self, zc, type_: str, name: str) -> None:
        info = ServiceInfo(type_, name)
        if not info.load_from_cache(zc):
            return
        peer_name = name.replace(f".{_MDNS_TYPE}", "")
        ip = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
        props = {k.decode(): v.decode() for k, v in info.properties.items()}
        with _peers_lock:
            _peers[peer_name] = {
                "name": peer_name,
                "ip": ip,
                "port": info.port,
                "online": True,
                "props": props,
            }

    def remove_service(self, zc, type_: str, name: str) -> None:
        peer_name = name.replace(f".{_MDNS_TYPE}", "")
        with _peers_lock:
            if peer_name in _peers:
                _peers[peer_name]["online"] = False

    def update_service(self, zc, type_: str, name: str) -> None:
        self.add_service(zc, type_, name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _zeroconf, _browser

    with open(CONFIG_PATH, "rb") as f:
        _cfg = tomllib.load(f)
    net = _cfg.get("network", {})
    my_name: str = net.get("name", "oi-speaker")
    my_port: int = int(net.get("port", 8000))
    my_ip: str = _local_ip()

    _zeroconf = AsyncZeroconf()
    _svc_name = f"{my_name}.{_MDNS_TYPE}"
    info = ServiceInfo(
        _MDNS_TYPE,
        _svc_name,
        addresses=[socket.inet_aton(my_ip)],
        port=my_port,
        properties={"name": my_name, "role": "worker" if "--worker" in sys.argv else "speaker"},
    )
    await _zeroconf.async_register_service(info)
    _browser = AsyncServiceBrowser(_zeroconf.zeroconf, _MDNS_TYPE, _PeerListener())

    spk.start()
    yield

    spk.shutdown()
    await _browser.async_cancel()
    await _zeroconf.async_unregister_service(info)
    await _zeroconf.async_close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    text: str


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    entry = {"role": "user", "text": req.text}
    spk.chat_history.append(entry)
    spk._save_history(spk.CHAT_HISTORY_PATH, entry)
    spk.start_perf_timer()
    loop = asyncio.get_event_loop()
    state = await loop.run_in_executor(None, spk.query_llm, spk.ctx.llm_client, spk.ctx.system, req.text)
    return {"response": f"[{state.value}]"}


@app.get("/history")
async def history():
    return spk.chat_history


@app.get("/play-history")
async def play_history(limit: int = 100):
    return spk._load_history(spk.PLAY_HISTORY_PATH, limit)


@app.get("/audio-devices")
async def audio_devices():
    return spk.enumerate_audio_devices()


_DEFAULT_CONFIDENTIAL = ["llm.anthropic_api_key"]

@app.get("/settings")
async def get_settings():
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)
    meta = cfg.setdefault("meta", {})
    meta.setdefault("confidential", _DEFAULT_CONFIDENTIAL)
    return cfg


@app.post("/settings")
async def update_settings(settings: dict[str, Any] = Body(...)):
    with open(CONFIG_PATH, "wb") as f:
        f.write(tomli_w.dumps(settings).encode())
    return {"ok": True}


@app.post("/stop")
async def stop():
    spk.stop_playback()
    return {"ok": True}


@app.get("/logs")
async def logs(since: int = 0):
    return spk.get_log_lines(since)


@app.get("/status")
async def status():
    with open(CONFIG_PATH, "rb") as f:
        _cfg = tomllib.load(f)
    port = int(_cfg.get("network", {}).get("port", 8000))
    return {
        "state": spk.ctx.speaker_state.value if spk.ctx else spk.SpeakerState.LISTEN_FOR_WAKE.value,
        "playing": spk._player.active,
        "ip": _local_ip(),
        "port": port,
    }


@app.get("/peers")
async def peers():
    with _peers_lock:
        return list(_peers.values())


@app.api_route("/speakers/{peer_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(peer_name: str, path: str, request: Request):
    with _peers_lock:
        peer = _peers.get(peer_name)
    if not peer or not peer["online"]:
        raise HTTPException(status_code=404, detail=f"peer '{peer_name}' not found or offline")
    target = f"http://{peer['ip']}:{peer['port']}/{path}"
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=target,
            params=dict(request.query_params),
            content=await request.body(),
            headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
            timeout=10.0,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# ── RPC endpoints ────────────────────────────────────────────────────────────


def _wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    # Streaming WAV header — data size set to max so clients don't need to seek.
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", 0xFFFFFFFF,
    )


@app.post("/rpc/transcribe")
async def rpc_transcribe(audio: UploadFile = File(...)):
    """Accept a WAV file and return the transcribed text."""
    if spk.ctx is None:
        raise HTTPException(status_code=503, detail="speaker not ready")
    spk.log("rpc: transcribing for remote caller")
    data = await audio.read()
    buf = io.BytesIO(data)
    sr, audio_array = wavfile.read(buf)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if audio_array.dtype in (np.float32, np.float64):
        audio_array = (audio_array * 32767).astype(np.int16)
    else:
        audio_array = audio_array.astype(np.int16)
    if sr != 16000:
        import soxr
        audio_array = (soxr.resample(audio_array.astype(np.float32) / 32767.0, sr, 16000) * 32767).astype(np.int16)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, spk._transcribe_audio, spk.ctx.whisper_model, audio_array, spk.ctx.worker_url)
    return {"text": text.strip()}


@app.post("/rpc/speak")
async def rpc_speak(req: SpeakRequest):
    """Accept text and stream back a WAV audio response."""
    if spk.ctx is None:
        raise HTTPException(status_code=503, detail="speaker not ready")
    spk.log(f"rpc: speaking for remote caller: {req.text[:60]}")
    voice_model = spk.ctx.voice_model
    text = spk._strip_markdown(req.text)

    def generate():
        yield _wav_header(voice_model.config.sample_rate)
        for chunk in voice_model.synthesize(text):
            yield chunk.audio_int16_bytes

    return StreamingResponse(generate(), media_type="audio/wav")


_LOCAL_ONLY_SECTIONS = {"audio", "network"}

@app.post("/rpc/sync-config")
async def rpc_sync_config(req: SyncConfigRequest):
    """Pull config from a peer and apply it locally, preserving audio + network sections."""
    with _peers_lock:
        peer = _peers.get(req.peer_name)
    if not peer or not peer["online"]:
        raise HTTPException(status_code=404, detail=f"peer '{req.peer_name}' not found or offline")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{peer['ip']}:{peer['port']}/settings", timeout=10.0)
        resp.raise_for_status()
    remote_cfg: dict = resp.json()
    with open(CONFIG_PATH, "rb") as f:
        local_cfg = tomllib.load(f)
    for section in _LOCAL_ONLY_SECTIONS:
        if section in local_cfg:
            remote_cfg[section] = local_cfg[section]
    with open(CONFIG_PATH, "wb") as f:
        f.write(tomli_w.dumps(remote_cfg).encode())
    return {"ok": True}


@app.post("/rpc/resolve")
async def rpc_resolve(req: ResolveRequest):
    """Resolve a URL to a direct streamable URL (YouTube via yt-dlp, others pass through)."""
    loop = asyncio.get_event_loop()
    try:
        resolved = await loop.run_in_executor(None, spk.resolve_url, req.url)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"url": resolved}


