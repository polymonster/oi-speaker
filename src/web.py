import asyncio
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
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

import speaker as spk

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_PATH = Path("config.toml")

_MDNS_TYPE = "_oi-speaker._tcp.local."
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
        properties={"name": my_name},
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
    spk.chat_history.append({"role": "user", "text": req.text})
    spk.start_timer()
    loop = asyncio.get_event_loop()
    state = await loop.run_in_executor(None, spk.query_llm, spk.ctx.llm_client, spk.ctx.system, req.text)
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
        "state": spk.speaker_state.value,
        "playing": spk._player_active,
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
