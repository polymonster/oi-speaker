"""Handing transcription and speech to a faster box on the network. Every node scores itself per
service (a model on a gpu beats any cpu, more cores beat fewer, or a number set by hand); a node
started with --offload polls the others' scores slowly in the background and sends each call to the
best one that beats its own, falling back down the list and finally to its own model when a call
fails. Choosing costs nothing per call — the poll did the work."""
import os
import threading
import time

import requests

POLL_INTERVAL = 30.0  # workers come and go rarely; a call that fails skips the worker until the next poll anyway
FIRST_POLL_DELAY = 3.0  # mdns needs a moment after start to find the others
PROBE_TIMEOUT = 2.0
SERVICES = ("transcribe", "voice")
GPU_SCORE = 1000  # added for a model on a gpu, so any gpu outranks any cpu
GPU_DEVICES = ("cuda", "mps")


class Offload:
    """The workers this node can hand calls to, ranked per service. `own_scores()` is this node's
    scores ({service: score}); `peers()` lists the nodes mdns has found ({name, ip, port, online})."""

    def __init__(self, name: str, own_scores, peers, log=print):
        self._name = name
        self._own_scores = own_scores
        self._peers = peers
        self._log = log
        self._lock = threading.Lock()
        self._workers: dict[str, dict] = {}  # url -> its /rpc/capabilities
        self._failed: dict[str, float] = {}  # url -> skipped until this time
        self._best: dict[str, str | None] = {}  # service -> url last logged as the first choice

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        time.sleep(FIRST_POLL_DELAY)
        while True:
            try:
                self.poll()
            except Exception as e:
                self._log(f"offload: poll failed: {e}")
            time.sleep(POLL_INTERVAL)

    def poll(self):
        found = {}
        for peer in self._peers():
            if not peer.get("online") or not peer.get("ip") or peer.get("name") == self._name:
                continue
            url = f"http://{peer['ip']}:{peer['port']}"
            try:
                resp = requests.get(f"{url}/rpc/capabilities", timeout=PROBE_TIMEOUT)
                resp.raise_for_status()
                caps = resp.json()
            except Exception:
                continue  # down, busy, or a version from before offloading — not a worker either way
            if caps.get("name") != self._name:
                found[url] = caps
        with self._lock:
            self._workers = found
        self._report()

    def candidates(self, service: str) -> list[str]:
        """Worker urls to try for `service`, best first: only those scoring above this node, and not
        one that failed since the last poll."""
        own = self._own_scores().get(service, 0)
        now = time.time()
        with self._lock:
            ranked = sorted(((caps.get("scores", {}).get(service, 0), url) for url, caps in self._workers.items()
                             if self._failed.get(url, 0) <= now), reverse=True)
        return [url for score, url in ranked if score > own]

    def failed(self, url: str, error):
        self._log(f"offload: {self.name_of(url)} failed ({error}), skipping it until the next poll")
        with self._lock:
            self._failed[url] = time.time() + POLL_INTERVAL

    def name_of(self, url: str) -> str:
        with self._lock:
            return self._workers.get(url, {}).get("name") or url

    def status(self) -> dict:
        """Per service, where calls go now: the workers in order, then this node."""
        own = self._own_scores()
        with self._lock:
            workers = dict(self._workers)
        out = {}
        for service in SERVICES:
            out[service] = [{"name": workers[url].get("name", url), "url": url,
                             "score": workers[url].get("scores", {}).get(service, 0),
                             "model": workers[url].get(service)} for url in self.candidates(service)]
            out[service].append({"name": self._name, "url": None, "score": own.get(service, 0), "model": "this node"})
        return out

    def _report(self):
        """Log when the first choice for a service changes, rather than on every call."""
        for service in SERVICES:
            urls = self.candidates(service)
            best = urls[0] if urls else None
            if best != self._best.get(service, "unset"):
                self._best[service] = best
                if best is None:
                    self._log(f"offload: {service} runs on this node")
                else:
                    with self._lock:
                        caps = self._workers.get(best, {})
                    self._log(f"offload: {service} goes to {caps.get('name', best)} "
                              f"({caps.get(service)}, score {caps.get('scores', {}).get(service)})")


def derive_score(device: str) -> int:
    """A rough speed for a model on `device` here: any gpu beats any cpu, then more cores win."""
    return (GPU_SCORE if device in GPU_DEVICES else 0) + (os.cpu_count() or 1)
