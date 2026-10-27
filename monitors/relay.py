"""Run a hosted in-loop monitor from the laptop, so no monitor API key ever reaches a Pod.

The laptop serves the frozen monitor (keys from its own .env, score cache on its disk); the Pod
reaches it through an SSH reverse tunnel (`ssh -R 8200:127.0.0.1:8200`) and scores through
`RelayMonitor`. The server refuses requests whose question hash differs from its frozen config, so
both ends provably use the same question and tau.

    python -m monitors.relay --config configs/monitor/frozen_cot_v1.yaml --cache cache/relay.sqlite
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from monitors.base import Monitor


@dataclass
class RelayMonitor(Monitor):
    """Client side. A tunnel outage is retried for up to `wait_s`; after that the call raises, so
    training stops instead of running without its monitor."""
    url: str = "http://127.0.0.1:8200"
    wait_s: float = 1200.0
    timeout_s: float = 900.0

    def __post_init__(self):
        d = self._call("GET", "/describe")
        if d["question_hash"] != self.question.text_hash:
            raise ValueError(f"relay serves question {d['question_hash']}, "
                             f"config has {self.question.text_hash}")
        if self.question.tau is not None and d["tau"] != self.question.tau:
            raise ValueError(f"relay tau {d['tau']} != config tau {self.question.tau}")
        self.remote = d
        self.name = d["name"]
        self.backbone = d["backbone"]

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        import httpx
        t0, delay = time.time(), 2.0
        while True:
            try:
                r = httpx.request(method, self.url + path, json=body, timeout=self.timeout_s)
                if r.status_code == 409:
                    raise ValueError(r.text)
                r.raise_for_status()
                return r.json()
            except ValueError:
                raise
            except Exception as exc:
                if time.time() - t0 > self.wait_s:
                    raise RuntimeError(f"monitor relay unreachable for {self.wait_s:.0f}s: {exc}") from exc
                self.stats["retries"] += 1
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    def _score_batch(self, states):
        raise NotImplementedError("RelayMonitor scores (prompt, completion) pairs via score()")

    def score(self, prompts: list[str], completions: list[str]) -> list[float]:
        assert len(prompts) == len(completions)
        t0 = time.time()
        d = self._call("POST", "/score", {"prompts": prompts, "completions": completions,
                                          "question_hash": self.question.text_hash})
        self.stats["calls"] += 1
        self.stats["items"] += len(prompts)
        self.stats["seconds"] += time.time() - t0
        scores = [float(s) for s in d["scores"]]
        if len(scores) != len(prompts) or not all(0.0 <= s <= 1.0 for s in scores):
            raise ValueError("relay returned malformed scores")
        return scores

    def describe(self) -> dict:
        return {**self.remote, "transport": "relay"}


def make_server(monitor: Monitor, host: str, port: int) -> ThreadingHTTPServer:
    lock = threading.Lock()   # one batch at a time; the monitor parallelises inside it

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path != "/describe":
                return self._send(404, {"error": "not found"})
            self._send(200, {**monitor.describe(), "question_hash": monitor.question.text_hash,
                             "stats": monitor.stats})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req.get("question_hash") != monitor.question.text_hash:
                return self._send(409, {"error": "question hash mismatch"})
            t0 = time.time()
            try:
                with lock:
                    scores = monitor.score(req["prompts"], req["completions"])
            except Exception as exc:   # surfaced to the client, which retries then gives up
                return self._send(502, {"error": repr(exc)[:500]})
            print(f"[relay] {len(scores)} scored in {time.time() - t0:.1f}s "
                  f"(flag rate {sum(s > (monitor.question.tau or 1.0) for s in scores) / max(len(scores), 1):.3f})",
                  flush=True)
            self._send(200, {"scores": scores, "seconds": time.time() - t0})

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    from monitors.registry import load_frozen
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    m = load_frozen(a.config, a.cache, role="in_loop")
    m.max_workers = a.workers
    print(f"[relay] serving {m.describe()} on 127.0.0.1:{a.port}", flush=True)
    make_server(m, "127.0.0.1", a.port).serve_forever()


if __name__ == "__main__":
    main()
