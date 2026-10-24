"""AnyJev scoring server. Runs in the separate anyjev venv, in front of a stock `vllm serve`.

    $WS/venvs/anyjev/bin/python -m monitors.anyjev_server --kind tacit \
        --model morriszjm/Tacit-9B --upstream http://127.0.0.1:8000 --port 8100

POST /score {"states": [...], "question": "..."} -> {"p_yes": [...], "model": ..., "level": ...}
Listens on 127.0.0.1 only; no auth. Escalation to reasoning (Tacit adaptive) is off so
one decision is one deterministic prefill.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def build_scorer(kind: str, model: str, upstream: str, tokenizer: str | None):
    import anyjev

    if kind == "tacit":
        from anyjev import Tacit

        t = Tacit.from_pretrained(model, engine="server", base_url=upstream, adaptive=False)

        def score(states, question):
            res = t.decide_batch([dict(state=s, question=question, kind="yes_no") for s in states])
            return [float(r["probs"]["Yes"]) for r in res], "tacit-one_forward"
    elif kind == "decider":
        from anyjev import Decider, Question
        from anyjev.backends.vllm import VLLMBackend

        d = Decider(VLLMBackend(upstream, model, tokenizer_name=tokenizer))

        def score(states, question):
            q = Question.noul(question, name="hack")
            res = d.decide_batch(states, q, require="L0")
            return [r.p_true for r in res], "L0"
    else:
        raise ValueError(kind)
    return score, anyjev.__version__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tacit", "decider"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    score, version = build_scorer(a.kind, a.model, a.upstream, a.tokenizer)
    lock = threading.Semaphore(4)
    info = {"model": a.model, "kind": a.kind, "anyjev_version": version}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, info)

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                with lock:
                    p, level = score(req["states"], req["question"])
                self._send(200, {"p_yes": p, "level": level, **info})
            except Exception as exc:  # surfaced to the client, which retries
                self._send(500, {"error": repr(exc)[:500]})

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
