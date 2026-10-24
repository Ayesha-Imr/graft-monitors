"""Hosted SystemOne-API decision models: Clef / Clef-flash (Workers AI) and Jev (Typesafe).

Both take {"state", "questions": {name: {"type": "noul", "instructions", "criteria"?}}}
and return answers[name].noul = P(yes). One state per request; concurrency
comes from the thread pool in Monitor.score.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from monitors.base import Monitor

_Q = "hack"
_KEEP_HEADERS = ("x-typesafe-request-id", "cf-ray", "cf-ai-req-id", "x-model-version", "date")


def _question_body(m: Monitor) -> dict:
    q = {"type": "noul", "instructions": m.question.question}
    if m.question.criteria_true or m.question.criteria_false:
        q["criteria"] = {k: v for k, v in (("true", m.question.criteria_true),
                                           ("false", m.question.criteria_false)) if v}
    return {_Q: q}


class _HTTPMonitor(Monitor):
    timeout_s: float = 60.0

    def _client(self) -> httpx.Client:
        c = getattr(self, "_http", None)
        if c is None:
            c = httpx.Client(timeout=self.timeout_s, http2=False,
                             limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
            self._http = c
        return c

    def _post(self, url: str, headers: dict, body: dict) -> tuple[dict, dict]:
        r = self._client().post(url, headers=headers, json=body)
        if r.status_code == 429 or r.status_code >= 500:
            raise RuntimeError(f"{self.name} transient HTTP {r.status_code}")
        if r.status_code >= 400:
            # Validation errors are not retried by design: they would repeat.
            raise ValueError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
        kept = {k: r.headers[k] for k in _KEEP_HEADERS if k in r.headers}
        return r.json(), kept


@dataclass
class ClefMonitor(_HTTPMonitor):
    model: str = "clef-flash"   # or "clef"
    batch_size: int = 1
    max_workers: int = 16

    def __post_init__(self):
        self.name = f"cf-{self.model}"
        self.backbone = {"clef-flash": "Qwen3.5-9B", "clef": "Qwen3.8-27B"}[self.model]
        self._acct = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        self._token = os.environ["CLOUDFLARE_API_TOKEN"]

    def _score_batch(self, states):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._acct}/ai/run/@cf/cloudflare/{self.model}"
        out, metas = [], []
        for s in states:
            data, hdr = self._post(url, {"Authorization": f"Bearer {self._token}"},
                                   {"state": s, "questions": _question_body(self)})
            if not data.get("success", False):
                raise RuntimeError(f"{self.name} unsuccessful: {str(data.get('errors'))[:300]}")
            res = data["result"]
            out.append(float(res["answers"][_Q]["noul"]))
            u = res.get("usage", {})
            self.stats["input_tokens"] += int(u.get("input_tokens", 0))
            metas.append({"model": res.get("model"), **hdr})
        return out, {"model": metas[0].get("model"), "headers": metas[0]}


@dataclass
class JevMonitor(_HTTPMonitor):
    """Audit only. Never used for routing or penalties (asserted in assert_in_loop_allowed)."""
    model: str = "jev-latest"
    batch_size: int = 1
    max_workers: int = 16

    def __post_init__(self):
        self.name = f"typesafe-{self.model}"
        self.backbone = "Jev (closed, non-autoregressive)"
        self.audit_only = True
        self._token = os.environ["TYPESAFE_API_KEY"]

    def _score_batch(self, states):
        out, metas = [], []
        for s in states:
            data, hdr = self._post("https://api.typesafe.ai/v1/systemone",
                                   {"Authorization": f"Bearer {self._token}"},
                                   {"model": self.model, "state": s, "questions": _question_body(self)})
            out.append(float(data["answers"][_Q]["noul"]))
            u = data.get("usage", {})
            self.stats["input_tokens"] += int(u.get("input_tokens", 0))
            metas.append({"model": data.get("model"), **hdr})
        return out, {"model": metas[0].get("model"), "headers": metas[0]}
