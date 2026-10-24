"""Client for monitors.anyjev_server (AnyJev Tacit or training-free Decider on a local GPU)."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from monitors.base import Monitor


@dataclass
class AnyJevMonitor(Monitor):
    model: str = "morriszjm/Tacit-9B"
    kind: str = "tacit"                    # "tacit" | "decider"
    url: str = "http://127.0.0.1:8100"
    batch_size: int = 32
    max_workers: int = 4
    timeout_s: float = 300.0

    def __post_init__(self):
        self.name = f"anyjev-{self.kind}-{self.model.split('/')[-1]}"
        self.backbone = self.backbone if self.backbone != "unknown" else self.model

    def _score_batch(self, states):
        r = httpx.post(self.url + "/score", json={"states": states, "question": self.question.question},
                       timeout=self.timeout_s)
        if r.status_code >= 500:
            raise RuntimeError(f"{self.name} server error: {r.text[:300]}")
        r.raise_for_status()
        d = r.json()
        return d["p_yes"], {"model": d["model"], "kind": d["kind"], "level": d["level"],
                            "anyjev_version": d["anyjev_version"]}
