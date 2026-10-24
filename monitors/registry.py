"""Build monitors from a frozen monitor config. The in-loop role is guarded against Jev."""
from __future__ import annotations

from pathlib import Path

import yaml

from monitors.anyjev import AnyJevMonitor
from monitors.base import Monitor, QuestionConfig, assert_in_loop_allowed
from monitors.cache import ScoreCache
from monitors.systemone import ClefMonitor, JevMonitor

ROLES = ("in_loop", "audit", "offline")


def build_monitor(spec: dict, question: QuestionConfig, cache: ScoreCache | None,
                  role: str) -> Monitor:
    """spec: {"type": "clef"|"jev"|"anyjev", ...type kwargs}. role "in_loop" routes or penalizes."""
    if role not in ROLES:
        raise ValueError(role)
    kind = spec["type"]
    kw = {k: v for k, v in spec.items() if k != "type"}
    if kind == "clef":
        m = ClefMonitor(question=question, cache=cache, **kw)
    elif kind == "jev":
        m = JevMonitor(question=question, cache=cache, **kw)
    elif kind == "anyjev":
        m = AnyJevMonitor(question=question, cache=cache, **kw)
    else:
        raise ValueError(f"unknown monitor type {kind}")
    if role == "in_loop":
        assert_in_loop_allowed(m)
    return m


def load_frozen(path: str | Path, cache_path: str | Path | None, role: str) -> Monitor:
    """A frozen config file holds the question, tau and the selected monitor spec."""
    d = yaml.safe_load(Path(path).read_text())
    q = QuestionConfig.load(path)
    if role == "in_loop" and q.tau is None:
        raise ValueError(f"{path} has no frozen tau")
    return build_monitor(d["monitor"], q, ScoreCache(cache_path) if cache_path else None, role)
