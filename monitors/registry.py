"""Build monitors from a frozen monitor config. The in-loop role is guarded against Jev."""
from __future__ import annotations

from pathlib import Path

import yaml

from monitors.anyjev import AnyJevMonitor
from monitors.base import Monitor, QuestionConfig, assert_in_loop_allowed
from monitors.cache import ScoreCache
from monitors.openrouter import OpenRouterMonitor
from monitors.systemone import ClefMonitor, JevMonitor
from monitors.views import VIEWS

ROLES = ("in_loop", "audit", "offline")


def build_monitor(spec: dict, question: QuestionConfig, cache: ScoreCache | None,
                  role: str) -> Monitor:
    """spec: {"type": "clef"|"jev"|"anyjev"|"openrouter", ...type kwargs}. role "in_loop" routes or penalizes."""
    if role not in ROLES:
        raise ValueError(role)
    kind = spec["type"]
    kw = {k: v for k, v in spec.items() if k != "type"}
    if kind == "clef":
        m = ClefMonitor(question=question, cache=cache, **kw)
    elif kind == "jev":
        m = JevMonitor(question=question, cache=cache, **kw)
    elif kind == "openrouter":
        m = OpenRouterMonitor(question=question, cache=cache, **kw)
    elif kind == "anyjev":
        m = AnyJevMonitor(question=question, cache=cache, **kw)
    else:
        raise ValueError(f"unknown monitor type {kind}")
    if role == "in_loop":
        assert_in_loop_allowed(m)
    return m


def load_frozen(path: str | Path, cache_path: str | Path | None, role: str,
                relay_url: str | None = None) -> Monitor:
    """A frozen config file holds the question, tau, the selected monitor spec and the view.
    With `relay_url` the monitor runs on the laptop behind monitors.relay (no keys on the Pod)."""
    d = yaml.safe_load(Path(path).read_text())
    q = QuestionConfig.load(path)
    if role == "in_loop" and q.tau is None:
        raise ValueError(f"{path} has no frozen tau")
    if relay_url:
        from monitors.relay import RelayMonitor
        return RelayMonitor(question=q, url=relay_url)
    return build_monitor(d["monitor"], q, ScoreCache(cache_path) if cache_path else None, role)


def frozen_view(path: str | Path) -> str:
    """Which part of the completion the frozen monitor sees (monitors.views); "full" if unset."""
    v = yaml.safe_load(Path(path).read_text()).get("view", "full")
    if v not in VIEWS:
        raise ValueError(f"{path}: unknown view {v!r}")
    return v
