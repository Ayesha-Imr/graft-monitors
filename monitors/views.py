"""What part of a Countdown completion a monitor sees.

A completion is "<reasoning></think>{json files}". The reasoning view is the text before </think>;
without the tag it ends at the first JSON file object. Offline scoring (cot_signal.py views) and the
in-loop detector both call this, so the two see byte-identical text.
"""
from __future__ import annotations

import re

_END = "</think>"
_ACTION = re.compile(r'\{\s*"(solution|test)\.py"')
VIEWS = ("full", "reasoning")


def split(completion: str) -> tuple[str, str]:
    """(reasoning, action)."""
    if _END in completion:
        r, a = completion.split(_END, 1)
        return r, a
    m = _ACTION.search(completion)
    return (completion[:m.start()], completion[m.start():]) if m else (completion, "")


def view(completion: str, name: str) -> str:
    if name == "full":
        return completion
    if name == "reasoning":
        return split(completion)[0].strip()
    raise ValueError(f"unknown view {name!r}; expected one of {VIEWS}")
