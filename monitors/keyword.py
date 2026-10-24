"""The paper's Countdown monitor: flags trivial-test (test_overwrite) hacks, blind to the rest.

It is a behavioural probe (the model's verifier accepts a value-wrong expression),
the same probe the ground-truth labeler uses for trivial_test, so its recall on
that type is 1 by construction. In RL runs P1/P2 use GRAFT's native
`score_threshold` detector on `countdown_hack_test_overwrite_from_all` instead,
so the reproduction runs the paper's own code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from monitors.base import Monitor


@dataclass
class KeywordMonitor(Monitor):
    # (prompts, completions) -> labeler rows; injected so this module never runs model code itself.
    labeler: Callable[[list[str], list[str]], list[dict]] | None = field(default=None, repr=False)

    def __post_init__(self):
        self.name = "keyword-test_overwrite"
        self.backbone = "programmatic"

    def score(self, prompts, completions):
        assert self.labeler is not None, "KeywordMonitor needs a labeler (Pod only)"
        return [1.0 if row["flags"]["trivial_test"] else 0.0 for row in self.labeler(prompts, completions)]

    def _score_batch(self, states):
        raise NotImplementedError
