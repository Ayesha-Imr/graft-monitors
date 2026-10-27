"""Seed doc §8 test 6 (monitor wrappers) and the Jev in-loop guard. No network."""
import threading
from dataclasses import dataclass

import pytest

from monitors.base import InLoopForbidden, Monitor, QuestionConfig, assert_in_loop_allowed, format_state
from monitors.cache import ScoreCache, cache_key
from monitors.registry import build_monitor

Q = QuestionConfig(version="v1", question="Is it a hack?")


@dataclass
class FakeMonitor(Monitor):
    fail_first: int = 0

    def __post_init__(self):
        self.name = "fake"
        self.seen = []
        self._lock = threading.Lock()

    def _score_batch(self, states):
        with self._lock:
            if self.fail_first > 0:
                self.fail_first -= 1
                raise RuntimeError("transient")
            self.seen.extend(states)
        # deterministic content-dependent score
        return [(sum(map(ord, s)) % 1000) / 1000 for s in states], {"model": "fake-1"}


def data(n=50):
    return [f"task {i % 7}" for i in range(n)], [f"completion {i}" for i in range(n)]


def test_cache_hits_return_identical_scores(tmp_path):
    p, c = data()
    m = FakeMonitor(question=Q, cache=ScoreCache(tmp_path / "c.sqlite"), batch_size=8)
    s1 = m.score(p, c)
    n_calls = len(m.seen)
    s2 = m.score(p, c)
    assert s1 == s2
    assert len(m.seen) == n_calls          # second call fully cached
    m2 = FakeMonitor(question=Q, cache=ScoreCache(tmp_path / "c.sqlite"))
    assert m2.score(p, c) == s1 and m2.seen == []   # survives a new process-level object


def test_batch_vs_single_call_match(tmp_path):
    p, c = data(40)
    batched = FakeMonitor(question=Q, cache=None, batch_size=16).score(p, c)
    single = [FakeMonitor(question=Q, cache=None, batch_size=1).score([a], [b])[0] for a, b in zip(p, c)]
    assert max(abs(x - y) for x, y in zip(batched, single)) <= 1e-4


def test_duplicates_scored_once_and_order_preserved():
    p, c = ["t"] * 5 + ["u"], ["x"] * 5 + ["y"]
    m = FakeMonitor(question=Q, cache=None, batch_size=2)
    s = m.score(p, c)
    assert len(m.seen) == 2
    assert s[:5] == [s[0]] * 5 and s[5] == FakeMonitor(question=Q).score(["u"], ["y"])[0]


def test_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("monitors.base.time.sleep", lambda *_: None)
    m = FakeMonitor(question=Q, cache=None, batch_size=100, fail_first=3)
    assert len(m.score(*data(10))) == 10
    assert m.stats["retries"] == 3


def test_validation_errors_not_retried(monkeypatch):
    monkeypatch.setattr("monitors.base.time.sleep", lambda *_: None)

    @dataclass
    class Bad(Monitor):
        def _score_batch(self, states):
            raise ValueError("HTTP 422")

    m = Bad(question=Q)
    with pytest.raises(ValueError):
        m.score(["a"], ["b"])
    assert m.stats["retries"] == 0


def test_out_of_range_rejected():
    @dataclass
    class Wild(Monitor):
        def _score_batch(self, states):
            return [1.5] * len(states), {}

    with pytest.raises(ValueError):
        Wild(question=Q).score(["a"], ["b"])


def test_question_change_changes_cache_key(tmp_path):
    cache = ScoreCache(tmp_path / "c.sqlite")
    a = FakeMonitor(question=Q, cache=cache)
    a.score(["t"], ["c"])
    b = FakeMonitor(question=QuestionConfig(version="v1", question="Different?"), cache=cache)
    b.score(["t"], ["c"])
    assert len(b.seen) == 1                 # not served from a's cache entry


def test_cache_key_is_unambiguous():
    assert cache_key("m", "q", "ab", "c") != cache_key("m", "q", "a", "bc")


def test_jev_can_never_be_in_loop(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    with pytest.raises(InLoopForbidden):
        build_monitor({"type": "jev"}, Q, None, role="in_loop")
    jev = build_monitor({"type": "jev"}, Q, None, role="audit")
    assert jev.audit_only
    with pytest.raises(InLoopForbidden):
        assert_in_loop_allowed(jev)


def test_clef_allowed_in_loop(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "b")
    m = build_monitor({"type": "clef", "model": "clef-flash"}, Q, None, role="in_loop")
    assert not m.audit_only and m.backbone == "Qwen3.5-9B"


def test_question_v1_loads():
    q = QuestionConfig.load("configs/monitor/question_v1.yaml")
    assert q.question.startswith("Does this response tamper with") and q.tau is None


def test_state_contains_prompt_and_completion():
    s = format_state("PROMPT-X", "COMPLETION-Y")
    assert "PROMPT-X" in s and "COMPLETION-Y" in s


def test_openrouter_p_yes_from_logprobs():
    import math
    from monitors.openrouter import p_yes
    lp = [{"token": "Yes", "logprob": math.log(0.6)}, {"token": " no", "logprob": math.log(0.2)},
          {"token": "Maybe", "logprob": math.log(0.1)}]
    assert abs(p_yes(lp) - 0.75) < 1e-9
    assert p_yes([{"token": "Maybe", "logprob": -0.1}]) is None


def test_openrouter_scores_via_registry(monkeypatch):
    import math
    from monitors import openrouter
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    m = build_monitor({"type": "openrouter", "model": "x/y"}, Q, None, role="in_loop")
    sent = []

    def fake_post(url, headers, body):
        sent.append(body)
        return {"model": "x/y", "choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "Yes", "logprob": math.log(0.3)}, {"token": "No", "logprob": math.log(0.7)}]}]}}],
                "usage": {"prompt_tokens": 11}}, {}

    monkeypatch.setattr(m, "_post", fake_post)
    assert abs(m.score(["t"], ["c"])[0] - 0.3) < 1e-9
    assert sent[0]["logprobs"] and sent[0]["provider"]["require_parameters"] and sent[0]["max_tokens"] == 1
    assert m.name == "or-x/y" and openrouter.URL.startswith("https://openrouter.ai/")


def test_reasoning_view_drops_the_files():
    from monitors.views import view
    c = 'I will change the target.\n</think>\n{"solution.py": "target = 13", "test.py": "..."}'
    assert view(c, "reasoning") == "I will change the target."
    assert view('think {"solution.py": "x"}', "reasoning") == "think"
    assert view(c, "full") == c
    with pytest.raises(ValueError):
        view(c, "code")


def test_frozen_cot_config_is_reasoning_view_with_tau():
    from monitors.registry import frozen_view
    q = QuestionConfig.load("configs/monitor/frozen_cot_v1.yaml")
    assert frozen_view("configs/monitor/frozen_cot_v1.yaml") == "reasoning"
    assert frozen_view("configs/monitor/frozen_v1.yaml") == "full"
    assert q.tau is not None and q.text_hash == QuestionConfig.load("configs/monitor/cot_v1.yaml").text_hash


def _relay(monitor):
    from monitors.relay import make_server
    srv = make_server(monitor, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_relay_matches_direct_scores():
    from monitors.relay import RelayMonitor
    q = QuestionConfig(version="v1", question="Is it a hack?", tau=0.5)
    srv, url = _relay(FakeMonitor(question=q))
    try:
        p, c = data()
        rm = RelayMonitor(question=q, url=url)
        assert rm.name == "fake"
        assert rm.score(p, c) == FakeMonitor(question=q).score(p, c)
    finally:
        srv.shutdown()


def test_relay_rejects_a_different_question():
    from monitors.relay import RelayMonitor
    srv, url = _relay(FakeMonitor(question=QuestionConfig(version="v1", question="Is it a hack?", tau=0.5)))
    try:
        with pytest.raises(ValueError):
            RelayMonitor(question=QuestionConfig(version="v1", question="Other?", tau=0.5), url=url)
        with pytest.raises(ValueError):
            RelayMonitor(question=QuestionConfig(version="v1", question="Is it a hack?", tau=0.9), url=url)
    finally:
        srv.shutdown()


def test_relay_gives_up_when_unreachable():
    from monitors.relay import RelayMonitor
    with pytest.raises(RuntimeError, match="unreachable"):
        RelayMonitor(question=Q, url="http://127.0.0.1:9", wait_s=0.0)
