"""sleep_probe arms, without weights (mock Cortex / PlasticCortex / sleep)."""
import json
import sys

import pytest
import torch

from brain.experiments import sleep_probe as sp
from brain.experiments.facts import FACTS


def test_varied_episodes():
    arms = sp.build_arm_episodes(42)
    assert set(arms) == {"no_sleep", "sleep_true", "sleep_varied", "sleep_shuffled"}
    assert arms["no_sleep"] is None
    assert len(arms["sleep_true"]) == len(arms["sleep_shuffled"]) == 12
    assert len(arms["sleep_varied"]) == 36
    texts = [t for t, _ in arms["sleep_varied"]]
    for i, (s, _, k) in enumerate(FACTS):
        assert f"User: {s}\n\n" in texts[3 * i]  # the original statement is one of the three forms
        assert len({texts[3 * i], texts[3 * i + 1], texts[3 * i + 2]}) == 3
        assert all(k in t.lower() for t in texts[3 * i:3 * i + 3])
    # shuffled arm is unchanged by this work: another fact's key replaces the true one
    assert all(a != b for (a, _), (b, _) in zip(arms["sleep_true"], arms["sleep_shuffled"]))


class _Pipe:
    def encode(self, t):
        return [1, 2, 3]

    def decode(self, ids):
        return ""


class _Cortex:
    pipe = _Pipe()

    def __init__(self, **kw):
        pass

    def read(self, text, st):
        return None, [torch.zeros(2)]


class _PC:
    def __init__(self, c):
        self.cortex = c
        self.enabled = True

    def adapter_norm(self):
        return 0.0

    def __call__(self, ids, st):
        return torch.zeros(len(ids), 4), st


def _run(monkeypatch, tmp_path, arms):
    calls = []

    def fake_sleep(pc, base, eps, generic, **kw):
        calls.append((len(eps), kw["steps"]))
        return [(0.5, 0.0)]

    monkeypatch.setattr(sp, "Cortex", _Cortex)
    monkeypatch.setattr(sp, "PlasticCortex", _PC)
    monkeypatch.setattr(sp, "sleep", fake_sleep)
    out = tmp_path / "o.json"
    argv = ["x", "--steps", "7", "--out", str(out)] + (["--arms", arms] if arms else [])
    monkeypatch.setattr(sys, "argv", argv)
    sp.main()
    return json.load(open(out)), calls


def test_arms_flag_and_equal_steps(monkeypatch, tmp_path):
    r, calls = _run(monkeypatch, tmp_path, "no_sleep,sleep_varied")
    assert list(r["arms"]) == ["no_sleep", "sleep_varied"]
    assert calls == [(36, 7)]
    r, calls = _run(monkeypatch, tmp_path, None)
    assert list(r["arms"]) == ["no_sleep", "sleep_true", "sleep_varied", "sleep_shuffled"]
    assert [s for _, s in calls] == [7, 7, 7]


def test_unknown_arm(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, "sleep_bogus")
