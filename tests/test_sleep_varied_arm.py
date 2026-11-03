"""sleep_varied arm: three surface forms per fact, --arms selection, same total replay steps. No model weights."""
import sys
import types

import torch
import pytest

try:  # brain.cortex imports the rwkv package at module level; the tests never load a model, so a stub is enough
    import rwkv.model  # noqa: F401
except ImportError:
    for _n in ("rwkv", "rwkv.model", "rwkv.utils"):
        sys.modules[_n] = types.ModuleType(_n)
    sys.modules["rwkv.model"].RWKV = object
    sys.modules["rwkv.utils"].PIPELINE = object

from brain.experiments import sleep_probe  # noqa: E402
from brain.experiments.facts import FACTS, PARA  # noqa: E402
from brain.experiments.facts_rewordings import REWORDINGS  # noqa: E402

STOP = {"a", "an", "the", "my", "of", "in", "at", "to", "is", "was", "are", "am", "i", "i'm", "it", "this",
        "that", "there", "any", "do", "does", "which", "what", "what's", "who", "who's", "where", "when",
        "can't", "im", "whats", "whos"}


def _content(text):
    return [w for w in "".join(c if c.isalpha() or c in "' " else " " for c in text.lower()).split()
            if w not in STOP and w.strip("'") not in STOP]


def _runs(words, n=3):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class _Pipe:
    def __init__(self):
        self.vocab = {}

    def encode(self, text):
        return [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


class FakeCortex:
    def __init__(self, model_path=None, device="cpu"):
        self.pipe = _Pipe()

    def read(self, text, state=None):
        return torch.tensor([float(i) for i in self.pipe.encode(text)]), [torch.zeros(4), torch.zeros(2), torch.zeros(4)]


class FakePlastic:
    def __init__(self, cortex):
        self.cortex = cortex
        self.enabled = True

    def __call__(self, ids, state):
        return torch.zeros(len(ids), 5), state

    def adapter_norm(self):
        return 0.0


def _fake_sleep(plastic, base, episodes, generic, *, steps, lr, kl_weight, seed, log):
    _fake_sleep.calls.append({"episodes": episodes, "steps": steps})
    return [(1.0, 1.0)]


_fake_sleep.calls = []


@pytest.fixture()
def run_main(monkeypatch, tmp_path):
    """Run sleep_probe.main() with mocked Cortex/PlasticCortex/sleep; returns (parsed json, sleep calls)."""
    import json

    def _run(argv):
        _fake_sleep.calls = []
        monkeypatch.setattr(sleep_probe, "Cortex", FakeCortex)
        monkeypatch.setattr(sleep_probe, "PlasticCortex", FakePlastic)
        monkeypatch.setattr(sleep_probe, "sleep", _fake_sleep)
        out = tmp_path / "out.json"
        monkeypatch.setattr(sys, "argv", ["sleep_probe", "--out", str(out)] + argv)
        sleep_probe.main()
        return json.loads(out.read_text()), list(_fake_sleep.calls)

    return _run


# --- facts_rewordings.py: shape, key words, and no PARA question wording -------------------------

def test_two_rewordings_per_fact_in_order():
    assert len(REWORDINGS) == len(FACTS)
    for (statement, _, _), (r1, r2) in zip(FACTS, REWORDINGS):
        assert r1 != statement and r2 != statement and r1 != r2


def test_rewordings_keep_the_key_word():
    for (_, _, key), forms in zip(FACTS, REWORDINGS):
        for r in forms:
            assert key in r.lower(), f"{r!r} lost the key {key!r}"


def test_no_para_question_content_run_of_three_appears_in_any_rewording():
    para_runs = {q: _runs(_content(q)) for q in PARA}
    for q, runs in para_runs.items():
        for r in [x for pair in REWORDINGS for x in pair]:
            for run in runs:
                assert run not in _runs(_content(r)), f"{run} from PARA question {q!r} appears in {r!r}"


def test_rewordings_do_not_reuse_a_para_question_verbatim():
    for q in PARA:
        for r in [x for pair in REWORDINGS for x in pair]:
            assert q.lower() not in r.lower()


# --- sleep_varied arm ---------------------------------------------------------------------------

def test_varied_episodes_three_forms_per_fact():
    eps = sleep_probe.varied_episodes()
    assert len(eps) == 3 * len(FACTS)
    for i, ((s, _, _), (r1, r2)) in enumerate(zip(FACTS, REWORDINGS)):
        forms = [s, r1, r2]
        for j, text in enumerate(forms):
            ep_text, strength = eps[3 * i + j]
            assert text in ep_text
            assert ep_text.startswith("User: ") and ep_text.endswith("Assistant: Okay, I'll remember that.\n\n")
            assert strength == 1.0


def test_varied_first_form_is_the_true_text():
    eps = sleep_probe.varied_episodes()
    true_texts = [f"User: {s}\n\nAssistant: Okay, I'll remember that.\n\n" for s, _, _ in FACTS]
    assert [eps[3 * i][0] for i in range(len(FACTS))] == true_texts


def test_varied_same_total_replay_steps_as_true(run_main):
    result, calls = run_main(["--seed", "7", "--steps", "240"])
    sleep_keys = [k for k in result["arms"] if k != "no_sleep"]
    by_arm = dict(zip(sleep_keys, calls))
    assert set(by_arm) == {"sleep_true", "sleep_shuffled", "sleep_varied"}  # no_sleep never calls sleep()
    assert by_arm["sleep_varied"]["steps"] == by_arm["sleep_true"]["steps"] == 240
    assert len(by_arm["sleep_varied"]["episodes"]) == 3 * len(by_arm["sleep_true"]["episodes"])


def test_default_runs_all_four_arms(run_main):
    result, calls = run_main([])
    assert list(result["arms"]) == ["no_sleep", "sleep_true", "sleep_shuffled", "sleep_varied"]
    assert len(calls) == 3  # one sleep per sleep-arm


def test_arms_flag_skips_arms(run_main):
    result, calls = run_main(["--arms", "no_sleep", "sleep_varied"])
    assert list(result["arms"]) == ["no_sleep", "sleep_varied"]
    assert len(calls) == 1
    assert len(calls[0]["episodes"]) == 3 * len(FACTS)


def test_arms_flag_rejects_unknown_arm(run_main):
    with pytest.raises(SystemExit):
        run_main(["--arms", "sleep_nope"])