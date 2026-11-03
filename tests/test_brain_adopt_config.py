"""Cortex size as configuration, per-home cortex record, home lock, episode storage on CPU and per file, memory
accounting, brain.night, weekly --shared-sleep, tools/brain_gpu.sh. No weights needed (mock cortex from
test_brain_sleep). brain/research/2026-11-03-adopt-1.5b-plan.md steps 3, 5, 7, 8, 11."""
import json
import os
import subprocess

import pytest
import torch

import brain.brain as bb
import brain.cortex as bc
from brain.brain import ADAPTERS, EPISODE_INDEX, EPISODES, BrainBusy, CortexMismatch, sleep_defaults
from brain.hippocampus import Hippocampus, episode_bytes_for
from tests.test_brain_sleep import _brain, cortex  # noqa: F401  (fixture)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MiB = 1024 * 1024
NAME_01 = "rwkv7-g1d-0.1b-20260129-ctx8192"
NAME_15 = "rwkv7-g1l-1.5b-20261031-ctx34304"


# ---- step 3: the model is configuration ----
def test_default_cortex_stays_01b():
    assert bc.model_size(bc.DEFAULT_MODEL) == "0.1b" or "BRAIN_CORTEX_MODEL" in os.environ


def test_resolve_model(tmp_path, monkeypatch):
    for f in (NAME_01, "rwkv7-g1d-0.4b-20260210-ctx8192", NAME_15):
        (tmp_path / f"{f}.pth").write_bytes(b"")
    monkeypatch.setattr(bc, "MODEL_DIR", str(tmp_path))
    assert bc.resolve_model("1.5b") == str(tmp_path / f"{NAME_15}.pth")
    assert bc.resolve_model("0.4B").endswith("0.4b-20260210-ctx8192.pth")
    assert bc.resolve_model("/x/y/m") == "/x/y/m.pth" and bc.resolve_model("m.pth") == "m.pth"
    assert bc.resolve_model(None) == bc.DEFAULT_MODEL
    with pytest.raises(ValueError):
        bc.resolve_model("7b")
    (tmp_path / "other-1.5b-x.pth").write_bytes(b"")
    with pytest.raises(ValueError):          # two checkpoints of that size: ambiguous, never a silent pick
        bc.resolve_model("1.5b")
    assert bc.model_name("/a/b/" + NAME_15 + ".pth") == NAME_15


def test_home_records_its_cortex_and_refuses_another(tmp_path, cortex):  # noqa: F811
    cortex.model_name = NAME_01
    b = _brain(tmp_path, cortex)
    b.hear("My cat is called Pixel.")
    b.save()
    assert json.load(open(tmp_path / "meta.json"))["cortex_model"] == NAME_01
    cortex.model_name = NAME_15
    with pytest.raises(CortexMismatch):
        _brain(tmp_path, cortex)
    cortex.model_name = NAME_01
    assert len(_brain(tmp_path, cortex, sleep_on_restart=False).hippocampus.episodes) == 4


def test_home_older_than_the_record_is_checked_by_trace_shape(tmp_path, cortex):  # noqa: F811
    b = _brain(tmp_path, cortex)
    b.hear("My cat is called Pixel.")
    b.save()
    meta = json.load(open(tmp_path / "meta.json"))
    meta.pop("cortex_model")
    json.dump(meta, open(tmp_path / "meta.json", "w"))
    assert _brain(tmp_path, cortex, sleep_on_restart=False).meta["cortex_model"] is None   # same shapes: loads
    idx = json.load(open(tmp_path / EPISODE_INDEX))
    f = tmp_path / EPISODES / idx[0]["file"]
    blob = torch.load(f, weights_only=False)
    blob["delta"] = [None if d is None else torch.zeros(4, 64, 64, dtype=torch.half) for d in blob["delta"]]
    torch.save(blob, f)
    with pytest.raises(CortexMismatch):
        _brain(tmp_path, cortex)


# ---- step 8: sleep settings per model, no silent 0.1B values ----
def test_sleep_defaults_table():
    assert sleep_defaults(NAME_15) == {"steps": 480, "lr": 5e-4, "kl_weight": 3.0, "rank": 8}
    assert sleep_defaults("rwkv7-g1d-0.4b-20260210-ctx8192") == {"steps": 480, "lr": 1e-3, "kl_weight": 3.0, "rank": 8}
    assert sleep_defaults(NAME_01) == {"steps": 240, "lr": 5e-4, "kl_weight": 1.0, "rank": 8}  # integration default
    with pytest.raises(KeyError):
        sleep_defaults("rwkv7-g1-2.9b-x")
    with pytest.raises(KeyError):
        sleep_defaults(None)


def test_brain_takes_sleep_settings_from_its_cortex_size(tmp_path, cortex):  # noqa: F811
    cortex.model_name = NAME_15
    b = bb.Brain(str(tmp_path / "a"), cortex=cortex)
    assert b.sleep_cfg == {"steps": 480, "lr": 5e-4, "kl_weight": 3.0} and b.adapter_rank == 8
    b = bb.Brain(str(tmp_path / "b"), cortex=cortex, sleep_steps=7)       # an explicit value wins
    assert b.sleep_cfg["steps"] == 7 and b.sleep_cfg["kl_weight"] == 3.0
    del cortex.model_name
    with pytest.raises(KeyError):                                         # unknown size, settings missing
        bb.Brain(str(tmp_path / "c"), cortex=cortex)


# ---- step 7: one process per home ----
def test_home_lock(tmp_path, cortex):  # noqa: F811
    a = _brain(tmp_path, cortex, lock=True)
    with pytest.raises(BrainBusy):
        _brain(tmp_path, cortex, lock=True)
    _brain(tmp_path, cortex)            # an unlocked reader (tests, experiments) is not refused
    a.close()
    _brain(tmp_path, cortex, lock=True).release()


# ---- step 5: episodes on CPU, one file each, written once ----
def test_episodes_live_on_cpu_and_are_written_once(tmp_path, cortex):  # noqa: F811
    b = _brain(tmp_path, cortex, sleep_on_restart=False)
    assert all(d is None or d.device.type == "cpu" for e in b.hippocampus.episodes for d in e.delta)
    files = sorted(os.listdir(tmp_path / EPISODES))
    assert len(files) == 3 and not os.path.exists(tmp_path / "memory.pt")
    stamp = {f: os.stat(tmp_path / EPISODES / f).st_mtime_ns for f in files}
    b.hear("My cat is called Pixel.")
    b.hear("What is my cat called?")
    b.save()
    after = sorted(os.listdir(tmp_path / EPISODES))
    assert len(after) == 4 and all(os.stat(tmp_path / EPISODES / f).st_mtime_ns == t for f, t in stamp.items())
    idx = json.load(open(tmp_path / EPISODE_INDEX))
    b2 = _brain(tmp_path, cortex, sleep_on_restart=False)
    for e, e2, row in zip(b.hippocampus.episodes, b2.hippocampus.episodes, idx):
        assert e.text == e2.text == row["text"] and e.uses == e2.uses and e.strength == e2.strength
        assert all((d is None and d2 is None) or torch.equal(d, d2) for d, d2 in zip(e.delta, e2.delta))
        assert torch.equal(e.key, e2.key)
    old = b2.hippocampus.episodes[3]
    old.key = old.key * 2                                     # a replaced key (a re-keyed hippocampus) is rewritten
    b2.save()
    assert torch.equal(torch.load(tmp_path / EPISODES / idx[3]["file"], weights_only=False)["key"], old.key)


def test_older_single_file_home_loads_and_migrates(tmp_path, cortex):  # noqa: F811
    b = _brain(tmp_path, cortex, sleep_on_restart=False)
    b.hear("I drive a red Volvo.")
    eps = [{k: getattr(e, k) for k in ("text", "delta", "key", "strength", "created", "last_used", "uses",
                                         "permanent")} for e in b.hippocampus.episodes]
    old = tmp_path / "old"
    old.mkdir()
    torch.save({"episodes": eps, "state": b.state}, old / "memory.pt")
    json.dump(b.meta, open(old / "meta.json", "w"))
    b2 = _brain(old, cortex, sleep_on_restart=False)
    assert [e.text for e in b2.hippocampus.episodes] == [e.text for e in b.hippocampus.episodes]
    b2.save()
    assert len(os.listdir(old / EPISODES)) == 4
    assert [e.text for e in _brain(old, cortex, sleep_on_restart=False).hippocampus.episodes][-1] == \
        "User: I drive a red Volvo."


def test_adapters_written_only_after_a_night(tmp_path, cortex):  # noqa: F811
    b = _brain(tmp_path, cortex)
    b.hear("My cat is called Pixel.")
    b.sleep()
    b.save()
    t = os.stat(tmp_path / ADAPTERS).st_mtime_ns
    b.hear("Hello there.")
    b.save()
    assert os.stat(tmp_path / ADAPTERS).st_mtime_ns == t


# ---- memory accounting ----
def test_episode_bytes_match_the_plan_table():
    assert round(episode_bytes_for(12, 12, 64) / MiB, 1) == 1.3     # 0.1B
    assert episode_bytes_for(24, 16, 64) / MiB == pytest.approx(3.25)   # 0.4B (plan: 3.3)
    assert round(episode_bytes_for(24, 32, 64) / MiB, 1) == 6.5     # 1.5B


class _Shaped:
    """A cortex of 1.5B state shape (24 layers, 32 heads x 64) without weights: read writes a random trace."""
    L, H, N, C = 24, 32, 64, 2048

    def base(self):
        return [torch.zeros(self.H, self.N, self.N) if i % 3 == 1 else torch.zeros(self.C) for i in range(3 * self.L)]

    def read(self, text, state):
        g = torch.Generator().manual_seed(len(text))
        return None, [s + torch.randn(s.shape, generator=g) for s in state]

    matrix_delta = bc.Cortex.matrix_delta
    key_of = staticmethod(bc.Cortex.key_of)
    add_delta = staticmethod(bc.Cortex.add_delta)


def test_15b_sized_episode_lives_on_cpu_with_the_planned_size():
    c, h = _Shaped(), Hippocampus()
    base = c.base()
    ep = h.store(c, base, "My cat is called Pixel.")
    assert all(d.device.type == "cpu" and d.dtype == torch.float16 for d in ep.delta if d is not None)
    assert ep.key.device.type == "cpu"
    assert h.nbytes() == {"cpu": episode_bytes_for(24, 32, 64)}
    st = h.reinstate(c, base, ep)
    assert st[1].dtype == torch.float32 and torch.allclose(st[1], 4.0 * ep.delta[1].float())


def test_memory_report(tmp_path, cortex):  # noqa: F811
    b = _brain(tmp_path, cortex)
    b.hear("My cat is called Pixel.")
    b.sleep()
    b.save()
    r = b.memory_report()
    assert r["episodes"] == 4 and set(r["episode_bytes"]) == {"cpu"} and r["adapter_bytes"] > 0
    assert r["bytes_per_episode"] * 4 == sum(r["episode_bytes"].values()) and r["home_disk_bytes"] > 0


# ---- step 7: the night as its own process ----
def test_night_sleeps_once_and_only_when_something_was_lived(tmp_path, cortex):  # noqa: F811
    from brain import night
    kw = dict(cortex=cortex, sleep_steps=4, sleep_lr=1e-2, sleep_kl_weight=1.0, adapter_rank=2)
    with pytest.raises(FileNotFoundError):
        night.run(str(tmp_path / "nobody"), **kw)
    b = _brain(tmp_path, cortex, sleep_on_restart=False)
    b.hear("My cat is called Pixel.")
    b.close()
    out = night.run(str(tmp_path), **kw)
    assert out["slept"] and out["pending"] == 1 and out["night"] == 1 and out["memory"]["episodes"] == 4
    assert os.path.exists(tmp_path / ADAPTERS)
    out = night.run(str(tmp_path), **kw)
    assert not out["slept"] and out["night"] == 1
    chat = _brain(tmp_path, cortex, sleep_on_restart=False, lock=True)   # a chat holds the home
    with pytest.raises(BrainBusy):
        night.run(str(tmp_path), **kw)
    chat.close()


# ---- step 11: weekly options ----
def test_weekly_shared_sleep_sleeps_once(tmp_path, cortex, monkeypatch):  # noqa: F811
    from brain.eval import weekly
    cortex.model_name = NAME_01
    monkeypatch.setitem(bb.SLEEP_DEFAULTS, "0.1b", {"steps": 2, "lr": 1e-2, "kl_weight": 1.0, "rank": 2})
    nights = []
    real = bb.Brain.sleep
    monkeypatch.setattr(bb.Brain, "sleep", lambda self, *a, **k: (nights.append(self.home), real(self, *a, **k))[1])
    out = tmp_path / "t.md"
    lines = weekly.main(["--out", str(out), "--homes", str(tmp_path / "homes"), "--shared-sleep"], cortex=cortex)
    assert len(nights) == 1
    text = out.read_text()
    assert NAME_01 in text and "one shared night" in text
    assert text.count("### session 2") == 2 and "session 1 (shared by both arms)" in text
    assert sum("**you:**" in line for line in lines) == len(weekly.SESSION_1) + 2 * len(weekly.SESSION_2)
    nights.clear()
    weekly.main(["--out", str(out), "--homes", str(tmp_path / "homes2")], cortex=cortex)
    assert len(nights) == 2                                   # the default protocol is unchanged: each arm sleeps
    assert "_cortex:" not in out.read_text()


# ---- tools/brain_gpu.sh ----
def _gpu(*args):
    return subprocess.run(["bash", os.path.join(REPO, "tools", "brain_gpu.sh"), *args], capture_output=True, text=True)


def test_brain_gpu_dry_run_lines():
    r = _gpu("night", "--home", "/h/my home", "--model", "1.5b", "--dry-run")
    assert r.returncode == 0, r.stderr
    line = r.stdout.strip()
    assert "tools/local_llm/llm.sh run bash tools/memcap.sh 16 --" in line
    assert "-m brain.night --device cuda --home /h/my\\ home --model 1.5b" in line
    r = _gpu("weekly", "--model", "1.5b", "--shared-sleep", "--dry-run")
    assert r.returncode == 0 and "-m brain.eval.weekly --device cuda --model 1.5b --shared-sleep" in r.stdout
    assert _gpu("night", "--dry-run").returncode == 2           # a night needs a home
    assert _gpu("bogus").returncode == 2
