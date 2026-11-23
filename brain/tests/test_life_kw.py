# Capability: Self and identity
# Record: brain/research/2026-11-23-biggest-defect-read.md
# Expected-fail: none (runner plumbing on the stub cortex; no model weights)
"""brain.experiments.life_kw on the stub cortex of test_life.py: absent extras = the plain life; extras reach every Brain
and are recorded; TEST splits, life-owned keys, unknown keys and mismatched resumes are refused."""
import json

import pytest

import brain.eval.life as life
from brain.experiments import life_kw
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)

TIMING = ("seconds", "live_seconds")


def _untimed(x):
    if isinstance(x, dict):
        return {k: _untimed(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [_untimed(v) for v in x]
    return x


def _argv(tmp_path, name, *more, days=2):
    return ["--seed", "7", "--days", str(days), "--full-probe-nights", "1", "--homes", str(tmp_path / name / "h"),
            "--out", str(tmp_path / name / "o" / "life_s7"), *more]


def _disk(tmp_path, name):
    with open(str(tmp_path / name / "o" / "life_s7") + ".json") as fh:
        return json.load(fh)


def test_absent_extra_is_the_plain_life(tmp_path):
    a = life.main(_argv(tmp_path, "plain"), cortex=StubCortex(), brain_kw=BKW)
    b = life_kw.main(_argv(tmp_path, "wrap"), cortex=StubCortex(), brain_kw=BKW)
    assert "brain_kw_extra" not in b
    da, db = _disk(tmp_path, "plain"), _disk(tmp_path, "wrap")
    # everything but wall-clock timings is equal, the Brain-kwarg echo and provenance included
    assert json.dumps(_untimed(da), sort_keys=True) == json.dumps(_untimed(db), sort_keys=True)
    assert json.dumps(_untimed(a), sort_keys=True) == json.dumps(_untimed(b), sort_keys=True)
    assert life_kw.main(_argv(tmp_path, "empty", "--brain-kw-extra", "{}"), cortex=StubCortex(), brain_kw=BKW) \
        and "brain_kw_extra" not in _disk(tmp_path, "empty")


def test_extra_reaches_every_brain_and_is_recorded(tmp_path, monkeypatch):
    seen = []
    real = life.Brain

    def spy(*a, **k):
        seen.append(dict(k))
        return real(*a, **k)
    monkeypatch.setattr(life, "Brain", spy)
    extra = {"speaker_contrast": 0.0, "episode": "tell"}     # the defaults: the life must equal the plain one
    res = life_kw.main(_argv(tmp_path, "x", "--brain-kw-extra", json.dumps(extra)), cortex=StubCortex(), brain_kw=BKW)
    assert seen and all(k.get("speaker_contrast") == 0.0 and k.get("episode") == "tell" for k in seen)
    disk = _disk(tmp_path, "x")
    for r in (res, disk):
        assert r["brain_kw_extra"] == extra and all(r["brain_kw"][k] == v for k, v in extra.items())
    assert life.write_outputs.__module__ == "brain.eval.life"       # the wrap is undone after the call
    plain = life.main(_argv(tmp_path, "p"), cortex=StubCortex(), brain_kw=BKW)
    drop = lambda r: {k: v for k, v in _untimed(r).items() if k not in ("brain_kw", "brain_kw_extra")}
    assert json.dumps(drop(res), sort_keys=True) == json.dumps(drop(plain), sort_keys=True)


def test_non_default_extra_changes_the_brain_kwargs(tmp_path, monkeypatch):
    seen = []
    real = life.Brain
    monkeypatch.setattr(life, "Brain", lambda *a, **k: (seen.append(dict(k)), real(*a, **k))[1])
    res = life_kw.main(_argv(tmp_path, "sc", "--brain-kw-extra", '{"speaker_contrast": 3.0}', days=1),
                       cortex=StubCortex(), brain_kw=BKW)
    assert res["brain_kw"]["speaker_contrast"] == 3.0 and all(k["speaker_contrast"] == 3.0 for k in seen)


@pytest.mark.parametrize("split", ["test2", "test"])
def test_test_splits_refused_before_anything_is_written(tmp_path, split):
    with pytest.raises(SystemExit, match="refused"):
        life_kw.main(_argv(tmp_path, "t", "--split", split, "--brain-kw-extra", '{"episode": "lived_key"}'),
                     cortex=StubCortex(), brain_kw=BKW)
    with pytest.raises(SystemExit, match="refused"):
        life_kw.main(_argv(tmp_path, "t", f"--split={split}", '--brain-kw-extra={"episode": "lived_key"}'),
                     cortex=StubCortex(), brain_kw=BKW)
    assert not (tmp_path / "t").exists()


@pytest.mark.parametrize("extra,msg", [('{"keep_bank": "prior"}', "set by brain.eval.life"),
                                       ('{"sleep_seed": 1}', "set by brain.eval.life"),
                                       ('{"no_such_kwarg": 1}', "takes no"), ('[1]', "JSON object")])
def test_bad_extras_refused(tmp_path, extra, msg):
    with pytest.raises(ValueError, match=msg):
        life_kw.main(_argv(tmp_path, "b", "--brain-kw-extra", extra), cortex=StubCortex(), brain_kw=BKW)


def test_resume_must_keep_the_extras(tmp_path):
    ex = '{"speaker_contrast": 0.0}'
    life_kw.main(_argv(tmp_path, "r", "--brain-kw-extra", ex, days=1), cortex=StubCortex(), brain_kw=BKW)
    with pytest.raises(SystemExit, match="resume"):
        life_kw.main(_argv(tmp_path, "r", "--resume", days=2), cortex=StubCortex(), brain_kw=BKW)
    with pytest.raises(SystemExit, match="resume"):
        life_kw.main(_argv(tmp_path, "r", "--resume", "--brain-kw-extra", '{"speaker_contrast": 1.0}', days=2),
                     cortex=StubCortex(), brain_kw=BKW)
    res = life_kw.main(_argv(tmp_path, "r", "--resume", "--brain-kw-extra", ex, days=2), cortex=StubCortex(), brain_kw=BKW)
    assert res["brain_kw_extra"] == {"speaker_contrast": 0.0} and max(n["night"] for n in res["nights"]) == 2
