# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (mechanics of the split switch)
"""The life script runs on the TEST split only while its sha256 equals the recorded hash; DEV is unchanged."""
import pytest

from brain.eval import life, life_bank, life_bank_test


def test_dev_schedule_unchanged_and_bank_restored():
    s1, l1 = life.build_life(311, 7, "dev")
    life.build_life(311, 7, "test")
    s2, l2 = life.build_life(311, 7, "dev")
    assert (s1, l1) == (s2, l2) and life.LB is life_bank


def test_test_split_uses_test_items():
    sched, ledger = life.build_life(42, 15, "test")
    assert life.LB is life_bank_test
    ab = [i for i, e in ledger.items() if e["cls"] == "AB"]
    assert len(ab) == 16 and all(i in life_bank_test.ALL for i in ab)
    texts = {t["text"] for d in sched for t in d["turns"]}
    assert not texts & {x.telling for x in life_bank.AB + life_bank.F + life_bank.E + life_bank.D}
    assert len([q for q in life.battery(42, 5, ledger) if q["kind"].startswith("open-")]) == 6
    life.use_split("dev")


def test_edited_test_split_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "h.sha256"
    f.write_text("0" * 64 + "  x\n")
    monkeypatch.setattr(life, "TEST_HASH_FILE", str(f))
    with pytest.raises(RuntimeError):
        life.use_split("test")
    life.use_split("dev")


# ---------------------------------------------------------------- TEST v2 (design 2026-11-22 sections 4 and 5)
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import brain.eval as BE  # noqa: E402
from brain.eval import life_cues as LC  # noqa: E402
from brain.eval import life_cues_split as CS  # noqa: E402
from brain.eval import life_interview, life_ladder  # noqa: E402

# json.dumps(..., sort_keys=True) sha256 of the v1 cue dicts at 4c9c0504f. life_cues.py itself stays byte-identical:
# V1_CUES_FILE is its registered sha256 (brain/research/2026-11-19-multiweek-life-design.md), recorded by every
# --cue-ladder life; TEST v2 is reached through life_cues_split, never by editing life_cues.py.
V1_CUES_FILE = "2055dc92678c7481cac99d067a24f079126b80feb5d7a799e467cf3a23b3247e"
V1_CUES_DEV = "dafaff4cd4cf072d13768593b3b05d79bd8e3503f878941036000b925863ccf3"
V1_CUES_TEST = "8b2e8e21cbf2420994843c73cad8572cb57f44905d6ad87f6b483467d91dd0ab"


def _fake_module(monkeypatch, tmp_path, name, attrs, text="# fake\n"):
    f = tmp_path / f"{name}.py"
    f.write_text(text)
    m = types.ModuleType(f"brain.eval.{name}")
    m.__file__ = str(f)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, f"brain.eval.{name}", m)
    monkeypatch.setattr(BE, name, m, raising=False)
    return m, str(f)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_test2_without_its_hash_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(life, "TEST2_HASH_FILE", str(tmp_path / "missing.sha256"))
    with pytest.raises(RuntimeError, match="not frozen"):
        life.use_split("test2")
    assert life.LB is life_bank or life.use_split("dev") is life_bank


def test_test2_bank_hash_guard(tmp_path, monkeypatch):
    m, f = _fake_module(monkeypatch, tmp_path, "life_bank_test2", {"ALL": {}})
    h = tmp_path / "t2.sha256"
    monkeypatch.setattr(life, "TEST2_HASH_FILE", str(h))
    h.write_text(f"{_sha(f)}  life_bank_test2.py\n")
    assert life.use_split("test2") is m and life.LB is m            # the frozen file is accepted
    h.write_text("0" * 64 + "  life_bank_test2.py\n")               # an edited v2 bank is refused
    with pytest.raises(RuntimeError, match="changed"):
        life.use_split("test2")
    assert life.use_split("test") is life_bank_test               # v1 stays runnable and hash-guarded
    life.use_split("dev")


def test_split_choices_and_unknown_split():
    with pytest.raises(SystemExit):
        life.main(["--seed", "1", "--split", "test3", "--homes", "x", "--out", "y"])
    with pytest.raises(ValueError):
        life.use_split("test3")
    assert life.SPLITS == ("dev", "test", "test2")


def test_test2_cues_hash_guard(tmp_path, monkeypatch):
    cues = {"f01": ("Where the tiles came from. Any ideas?", "The tiles at the pool. Any ideas?")}
    m, f = _fake_module(monkeypatch, tmp_path, "life_cues_test2", {"TEST2": cues})
    h = tmp_path / "c2.sha256"
    monkeypatch.setattr(life_ladder, "CUES_TEST2_HASH_FILE", str(h))
    with pytest.raises(RuntimeError, match="not frozen"):         # no hash file: fail closed
        life_ladder.load_cues("test2")
    h.write_text(f"{_sha(f)}  life_cues_test2.py\n")
    assert life_ladder.load_cues("test2") is cues and life_ladder.cues_sha("test2") == _sha(f)
    h.write_text("0" * 64 + "  life_cues_test2.py\n")
    with pytest.raises(RuntimeError, match="changed"):             # an edited v2 cue file is refused
        life_ladder.load_cues("test2")
    assert life_ladder.cues_sha() == life_ladder.cues_sha("test") == _sha(LC.__file__)


def test_cues_test2_is_lazy_and_dev_test_unchanged(monkeypatch):
    monkeypatch.setattr(CS, "LAZY", {"test2": ("brain.eval._no_such_cue_file", "TEST2")})
    assert CS.cues("test2") is None                                  # a missing v2 file never breaks dev/test
    with pytest.raises(ValueError, match="no cue texts"):
        life_ladder.load_cues("test2")
    assert CS.cues("dev") is LC.DEV and CS.cues("test") is LC.TEST
    assert type(LC.CUES) is dict and LC.CUES == {"dev": LC.DEV, "test": LC.TEST}
    assert _sha(LC.__file__) == V1_CUES_FILE                         # life_cues.py is the registered, frozen file
    assert open(life_ladder.CUES_HASH_FILE).read().split()[0] == V1_CUES_FILE
    assert life_ladder.cues_sha() == life_ladder.cues_sha("dev") == life_ladder.cues_sha("test") == V1_CUES_FILE
    for d, want in ((LC.DEV, V1_CUES_DEV), (LC.TEST, V1_CUES_TEST)):
        assert hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() == want
    assert life_ladder.load_cues("test") is LC.TEST                 # the registered v1 hash matches the file
    assert LC.overlaps("test") == LC.overlaps("test", life_bank_test) == CS.overlaps("test")
    assert LC.overlaps("dev") == LC.overlaps("dev", life_bank) == CS.overlaps("dev")


def test_overlaps_test2_reads_the_v2_bank(tmp_path, monkeypatch):
    it = types.SimpleNamespace(telling="I keep my spare tiles in the garden shed.")
    _fake_module(monkeypatch, tmp_path, "life_bank_test2", {"ALL": {"f01": it}})
    cues = {"f01": ("The spare tiles in my garden. Any ideas?", "A shed full of garden tools. Any ideas?")}
    _fake_module(monkeypatch, tmp_path, "life_cues_test2", {"TEST2": cues})
    ov = CS.overlaps("test2")
    assert ov == {"f01": (3, 2, 0)}          # spare/tiles/garden vs shed/garden


def test_interview_never_topics_from_the_bank():
    q = "What is my favourite synthetic thing?"
    bank = types.SimpleNamespace(NEVER=[("nv1", q, ())], NEVER_TOPICS={q: "your favourite synthetic thing"}, ALL={})
    sel = {"picks": {b: (["nv1"] if b == "never" else []) for b in life_interview.SP.BUCKETS}}
    us = life_interview.units(sel, bank)
    assert us[0]["topic"] == "your favourite synthetic thing" and us[1]["text"] == q
    v1 = types.SimpleNamespace(NEVER=life_bank_test.NEVER, ALL=life_bank_test.ALL)     # no NEVER_TOPICS: the spec's
    sel = {"picks": {b: ([life_bank_test.NEVER[0][0]] if b == "never" else []) for b in life_interview.SP.BUCKETS}}
    assert life_interview.units(sel, v1)[0]["topic"] == life_interview.SP.NEVER_TOPICS[life_bank_test.NEVER[0][1]]
