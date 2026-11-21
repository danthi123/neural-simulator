# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (mechanics of the --lang-guard read-out)
"""--lang-guard: the frozen held-out set (hash, no bank content, not a sleep replay text), the KL/NLL read, and the life
harness (flag off = absent, flag on = the same life plus the lang_guard keys, resume, md, table)."""
import json
import re

import pytest
import torch

import brain.brain as bb
import brain.eval.life as life
from brain import clock
from brain.brain import ORDINARY, Brain
from brain.eval import life_bank, life_bank_test, life_heldout, life_langguard, life_table
from brain.tests.test_brain_key_switch import KW, StubCortex
from brain.tests.test_life import _scrub

BKW = {k: v for k, v in KW.items() if k != "sleep_on_restart"}
REAL_BUILD_LIFE = life.build_life
W = lambda s: set(re.findall(r"[a-z']+", s.lower()))  # noqa: E731
STOP = set("a an the and or but so if of to in on at by for from with without about as is are was were be been am do does did "
           "have has had it its this that these those i me my you your we our they their he she his her them us not no yes "
           "just also then than there here what which who how why when where can could would should will may might must very "
           "more most some any all each every other such too up out off over into only one two user assistant".split())
# everyday words that happen to occur in a bank telling; any other shared word is a leak
COMMON = set("air along because before case day far go keep last least like little long mostly new night old take time times "
             "way week work works evening morning light hot bright mild wear learn listen water".split())


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(bb, "replay_into_cortex", lambda plastic, base, eps, generic, **k: [])
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: [], "B_k": {}, "B_v": {}})())

    def short(seed, days, split="dev"):
        sched, ledger = REAL_BUILD_LIFE(seed, days, split)
        for d in sched:
            d["turns"] = d["turns"][:3]
        return sched, ledger
    monkeypatch.setattr(life, "build_life", short)
    yield
    clock.set(None)
    life.use_split("dev")


# ---------------------------------------------------------------- the set
def test_heldout_frozen_and_wellformed():
    assert life_heldout.verify() == life_heldout.sha()
    H = life_heldout.HELDOUT
    assert len(H) == 24 == len(set(H))
    for t in H:
        assert re.fullmatch(r"User: [^\n]+\n\nAssistant: [^\n]+\n\nUser: [^\n]+\n\nAssistant: [^\n]+\n\n", t), t


def test_heldout_has_no_bank_content():
    hw = set().union(*[W(t) for t in life_heldout.HELDOUT]) - STOP
    keys, tell, names = set(), set(), set()
    for M in (life_bank, life_bank_test):
        for it in M.ALL.values():
            for k in it.keys + it.gist_keys:
                keys |= W(k)
            tell |= W(it.telling or "") | W(it.context or "")
            names |= set(re.findall(r"\b[A-Z][a-z]+\b", it.telling or ""))
    assert not hw & keys, sorted(hw & keys)
    assert not hw & (tell - STOP - COMMON), sorted(hw & (tell - STOP - COMMON))
    text = " ".join(life_heldout.HELDOUT)
    ok = {"The", "It", "My", "At", "Every", "A", "I", "Is", "Not"}
    assert not any(re.search(rf"\b{n}\b", text) for n in names - ok)    # no proper name used in a bank telling


def test_heldout_is_not_a_sleep_replay_text():
    replay = list(ORDINARY)
    try:
        from brain.experiments import sleep_probe
        replay += list(sleep_probe.GENERIC)
    except Exception:
        pass
    assert not any(h.strip() == r.strip() or h.strip() in r or r.strip() in h for h in life_heldout.HELDOUT for r in replay)
    first = {t.split("\n")[0] for t in replay}
    assert not any(h.split("\n")[0] in first for h in life_heldout.HELDOUT)


def test_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    f = tmp_path / "h.sha256"
    f.write_text("0" * 64 + "  life_heldout.py\n")
    monkeypatch.setattr(life_heldout, "HASH_FILE", str(f))
    with pytest.raises(RuntimeError):
        life_heldout.verify()


# ---------------------------------------------------------------- the read
class _Pipe:
    def encode(self, t):
        return [ord(c) % 20 for c in t]


class _P:
    """Fake plastic: logits = table[ids] (+ a bump when enabled); a call draws no RNG and does not touch the base."""
    def __init__(self, bump):
        self.enabled, self.bump = True, bump
        self.table = torch.randn(20, 20, generator=torch.Generator().manual_seed(1))

    def __call__(self, ids, base):
        out = self.table[torch.tensor(ids)]
        return (out + self.bump if self.enabled else out), base


class _C:
    pipe = _Pipe()

    def __init__(self, plastic):
        self.plastic = plastic


def test_measure_kl_and_nll_and_restore():
    p = _P(torch.zeros(20))
    r = life_langguard.measure(_C(p), [])
    assert r["kl"] == 0 and r["nll_on"] == r["nll_off"] and r["n_texts"] == 24       # adapters changing nothing: KL 0
    p = _P(torch.linspace(0, 3, 20))
    p.enabled = False                                                                   # the state is restored
    r = life_langguard.measure(_C(p), [], texts=life_heldout.HELDOUT[:3])
    assert p.enabled is False and r["kl"] > 0 and r["n_texts"] == 3 and r["nll_on"] != r["nll_off"]
    ids = _Pipe().encode(life_heldout.HELDOUT[0])
    on = torch.log_softmax(p.table[torch.tensor(ids)] + p.bump, -1)
    off = torch.log_softmax(p.table[torch.tensor(ids)], -1)
    want = float(torch.nn.functional.kl_div(on, off, log_target=True, reduction="batchmean"))   # sleep.py's keep term
    assert life_langguard.measure(_C(p), [], texts=life_heldout.HELDOUT[:1])["kl"] == pytest.approx(want, abs=1e-5)
    assert life_langguard.measure(_C(None), [])["kl"] == 0.0


# ---------------------------------------------------------------- the life
def _fake(calls):
    def f(home, cortex, night, sim, brain_kw, work, weekend=False):
        calls.append(night)
        return {"kl": round(0.01 * night, 6), "nll_on": 3.0 + night, "nll_off": 3.0, "n_texts": 24, "seconds": 0.0}
    return f


def _run(tmp_path, name, days=3, **kw):
    return life.run(StubCortex(), 7, days, "dev", str(tmp_path / name / "h"), str(tmp_path / name / "o" / "life_s7"),
                    kw.pop("full", {1}), brain_kw=BKW, log=lambda *_: None, **kw)


def _strip(r):
    r = json.loads(json.dumps(_scrub(r)))
    r.pop("lang_guard", None)
    for n in r["nights"]:
        n.pop("lang_guard", None)
    return json.dumps(r, sort_keys=True)


def test_flag_on_equals_off_outside_keys_and_off_is_absent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(life_langguard, "night_read", _fake(calls))
    off = _run(tmp_path, "off", lang_guard=False)
    absent = _run(tmp_path, "absent")
    on = _run(tmp_path, "on", lang_guard=True)
    assert json.dumps(_scrub(off), sort_keys=True) == json.dumps(_scrub(absent), sort_keys=True)
    assert "lang_guard" not in json.dumps(off) and "KL" not in open(str(tmp_path / "off" / "o" / "life_s7.md")).read()
    assert calls == [1, 2, 3]                                   # every trunk night, trunk line only
    assert _strip(on) == _strip(off)
    assert all(n["lang_guard"]["n_texts"] == 24 for n in on["nights"] if n["line"] == "trunk")
    assert on["lang_guard"]["heldout_sha256"] == life_heldout.sha()
    assert "KL 0.02; NLL on/off 5.0/3.0" in open(str(tmp_path / "on" / "o" / "life_s7.md")).read()


def test_nosleep_line_not_measured(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(life_langguard, "night_read", _fake(calls))
    res = _run(tmp_path, "ns", 2, nosleep_arm=True, lang_guard=True)
    assert calls == [1, 2] and all("lang_guard" not in n for n in res["nights"] if n["line"] == "nosleep")


def test_real_night_read_restores_everything(tmp_path):
    """night_read on a real trunk copy: the trunk home, the global RNG and the adapter switch are unchanged."""
    import random
    _run(tmp_path, "r", 1)
    home = str(tmp_path / "r" / "h" / "trunk")
    sim = life.SimClock(life.day_start(2))
    clock.set(sim.now)
    random.seed(5)
    torch.manual_seed(5)
    st = (random.getstate(), torch.get_rng_state())
    import os
    before = sorted(os.listdir(home))
    r = life_langguard.night_read(home, StubCortex(), 1, sim, {**BKW, "sleep_seed": 7}, str(tmp_path / "w"))
    assert r["kl"] == 0.0 and r.get("no_adapters")            # the stub has no adapters: pretrained is the cortex
    assert random.getstate() == st[0] and torch.equal(torch.get_rng_state(), st[1])
    assert sorted(os.listdir(home)) == before and not os.path.exists(str(tmp_path / "w" / "langguard_L1"))


def test_resume_and_flag_checks(tmp_path, monkeypatch):
    whole_calls = []
    monkeypatch.setattr(life_langguard, "night_read", _fake(whole_calls))
    whole = _run(tmp_path, "w", 3, lang_guard=True)
    calls, cut = [], {"on": True}
    inner = _fake(calls)

    def cutting(home, cortex, night, *a, **k):
        if cut["on"] and night == 3:
            cut["on"] = False
            raise RuntimeError("cut")
        return inner(home, cortex, night, *a, **k)
    monkeypatch.setattr(life_langguard, "night_read", cutting)
    with pytest.raises(RuntimeError):
        _run(tmp_path, "c", 3, lang_guard=True)
    assert calls == [1, 2]
    res = _run(tmp_path, "c", 3, resume=True)                   # the flag is restored from res
    assert calls == [1, 2, 3]                                   # nights 1-2 are not re-measured
    assert json.dumps(_scrub(res), sort_keys=True) == json.dumps(_scrub(whole), sort_keys=True)
    assert len([n for n in res["nights"] if n["line"] == "trunk"]) == 3
    _run(tmp_path, "z", 2)
    with pytest.raises(ValueError):                             # a life started without the flag
        _run(tmp_path, "z", 2, resume=True, lang_guard=True)


def test_cli_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(life_langguard, "night_read", _fake([]))
    res = life.main(["--seed", "7", "--days", "2", "--full-probe-nights", "1", "--homes", str(tmp_path / "h"),
                     "--out", str(tmp_path / "o" / "life_s7"), "--lang-guard"], cortex=StubCortex(), brain_kw=BKW)
    assert "lang_guard" in res


def test_table_column_only_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(life_langguard, "night_read", _fake([]))
    on = _run(tmp_path, "on", 3, lang_guard=True)
    off = _run(tmp_path, "off", 3)
    t_on, t_off = life_table.render([on]), life_table.render([off])
    assert "| KL | KL / night-1 |" in t_on and "| 0.03 | 3.0 |" in t_on
    assert "KL" not in t_off
    old = json.loads(json.dumps(off))                           # an old-style res: no lang_guard anywhere
    assert life_table.render([old]) == t_off
    assert [r.get("kl") for r in life_table.seed_rows(on)] == [0.01, 0.02, 0.03]
    assert all("kl" not in r for r in life_table.seed_rows(old))


def _all_strings(o, out):
    if isinstance(o, str):
        out.append(o)
    elif isinstance(o, (list, tuple, set)):
        for x in o:
            _all_strings(x, out)
    elif isinstance(o, dict):
        for k, v in o.items():
            _all_strings(k, out)
            _all_strings(v, out)
    elif hasattr(o, "__dataclass_fields__"):
        for f in o.__dataclass_fields__:
            _all_strings(getattr(o, f), out)
    return out


def test_heldout_shares_no_4gram_with_any_bank_or_replay_text():
    """Every string of both banks (tellings, questions, small talk, open prompts, incidents, never-asked), ORDINARY and
    the sleep_probe GENERIC set: no shared word 4-gram (a lived or replayed frame would bias the guard toward 0)."""
    from brain.experiments import sleep_probe
    src = []
    for M in (life_bank, life_bank_test):
        for n in dir(M):
            if n.isupper():
                _all_strings(getattr(M, n), src)
    src += list(ORDINARY) + list(sleep_probe.GENERIC)
    assert len(src) > 300

    def grams(s):
        w = re.findall(r"[a-z']+", s.lower())
        return {tuple(w[i:i + 4]) for i in range(len(w) - 3)} - {("user",) * 4}
    held = set().union(*[grams(h) for h in life_heldout.HELDOUT])
    hits = [(s[:60], sorted(grams(s) & held)) for s in src if grams(s) & held]
    assert not hits, hits


def test_changed_heldout_file_fails_closed(tmp_path, monkeypatch):
    """A one-character edit of the data file (not of the hash file) must make verify() raise."""
    src = open(life_heldout.__file__).read()
    p = tmp_path / "life_heldout.py"
    p.write_text(src.replace("Will it be warm tomorrow?", "Will it be warm tomorrow!", 1))
    assert p.read_text() != src
    monkeypatch.setattr(life_heldout, "__file__", str(p))
    with pytest.raises(RuntimeError):
        life_heldout.verify()


def test_real_night_read_in_a_life_equals_off_outside_keys(tmp_path):
    """Not faked: the real night_read (disk copy, Brain open, clock move, RNG save/restore) inside a 3-day life leaves
    every other key equal to the flag-off life."""
    off = _run(tmp_path, "off", 3, lang_guard=False)
    on = _run(tmp_path, "on", 3, lang_guard=True)
    assert [n["night"] for n in on["nights"] if "lang_guard" in n] == [1, 2, 3]
    assert _strip(on) == _strip(off)
    assert not any(x.startswith("langguard_") for _, ds, _ in __import__("os").walk(str(tmp_path / "on")) for x in ds)
