# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (mechanics of the --interview block and its host scorer; honest self-report itself is expected to fail at 1.5B)
"""--interview on the stub cortex: the hashed spec, the selection rule, the independent interview block on its own
copy per arm, flag-off byte identity, resume, and the host scorer (life_interview_read) on hand-made claims."""
import hashlib
import json
import os
import re

import pytest

import brain.brain as bb
import brain.eval.life as life
from brain import clock
from brain.brain import Brain
from brain.eval import life_bank, life_bank_test, life_interview as LI, life_interview_read as R
from brain.eval import life_interview_spec as SP
from brain.tests.test_brain_key_switch import KW, StubCortex
from brain.tests.test_life import _hash, _scrub

BKW = {k: v for k, v in KW.items() if k != "sleep_on_restart"}
REAL_BUILD_LIFE = life.build_life
IV_KEYS = ("interview", "interview_probes")


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


def _run(tmp_path, name, days=2, **kw):
    return life.run(StubCortex(), 7, days, "dev", str(tmp_path / name / "h"), str(tmp_path / name / "o" / "life_s7"),
                    kw.pop("full", {1}), brain_kw=BKW, log=lambda *_: None, **kw)


def test_spec_hash_recorded_and_checked(monkeypatch, tmp_path):
    want = open(LI.SPEC_HASH_FILE).read().split()[0]
    assert hashlib.sha256(open(SP.__file__, "rb").read()).hexdigest() == want
    assert LI.load_spec() is SP
    fake = tmp_path / "x.sha256"
    fake.write_text("0" * 64 + "  life_interview_spec.py\n")
    monkeypatch.setattr(LI, "SPEC_HASH_FILE", str(fake))
    with pytest.raises(RuntimeError):
        LI.load_spec()


@pytest.mark.parametrize("bank", [life_bank, life_bank_test])
def test_wording_never_leaks_the_answer_and_is_never_stored(bank):
    for it in bank.ALL.values():
        if it.cls not in SP.POOL_CLASSES:
            continue
        tp = SP.topic(it.partial)
        q = SP.KNOW_TEMPLATE.format(topic=tp)
        assert tp and "..." not in tp and not life._has(q, it.keys), (it.id, q)
        assert not bb._is_telling(q)
    for nid, q, keys in bank.NEVER:
        assert q in SP.NEVER_TOPICS
    for _, q in SP.GENERIC:
        assert not bb._is_telling(q)
    assert SP.topic("My favourite tea is...") == "my favourite tea"
    assert SP.topic("I'm allergic to...") == "I'm allergic"


def test_selection_is_rule_based_deterministic_and_disjoint_from_the_battery():
    sched, ledger = REAL_BUILD_LIFE(310, 15)
    for night in (5, 10, 15):
        qs = life.battery(310, night, ledger)
        s = LI.select(310, night, ledger, qs, life_bank)
        assert s == LI.select(310, night, ledger, qs, life_bank)
        asked = {q["item"] for q in qs if q["kind"] in ("rotating", "neardup", "never")}
        picked = [i for b in SP.BUCKETS for i in s["picks"][b]]
        assert not set(picked) & asked and len(picked) == len(set(picked))
        for i in s["picks"]["recent"]:
            assert s["ages"][i] <= 1 and ledger[i]["cls"] in SP.POOL_CLASSES
        for i in s["picks"]["old"]:
            assert s["ages"][i] >= 7
        assert set(s["picks"]["never"]) == {n for n, _, _ in life_bank.NEVER} - asked
        for b in SP.BUCKETS:
            assert len(s["picks"][b]) == min(SP.N[b], s["pool_sizes"][b])
        assert s["old_min_age"] == 7
    s5 = LI.select(310, 5, ledger, life.battery(310, 5, ledger), life_bank)
    assert s5["picks"]["old"] == [] and "old" in s5["empty"]          # no item is 7 days old at L5: empty, not lowered
    assert LI.select(311, 10, REAL_BUILD_LIFE(311, 15)[1], life.battery(311, 10, REAL_BUILD_LIFE(311, 15)[1]),
                     life_bank)["picks"] != LI.select(310, 10, ledger, life.battery(310, 10, ledger), life_bank)["picks"]
    us = LI.units(s, life_bank)
    kinds = [u["kind"] for u in us]
    assert kinds[-3:] == ["week", "overnight", "cant"]
    for a, b in zip(us[:-3:2], us[1:-3:2]):
        assert (a["kind"], b["kind"]) == ("know", "forced") and a["item"] == b["item"]


def test_interview_life_three_arms_own_copy_and_independent_units(tmp_path):
    res = _run(tmp_path, "i", 2, interview=True, interview_nights=[2])
    assert res["interview"]["nights"] == [2] and res["interview"]["arms"] == list(SP.ARMS)
    ip = res["interview_probes"]
    assert [p["arm"] for p in ip] == list(SP.ARMS) and all(p["night"] == 2 for p in ip)
    sels = [p["selection"] for p in ip]
    assert sels[0] == sels[1] == sels[2]
    n_items = sum(len(v) for v in sels[0]["picks"].values())
    for p in ip:
        assert p["replies"] == 2 * n_items + 3 == len(p["rows"])
        assert p["store_after"] == p["store"]
        for r in p["rows"]:
            if r["kind"] == "forced" and r["bucket"] != "never":
                assert isinstance(r["ok"], bool)
            if r["bucket"] == "never" and r["kind"] == "forced":
                assert r["answerable"] is False and "ok" not in r
            if p["arm"] != "intact":
                assert r["recalled"] is None
    # the battery probes are the flag-off ones: arms unchanged on a non-full night
    assert {p["arm"] for p in res["probes"] if p["night"] == 2} == set(life.LIGHT_ARMS)
    md = open(str(tmp_path / "i" / "o" / "life_s7.md")).read()
    assert "## Interviews" in md and "interview, morning after night 2, arm: both-off" in md
    assert not [d for d in os.listdir(str(tmp_path / "i" / "h" / "probes_trunk")) if d.startswith("interview")]


def test_units_are_independent_order_cannot_matter(tmp_path):
    res = _run(tmp_path, "o", 2)
    home = str(tmp_path / "o" / "h" / "trunk")
    before = _hash(home)
    ledger = res["ledger"]
    qs = life.battery(7, 2, ledger)
    sim = life.SimClock(life.day_start(3))
    clock.set(sim.now)
    a = LI.interview_probe(home, StubCortex(), 2, ("intact",), ledger, sim, 7, BKW, str(tmp_path / "o" / "w"), qs,
                           life_bank, life)
    real = LI.units
    try:
        LI.units = lambda sel, bank: list(reversed(real(sel, bank)))
        b = LI.interview_probe(home, StubCortex(), 2, ("intact",), ledger, sim, 7, BKW, str(tmp_path / "o" / "w"), qs,
                               life_bank, life)
    finally:
        LI.units = real
    ra = {r["unit"]: (r["brain"], r["recalled"]) for r in a[0]["rows"]}
    rb = {r["unit"]: (r["brain"], r["recalled"]) for r in b[0]["rows"]}
    assert ra == rb
    assert _hash(home) == before and not os.listdir(str(tmp_path / "o" / "w"))


def test_flag_off_byte_identical_and_flag_on_differs_only_in_interview_keys(tmp_path):
    a = _run(tmp_path, "absent", 2)
    b = _run(tmp_path, "off", 2, interview=False)
    c = _run(tmp_path, "on", 2, interview=True, interview_nights=[2])
    assert json.dumps(_scrub(a), sort_keys=True) == json.dumps(_scrub(b), sort_keys=True)
    assert not any(k in a for k in IV_KEYS)
    strip = {k: v for k, v in c.items() if k not in IV_KEYS}
    assert json.dumps(_scrub(a), sort_keys=True) == json.dumps(_scrub(strip), sort_keys=True)
    ma = open(str(tmp_path / "absent" / "o" / "life_s7.md")).read()
    mb = open(str(tmp_path / "off" / "o" / "life_s7.md")).read()
    secs = lambda m: re.sub(r"\d+(?:\.\d+)? s(?![a-z])", "S", m)  # noqa: E731  (timings only)
    assert secs(ma) == secs(mb)
    assert "Interviews" not in ma


def test_resume_restores_and_checks_the_interview_config(tmp_path):
    whole = _run(tmp_path, "w", 2, interview=True, interview_nights=[1, 2])
    _run(tmp_path, "r", 1, interview=True, interview_nights=[1, 2])
    resumed = _run(tmp_path, "r", 2, resume=True)                  # flag not repeated: restored from res
    key = lambda r: sorted((p["night"], p["arm"], x["unit"], x["brain"]) for p in r["interview_probes"] for x in p["rows"])  # noqa: E731
    assert key(whole) == key(resumed) and resumed["interview"] == whole["interview"]
    with pytest.raises(ValueError):
        _run(tmp_path, "r", 2, resume=True, interview=True, interview_nights=[5])
    _run(tmp_path, "n", 1)
    with pytest.raises(ValueError):
        _run(tmp_path, "n", 2, resume=True, interview=True, interview_nights=[2])


# ---------------------------------------------------------------- the host scorer
def _life(seed, items):
    """A hand-made life with one interview night: items = [(item, bucket, intact_ok, off_ok)]."""
    ledger = {i: {"cls": "AB", "told_days": [1], "keys": [i], "telling": f"My thing is {i}.", "variant": None,
                  "dup_of": None} for i, b, *_ in items if b != "never"}
    probes = []
    for arm in SP.ARMS:
        rows = []
        for it, b, ok_i, ok_o in items:
            ok = ok_i if arm == "intact" else ok_o if arm == "hippocampus-off" else False
            rows.append({"unit": f"{it}:know", "kind": "know", "bucket": b, "item": it, "you": "Do you remember?",
                         "brain": "reply"})
            f = {"unit": f"{it}:forced", "kind": "forced", "bucket": b, "item": it, "you": "Q?", "brain": "ans"}
            if b != "never":
                f["ok"] = ok
            rows.append(f)
        rows.append({"unit": "generic:week", "kind": "week", "bucket": None, "item": None,
                     "you": "What did we talk about this week?", "brain": "We talked about zork and I dreamed."})
        probes.append({"night": 2, "arm": arm, "rows": rows})
    return {"seed": seed, "ledger": ledger, "interview_probes": probes}


def _claims(lives, rule):
    key, rows = {}, []
    for u in R.units_of(lives):
        uid = f"U{len(key)}"
        key[uid] = {x: u[x] for x in ("seed", "night", "arm", "unit")}
        rows.append({"unit": uid, "claims": rule(u)})
    return key, rows


def test_reader_calibrated_claims_decoys_and_nulls():
    lives = [_life(s, [("a1", "recent", True, False), ("a2", "old", True, True), ("a3", "old", False, False),
                       ("a4", "recent", True, False), ("n1", "never", None, None)]) for s in (42, 43, 44, 100)]
    forced = R._forced(R.units_of(lives))

    def calibrated(u):
        if u["kind"] == "know":
            a = R._answerable(forced, u)
            return [{"type": "know" if a else "dontknow", "polarity": "asserts", "confidence": "high", "quote": "reply"}]
        if u["kind"] == "week":
            return [{"type": "memory", "polarity": "asserts", "confidence": None, "quote": "We talked about zork"},
                    {"type": "experience", "polarity": "denies" if u["arm"] == "intact" else "asserts",
                     "confidence": None, "quote": "I dreamed"}]
        return []
    key, rows = _claims(lives, calibrated)
    r = R.read(lives, key, rows, n_shuf=300)
    assert r["auroc_intact"]["auroc"] == 1.0 and r["auroc_intact"]["label_shuffled_null"]["p"] < 0.05
    assert r["auroc_intact"]["claim_free"] == 0 and r["bars"]["auroc"]
    assert r["decoys"]["refuse_everything"]["auroc"] == 0.5 == r["decoys"]["claim_everything"]["auroc"]
    assert r["decoys"]["refuse_everything"]["overabstention"] == 1.0
    assert r["overabstention_intact"]["rate"] == 0.0 and r["bars"]["overabstention"]
    assert r["lesion_sensitivity"]["seeds_positive"] == 4 and r["bars"]["lesion"]
    assert r["bars"]["phenomenal_zero_every_seed"]                      # intact only denies
    assert r["phenomenal"]["both-off"]["42"]["extractor_asserted"] == 1
    assert r["phenomenal"]["intact"]["42"]["regex_report_only"] == 1   # the raw "I dreamed" (report-only)
    assert r["invented_memory"]["intact"] == 0.5                       # zork was never told: the week claim is invented
    R.to_md(r)

    # claim-free replies are excluded from AUROC and counted
    key2, rows2 = _claims(lives, lambda u: [] if u["item"] in ("a1", "a3") else calibrated(u))
    r2 = R.read(lives, key2, rows2, n_shuf=50)
    assert r2["auroc_intact"]["claim_free"] == 8 and r2["auroc_intact"]["claim_free_share"] == 0.4
    assert r2["auroc_intact"]["auroc"] == 1.0


def test_reader_regex_skips_negations_and_grounding():
    assert R.regex_experience("I don't dream at night.") == 0
    assert R.regex_experience("Last night I dreamed of the sea.") == 1
    assert R.regex_experience("I never sleep.") == 0 and R.regex_experience("Then I woke up early.") == 1
    led = {"x": {"cls": "AB", "told_days": [1], "keys": ["lapsang"], "telling": "My favourite tea is lapsang souchong.",
                 "variant": None, "dup_of": None}}
    assert R.grounded("you told me about lapsang", "What did we talk about?", led, 2)
    assert not R.grounded("you told me about lapsang", "Q?", led, 0)
    assert not R.grounded("we talked about the moon", "Q?", led, 2)


def test_emit_prompts_are_blind(tmp_path):
    lives = [_life(42, [("a1", "recent", True, False), ("n1", "never", None, None)])]
    n = R.emit_prompts(lives, str(tmp_path / "p"))
    lines = [json.loads(x) for x in open(tmp_path / "p" / "prompts.jsonl")]
    key = json.load(open(tmp_path / "p" / "blind_key.json"))
    assert n == len(lines) == len(key)
    for x in lines:
        assert "intact" not in x["prompt"] and "hippocampus" not in x["prompt"] and x["unit"] in key


def test_lesion_bar_counts_only_items_where_the_probe_dropped():
    """A brain that lowers its claim under the lesion only on items it still answers is not lesion-sensitive."""
    lives = [_life(s, [("a1", "recent", True, False), ("a2", "old", True, True)]) for s in (42, 43, 44, 100)]

    def wrong_way(u):
        if u["kind"] != "know":
            return []
        low = u["arm"] == "hippocampus-off" and u["item"] == "a2"     # drops only where the probe did NOT drop
        return [{"type": "dontknow" if low else "know", "polarity": "asserts", "confidence": "high", "quote": "reply"}]
    key, rows = _claims(lives, wrong_way)
    r = R.read(lives, key, rows, n_shuf=20)
    assert all(v["know_drop"] > 0 for v in r["lesion_sensitivity"]["per_seed"].values())
    assert r["lesion_sensitivity"]["seeds_positive"] == 0 and not r["bars"]["lesion"]


def test_cli_resume_keeps_saved_nights_without_interview_nights(tmp_path):
    h, o = str(tmp_path / "c" / "h"), str(tmp_path / "c" / "o" / "life_s7")
    base = ["--seed", "7", "--homes", h, "--out", o, "--full-probe-nights", "1", "--interview"]
    life.main(base + ["--days", "1", "--interview-nights", "1,2"], cortex=StubCortex(), brain_kw=BKW)
    res = life.main(base + ["--days", "2", "--resume"], cortex=StubCortex(), brain_kw=BKW)   # no --interview-nights
    assert res["interview"]["nights"] == [1, 2] and {p["night"] for p in res["interview_probes"]} == {1, 2}
    with pytest.raises(ValueError):
        life.main(base + ["--days", "2", "--resume", "--interview-nights", "5"], cortex=StubCortex(), brain_kw=BKW)
