"""Regression pin for the 2026-09-24 wm-binding-advanced hold-query INTEGRITY-SMOKE relabel.

Adversarial re-review (journal key v2:7512414c9a65ead57a205e7a2e166bcf476472d9dfbc61180205447d18d460df, label
'rereview:wm-binding') found: "I deleted `res["verdict"] = "integrity-smoke"` and `--selftest` still printed
VERDICT: PASS. tests/ has no test for it either." The hold-query probe (LB_WMB_HOLDQUERY_PROBE) is
PASS-BY-CONSTRUCTION: its reply is a host template whose only input is the buffer the lesion disables, so once
the route is reached a reply change is guaranteed. `measure_faculty`'s `_wmb_on` tail (research/runners/
load_bearing_fraction.py) is what keeps that predetermined result OUT of run()'s load-bearing numerator and
denominator: it unconditionally forces `res["verdict"] = "integrity-smoke"` and `res["load_bearing"] = None`
after the pre-registered adequacy gate, regardless of what the underlying treatment/control comparison read.
That force was covered by no test and no gate mutation, so a later edit could silently drop it and both
`--selftest` and every existing check would keep passing while a pass-by-construction probe re-entered the #1
metric as a real GO.

REVIEW FOLLOW-UP (round 2): the first version of this test drove EVERY arm/turn through one flat `_fake_spawn`
that returned the SAME dict regardless of `env`/`turn_labels`, so with no `multiref` field on any turn the
PRE-relabel verdict already read `probe-inadequate:route` (the adequacy gate's own UNDEFINED path) -- the test
exercised only that UNDEFINED branch, never the "a real countable verdict got relabeled" branch it claims to
pin. `_wmb_synthetic_spawn` below instead varies its reply by BOTH the lesion env and the turn label, so the
PRE-relabel path reads a genuine, adequacy-gate-PASSING "regressed"/load_bearing=True (a route+two-referents
hold-query change on 'wmb_ask' that a specificity-clean 'wmb1_ask' control does not share) -- this is the
result the relabel exists to keep out of run()'s fraction. This test FAILS the moment the three relabel lines
(`integrity_smoke` / `integrity_smoke_verdict` / the `verdict` and `load_bearing` overwrite) are removed,
because the identical synthetic arms below then leave `res["verdict"] == "regressed"` and
`res["load_bearing"] is True` -- real, non-null values that trip every assertion below (mutation-checked by hand:
deleting the three lines flips `test_wmb_holdquery_tail_forces_integrity_smoke_not_a_verdict`'s first assertion
from PASS to `AssertionError: {'verdict': 'regressed', 'load_bearing': True, ...}`, confirmed then reverted)."""
import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import load_bearing_fraction as lbf  # noqa: E402


def _wmb_synthetic_spawn(env, turn_labels, out_path):
    """Fake `_spawn_arm`. Varies its reply by BOTH `env` (is the multiref lesion armed?) and the turn label, so the
    synthetic arms carry the real `multiref` route fields A1/A2/S1 need:

    - drive turn 'wmb_ask': INTACT (env has no lesion) reads a hold-query read-out ('multiref.kind'=='query',
      'is_hold_query'=True) naming BOTH referents ('the fox and the wolf', n_referents=2) -- satisfies A1+A2.
      LESIONED (env[flag]==val) reads the SAME route (kind='query', is_hold_query=True, so A1 also holds on the
      lesion arm per the review's v2:7a3b94367 fix) but a DIFFERENT answer ('the wolf', n_referents=1) -- this is
      the treatment change (`compare()` -> verdict='regressed').
    - control turn 'wmb1_ask': identical reply regardless of `env` ('i do not know', multiref.kind=None -- NOT a
      query route) -- the organ is out of scope on the 1-referent intro, so intact/intact-rebuild/lesion all read
      byte-identical -> S1 (specificity) holds clean on both null and lesion.
    - intro turns ('wmb_intro'/'wmb1_intro'): placeholder content; no compared field ever reads them.

    Both `wmb_ask` intact calls (`intact_a`, `intact_b`) pass env={} (no lesion) -> identical replies -> the null
    control is clean. Net effect BEFORE the relabel: verdict='regressed', load_bearing=True, adequacy override=None
    (every A1/A2/S1 condition holds) -- a real, countable, adequacy-gate-PASSING result. The relabel is the ONLY
    thing that then keeps it out of run()'s numerator and denominator."""
    lesioned = env.get("BRAIN_MULTIREF_LESION") == "1"
    out = {}
    for label in turn_labels:
        if label == "wmb_ask":
            if lesioned:
                out[label] = {"answer": "the wolf", "abstained": False,
                              "multiref": {"kind": "query", "is_hold_query": True, "n_referents": 1}}
            else:
                out[label] = {"answer": "the fox and the wolf", "abstained": False,
                              "multiref": {"kind": "query", "is_hold_query": True, "n_referents": 2}}
        elif label == "wmb1_ask":
            # organ out of scope on the 1-referent intro -- SAME reply regardless of `env` (specificity: the
            # confined lesion must not change it either).
            out[label] = {"answer": "i do not know", "abstained": False, "multiref": {"kind": None}}
        else:  # 'wmb_intro' / 'wmb1_intro' -- never read by the compared field or the adequacy gate
            out[label] = {"answer": "ok", "abstained": False}
    return out


def _patch_wmb_on(monkeypatch):
    monkeypatch.setattr(lbf, "LB_WMB_HOLDQUERY", True)
    monkeypatch.setattr(lbf, "_spawn_arm", _wmb_synthetic_spawn)


def test_wmb_holdquery_tail_forces_integrity_smoke_not_a_verdict(monkeypatch, tmp_path):
    _patch_wmb_on(monkeypatch)
    res = lbf.measure_faculty("wm-binding-advanced", str(tmp_path), repeats=1, intact_cache={}, seed=42)
    # The relabel: verdict must be the sentinel, never a countable "regressed"/"pass", and load_bearing must be
    # unset. Removing `res["verdict"] = "integrity-smoke"` / `res["load_bearing"] = None` leaves this at
    # verdict="regressed", load_bearing=True (the synthetic arms carry a REAL, adequacy-gate-passing route+
    # two-referents hold-query change) -- real values that fail these asserts instead of vacuously passing.
    assert res["verdict"] == "integrity-smoke", res
    assert res["load_bearing"] is None, res
    assert res.get("integrity_smoke") is True, res
    # The smoke's own (pre-relabel) outcome is still recorded, just not under the counted key -- and it is the
    # genuine "regressed" read the relabel is suppressing, not an UNDEFINED probe-inadequate fallback.
    assert res.get("integrity_smoke_verdict") == "regressed", res
    assert res.get("wmb_adequacy", {}).get("A1_route") is True, res
    assert res.get("wmb_adequacy", {}).get("A2_two_referents") is True, res


def test_wmb_holdquery_is_excluded_from_runs_load_bearing_fraction(monkeypatch, tmp_path):
    """Integration-level pin: run()'s own exercised filter (`verdict in ("regressed", "pass", "trace-only")`) must
    exclude the hold-query probe from n_exercised / n_load_bearing / load_bearing_fraction. wm-binding-advanced's
    FACULTY_LESIONS kind is "neural-lesion" (a coverable kind), so without the relabel this probe's REAL
    "regressed"/load_bearing=True read would count in BOTH the numerator and the denominator -- the over-credit
    the review named."""
    _patch_wmb_on(monkeypatch)
    report = lbf.run(out_dir=str(tmp_path), only=["wm-binding-advanced"], repeats=1, seed=42)
    per = report["per_faculty"][0]
    assert per["verdict"] == "integrity-smoke", per
    assert report["counts"]["n_coverable_env_lesion"] == 1, report["counts"]
    assert report["counts"]["n_exercised"] == 0, report["counts"]
    assert report["counts"]["n_load_bearing"] == 0, report["counts"]
    assert report["counts"]["n_not_load_bearing"] == 0, report["counts"]
    assert report["load_bearing_fraction"] is None, report
    assert "wm-binding-advanced" not in report["load_bearing_faculties"]
    assert "wm-binding-advanced" not in report["not_load_bearing_faculties"]
