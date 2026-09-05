"""VALUE-CHOICE REAL TRAINED CRITIC on the shared spiking salience context -- the 6-seed gate rank-4's own
plumbing proof WAIVED to seed-42-only (scaffold-retirement backlog rank-20, 2026-09-05 follow-on to rank-4).

THE GAP THIS CLOSES. `research/findings/2026-09-05-shared-spiking-salience-afferent-wired-GO.md` (rank-4) wired
`value_choice_production_organ.py::default_context_fn()` through the shared ASK-pool spiking salience organ
(`BRAIN_SHARED_SALIENCE`, default-OFF) and 6-seed-GO'd the CONTEXT FUNCTION itself (its own "value_choice_context"
gate, `_shared_salience_afferent_derisk.py::run_seed`'s section (3)). But the ONE thing that ran through the REAL
HEAVY trained `striosome_value` critic end-to-end (`ValueChoiceProductionOrgan.choose()`, not just its upstream
`default_context_fn()`) was WAIVED to a single seed (42) -- documented explicitly in that finding's own frontmatter
seed-waiver: "a full end-to-end pass through value_choice_production_organ's REAL trained striosome_value critic
... is run ONCE at seed 42 only ... because the critic's OWN sensitivity to its engagement input is a pre-existing,
already-6-seed-GO'd mechanism this work does not modify ... and a single trained build costs ~5 CPU-minutes."

That waiver is legitimate for the CRITIC'S OWN sensitivity-to-input mechanism (RANK-1 GO, 2026-07-23, already
6-seed). It is NOT automatically legitimate for THIS specific question: does the REAL trained critic, when fed the
shared-organ-mediated context INSTEAD of the bare host recency ratio, still (a) reach a decisive commit that
matches-or-improves-on what the host ratio produces, and (b) stay load-bearing on the shared afferent specifically
(vary the afferent -> the commit responds; lesion the afferent -> it reverts) -- ACROSS SEEDS, not just once. This
module runs that extension: the SAME `run_plumbing_proof` computation `_shared_salience_afferent_derisk.py`
introduced, replicated at all 6 project-standard seeds, over 4 candidate/recency SCENARIOS per seed (not just the
1 scenario the seed-42 proof used) so "vary it, the choice changes" gets more than one data point per seed.

SCOPE, deliberately narrow (docs/TERMS.md "closed"/"selective" discipline). This does NOT flip any default (both
`BRAIN_VALUE_CHOICE` -- already default-ON since 2026-08-26 -- and `BRAIN_SHARED_SALIENCE` -- default-OFF since
rank-4 -- are used at THEIR EXISTING defaults/flags; no code in `value_choice_production_organ.py` or
`shared_salience_afferent.py` is modified by this module). It does NOT re-verify the RANK-1 GO's own anti-cheats
(G_UNTRAINED: the untrained-critic control) because that mechanism does not depend on the context's PROVENANCE
(host ratio vs shared-organ read) and is already 6-seed-GO'd
(`research/findings/2026-07-23-value-critic-closure-RANK1-GO.md`) -- re-running it here would cost a SECOND
267s-class build per seed for a question this module does not ask. It DOES cheaply re-check the critic's OWN
G_LESION (`BRAIN_VALUE_CHOICE_LESION`, the mean-pin) against the shared-organ-mediated context, at zero extra
build cost (same organ instance, `choose(..., lesion=True)`), as a non-regression sanity check that the
pre-existing critic-level anti-cheat still collapses decisiveness when the input is neurally-mediated (expected
TRIVIALLY true by construction -- the mean-pin discards the fed array's content entirely -- so it is reported as
a sanity confirmation, not counted toward the headline gates).

THE 4 SCENARIOS (per seed, against ONE trained organ build -- only the `choose()` call is repeated, not the
267s-class build+value-train):
  S1 baseline        3 candidates, recency [0.0, 0.5, 1.0] -- reproduces the rank-4 seed-42 plumbing proof exactly.
  S2 near_tie_low    3 candidates, recency [0.0, 0.111, 1.0] -- an ASYMMETRIC spacing (two facts stored close
                     together, one much later) exercising a different region of the shared organ's nonlinearity
                     than the evenly-spaced S1 (`default_context_fn` always renormalizes to the SELECTED
                     candidates' own min/max index, so an evenly-spaced N always reduces to the same [0..1]
                     ladder -- asymmetric storage order is required to reach a non-uniform pattern).
  S3 referent_tie    3 candidates, recency [0.0, 0.5, 1.0], `chat.is_multiturn=True` + the discourse referent
                     bound to the MIDDLE-recency candidate -- exercises the `+0.5` referent-boost branch
                     (untested by the seed-42 plumbing proof), which clips the middle and top candidate to the
                     SAME host engagement (1.0, 1.0) -- a host-level TIE the shared organ's spiking jitter may or
                     may not break differently.
  S4 four_candidate  4 candidates, recency [0.0, 0.333, 0.667, 1.0] -- a larger competitive set, exercising a
                     freshly-built n=4 spiking value-WTA (`_wta_for(4)`, distinct from S1-S3's cached n=3 WTA).

GATES (per seed; the verdict needs >= 5/6 seeds, the project's own established bar --
`research/findings/2026-07-23-value-critic-closure-RANK1-GO.md`'s own "VERDICT: GO (>= 5/6 seeds per gate)"):
  g_off_identical      every scenario's OFF-arm engagement EXACTLY matches an independently-computed host
                       recency/referent formula (byte-identical-off, computed from source, not a hand-picked
                       literal) -- BRAIN_SHARED_SALIENCE unset must not perturb the pre-existing default.
  g_on_loadbearing     every scenario's ON-arm engagement measurably DIFFERS from OFF (the shared organ is
                       genuinely in the path feeding the REAL critic, not a coincidental no-op).
  g_matches_or_improves at least 3/4 scenarios: the ON commit either MATCHES the OFF commit (the neurally-
                       mediated context preserves the host ratio's ordering through the real critic) OR reaches a
                       decisive commit where OFF failed to (improves) -- "the choice task" the RANK-1 GO defines
                       (a decisive value-driven commit), not an external ground-truth label (there is none in
                       production -- OFF IS today's production reference, not a labelled correct answer).
  g_lesion_loadbearing at least 3/4 scenarios: `attributable_to` the ON-arm's fed-value spread (the gradient the
                       critic needs to be decisive) vs the ON+SHARED-LESION spread reports a HIGH fraction
                       (the spread the shared organ contributes actually vanishes under its own lesion, at the
                       REAL critic's readout, not just at the upstream context function rank-4 already proved).

Run (controller, the 6-seed gate; ~267s/seed x 6 ~= 27 CPU-minutes, numpy-CPU, cost-routed off the GPU lane):
  SIM_BACKEND=numpy python -m research.runners._value_choice_neural_context_6seed_derisk \\
      --seeds 42 43 44 100 101 102 \\
      --out research/findings/raw/_value_choice_neural_context/verify_6seed.json

Run (single-seed worker -- what the controller subprocess-fans; also runnable standalone):
  SIM_BACKEND=numpy python -m research.runners._value_choice_neural_context_6seed_derisk --seed 42 \\
      --out research/findings/raw/_value_choice_neural_context/verify_seed42.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# WHOSE-the-difference attribution (tools.lab, the gap#5 lesson): the lesion arm and the ON arm are both measured
# below (a treatment/control pair) -- ask how much of the ON-arm spread survives the shared-afferent lesion rather
# than banking both numbers unattributed.
from tools.lab import attributable_to  # noqa: E402

# REUSE-BY-IMPORT (no sim/ edit, no duplicated flag logic): the SAME env-flag helpers rank-4's own de-risk uses.
from research.runners._shared_salience_afferent_derisk import _clear_flags, _set_flags  # noqa: E402

RAW_DIR = "research/findings/raw/_value_choice_neural_context"


class _FakeAgent:
    def __init__(self, referent=None):
        self._referent = referent

    def held_referent(self):
        return (self._referent, None)


class _FakeChat:
    """Mirrors `_shared_salience_afferent_derisk._FakeChat`, extended with a configurable `stored_facts` order
    and an optional discourse referent (S3 needs both; the sibling de-risk's fixture only needed one fixed order,
    no referent, no is_multiturn)."""
    def __init__(self, stored_facts, referent=None):
        self.stored_facts = list(stored_facts)
        self.is_multiturn = referent is not None
        self.agent = _FakeAgent(referent=referent)


def _host_formula(stored_facts, a, v, cands, referent=None):
    """The pre-existing recency/referent engagement formula, computed INDEPENDENTLY from source (verbatim re-
    implementation of `default_context_fn`'s host arithmetic) so `g_off_identical` does not depend on calling the
    same code it is meant to check against."""
    order = {}
    for i, (fa, fv, fp) in enumerate(stored_facts):
        if fa == a and fv == v:
            order[fp] = i
    idxs = [order.get(p, 0) for p in cands]
    lo, hi = min(idxs), max(idxs)
    eng = [((ix - lo) / (hi - lo) if hi > lo else 0.5) for ix in idxs]
    eng = [min(1.0, e + (0.5 if p == referent else 0.0)) for e, p in zip(eng, cands)]
    return eng


def _scenarios():
    """The 4 candidate/recency scenarios (see module docstring). Each entry: (name, stored_facts, cands, referent)."""
    s1_facts = [("dog", "chase", "cat"), ("dog", "chase", "ball"), ("dog", "chase", "shoe")]
    s1_cands = ["cat", "ball", "shoe"]
    # S2: an ASYMMETRIC storage order -- cat/ball stored back-to-back (indices 0,1), shoe stored much later
    # (index 9, with 7 unrelated intervening facts for a different (agent,action) pair that default_context_fn's
    # own per-(a,v) filter must skip) -> recency = [0.0, 1/9, 1.0], a genuine near-tie at the low end.
    s2_facts = ([("dog", "chase", "cat"), ("dog", "chase", "ball")]
                + [("cat", "chase", "mouse")] * 7
                + [("dog", "chase", "shoe")])
    s2_cands = ["cat", "ball", "shoe"]
    s3_facts = s1_facts
    s3_cands = ["cat", "ball", "shoe"]
    s4_facts = [("dog", "chase", "cat"), ("dog", "chase", "ball"), ("dog", "chase", "shoe"),
                ("dog", "chase", "stick")]
    s4_cands = ["cat", "ball", "shoe", "stick"]
    return [
        ("S1_baseline", s1_facts, s1_cands, None),
        ("S2_near_tie_low", s2_facts, s2_cands, None),
        ("S3_referent_tie", s3_facts, s3_cands, "ball"),
        ("S4_four_candidate", s4_facts, s4_cands, None),
    ]


def _decisive(meta):
    return bool(meta.get("decisive"))


def _spread(meta):
    return float(meta.get("fed_spread_hz", 0.0))


def run_seed(seed: int, value_train_trials: int = 40) -> dict:
    """Build ONE real trained ValueChoiceProductionOrgan at `seed` (~267s-class, matches the RANK-1 GO default
    `value_train_trials=40`), then run all 4 scenarios' OFF/ON/ON+LESION arms against it (cheap -- only the tiny
    spiking value-WTA `choose()` call repeats, not the build+value-train)."""
    _clear_flags()
    out = {"seed": int(seed), "value_train_trials": int(value_train_trials)}

    from research.runners.value_choice_production_organ import ValueChoiceProductionOrgan
    import research.runners.value_choice_production_organ as VC

    t0 = time.time()
    vco = ValueChoiceProductionOrgan(seed=seed, value_train_trials=value_train_trials)
    vco.ensure_built()
    build_s = time.time() - t0

    scenario_results = []
    for name, facts, cands, referent in _scenarios():
        fchat = _FakeChat(facts, referent=referent)

        _clear_flags()
        ctx_off = VC.default_context_fn(fchat)
        off_eng = ctx_off("dog", "chase", cands)
        expect_off = _host_formula(facts, "dog", "chase", cands, referent=referent)

        _set_flags(on=True)
        ctx_on = VC.default_context_fn(fchat)
        on_eng = ctx_on("dog", "chase", cands)

        _set_flags(on=True, lesion=True)
        ctx_lesion = VC.default_context_fn(fchat)
        lesion_eng = ctx_lesion("dog", "chase", cands)
        _clear_flags()

        chosen_off, meta_off = vco.choose(cands, off_eng, lesion=False)
        chosen_on, meta_on = vco.choose(cands, on_eng, lesion=False)
        chosen_on_shared_lesion, meta_on_shared_lesion = vco.choose(cands, lesion_eng, lesion=False)
        # cheap sanity re-check of the PRE-EXISTING critic-level G_LESION (mean-pin) against the neurally-
        # mediated input -- expected trivially decisive-collapse by construction (fed = full(n, mean(V))
        # discards the input array's content regardless of provenance); reported, not gated on.
        chosen_on_critic_lesion, meta_on_critic_lesion = vco.choose(cands, on_eng, lesion=True)

        off_spread = _spread(meta_off)
        on_spread = _spread(meta_on)
        lesion_spread = _spread(meta_on_shared_lesion)
        spread_attrib = attributable_to(
            "seed %d %s: REAL-critic fed spread attributable to the shared-salience pathway" % (seed, name),
            on_spread, lesion_spread)

        match = bool(chosen_on == chosen_off)
        improves = bool(_decisive(meta_on) and not _decisive(meta_off))
        matches_or_improves = bool(match or improves)
        lesion_reverts = bool(
            (spread_attrib is not None and spread_attrib >= 0.5)
            or (chosen_on_shared_lesion != chosen_on and _decisive(meta_on)))

        scenario_results.append({
            "name": name, "candidates": cands,
            "off_eng": off_eng, "on_eng": on_eng, "shared_lesion_eng": lesion_eng,
            "off_eng_matches_independent_host_formula": bool(
                all(abs(a - b) < 1e-9 for a, b in zip(off_eng, expect_off))),
            "on_eng_differs_from_off": bool(any(abs(a - b) > 1e-6 for a, b in zip(on_eng, off_eng))),
            "chosen_off": chosen_off, "chosen_on": chosen_on,
            "chosen_on_shared_lesion": chosen_on_shared_lesion,
            "chosen_on_critic_lesion (sanity, not gated)": chosen_on_critic_lesion,
            "meta_off": meta_off, "meta_on": meta_on, "meta_on_shared_lesion": meta_on_shared_lesion,
            "meta_on_critic_lesion (sanity, not gated)": meta_on_critic_lesion,
            "off_spread_hz": off_spread, "on_spread_hz": on_spread, "shared_lesion_spread_hz": lesion_spread,
            "spread_attributable_to_shared_pathway": spread_attrib,
            "match": match, "improves_over_off": improves, "matches_or_improves": matches_or_improves,
            "lesion_reverts": lesion_reverts,
            "critic_lesion_sanity_collapses (expected trivially True)": bool(not _decisive(meta_on_critic_lesion)),
        })

    n = len(scenario_results)
    g_off_identical = all(s["off_eng_matches_independent_host_formula"] for s in scenario_results)
    g_on_loadbearing = all(s["on_eng_differs_from_off"] for s in scenario_results)
    n_matches_or_improves = sum(1 for s in scenario_results if s["matches_or_improves"])
    n_lesion_reverts = sum(1 for s in scenario_results if s["lesion_reverts"])
    g_matches_or_improves = n_matches_or_improves >= 3
    g_lesion_loadbearing = n_lesion_reverts >= 3

    out["build_seconds"] = round(build_s, 2)
    out["scenarios"] = scenario_results
    out["n_scenarios"] = n
    out["n_matches_or_improves"] = n_matches_or_improves
    out["n_lesion_reverts"] = n_lesion_reverts
    out["g_off_identical"] = bool(g_off_identical)
    out["g_on_loadbearing"] = bool(g_on_loadbearing)
    out["g_matches_or_improves"] = bool(g_matches_or_improves)
    out["g_lesion_loadbearing"] = bool(g_lesion_loadbearing)
    out["all_gates_pass"] = bool(g_off_identical and g_on_loadbearing and g_matches_or_improves
                                 and g_lesion_loadbearing)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="single-seed worker mode")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="controller mode: subprocess-fan these seeds")
    ap.add_argument("--value-train-trials", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    if a.seeds:
        # CONTROLLER: subprocess-fan one worker per seed (process isolation -- the trained critic + the process-
        # shared curiosity organ singleton must each be a fresh build per seed; see the sibling rank-4 de-risk's
        # module docstring for the identical rationale).
        per_seed = {}
        for s in a.seeds:
            t0 = time.time()
            r = subprocess.run(
                [sys.executable, "-m", "research.runners._value_choice_neural_context_6seed_derisk",
                 "--seed", str(s), "--value-train-trials", str(a.value_train_trials)],
                cwd=str(_REPO), capture_output=True, text=True, timeout=900,
                env={**os.environ, "SIM_NO_PROVENANCE": "1"},   # the controller's OWN --out carries provenance
            )
            if r.returncode != 0:
                per_seed[str(s)] = {"seed": s, "error": r.stderr[-4000:], "returncode": r.returncode}
                continue
            line = None
            for ln in r.stdout.splitlines():
                if ln.startswith("RESULT_JSON:"):
                    line = ln[len("RESULT_JSON:"):]
            per_seed[str(s)] = json.loads(line) if line else {"seed": s, "error": "no RESULT_JSON line",
                                                               "stdout_tail": r.stdout[-2000:]}
            per_seed[str(s)]["wall_seconds"] = round(time.time() - t0, 2)
        n_pass = sum(1 for s in a.seeds if per_seed.get(str(s), {}).get("all_gates_pass"))
        n_seeds = len(a.seeds)
        result = {"mode": "controller", "seeds": a.seeds, "n_seeds": n_seeds, "n_pass": n_pass,
                  "verdict": "GO" if n_pass >= max(5, n_seeds - 1) and n_seeds >= 5 else "NO-GO",
                  "all_seeds_pass": bool(n_pass == n_seeds), "per_seed": per_seed}
    elif a.seed is not None:
        result = run_seed(a.seed, value_train_trials=a.value_train_trials)
        print("RESULT_JSON:" + json.dumps(result))
    else:
        ap.error("pass --seed N (worker) or --seeds N N N (controller)")
        return

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"wrote {a.out}")
    if a.seed is None:
        print(json.dumps({k: v for k, v in result.items() if k not in ("per_seed",)}, indent=2, default=str))
        if "per_seed" in result:
            for s, r in result["per_seed"].items():
                print(f"  seed {s}: all_gates_pass={r.get('all_gates_pass')} build_s={r.get('build_seconds')}"
                      + ("" if "error" not in r else f"  ERROR={r['error'][:200]}"))


if __name__ == "__main__":
    main()
