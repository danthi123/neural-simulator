"""DA-GATED ENCODING — production magnitude-store NO-REGRESSION soak (the flip gate, 2026-08-21).

WHY THIS EXISTS. da-gated-encoding is de-risked + wired default-OFF (research/runners/_da_encoding_wired_verify.py GO:
off-identical, on-load-bearing, lesion-severable). Its two WAVE-0 siblings (curiosity/ideation) flipped default-ON
because they change NO response content by construction. This one is DIFFERENT: on the production onebrain MAGNITUDE
store it scales a fact's stored |w| by a DA-derived gain g=clip(0.5,3.0,1+2(DA-0.5)) — so a below-tonic-DA fact is
stored WEAKER (g<1, floored 0.5). Flipping it default-ON can therefore change FUTURE recall. The composed no-regression
(_wave4_composed_flip_noregression.py) ran on the rf FAST-path where recall is magnitude-INVARIANT -> the da-encoding
lever moved ZERO variables there -> NOT evidence for this flip. This runner is the missing production-store soak.

THE HONEST QUESTION. Flipping encoding ON must not make production recall WORSE. Two regimes:
  (1) CLEAN read (sigma=0) — the dominant production case for a modest fact store: the RF substrate readout is a PHASE
      read, magnitude-INVARIANT, so g does not affect WHICH fact is recalled. Prediction: ZERO regression (every fact
      recalled OFF is recalled ON). If this holds, the flip is safe for the clean-read path.
  (2) READ STRESS (sigma>0) — the I-7-b knee: higher-gain facts have higher SNR and survive the read floor; a
      g=0.5 low-DA fact has LOWER SNR than its g=1 OFF counterpart, so it can regress. This is the BIOLOGICALLY
      INTENDED salience-gating (Lisman-Grace/Kandel D.16: DA gates entry into long-term memory), NOT a bug — BUT it is a
      real behaviour change. We characterise it over a REALISTIC DA distribution and ask whether the redistribution is
      NET-neutral-or-positive (the high-DA gains at least offset the low-DA losses), and WHERE regressions begin.

THE BATTERY. M distinct SVO facts on a shared magnitude store (real interference, unlike the I-7-b 2-fact probe), each
taught at a DA drawn from a realistic teaching distribution (mostly ~tonic, a salient minority high-DA, a low-engagement
minority low-DA). OFF arm: encoding_gain_fn=None (all g=1 == today's default). ON arm: encoding_gain_fn returns the
per-fact g. Both arms are the SAME production OneBrainComposer with the SAME facts + SAME read-damage draws; the ONLY
difference is the write gain. 6 seeds (42/43/44/100/101/102).

VERDICT (GO = the flip is safe):
  * CLEAN (sigma=0): n_regressed == 0 on every seed (magnitude-invariant -> the flip cannot hurt a clean read).  [HARD]
  * MOAT: an UNSTORED cue abstains on both arms at every sigma (encoding never manufactures a fact).                [HARD]
  * STRESS net: at every swept sigma, aggregate recall_ON >= recall_OFF (the salience redistribution is net-neutral-or-
    positive on the realistic distribution — the high-DA durability gains offset the low-DA losses).                [HARD]
  * Characterisation (reported, not gated): per-sigma n_regressed / n_improved and the sigma where regressions begin.

Run (numpy-CPU, foreground/background, 0 agent tokens):
  SIM_BACKEND=numpy python -u -m research.runners._da_encoding_noregression_soak
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging
logging.getLogger().setLevel(logging.ERROR)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402
from research.runners._burndown_I7_dopamine_encoding_deploy_derisk import (  # noqa: E402
    da_to_encoding_gain, _query_under_damage,
)

SEEDS = [42, 43, 44, 100, 101, 102]
D = 64
K_DA = 2.0            # == the I-7-b / consolidation-probe2 production default
DA_BASELINE = 0.5     # tonic
SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]   # read-stress sweep (0 = clean production case)

# a vocab large enough to mint M distinct SVO facts with distinct (agent, action) cues (so a cue picks out ONE fact).
_AGENTS = ["dog", "cat", "bird", "fish", "wolf", "deer", "frog", "hawk", "mouse", "goat", "seal", "lion"]
_ACTIONS = ["eat", "chase", "see", "find", "want", "hear", "reach", "hold", "push", "carry", "watch", "lead"]
_PATIENTS = ["grass", "apple", "river", "home", "north", "seed", "leaf", "rock", "cloud", "root", "sand", "hill"]

# a REALISTIC teaching-DA distribution: mostly ~tonic, a salient minority high, a low-engagement minority low.
# (fractions sum to 1; the low block is the regression risk — g floored 0.5.)
_DA_HIGH, _DA_TONIC, _DA_LOW = 1.24, 0.5, 0.05           # per da_mode_drives_chat's afferent calibration
_FRAC_HIGH, _FRAC_LOW = 0.30, 0.30                       # 30% salient, 30% low-engagement, 40% tonic


def _battery(m):
    """M distinct facts with distinct (agent, action) cues + a per-fact DA from the realistic distribution. Deterministic
    (no RNG) so OFF and ON see the identical facts + DA schedule. Returns (facts, das, unstored_cue)."""
    facts, das = [], []
    n_high = int(round(m * _FRAC_HIGH))
    n_low = int(round(m * _FRAC_LOW))
    for i in range(m):
        a, act, p = _AGENTS[i % len(_AGENTS)], _ACTIONS[i % len(_ACTIONS)], _PATIENTS[(i * 7) % len(_PATIENTS)]
        facts.append((a, act, p))
        if i < n_high:
            das.append(_DA_HIGH)
        elif i < n_high + n_low:
            das.append(_DA_LOW)
        else:
            das.append(_DA_TONIC)
    unstored = ("river", "swim")   # a cue never taught (moat probe): 'swim' is not in _ACTIONS
    return facts, das, unstored


def _build(seed, facts, das, *, encoding_on):
    """Build a production OneBrainComposer and store the battery. encoding_on=False -> encoding_gain_fn=None (all g=1 ==
    today's default). encoding_on=True -> per-fact g from the DA schedule (holder set before each store)."""
    vocab = sorted(set(_AGENTS + _ACTIONS + _PATIENTS + ["swim"]))
    holder = {"g": 1.0}
    gain_fn = None if not encoding_on else (lambda: holder["g"])
    c = OneBrainComposer(seed=seed, D=D, vocab=vocab, k_max=len(facts) + 4, enable_batched=False,
                         enable_rf_cudagraph=False, enable_csr_cache=False, enable_spiking_cleanup=False,
                         encoding_gain_fn=gain_fn)
    for (a, act, p), da in zip(facts, das):
        holder["g"] = da_to_encoding_gain(da, DA_BASELINE, K_DA)
        c.store(a, act, p)
    return c


def _recall_set(comp, facts, sigma, seed):
    """Per-fact recall under a fixed read-stress sigma. Returns a boolean list (fact i recalled correctly). sigma=0 ->
    the clean magnitude-invariant read. Each fact gets its own reproducible damage draw (stable across arms)."""
    ok = []
    for i, (a, act, p) in enumerate(facts):
        if sigma <= 0.0:
            rec = comp.query_patient(a, act)
        else:
            rec = _query_under_damage(comp, a, act, sigma, seed * 100003 + i)
        ok.append(rec == p)
    return ok


def _moat_holds(comp, unstored, sigma, seed):
    a, act = unstored
    if sigma <= 0.0:
        return comp.query_patient(a, act) is None
    return _query_under_damage(comp, a, act, sigma, seed * 100003 + 99999) is None


def main():
    m = 20   # battery size (>> the I-7-b 2 facts -> real shared-store interference)
    facts, das, unstored = _battery(m)
    per_seed = []
    clean_regressions_total = 0
    moat_fail_total = 0
    stress_net_violations = 0   # sigmas where recall_ON < recall_OFF

    for seed in SEEDS:
        c_off = _build(seed, facts, das, encoding_on=False)
        c_on = _build(seed, facts, das, encoding_on=True)
        rows = []
        for sigma in SIGMAS:
            off = _recall_set(c_off, facts, sigma, seed)
            on = _recall_set(c_on, facts, sigma, seed)
            regressed = [i for i in range(m) if off[i] and not on[i]]     # recalled OFF, lost ON (harmful)
            improved = [i for i in range(m) if on[i] and not off[i]]      # recalled ON, gained vs OFF (beneficial)
            moat_off = _moat_holds(c_off, unstored, sigma, seed)
            moat_on = _moat_holds(c_on, unstored, sigma, seed)
            rows.append({
                "sigma": sigma, "recall_off": int(sum(off)), "recall_on": int(sum(on)), "of": m,
                "n_regressed": len(regressed), "n_improved": len(improved),
                "regressed_idx": regressed, "improved_idx": improved,
                "moat_off": bool(moat_off), "moat_on": bool(moat_on),
            })
            if sigma <= 0.0:
                clean_regressions_total += len(regressed)
            else:
                if sum(on) < sum(off):
                    stress_net_violations += 1
            if not (moat_off and moat_on):
                moat_fail_total += 1
        per_seed.append({"seed": seed, "sweep": rows})

    go_clean = (clean_regressions_total == 0)
    go_moat = (moat_fail_total == 0)
    go_stress_net = (stress_net_violations == 0)
    go = bool(go_clean and go_moat and go_stress_net)

    # characterisation: the smallest sigma at which ANY seed shows a regression (where salience-gating starts to bite).
    first_regress_sigma = None
    for sigma in SIGMAS:
        if sigma <= 0.0:
            continue
        if any(any(r["sigma"] == sigma and r["n_regressed"] > 0 for r in ps["sweep"]) for ps in per_seed):
            first_regress_sigma = sigma
            break

    from tools.verdict import Verdict
    v = Verdict("DA-gated encoding default-ON is safe on the production magnitude store (no-regression soak)")
    v.require("CLEAN read (sigma=0): zero facts regress OFF->ON on every seed (magnitude-invariant)",
              clean_regressions_total, expect=0,
              note="the dominant production case for a modest fact store: a phase read is magnitude-invariant")
    v.require("MOAT: an unstored cue abstains on both arms at every sigma (encoding never manufactures a fact)",
              moat_fail_total, expect=0)
    v.require("STRESS net: recall_ON >= recall_OFF at every swept sigma (salience redistribution net-neutral-or-positive)",
              stress_net_violations, expect=0,
              note=f"realistic DA dist: {int(_FRAC_HIGH*100)}% high / {int(_FRAC_LOW*100)}% low / rest tonic")
    v.disabled("the spiking-cleanup read (enable_spiking_cleanup=False here for speed)",
               why="the magnitude-sensitivity lives in the substrate store's |w| + the read floor, exercised by the "
                   "host read-damage sweep (the I-7-b instrument); the spiking cleanup adds intrinsic noise of the "
                   "same KIND (a further read-stress point on this same sweep), not a different mechanism")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "research/runners/_da_encoding_noregression_soak.py",
        "coupling": "DA-gated encoding, production magnitude-store no-regression soak (the default-ON flip gate)",
        "config": {"seeds": SEEDS, "D": D, "k_da": K_DA, "battery_m": m, "sigmas": SIGMAS,
                   "da_high": _DA_HIGH, "da_tonic": _DA_TONIC, "da_low": _DA_LOW,
                   "frac_high": _FRAC_HIGH, "frac_low": _FRAC_LOW},
        "VERDICT": "GO" if go else "NO-GO", "status": decided["status"],
        "go_clean_zero_regression": go_clean, "go_moat": go_moat, "go_stress_net_nonnegative": go_stress_net,
        "clean_regressions_total": clean_regressions_total, "moat_fail_total": moat_fail_total,
        "stress_net_violations": stress_net_violations,
        "first_regression_sigma": first_regress_sigma,
        "per_seed": per_seed,
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
    }
    op = os.path.join(_REPO, "research", "findings", "raw", "_da_encoding_noreg", "soak.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 100
    print("\n" + bar, flush=True)
    print("  DA-GATED ENCODING — production magnitude-store NO-REGRESSION soak", flush=True)
    print(bar, flush=True)
    print(f"  battery={m} facts, {len(SEEDS)} seeds, DA dist {int(_FRAC_HIGH*100)}%hi/{int(_FRAC_LOW*100)}%lo/rest-tonic",
          flush=True)
    print(f"  CLEAN (sigma=0) regressions: {clean_regressions_total} (expect 0)", flush=True)
    print(f"  MOAT failures: {moat_fail_total} (expect 0)", flush=True)
    print(f"  STRESS net violations (recall_ON<recall_OFF): {stress_net_violations} (expect 0)", flush=True)
    print(f"  first regression appears at sigma: {first_regress_sigma}", flush=True)
    # a compact per-sigma aggregate (seed 42) for the log
    s42 = per_seed[0]["sweep"]
    for r in s42:
        print(f"    seed42 sigma={r['sigma']:.2f}  off={r['recall_off']:2d} on={r['recall_on']:2d}/{r['of']}  "
              f"regressed={r['n_regressed']} improved={r['n_improved']}  moat={r['moat_off']}&{r['moat_on']}", flush=True)
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'} ({decided['status']})", flush=True)
    print(f"  [saved] {op}\n" + bar, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
