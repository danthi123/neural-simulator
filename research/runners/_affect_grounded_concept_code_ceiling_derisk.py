"""GROUNDED-CONCEPT-CODE CEILING (2026-09-05) — the REQUIREMENTS de-risk for the surpass the affect-gate BOUNDARY named.

WHAT THE PRIOR BOUNDARY LEFT OPEN (do NOT re-derive it).
`2026-09-05-affect-gate-embodied-US-necessary-not-sufficient-concept-code-must-be-grounded-BOUNDARY.md` proved
(6-seed) that even a PERFECT embodied US cannot retire the affect SALIENCE GATE while the CONCEPT CODE is derived
from TEXT co-occurrence: a supervised, label-given, noise-free ridge CEILING over the text code reads ~0.000
worst-case recall@FP0 on the 164-word partition (102 affect / 62 neutral). The ceiling INSTRUMENT is validated
(1.000 worst-case on a cleanly-separable synthetic grounded code), so the ~0 is a genuine property of the TEXT
CODE, not a probe artifact. The named next mechanism: the concept code must be GROUNDED (a grounded-perception
teacher), not derived from text co-occurrence — an embodied US is NECESSARY but NOT SUFFICIENT.

THE OPEN, DECISION-RELEVANT QUESTION THIS RUNNER ANSWERS (which the BOUNDARY did not measure). The synthetic
control in that runner shows a FULLY-clean, FULLY-covered grounded code -> ceiling 1.0. But a REAL grounded-
perception teacher cannot deliver that: it will ground only a FRACTION of affect concepts (only some concepts'
referents are ever EXPERIENCED with a bodily consequence in the teacher), with NOISE, and it will FUSE a grounded
axis onto the (still register-confounded) text code rather than replace it. So the buildable next arc has a
REQUIREMENTS question the record has never quantified:
    * FUSION vs REPLACEMENT — does ADDING a grounded body-state axis to the confounded text code lift the ceiling,
      or must the text code be discarded? (Additive/cheap vs a full perception retrain — a big scoping fork.)
    * COVERAGE rho — what fraction of affect concepts must the teacher actually ground before the ceiling clears?
    * NOISE sigma — how noisy may the interoceptive/embodied signal be and still lift the ceiling?
This runner maps that (rho x sigma) frontier, reusing the SAME validated ceiling instrument, so the next builder
knows the SPEC a grounded teacher must hit BEFORE building the (expensive) grounded-experience stream.

THIS IS A REQUIREMENTS / DOSE-RESPONSE DE-RISK, NOT A GATE RETIREMENT. It does NOT claim the gate is retired: the
grounded body-state feature is a STAND-IN (declared) for what a grounded-perception teacher would produce — exactly
the oracle-stand-in discipline the embodied-US runner used for its US. The claim under test is: "IF grounded
perception adds a body-state axis to the concept code at coverage rho / noise sigma, does the code become
SEPARABLE (ceiling >= 0.5) where the text code is not (~0)?" — and at what (rho, sigma). A YES with a realistic
operating point unblocks + SPECS the grounded-teacher arc; a NO (needs near-perfect grounding) reshapes it.

THE GROUNDED BODY-STATE FEATURE (host is legit ONLY for world/body/perception rendering — the SAME boundary the
embodied-US runner + the board #49/#84 interoceptive relay use). Biologically motivated by the interoceptive-
grounding pattern already in production (`2026-08-19-embodied-affect-interoception-GO`): concepts whose referents
are experienced with the SAME bodily consequence come to SHARE interoceptive feature dimensions (feels-good
concepts cluster on one body-state axis, feels-bad on another) — the exact structure the BOUNDARY's synthetic
control models and that a text co-occurrence code lacks. We build a small G-dim grounded feature block where
+affect concepts load on axis 0, -affect on axis 1 (shared-consequence structure), scaled by the concept's affect
magnitude, degraded by:
    * COVERAGE rho: only a random rho-fraction of AFFECT concepts are grounded (experienced); the rest load ~0
      on the body-state axes (ungrounded -> look neutral there), as a real teacher that never experiences some
      abstract affect words would leave them.
    * NOISE sigma: additive Gaussian on every concept's body-state block (incl. neutral -> false grounding).
The concept representation under test is the FUSION: [ text_code | grounded_block ], row-L2-normalized (the SAME
treatment the text codes get), then read by the validated supervised ceiling probe.

WHY THIS IS NOT HOLLOW / CIRCULAR (the controls ARE the result):
  * NO-GROUNDING control (rho=0): the grounded block is the SAME-scale block with its SIGNAL zeroed (pure noise).
    If the fused ceiling at rho=0 is ~ the text ceiling (~0), then any lift at rho>0 is the grounding SIGNAL, not
    "extra dimensions help the ridge". THIS is the load-bearing anti-hollow control.
  * SHUFFLE control: permute the body-state signs across concepts (destroy the concept<->body-state binding).
    The ceiling must collapse — a positive result must depend on the RIGHT concept carrying the RIGHT body-state.
  * TEXT-ONLY baseline: reproduce the BOUNDARY's ~0 text ceiling on the SAME partition + seeds (like-for-like).
  * INSTRUMENT validation (reused verbatim): `synthetic_separable_gate` reads ceiling ~1.0 on a clean grounded
    code AND the text ceiling reads ~0 -> the probe DISCRIMINATES (not stuck at either rail).
  * GROUNDED-ONLY (replacement) arm: the body-state block ALONE, so FUSION-vs-REPLACEMENT is directly comparable.

MECHANISM DISCIPLINE. Reuse-by-import ONLY — the corpus, the 164-word partition, the text concept code, and the
validated ceiling instrument all come from the affect de-risk lane unchanged (NO reimplementation, NO `sim/` edit,
numpy-CPU). The ceiling probe is a supervised UPPER BOUND (the instrument, exactly as in the BOUNDARY finding);
the grounded feature is the world/body stand-in whose REQUIRED quality this runner measures. NOT WIRED: nothing
here touches affect_production_organ.py / wkv_mouth_generator.py (byte-unchanged; `_STRONG_MARGIN` stays).

PRE-REGISTERED GO GATE (fixed BEFORE the 6-seed; a REQUIREMENTS verdict, not a gate-retirement verdict):
  G1 LIFT           at a REALISTIC operating point (rho >= RHO_REAL, sigma <= SIGMA_REAL) the grounded-FUSED code's
                    worst-case ceiling (min across seeds) >= CEIL_GO_BAR (0.5), i.e. it CLEARS where text is ~0.
  G2 ATTRIBUTABLE   the lift is the GROUNDING SIGNAL: the no-grounding control (rho=0) AND the shuffle control both
                    stay <= text_ceiling + ATTRIB_MARGIN (the extra dims / a random axis do NOT manufacture it).
  G3 INSTRUMENT     the ceiling instrument DISCRIMINATES: synthetic clean-code ceiling >= 0.5 AND text ceiling < 0.2
                    on the SAME partition + seeds (it is not stuck at either rail).
GO iff G1 AND G2 AND G3  ==> "the grounded-perception teacher arc is de-risked; its required signal spec is the
                             (rho, sigma) frontier reported below." (NOT "the gate is retired".)

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._affect_grounded_concept_code_ceiling_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._affect_grounded_concept_code_ceiling_derisk \
                  --seeds 42 43 44 100 101 102 \
                  --out research/findings/raw/_affect_grounded_concept_code_ceiling_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import logging as _logging
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

# --- reuse-by-import: the SAME de-risked corpus / partition / code / ceiling primitives (NO reimplementation) ----
from research.runners._affect_distributional_tag_derisk import (  # noqa: E402
    WARRINER, load_stories,
)
from research.runners._affect_experienced_opponent_gate_derisk import (  # noqa: E402
    _STRONG_MARGIN, CANONICAL_SEEDS, resample_stories, build_partition, _codes_for,
)
from research.runners._affect_embodied_us_gate_derisk import (  # noqa: E402
    code_separability_ceiling, synthetic_separable_gate,
)
from tools.lab import void_if, undefined_if_empty, attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_grounded_concept_code_ceiling.json"

CEIL_GO_BAR = 0.5        # pre-registered: the ceiling the grounded-fused code must clear (== the boundary's bar)
RHO_REAL = 0.6           # "realistic" coverage a grounded teacher could plausibly deliver (grounds >=60% of affect)
SIGMA_REAL = 1.0         # "realistic" body-state noise at the realistic operating point
ATTRIB_MARGIN = 0.15     # G2: control ceilings must stay within this of the text ceiling (no spurious lift)
TEXT_CEIL_MAX = 0.20     # G3: the text code ceiling must read low (reproduce the boundary) for the instrument test

RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SIGMA_GRID = (0.0, 0.5, 1.0, 2.0)


def grounded_block(part_words, raw_gate, seed, rho, sigma, G=8, sep=6.0, shuffle=False):
    """Build the grounded body-state FEATURE block for the partition words (host = world/body boundary, a declared
    stand-in for a grounded-perception teacher's output). +affect concepts load on axis 0, -affect on axis 1
    (shared-bodily-consequence structure -- the interoceptive-grounding pattern), scaled by affect magnitude;
    only a rho fraction of AFFECT concepts are grounded (coverage); Gaussian noise sigma on every concept
    (incl. neutral = false grounding). Returns a (n x G) non-negative-ish block (PPMI-like), same construction as
    the validated synthetic instrument so the ceiling reads it fairly.

    rho=0 -> the SIGNAL is fully absent (pure-noise block, same scale) = the anti-hollow NO-GROUNDING control.
    shuffle=True -> the body-state sign/magnitude is permuted across concepts (binding destroyed) = the SHUFFLE
    control. Neither should lift the ceiling above the text baseline if the lift is genuine grounding signal."""
    rng = np.random.default_rng(seed + 91_000)
    n = len(part_words)
    val = np.array([(WARRINER[w][0] - 5.0) / 4.0 for w in part_words])    # signed true affect magnitude in ~[-1,1]
    sign = np.sign(val)
    mag = np.abs(val)
    aff_idx = np.where(raw_gate)[0]
    # COVERAGE: only a rho-fraction of AFFECT concepts are actually grounded (experienced). Neutral concepts are
    # never grounded (mag already ~small for them, and they get zeroed on the signal axes below anyway).
    grounded = np.zeros(n, bool)
    if rho > 0 and len(aff_idx) > 0:
        k = int(round(rho * len(aff_idx)))
        chosen = rng.choice(aff_idx, size=k, replace=False) if k > 0 else np.array([], int)
        grounded[chosen] = True
    if shuffle:                                                           # destroy the concept<->body-state binding
        perm = rng.permutation(n)
        sign = sign[perm]; mag = mag[perm]; grounded = grounded[perm]
    block = np.abs(rng.standard_normal((n, G))) * sigma                   # base body-state noise (PPMI-like, >=0)
    signal = np.zeros(n)
    signal[grounded] = mag[grounded]                                      # only grounded concepts carry the axis load
    pos = grounded & (sign > 0)
    neg = grounded & (sign < 0)
    block[pos, 0] += sep * signal[pos]                                    # +affect concepts share body-state axis 0
    block[neg, 1] += sep * signal[neg]                                    # -affect concepts share body-state axis 1
    return block, int(grounded.sum())


def _fuse(text_codes, block):
    """FUSION: concatenate the text concept code and the grounded body-state block, then row-L2-normalize (the SAME
    treatment the text codes receive) so neither block is scale-privileged for the ridge ceiling."""
    fused = np.concatenate([text_codes, block], axis=1)
    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12)
    return fused


def run_seed(seed, stories, part_words, raw_gate, n_hub, window, min_count, resample_frac, G, verbose=False):
    """One seed: build the text code; measure text-only, grounded-only, and FUSED ceilings across the (rho,sigma)
    grid; + the shuffle control at the realistic operating point; + the reused synthetic instrument validation."""
    sub = resample_stories(stories, resample_frac, seed)
    vocab, codes, _codes_read, _rel = _codes_for(sub, n_hub, window, min_count)
    widx = {w: i for i, w in enumerate(vocab)}
    part_idx = np.array([widx[w] for w in part_words])
    text_codes = np.asarray(codes[part_idx], float)
    D = text_codes.shape[1]

    # TEXT-ONLY baseline (reproduce the boundary's ~0 ceiling, like-for-like)
    text_ceiling = code_separability_ceiling(text_codes, raw_gate, seed)

    # (rho x sigma) grid: FUSED ceiling + GROUNDED-ONLY ceiling
    grid = []
    for rho in RHO_GRID:
        for sigma in SIGMA_GRID:
            block, n_grounded = grounded_block(part_words, raw_gate, seed, rho, sigma, G=G)
            fused = _fuse(text_codes, block)
            c_fused = code_separability_ceiling(fused, raw_gate, seed)
            gonly = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-12)
            c_gonly = code_separability_ceiling(gonly, raw_gate, seed)
            grid.append({"rho": rho, "sigma": sigma, "n_grounded": n_grounded,
                         "fused_ceiling": c_fused, "grounded_only_ceiling": c_gonly})

    # SHUFFLE control at the realistic operating point (binding destroyed)
    blk_sh, _ = grounded_block(part_words, raw_gate, seed, RHO_REAL, SIGMA_REAL, G=G, shuffle=True)
    shuffle_ceiling = code_separability_ceiling(_fuse(text_codes, blk_sh), raw_gate, seed)

    # NO-GROUNDING control (rho=0 at the realistic sigma) is already in the grid; pull it out explicitly
    blk0, _ = grounded_block(part_words, raw_gate, seed, 0.0, SIGMA_REAL, G=G)
    nogrounding_ceiling = code_separability_ceiling(_fuse(text_codes, blk0), raw_gate, seed)

    # INSTRUMENT validation (reused verbatim): the clean synthetic grounded code must read ~1.0
    synth = synthetic_separable_gate(seed, raw_gate, D)

    def _cell(rho, sigma):
        return next(g["fused_ceiling"] for g in grid if g["rho"] == rho and g["sigma"] == sigma)

    real_fused = _cell(RHO_REAL, SIGMA_REAL)
    if verbose:
        print(f"  [seed {seed}] D={D} text_ceiling={text_ceiling:.3f} | fused@(rho={RHO_REAL},sig={SIGMA_REAL})="
              f"{real_fused:.3f} | no-grounding={nogrounding_ceiling:.3f} shuffle={shuffle_ceiling:.3f} | "
              f"synth_ceiling={synth['code_ceiling']:.3f}", flush=True)
    return {"seed": int(seed), "code_dim": int(D), "text_ceiling": text_ceiling, "grid": grid,
            "real_fused_ceiling": real_fused, "nogrounding_ceiling": nogrounding_ceiling,
            "shuffle_ceiling": shuffle_ceiling, "synth_code_ceiling": float(synth["code_ceiling"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=CANONICAL_SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny corpus -- proves it RUNS + controls live")
    ap.add_argument("--max-stories", type=int, default=60000)
    ap.add_argument("--resample-frac", type=float, default=0.8)
    ap.add_argument("--n-hub", type=int, default=64, help="concept code dim (matches the affect lane operating point)")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--grounded-dim", type=int, default=8, help="body-state feature block width G")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = [a.seeds[0]] if a.smoke else a.seeds
    max_stories = min(a.max_stories, 8000) if a.smoke else a.max_stories
    min_count = 2 if a.smoke else a.min_count

    t0 = time.time()
    print(f"[grounded-code-ceiling] seeds={seeds} smoke={a.smoke} max_stories={max_stories} n_hub={a.n_hub} "
          f"G={a.grounded_dim} backend={os.environ.get('SIM_BACKEND')}", flush=True)
    stories = load_stories(max_stories)
    part_words, raw_gate = build_partition(stories, seeds, a.resample_frac, min_count)
    void_if(len(part_words) < 20, f"only {len(part_words)} common partition words")
    n_pos, n_neg = int(raw_gate.sum()), int((~raw_gate).sum())
    void_if(n_pos == 0 or n_neg == 0, f"degenerate partition n_pos={n_pos} n_neg={n_neg}")
    print(f"  partition: {len(part_words)} common words | raw-gated(affect)={n_pos} raw-excluded(neutral)={n_neg}",
          flush=True)

    rows = [run_seed(s, stories, part_words, raw_gate, a.n_hub, a.window, min_count, a.resample_frac,
                     a.grounded_dim, verbose=True) for s in seeds]

    # ── aggregate the (rho x sigma) frontier (worst-case + mean across seeds) ─────────────────────────────────────
    frontier = []
    for rho in RHO_GRID:
        for sigma in SIGMA_GRID:
            vals = [next(g["fused_ceiling"] for g in r["grid"] if g["rho"] == rho and g["sigma"] == sigma)
                    for r in rows]
            gonly = [next(g["grounded_only_ceiling"] for g in r["grid"] if g["rho"] == rho and g["sigma"] == sigma)
                     for r in rows]
            frontier.append({"rho": rho, "sigma": sigma, "fused_worst": float(min(vals)),
                             "fused_mean": float(np.mean(vals)), "grounded_only_worst": float(min(gonly)),
                             "clears_go_bar": bool(min(vals) >= CEIL_GO_BAR)})

    text_ceiling_worst = float(max(r["text_ceiling"] for r in rows))      # worst = HIGHEST (hardest to call it low)
    text_ceiling_mean = float(np.mean([r["text_ceiling"] for r in rows]))
    real_fused_worst = float(min(r["real_fused_ceiling"] for r in rows))
    real_fused_mean = float(np.mean([r["real_fused_ceiling"] for r in rows]))
    nogrounding_worst = float(max(r["nogrounding_ceiling"] for r in rows))
    shuffle_worst = float(max(r["shuffle_ceiling"] for r in rows))
    synth_ceiling_worst = float(min(r["synth_code_ceiling"] for r in rows))

    # smallest rho that clears the GO bar (worst-case) at each sigma -> the teacher's required coverage spec
    coverage_spec = {}
    for sigma in SIGMA_GRID:
        clearing = [f["rho"] for f in frontier if f["sigma"] == sigma and f["clears_go_bar"]]
        coverage_spec[str(sigma)] = (min(clearing) if clearing else None)

    # ── GO CRITERIA (pre-registered) ──────────────────────────────────────────────────────────────────────────────
    g1 = bool(real_fused_worst >= CEIL_GO_BAR)
    g2 = bool(nogrounding_worst <= text_ceiling_worst + ATTRIB_MARGIN
              and shuffle_worst <= text_ceiling_worst + ATTRIB_MARGIN)
    g3 = bool(synth_ceiling_worst >= CEIL_GO_BAR and text_ceiling_worst < TEXT_CEIL_MAX)
    go = bool(g1 and g2 and g3)

    v = Verdict("grounded-concept-code ceiling: is the (grounded-fused) lift interpretable + attributable?")
    v.require("partition non-degenerate (affect + neutral both present)", measured=(n_pos > 0 and n_neg > 0),
              expect=True)
    v.require("the ceiling INSTRUMENT discriminates (synthetic clean code reads >=0.5, text code reads <0.2)",
              measured=(synth_ceiling_worst >= CEIL_GO_BAR and text_ceiling_worst < TEXT_CEIL_MAX), expect=True)
    v.control("the LIFT is the grounding SIGNAL (fused@realistic vs the no-grounding rho=0 control)",
              treatment=real_fused_mean, control=nogrounding_worst, min_separation=0.2)
    verdict_earned = v.decide(go=go, verbose=False)

    attributable_to("grounded-fused ceiling (vs the text-only ceiling)", real_fused_mean, text_ceiling_mean)
    attributable_to("grounded-fused ceiling (vs the no-grounding rho=0 control)", real_fused_mean, nogrounding_worst)
    attributable_to("grounded-fused ceiling (vs the shuffle-binding control)", real_fused_mean, shuffle_worst)

    tag = f"{len(seeds)}-seed" if not a.smoke else "SMOKE(1-seed)"
    if go:
        verdict = (
            f"GO ({tag}) -- the grounded-perception concept-code arc is DE-RISKED (a REQUIREMENTS verdict, NOT a gate "
            f"retirement). FUSING a grounded body-state axis onto the register-confounded TEXT code lifts the "
            f"supervised separability ceiling from {text_ceiling_worst:.3f} worst-case (text-only, reproducing the "
            f"BOUNDARY) to {real_fused_worst:.3f} worst-case at a REALISTIC operating point (coverage rho>={RHO_REAL}, "
            f"noise sigma<={SIGMA_REAL}) -- CLEARING the {CEIL_GO_BAR} bar the text code cannot. The lift is the "
            f"grounding SIGNAL: the no-grounding control (rho=0, {nogrounding_worst:.3f}) and the shuffle-binding "
            f"control ({shuffle_worst:.3f}) both stay at the text baseline, so it is not the extra dimensions or a "
            f"random axis. FUSION suffices (the text code need NOT be discarded -- the grounded axis is ADDITIVE). "
            f"REQUIRED SPEC for the grounded teacher = the (rho,sigma) frontier + coverage_spec in the artifact: the "
            f"minimum coverage that clears the bar at each noise level. NEXT: build the grounded-experience stream "
            f"(the embodied/interoceptive US delivering a per-concept body-state via the board #49/#84 relay + a "
            f"Hebbian convergence binding it as a concept feature, the vision->concept `_genfrontier_capstone` "
            f"template) to hit that spec. Brain-based (the body-state is the world/body boundary; the ceiling is the "
            f"instrument); NO sim/ edit; NOT wired.")
    else:
        miss = [k for k, ok in (("G1_lift_clears_bar", g1), ("G2_attributable", g2), ("G3_instrument", g3)) if not ok]
        verdict = (
            f"BOUNDARY/PARTIAL ({tag}, build-informative) -- grounded-fused ceiling {real_fused_worst:.3f} worst-case "
            f"at the realistic operating point vs text-only {text_ceiling_worst:.3f}; no-grounding {nogrounding_worst:.3f}; "
            f"shuffle {shuffle_worst:.3f}; synthetic instrument {synth_ceiling_worst:.3f}. FAILED: {miss}. See the "
            f"(rho,sigma) frontier for what coverage/noise WOULD clear the bar. The fixed _STRONG_MARGIN gate in "
            f"affect_production_organ.py is UNCHANGED (this file wires nothing).")

    summary = {
        "probe": "affect_grounded_concept_code_ceiling_derisk (REQUIREMENTS de-risk: what must a grounded teacher deliver?)",
        "verdict": verdict, "GO": go,
        "G1_lift_clears_bar": g1, "G2_attributable": g2, "G3_instrument": g3,
        "text_ceiling_worst": text_ceiling_worst, "text_ceiling_mean": text_ceiling_mean,
        "grounded_fused_realistic_worst": real_fused_worst, "grounded_fused_realistic_mean": real_fused_mean,
        "nogrounding_control_worst": nogrounding_worst, "shuffle_control_worst": shuffle_worst,
        "synthetic_instrument_ceiling_worst": synth_ceiling_worst,
        "ceiling_go_bar": CEIL_GO_BAR, "rho_realistic": RHO_REAL, "sigma_realistic": SIGMA_REAL,
        "attrib_margin": ATTRIB_MARGIN, "text_ceil_max": TEXT_CEIL_MAX,
        "rho_sigma_frontier": frontier,
        "required_coverage_spec_by_sigma": coverage_spec,
        "n_pos_raw_gated": n_pos, "n_neg_raw_excluded": n_neg, "n_partition_words": len(part_words),
        "per_seed": [{"seed": r["seed"], "code_dim": r["code_dim"], "text_ceiling": r["text_ceiling"],
                      "real_fused_ceiling": r["real_fused_ceiling"], "nogrounding_ceiling": r["nogrounding_ceiling"],
                      "shuffle_ceiling": r["shuffle_ceiling"], "synth_code_ceiling": r["synth_code_ceiling"]}
                     for r in rows],
        "preconditions": verdict_earned["preconditions"], "verdict_earned_status": verdict_earned["status"],
        "verdict_undefined_reasons": verdict_earned["undefined_reasons"],
        "config": {"seeds": seeds, "smoke": a.smoke, "max_stories": max_stories, "resample_frac": a.resample_frac,
                   "n_hub": a.n_hub, "window": a.window, "min_count": min_count, "grounded_dim": a.grounded_dim,
                   "rho_grid": list(RHO_GRID), "sigma_grid": list(SIGMA_GRID), "backend": os.environ.get("SIM_BACKEND")},
        "mechanism": "TEXT concept code = build_cooccurrence -> codes_from_cooccurrence (the register-confounded "
                     "PPMI stream code, reused). GROUNDED body-state block = a G-dim interoceptive feature (host "
                     "world/body boundary, declared STAND-IN) where +/-affect concepts share body-state axes, at "
                     "coverage rho + noise sigma. FUSED = row-L2-normalized concat. CEILING = the validated "
                     "supervised ridge k-fold probe (reused from the embodied-US runner) = the upper bound on any "
                     "readout of the representation. Controls: rho=0 no-grounding (pure-noise same-scale block), "
                     "shuffle (binding destroyed), grounded-only (replacement), synthetic clean code (instrument).",
        "sources": [
            "2026-09-05-affect-gate-embodied-US-necessary-not-sufficient-concept-code-must-be-grounded-BOUNDARY.md "
            "-- proved the text code ceiling is ~0 (an embodied US is necessary-but-not-sufficient; the concept code "
            "must be grounded); this runner measures the REQUIRED grounded-signal spec for that named next arc.",
            "2026-08-19-embodied-affect-interoception-GO.md -- the interoceptive-grounding pattern (concepts sharing "
            "a bodily consequence share a body-state axis) that motivates the grounded feature structure; the board "
            "#49/#84 relay is the production delivery path for a real grounded US.",
            "2026-07-02-emerge34-perception-grounded-emergence-GO.md + _genfrontier_capstone_vision_to_concept_derisk "
            "-- the perception->concept convergence template (Hebbian, spiking) a grounded-affect teacher would reuse.",
            "Namburi, Tye et al. (2015, Nature) -- opponent valence populations bound to a real US, not lexical company.",
        ],
        "production_wiring": "NONE -- affect_production_organ.py and wkv_mouth_generator.py are byte-unchanged; this "
                             "is a standalone REQUIREMENTS de-risk (reuse-by-import only). _STRONG_MARGIN unchanged.",
        "HONEST_RESIDUALS": "(1) the grounded body-state feature is a declared STAND-IN for a grounded-perception "
                            "teacher's output (the SAME oracle-stand-in discipline the embodied-US runner used); this "
                            "runner measures the REQUIRED signal quality, it does NOT deliver a real grounded stream "
                            "(no world grounds the TinyStories vocabulary with bodily consequences today -- that is "
                            "the named next build). (2) GO here means the grounded-teacher ARC is de-risked + specced, "
                            "NOT that the gate is retired. (3) the ceiling is a linear upper bound (the spiking "
                            "opponent's mild nonlinearity was measured NOT to help by the prior boundaries). (4) the "
                            "shared-axis body-state structure is the simplest grounded model; a real interoceptive "
                            "code may be richer but this is the conservative case (2 shared axes).",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    undefined_if_empty("partition-words", len(part_words), len(part_words), len(part_words))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[grounded-code-ceiling] text={text_ceiling_worst:.3f} | fused@realistic(rho={RHO_REAL},sig={SIGMA_REAL})="
          f"{real_fused_worst:.3f} | no-grounding={nogrounding_worst:.3f} shuffle={shuffle_worst:.3f} | "
          f"synth-instrument={synth_ceiling_worst:.3f}", flush=True)
    print(f"[grounded-code-ceiling] required coverage spec by sigma: {coverage_spec}", flush=True)
    print(f"[grounded-code-ceiling] VERDICT: {verdict}", flush=True)
    print(f"[grounded-code-ceiling] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
