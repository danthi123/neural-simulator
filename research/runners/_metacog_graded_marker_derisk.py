"""DE-RISK (§8 honesty=STATE-fidelity, Sub-arc B FIRST BUILD): does the metacog workspace's OWN spiking
balance-of-evidence margin (`metacog_production_organ.nmda_norm_margin` — the divisive-normalized NMDA-conductance
read, `2026-08-13-metacog-robust-confidence-GO.md`) resolve THREE ordinal confidence bands
{SPECULATE / HEDGE / ASSERT}, not just the two the production organ ships today (`confident` bool)?

THIS IS A MEASUREMENT DE-RISK, NOT A PRODUCTION WIRE-IN. Nothing here touches `webapp/server.py` / `judge()`'s
call sites; the graded marker ladder lives ENTIRELY in this file. `metacog_production_organ.py` is imported
UNCHANGED (verified below, `ORGAN_FILE_BASELINE_HASH`) — pure reuse-by-import of `MetacogProductionOrgan`,
`nmda_norm_margin`, `SIG_LO`/`SIG_HI`/`BASE_PA`/`READ_REPS`, and `judge()` (via the organ instance).

WHY THIS IS A GENUINELY DIFFERENT QUESTION from the two prior metacog residuals in this lane (both read via
`before_you_build.sh` before writing a line here):
  * `2026-08-27-metacog-hedge-confidence-band-recalibration-GO.md` / `2026-08-13-metacog-robust-confidence-GO.md`
    calibrate the EXISTING 2-band (confident/uncertain) split of THIS SAME `nmda_norm` margin. This asks whether
    that margin's dynamic range additionally resolves a THIRD, MIDDLE band — a strictly harder ordinal-separation
    question the 2-band GO never measured (a clean 2-way gap says nothing about whether the gap has a
    discriminable interior).
  * `2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md` / `2026-09-05-metacog-accumulation-to-bound-
    middle-band-PARTIAL.md` target a DIFFERENT, upstream mechanism — the EVIDENCE DERIVATION
    (`RFPhasorComposer._spiking_margin[_accum]`, a recall-competition spike-count read that feeds the workspace's
    evidence scalar) — and found ITS middle band nearly at chance (AUC~0.48-0.585). That is a different substrate
    (recall-competition spike counts) from the one this script measures (the workspace's own settled divisive-
    normalized NMDA-conductance balance, downstream of wherever the evidence came from). The task brief explicitly
    flags this as the "twice-measured hard case" to not silently re-run — this script re-uses NEITHER runner's
    composer/capture machinery; it drives `nmda_norm_margin` directly, exactly as the two GO findings above did.

THE MARKER LADDER (functional read-out phrasing only, per the honesty boundary — see `graded_marker_prefix`):
  ASSERT     margin >= boundary_med_hi   -> bare, unqualified answer
  HEDGE      boundary_lo_med <= margin < boundary_med_hi -> "I think -- my decision-margin reads this as ..."
  SPECULATE  margin <  boundary_lo_med   -> "I'd only guess -- my decision-margin reads this as ..."
Never a felt/phenomenal/conscious claim (`_HONESTY_BANNED_WORDS`, checked mechanically below, not just by eye).

SELF-CALIBRATION (no hand-set thresholds): at build time, on the SAME organ instance under test, a synthetic
LOW / MED / HIGH evidence battery (`CAL_LOW_EVIDENCE` / `CAL_MED_EVIDENCE` / `CAL_HIGH_EVIDENCE`, 6 points each,
spanning [0,1] with a fixed 0.125 buffer between adjacent bands so no battery point sits on a future boundary by
construction) is read through `nmda_norm_margin` and the two boundaries are placed exactly as the organ's own
`ensure_built()` places its single boundary today (midpoint of the clean gap between adjacent bands, or the
class-mean midpoint if the gap is not clean) -- `_boundary_between`, a literal generalization of that one function
to a second adjacent pair. All six battery bounds and both separability bars (`ORDINAL_AUC_BAR`, `SPEARMAN_BAR`,
`SHUFFLE_RHO_BAR`, `LESION_RHO_BAR`, `MIN_SEEDS_PASS`) are FIXED in this file BEFORE the first seed's numbers were
read (a single seed-42 smoke of the UNRELATED 2-band organ was read while scoping the reuse surface, at
`SIG_LO`/`SIG_HI`/`READ_REPS` -- the constants already shipped in `metacog_production_organ.py` -- never at this
script's own 3-band battery, which is what HARD RULE 2 forbids tuning against).

Note on the brief's "ride the scale-invariant margin_snr / SNR anchors" parenthetical: those anchors exist to
compare a DIFFERENT quantity (`mean_role_confidence`'s host cosine-similarity score) across CODEBOOK SIZES that
change that quantity's absolute scale (`SNR_LO`/`SNR_HI` in `metacog_production_organ.py`). `nmda_norm_margin` is
already a ratio of two conductances (Carandini & Heeger divisive normalization) with no codebook-size dependence
of its own -- there is no analogous scale-drift for this script's battery to anchor against, so no SNR-style remap
is applied; the two boundaries are self-calibrated directly from the margin's own LOW/MED/HIGH distributions.

SWEEP: `SWEEP_EVIDENCE_LEVELS` (9 points spanning evidence 0..1, i.e. drive SIG_LO=40..SIG_HI=260 pA per
`nmda_norm_margin`'s own evidence->pA map) reads `organ.judge(evidence, lesion=...)` at every level -- exercising
the ACTUAL production call path, not a bespoke read.

GO-GATE (pre-registered; see the constants above -- this docstring is written before the first 6-seed number was
read): on >=5/6 seeds, (1) the emitted band is monotone-non-decreasing across the sweep AND
Spearman(margin, evidence) >= 0.80; AND (2) BOTH adjacent-band gaps in the calibration battery are ordinally
separable -- pairwise AUC(low,med) and AUC(med,high) >= 0.75 AND each adjacent median-margin gap exceeds the
wider of its two flanking within-band standard deviations. A NO-GO here is a verdict on THIS single-snapshot
method (banked, not a capability retreat) -- the pre-named next mechanism is the accumulation-to-bound temporal
read already explored one level upstream (`_metacog_accumulation_to_bound_derisk.py`), applied to THIS workspace's
own settled competition instead of the recall composer's.

ANTI-CHEATS (all six implemented; see `run_seed_arm` / `aggregate`):
  (1) SHUFFLE/YOKE control -- permute the intact sweep's own margins across the evidence axis (fixed per-seed
      RNG); the shuffled arrangement must fail to track evidence (non-monotone or |rho| < SHUFFLE_RHO_BAR).
  (2) LESION-load-bearing -- a fresh-subprocess rebuild at the SAME seed with `nmda_norm_margin(..., lesion=True)`
      at every sweep level (removes the evidence DIFFERENTIAL, per the organ's own existing lesion contract);
      classified under the INTACT run's OWN boundaries (same seed -> same underlying network), it must collapse
      to all-SPECULATE and its rho-vs-evidence must fall near zero.
  (3) NOT-a-2-level-relabel -- the GO gate's condition (2) requires the MIDDLE boundary (low/med AND med/high)
      separable above the within-band spread, scored across the calibration battery's full distribution, never a
      single templated evidence level.
  (4) DETERMINISM/null -- the intact arm is run TWICE per seed in independent fresh subprocesses; every
      calibration number and every sweep margin must compare exactly equal between the two repeats.
  (5) BYTE-IDENTICAL OFF -- `metacog_production_organ.py` is never edited; `ORGAN_FILE_BASELINE_HASH` (captured
      via `git hash-object` before this file was written) is re-verified by `git hash-object` at aggregate time,
      an exact-compare per `docs/TERMS.md`'s byte-identical condition, not an inference from "we didn't edit it".
  (6) HONESTY BOUNDARY -- `graded_marker_prefix` phrasings are scanned by `_honesty_scan` against a banned-word
      list (feel/conscious/aware/experience/sentient/subjective/...); every band's phrase must score clean.

COMPUTE: numpy/CPU only (`SIM_BACKEND=numpy`, `CUDA_VISIBLE_DEVICES=''`, forced below); each arm subprocess is
wrapped in `tools/memcap.sh` (falls back to unwrapped with a recorded reason if the systemd --user scope is
unavailable, never silently). Two ~80-neuron assemblies + a meta pool (290 neurons total) -- each arm build+
battery+sweep runs in low single-digit seconds; 6 seeds x 3 arms (intact-a, intact-b, lesion) = 18 fresh builds.

Usage:
  # fast structural self-test (no brain build):
  python -m research.runners._metacog_graded_marker_derisk --selftest

  # one arm directly (used internally by --mode aggregate; runs one fresh build):
  python -m research.runners._metacog_graded_marker_derisk --mode arm --seed 42 --arm intact --repeat a \
      --out research/findings/raw/_metacog_graded_marker_derisk/s42_intact_a.json

  # full 6-seed de-risk (spawns the 18 arm subprocesses above, aggregates, decides the Verdict):
  python -m research.runners._metacog_graded_marker_derisk --mode aggregate \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/_metacog_graded_marker_derisk_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

# ── reuse-by-import, UNCHANGED (no sim/ edit; no metacog_production_organ.py edit at all) ──────────────────────
from research.runners.metacog_production_organ import (  # noqa: E402
    MetacogProductionOrgan, nmda_norm_margin, SIG_LO, SIG_HI, BASE_PA, READ_REPS,
)
from research.runners._self_schema_region_derisk import _auc, _spearman  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

# ── byte-identical-off anti-cheat: the exact git blob hash of metacog_production_organ.py captured via
# `git hash-object research/runners/metacog_production_organ.py` BEFORE this file was written, and never edited
# since (this script makes zero edits to that file -- pure reuse-by-import). Re-verified in `aggregate()`.
ORGAN_FILE_BASELINE_HASH = "803c5a28b926ec55094942a3dd7999986f48977e"
ORGAN_FILE_REL = os.path.join("research", "runners", "metacog_production_organ.py")

# ── the 3-band ladder ────────────────────────────────────────────────────────────────────────────────────────
BAND_SPECULATE, BAND_HEDGE, BAND_ASSERT = "SPECULATE", "HEDGE", "ASSERT"
BAND_ORDINAL = {BAND_SPECULATE: 0, BAND_HEDGE: 1, BAND_ASSERT: 2}

# ── self-calibration battery (LOW / MED / HIGH), fixed BEFORE any 3-band number was read (see docstring) ──────
# 6 points/band, 0.125 buffer between adjacent bands so no calibration point can straddle a future boundary.
CAL_LOW_EVIDENCE = tuple(float(x) for x in np.linspace(0.0, 0.25, 6))
CAL_MED_EVIDENCE = tuple(float(x) for x in np.linspace(0.375, 0.625, 6))
CAL_HIGH_EVIDENCE = tuple(float(x) for x in np.linspace(0.75, 1.0, 6))

# ── the sweep (>=3 required by the brief; 9 chosen for a robust Spearman/monotonicity read) ────────────────────
SWEEP_EVIDENCE_LEVELS = tuple(float(x) for x in np.linspace(0.0, 1.0, 9))

# ── pre-registered gate bars (fixed before the first 3-band number was read) ───────────────────────────────────
SPEARMAN_BAR = 0.80          # brief's own bar for condition (1)
ORDINAL_AUC_BAR = 0.75       # stricter than the sibling 2-band GO's type2_auc>=0.65 bar (a synthetic, noise-
                              # engineered 3-way split should be cleaner than a natural type-2 correctness read)
MIN_SEEDS_PASS = 5           # of 6, per the brief's GO gate
SHUFFLE_RHO_BAR = 0.30       # a shuffled/yoked margin arrangement must fall (well) below this |rho| to "fail to track"
LESION_RHO_BAR = 0.30        # same bar, applied to the lesion arm's rho-vs-evidence
LESION_SPECULATE_FRAC = 0.85 # >= this fraction of lesioned sweep points must classify SPECULATE

SEEDS_DEFAULT = (42, 43, 44, 100, 101, 102)

# ── honesty-boundary phrasing (functional read-out only; never phenomenal) ─────────────────────────────────────
_HONESTY_BANNED_WORDS = ("feel", "feels", "felt", "conscious", "aware", "experienc", "sentien", "subjectiv",
                          "qualia", "phenomenal")


def graded_marker_prefix(band: str) -> str:
    """The honest functional prefix for a graded marker band. Mirrors `metacog_production_organ.hedge_prefix`'s
    existing phrasing convention ("my decision-margin reads this as...") -- a read of the spiking margin, never a
    claim about what it is like to hold the belief."""
    if band == BAND_ASSERT:
        return ""  # bare -- the answer stands unqualified
    if band == BAND_HEDGE:
        return ("I think — my decision-margin reads this as moderately confident, so take it as "
                 "likely-but-not-certain: ")
    if band == BAND_SPECULATE:
        return ("I'd only guess — my decision-margin reads this as low-confidence, so take it as a "
                 "speculative guess, not an assertion: ")
    raise ValueError(f"unknown band {band!r}")


def _honesty_scan(text: str):
    """Mechanically scan a marker phrase for a banned phenomenal/experiential word. Returns the hit list (empty
    = clean). Run over EVERY band's phrase, not eyeballed."""
    low = text.lower()
    return [w for w in _HONESTY_BANNED_WORDS if w in low]


def classify_band(margin: float, boundary_lo_med: float, boundary_med_hi: float) -> str:
    if margin < boundary_lo_med:
        return BAND_SPECULATE
    if margin < boundary_med_hi:
        return BAND_HEDGE
    return BAND_ASSERT


def _boundary_between(lower_vals, upper_vals):
    """Place one boundary between two adjacent evidence-level battery distributions, EXACTLY generalizing
    `MetacogProductionOrgan.ensure_built`'s own 2-point placement rule: midpoint of the clean gap (max(lower) <
    min(upper)) when the gap is clean, else the midpoint of the two class means. Applied twice (low/med, med/high)
    for the 3-band ladder."""
    lower_vals = np.asarray(lower_vals, dtype=np.float64)
    upper_vals = np.asarray(upper_vals, dtype=np.float64)
    max_lower, min_upper = float(lower_vals.max()), float(upper_vals.min())
    mean_lower, mean_upper = float(lower_vals.mean()), float(upper_vals.mean())
    clean_gap = min_upper > max_lower
    boundary = 0.5 * (min_upper + max_lower) if clean_gap else 0.5 * (mean_upper + mean_lower)
    return {"boundary": float(boundary), "clean_gap": bool(clean_gap),
            "mean_lower": mean_lower, "mean_upper": mean_upper,
            "max_lower": max_lower, "min_upper": min_upper}


def three_band_calibration(bridge, xp, idx, snap):
    """Self-calibrate the two 3-band boundaries from a LOW/MED/HIGH synthetic evidence battery, reading
    `nmda_norm_margin` (UNCHANGED, reuse-by-import) at each battery point on the SAME built bridge under test.
    No hand-set thresholds -- both boundaries and both separability scores are DERIVED from this battery."""
    low_m = [nmda_norm_margin(bridge, xp, idx, snap, e, lesion=False) for e in CAL_LOW_EVIDENCE]
    med_m = [nmda_norm_margin(bridge, xp, idx, snap, e, lesion=False) for e in CAL_MED_EVIDENCE]
    high_m = [nmda_norm_margin(bridge, xp, idx, snap, e, lesion=False) for e in CAL_HIGH_EVIDENCE]

    lo_med = _boundary_between(low_m, med_m)
    med_hi = _boundary_between(med_m, high_m)

    auc_lo_med = _auc(np.array(low_m + med_m), np.array([False] * len(low_m) + [True] * len(med_m)))
    auc_med_hi = _auc(np.array(med_m + high_m), np.array([False] * len(med_m) + [True] * len(high_m)))

    gap_lo_med = float(np.median(med_m) - np.median(low_m))
    spread_lo_med = float(max(np.std(low_m), np.std(med_m)))
    gap_med_hi = float(np.median(high_m) - np.median(med_m))
    spread_med_hi = float(max(np.std(med_m), np.std(high_m)))

    return {
        "low_evidence_levels": list(CAL_LOW_EVIDENCE), "low_margins": [float(x) for x in low_m],
        "med_evidence_levels": list(CAL_MED_EVIDENCE), "med_margins": [float(x) for x in med_m],
        "high_evidence_levels": list(CAL_HIGH_EVIDENCE), "high_margins": [float(x) for x in high_m],
        "boundary_lo_med": lo_med["boundary"], "clean_gap_lo_med": lo_med["clean_gap"],
        "boundary_med_hi": med_hi["boundary"], "clean_gap_med_hi": med_hi["clean_gap"],
        "lo_med_detail": lo_med, "med_hi_detail": med_hi,
        "auc_lo_med": float(auc_lo_med), "auc_med_hi": float(auc_med_hi),
        "median_gap_lo_med": gap_lo_med, "spread_lo_med": spread_lo_med,
        "median_gap_med_hi": gap_med_hi, "spread_med_hi": spread_med_hi,
        "ordinal_separable_lo_med": bool(auc_lo_med >= ORDINAL_AUC_BAR and gap_lo_med > spread_lo_med),
        "ordinal_separable_med_hi": bool(auc_med_hi >= ORDINAL_AUC_BAR and gap_med_hi > spread_med_hi),
    }


# ── one arm, one fresh process (build once, no other arm shares this bridge) ────────────────────────────────────
def run_seed_arm(seed: int, arm: str, repeat: str) -> dict:
    assert arm in ("intact", "lesion")
    t0 = time.time()
    organ = MetacogProductionOrgan(seed=seed)
    organ.ensure_built()
    build_s = time.time() - t0

    result = {
        "seed": seed, "arm": arm, "repeat": repeat, "confidence_read": organ.confidence_read,
        "organ_calib_2band": organ.calib, "organ_threshold_2band": organ.threshold,
        "build_seconds": build_s, "sig_lo_pa": SIG_LO, "sig_hi_pa": SIG_HI, "base_pa": BASE_PA,
        "read_reps": READ_REPS,
    }

    boundaries = None
    if arm == "intact":
        cal = three_band_calibration(organ.bridge, organ.xp, organ.idx, organ.snap)
        result["calibration"] = cal
        boundaries = (cal["boundary_lo_med"], cal["boundary_med_hi"])

    lesion_flag = (arm == "lesion")
    sweep = []
    for e in SWEEP_EVIDENCE_LEVELS:
        j = organ.judge(float(e), lesion=lesion_flag)  # reuse-by-import: the PRODUCTION judge() call path
        row = {"evidence": float(e), "margin": j["balance"], "threshold_2band": j["threshold"],
               "confident_2band": j["confident"]}
        if boundaries is not None:
            row["band"] = classify_band(j["balance"], *boundaries)
        sweep.append(row)
    result["sweep"] = sweep
    result["elapsed_seconds"] = time.time() - t0
    return result


def _honesty_selftest():
    """Fast, no-brain-build structural check: every band's phrase is clean, `classify_band`/`_boundary_between`
    behave as specified, and the production organ file is byte-identical to the captured baseline."""
    problems = []
    for band in (BAND_SPECULATE, BAND_HEDGE, BAND_ASSERT):
        phrase = graded_marker_prefix(band)
        hits = _honesty_scan(phrase)
        if hits:
            problems.append(f"band {band} phrase contains banned word(s) {hits}: {phrase!r}")
    # classify_band boundary arithmetic
    if classify_band(0.0, 0.1, 0.2) != BAND_SPECULATE:
        problems.append("classify_band: below lo_med boundary must be SPECULATE")
    if classify_band(0.15, 0.1, 0.2) != BAND_HEDGE:
        problems.append("classify_band: between boundaries must be HEDGE")
    if classify_band(0.25, 0.1, 0.2) != BAND_ASSERT:
        problems.append("classify_band: at/above med_hi boundary must be ASSERT")
    # _boundary_between: clean gap -> exact midpoint; non-clean -> mean midpoint
    b = _boundary_between([0.0, 0.1], [0.3, 0.4])
    if not b["clean_gap"] or abs(b["boundary"] - 0.2) > 1e-9:
        problems.append(f"_boundary_between clean-gap case wrong: {b}")
    b2 = _boundary_between([0.0, 0.3], [0.1, 0.4])  # overlapping -> not clean
    if b2["clean_gap"]:
        problems.append(f"_boundary_between should have detected an overlapping (non-clean) gap: {b2}")
    # byte-identical-off: the production organ file must be untouched
    try:
        out = subprocess.run(["git", "hash-object", ORGAN_FILE_REL], cwd=_REPO,
                              capture_output=True, text=True, timeout=30)
        cur_hash = out.stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        cur_hash = f"<git hash-object failed: {exc}>"
    organ_unchanged = (cur_hash == ORGAN_FILE_BASELINE_HASH)
    if not organ_unchanged:
        problems.append(f"metacog_production_organ.py hash CHANGED: baseline={ORGAN_FILE_BASELINE_HASH} "
                         f"current={cur_hash}")
    return problems, organ_unchanged, cur_hash


def cmd_selftest(_args):
    problems, organ_unchanged, cur_hash = _honesty_selftest()
    print(f"organ file hash: baseline={ORGAN_FILE_BASELINE_HASH} current={cur_hash} "
          f"unchanged={organ_unchanged}")
    if problems:
        print("SELFTEST FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print("SELFTEST OK: honesty phrasing clean, classify_band/_boundary_between correct, organ file byte-identical.")
    return 0


def cmd_arm(args):
    result = run_seed_arm(args.seed, args.arm, args.repeat)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"[arm] seed={args.seed} arm={args.arm} repeat={args.repeat} -> {args.out} "
          f"({result['elapsed_seconds']:.2f}s)")
    return 0


# ── aggregate: spawn the 18 fresh-subprocess arms, then score ───────────────────────────────────────────────────
def _memcap_wrap(cmd, memcap_gb):
    memcap_script = os.path.join(_REPO, "tools", "memcap.sh")
    if memcap_gb and memcap_gb > 0 and os.path.exists(memcap_script):
        return ["bash", memcap_script, str(memcap_gb), "--"] + cmd, True
    return cmd, False


def _spawn_arm(python_exe, seed, arm, repeat, out_path, memcap_gb):
    base_cmd = [python_exe, "-u", "-m", "research.runners._metacog_graded_marker_derisk",
                "--mode", "arm", "--seed", str(seed), "--arm", arm, "--repeat", repeat, "--out", out_path]
    cmd, wrapped = _memcap_wrap(base_cmd, memcap_gb)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as exc:
        return {"seed": seed, "arm": arm, "repeat": repeat, "ok": False, "memcap_wrapped": wrapped,
                "error": f"timeout: {exc}", "elapsed": time.time() - t0}
    fallback_reason = None
    # memcap.sh refuses (exit 3) if systemd --user is genuinely unavailable -- fall back to unwrapped rather
    # than silently dropping the whole arm (memcap.sh's own doc: never silently drop the cap; at OUR level we
    # record the fallback loudly instead of hard-failing a cheap CPU-only tiny-network run over it).
    if wrapped and proc.returncode == 3 and "cannot enforce a cap" in (proc.stderr or ""):
        fallback_reason = "memcap.sh refused (systemd --user scope unavailable); ran UNWRAPPED"
        proc = subprocess.run(base_cmd, cwd=_REPO, capture_output=True, text=True, timeout=600)
        wrapped = False
    dt = time.time() - t0
    ok = (proc.returncode == 0) and os.path.exists(out_path)
    return {"seed": seed, "arm": arm, "repeat": repeat, "returncode": proc.returncode, "elapsed": dt,
            "ok": ok, "memcap_wrapped": wrapped, "memcap_fallback_reason": fallback_reason,
            "stderr_tail": "" if ok else (proc.stderr or "")[-3000:], "out_path": out_path}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _compare_intact_repeats(a, b):
    diffs = []
    for k in ("boundary_lo_med", "boundary_med_hi"):
        if a["calibration"][k] != b["calibration"][k]:
            diffs.append(k)
    for k in ("low_margins", "med_margins", "high_margins"):
        if a["calibration"][k] != b["calibration"][k]:
            diffs.append(k)
    sa = [r["margin"] for r in a["sweep"]]
    sb = [r["margin"] for r in b["sweep"]]
    if sa != sb:
        diffs.append("sweep_margins")
    ba = [r.get("band") for r in a["sweep"]]
    bb = [r.get("band") for r in b["sweep"]]
    if ba != bb:
        diffs.append("sweep_bands")
    return {"byte_identical": len(diffs) == 0, "diffs": diffs}


def _cond1(intact_a):
    margins = np.array([r["margin"] for r in intact_a["sweep"]])
    evidence = np.array([r["evidence"] for r in intact_a["sweep"]])
    ordinals = [BAND_ORDINAL[r["band"]] for r in intact_a["sweep"]]
    monotone = all(ordinals[i] <= ordinals[i + 1] for i in range(len(ordinals) - 1))
    rho = _spearman(margins, evidence)
    passed = bool(monotone and rho >= SPEARMAN_BAR)
    return {"monotone_non_decreasing": monotone, "spearman_rho": float(rho), "pass": passed}


def _cond2(intact_a):
    cal = intact_a["calibration"]
    passed = bool(cal["ordinal_separable_lo_med"] and cal["ordinal_separable_med_hi"])
    return {"auc_lo_med": cal["auc_lo_med"], "auc_med_hi": cal["auc_med_hi"],
            "median_gap_lo_med": cal["median_gap_lo_med"], "spread_lo_med": cal["spread_lo_med"],
            "median_gap_med_hi": cal["median_gap_med_hi"], "spread_med_hi": cal["spread_med_hi"],
            "ordinal_separable_lo_med": cal["ordinal_separable_lo_med"],
            "ordinal_separable_med_hi": cal["ordinal_separable_med_hi"], "pass": passed}


def _shuffle_control(intact_a, seed):
    margins = np.array([r["margin"] for r in intact_a["sweep"]])
    evidence = np.array([r["evidence"] for r in intact_a["sweep"]])
    rng = np.random.default_rng(900000 + int(seed))
    perm = rng.permutation(len(margins))
    shuffled = margins[perm]
    boundaries = (intact_a["calibration"]["boundary_lo_med"], intact_a["calibration"]["boundary_med_hi"])
    bands = [classify_band(float(m), *boundaries) for m in shuffled]
    ordinals = [BAND_ORDINAL[b] for b in bands]
    monotone = all(ordinals[i] <= ordinals[i + 1] for i in range(len(ordinals) - 1))
    rho = _spearman(shuffled, evidence)
    fails_to_track = bool((not monotone) or abs(rho) < SHUFFLE_RHO_BAR)
    return {"shuffled_margins": [float(x) for x in shuffled], "bands": bands, "monotone": monotone,
            "rho": float(rho), "fails_to_track": fails_to_track}


def _lesion_control(intact_a, lesion_run, seed):
    """The lesion/intact PAIR, attributed rather than merely subtracted (`tools.lab.attributable_to`, the
    gap#5 lesson: two control-shaped numbers sitting one key apart in a JSON is not the same as asking WHOSE
    the difference is). `treatment` = the intact arm's own evidence-tracking (|rho|); `control` = the SAME
    quantity with the evidence differential removed (`lesion=True`). `attribution` -> ~1.0 means (nearly) ALL
    of the intact arm's evidence-tracking is attributable to the lesionable evidence differential itself, not
    to something running identically in both arms (a clamp/bound/init, the gap#5 shape)."""
    boundaries = (intact_a["calibration"]["boundary_lo_med"], intact_a["calibration"]["boundary_med_hi"])
    margins = np.array([r["margin"] for r in lesion_run["sweep"]])
    evidence = np.array([r["evidence"] for r in lesion_run["sweep"]])
    bands = [classify_band(float(m), *boundaries) for m in margins]
    frac_speculate = float(np.mean([b == BAND_SPECULATE for b in bands]))
    rho_lesion = _spearman(margins, evidence)
    intact_margins = np.array([r["margin"] for r in intact_a["sweep"]])
    rho_intact = _spearman(intact_margins, evidence)
    collapses = bool(frac_speculate >= LESION_SPECULATE_FRAC and abs(rho_lesion) < LESION_RHO_BAR)
    attribution = attributable_to(f"seed {seed}: evidence-tracking (|rho|) owed to the lesionable "
                                   f"evidence differential", abs(rho_intact), abs(rho_lesion))
    return {"lesion_bands": bands, "lesion_margins": [float(x) for x in margins],
            "frac_speculate_lesion": frac_speculate, "rho_lesion": float(rho_lesion),
            "rho_intact": float(rho_intact), "collapses": collapses,
            "attribution_to_lesionable_differential": attribution}


def cmd_aggregate(args):
    seeds = args.seeds
    python_exe = args.python or sys.executable
    workdir = args.workdir or os.path.join(_REPO, "research", "findings", "raw",
                                            "_metacog_graded_marker_derisk")
    os.makedirs(workdir, exist_ok=True)

    spawn_records = []
    per_seed = {}
    for seed in seeds:
        paths = {
            "intact_a": os.path.join(workdir, f"s{seed}_intact_a.json"),
            "intact_b": os.path.join(workdir, f"s{seed}_intact_b.json"),
            "lesion": os.path.join(workdir, f"s{seed}_lesion.json"),
        }
        recs = {
            "intact_a": _spawn_arm(python_exe, seed, "intact", "a", paths["intact_a"], args.memcap_gb),
            "intact_b": _spawn_arm(python_exe, seed, "intact", "b", paths["intact_b"], args.memcap_gb),
            "lesion": _spawn_arm(python_exe, seed, "lesion", "x", paths["lesion"], args.memcap_gb),
        }
        spawn_records.extend(recs.values())
        for k, r in recs.items():
            if not r["ok"]:
                raise RuntimeError(f"arm {k} seed={seed} FAILED (rc={r.get('returncode')}): "
                                   f"{r.get('error', r.get('stderr_tail'))}")
        intact_a = _load(paths["intact_a"])
        intact_b = _load(paths["intact_b"])
        lesion = _load(paths["lesion"])

        determinism = _compare_intact_repeats(intact_a, intact_b)
        cond1 = _cond1(intact_a)
        cond2 = _cond2(intact_a)
        shuffle = _shuffle_control(intact_a, seed)
        lesion_ctrl = _lesion_control(intact_a, lesion, seed)

        per_seed[str(seed)] = {
            "intact_a": intact_a, "intact_b_paths": paths["intact_b"], "lesion": lesion,
            "determinism": determinism, "cond1": cond1, "cond2": cond2,
            "shuffle_control": shuffle, "lesion_control": lesion_ctrl,
            "both_conditions_pass": bool(cond1["pass"] and cond2["pass"]),
        }
        print(f"[seed {seed}] cond1(monotone+rho>=0.80)={cond1['pass']} (rho={cond1['spearman_rho']:.3f}) "
              f"cond2(ordinal-separable)={cond2['pass']} shuffle_fails_to_track={shuffle['fails_to_track']} "
              f"lesion_collapses={lesion_ctrl['collapses']} determinism_ok={determinism['byte_identical']}")

    n = len(seeds)
    n_cond1 = sum(1 for s in per_seed.values() if s["cond1"]["pass"])
    n_cond2 = sum(1 for s in per_seed.values() if s["cond2"]["pass"])
    n_both = sum(1 for s in per_seed.values() if s["both_conditions_pass"])
    n_shuffle_fails = sum(1 for s in per_seed.values() if s["shuffle_control"]["fails_to_track"])
    n_lesion_collapses = sum(1 for s in per_seed.values() if s["lesion_control"]["collapses"])
    n_determinism_ok = sum(1 for s in per_seed.values() if s["determinism"]["byte_identical"])

    # byte-identical-off (anti-cheat 5): re-verify the production organ file's hash, in the data, right now.
    _, organ_unchanged, organ_cur_hash = _honesty_selftest()[1], None, None
    problems, organ_unchanged, organ_cur_hash = _honesty_selftest()
    honesty_scan_clean = not any("banned word" in p for p in problems)

    v = Verdict(
        "metacog graded 3-band marker (SPECULATE/HEDGE/ASSERT) tracks the workspace's own nmda_norm spiking "
        "balance-of-evidence margin, 6-seed measurement de-risk")
    v.require(f"n_seeds cond1 (monotone AND rho>=0.80), need >={MIN_SEEDS_PASS}/{n}", n_cond1,
               expect=lambda x, _n=n: x >= MIN_SEEDS_PASS)
    v.require(f"n_seeds cond2 (3-band ordinally separable), need >={MIN_SEEDS_PASS}/{n}", n_cond2,
               expect=lambda x: x >= MIN_SEEDS_PASS)
    v.require(f"n_seeds BOTH conditions, need >={MIN_SEEDS_PASS}/{n}", n_both,
               expect=lambda x: x >= MIN_SEEDS_PASS)
    v.require(f"shuffle/yoke control fails to track on all {n} seeds (anti-cheat 1)", n_shuffle_fails,
               expect=lambda x: x == n)
    v.require(f"lesion collapses to SPECULATE + rho~0 on all {n} seeds (anti-cheat 2)", n_lesion_collapses,
               expect=lambda x: x == n)
    v.require(f"intact-repeat byte-identical determinism on all {n} seeds (anti-cheat 4)", n_determinism_ok,
               expect=lambda x: x == n)
    v.require("metacog_production_organ.py byte-identical to baseline hash (anti-cheat 5)", organ_unchanged,
               expect=True)
    v.require("graded-marker phrasing carries no phenomenal/experiential word (anti-cheat 6)", honesty_scan_clean,
               expect=True)
    for seed, s in per_seed.items():
        v.control(f"seed {seed}: lesion collapses evidence-tracking (|rho| gap)",
                   treatment=abs(s["lesion_control"]["rho_intact"]), control=abs(s["lesion_control"]["rho_lesion"]),
                   min_separation=0.4)

    go = bool(n_both >= MIN_SEEDS_PASS and n_shuffle_fails == n and n_lesion_collapses == n
              and n_determinism_ok == n and organ_unchanged and honesty_scan_clean)
    decided = v.decide(go=go)

    out_obj = {
        "runner": "research/runners/_metacog_graded_marker_derisk.py",
        "seeds": list(seeds),
        "sig_lo_pa": SIG_LO, "sig_hi_pa": SIG_HI, "read_reps": READ_REPS,
        "sweep_evidence_levels": list(SWEEP_EVIDENCE_LEVELS),
        "gate_bars": {"spearman_bar": SPEARMAN_BAR, "ordinal_auc_bar": ORDINAL_AUC_BAR,
                      "min_seeds_pass": MIN_SEEDS_PASS, "shuffle_rho_bar": SHUFFLE_RHO_BAR,
                      "lesion_rho_bar": LESION_RHO_BAR, "lesion_speculate_frac": LESION_SPECULATE_FRAC},
        "n_seeds": n, "n_cond1_pass": n_cond1, "n_cond2_pass": n_cond2, "n_both_pass": n_both,
        "n_shuffle_fails_to_track": n_shuffle_fails, "n_lesion_collapses": n_lesion_collapses,
        "n_determinism_ok": n_determinism_ok,
        "organ_file_baseline_hash": ORGAN_FILE_BASELINE_HASH, "organ_file_current_hash": organ_cur_hash,
        "organ_file_unchanged": organ_unchanged, "honesty_scan_clean": honesty_scan_clean,
        "marker_phrases": {b: graded_marker_prefix(b) for b in (BAND_SPECULATE, BAND_HEDGE, BAND_ASSERT)},
        "per_seed": per_seed,
        "spawn_records": spawn_records,
        "verdict": decided["status"], "go": decided["go"],
        "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2, sort_keys=True)
    print(f"\n=== VERDICT: {decided['status']} (go={decided['go']}) ===")
    print(f"wrote {args.out}")
    return 0


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode")

    p_self = sub.add_parser("selftest")
    p_self.set_defaults(func=cmd_selftest)

    p_arm = sub.add_parser("arm")
    p_arm.add_argument("--seed", type=int, required=True)
    p_arm.add_argument("--arm", choices=("intact", "lesion"), required=True)
    p_arm.add_argument("--repeat", default="a")
    p_arm.add_argument("--out", required=True)
    p_arm.set_defaults(func=cmd_arm)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p_agg.add_argument("--out", required=True)
    p_agg.add_argument("--workdir", default=None)
    p_agg.add_argument("--python", default=None)
    p_agg.add_argument("--memcap-gb", type=int, default=4)
    p_agg.set_defaults(func=cmd_aggregate)

    # `--selftest`/`--mode` top-level convenience aliases (matches most runners in this repo).
    p.add_argument("--selftest", action="store_true", help="alias for `selftest` subcommand")
    p.add_argument("--mode", choices=("selftest", "arm", "aggregate"), default=None)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # allow `--selftest` / `--mode X ...` top-level spelling in addition to the subcommand spelling
    if argv and argv[0] == "--selftest":
        return cmd_selftest(argparse.Namespace())
    if argv and argv[0] == "--mode":
        argv = [argv[1]] + argv[2:]
    p = build_argparser()
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
