"""SELF-MODEL / METACOGNITION — an honest FUNCTIONAL confidence read-out wired into the PRODUCTION turn
(Gate-B, E1, 2026-08-12).

The owner's honesty-boundary self-model: a brain that reads the CONFIDENCE of the answer it is about to
give and, when that confidence is LOW, honestly QUALIFIES it ("my decision-margin reads this as
low-confidence — take it as uncertain") instead of asserting a marginal recall as if it were certain. This
is a FUNCTIONAL metacognition read-out — never a phenomenal claim, never a fabricated fact.

THE CONFIDENCE READ (default = `nmda_norm`, 2026-08-13). Confidence is a DIVISIVE-NORMALIZED balance of the
first-order workspace WTA competition, read off the two class assemblies' slow-NMDA RECURRENT CONDUCTANCE:
`|g_nmda(asm1) − g_nmda(asm0)| / (g_nmda(asm1) + g_nmda(asm0) + eps)` over the evidence window (Carandini &
Heeger 2012 divisive normalization; Wang persistent-NMDA accumulator — the graded "balance-of-evidence
integrator" this faculty was designed around). Both terms are genuine spiking-substrate state (the NMDA
conductance `cp_conductance_g_nmda` is driven by presynaptic spikes through NMDA synapses, NOT the injected
current). This SUPERSEDES the original ABSOLUTE spike-rate margin `|rate(asm1) − rate(asm0)|` off
`cp_firing_states` (the E1 6/6 GO — `_second_order_metacog_monitor_derisk`, confidence_read='balance',
type2_auc 0.67–0.82, meta-d' > 0), kept as the escape `BRAIN_METACOG_READ=balance`: that spike-count margin
sits at the workspace's ~0.1%-firing NOISE FLOOR (near-random monotonicity ~0.5), which made the
confident/uncertain call non-monotone / seed-fragile and NOT invariant to the pool-#2 per-region init
re-draw. The NMDA read tracks evidence monotonically in BOTH the standalone and the merged build, so the
self-calibrated threshold lands at the SAME evidence boundary — a seed-robust, de-noised confidence code. No
downstream pool, no host confidence formula — the confidence IS a synaptic-conductance balance. (Adopted from
`_metacog_robust_confidence_derisk.py`, GO `2026-08-13-metacog-robust-confidence-GO.md`; reuse-by-import of
`nmda_norm_margin` — the derisk's `RobustMetacogProductionOrgan` reuses THIS canonical function.)

HOW IT MAPS ONTO A TURN: the brain's OWN spiking recall produces a graded confidence signal for the answer
it gives — the mean role-decode confidence of the thematic-role parse (the parser's per-role assignment
certainty, read off the composer's trace). That evidence scalar is delivered as the graded drive to the
metacog workspace competition (the correct assembly gets `base + evidence`, the other `base`); the settled
divisive-normalized NMDA-conductance balance IS the confidence. A calibrated threshold splits confident vs
uncertain; a low-confidence answer gets an honest hedge PREPENDED. The load-bearing SPIKING part is the
conductance balance read; the evidence DERIVATION (role-decode confidence) is a declared host boundary
(exactly as affect's appraisal injection and the surprise organ's sensory encoding are declared boundaries
with the spiking read-back load-bearing).

MOAT-SAFE + ADDITIVE by CONSTRUCTION: metacog runs AFTER the gate/moat has already produced (or refused) an
answer. It only QUALIFIES an already-produced, moat-verified answer with an honest hedge — it NEVER
manufactures a fact, flips an abstain into an assert (an abstain has no answer to qualify -> skipped), or
changes WHICH answer the recall produced. Default-ON; `BRAIN_METACOG=0` -> the byte-identical oracle.

LESION-LOAD-BEARING: the balance-of-evidence confidence lives in the EVIDENCE ENCODING (the de-risk's honest
signature: loop-ablation does NOT collapse it — it is a genuine decision-variable read, NOT type-1/type-2
DISSOCIABLE). So the load-bearing lesion is on that encoding: `BRAIN_METACOG_LESION=1` removes the evidence
DIFFERENTIAL from the workspace (drives both assemblies at `base`), so the settled margin collapses to ~0
regardless of the answer's true evidence -> a would-be-CONFIDENT answer FLIPS to a hedge. Verified: the same
high-evidence answer reads confident intact and uncertain lesioned -> the confident/uncertain discrimination
is caused by the SPIKING margin reading the evidence, not by a host threshold on the raw role-conf scalar.

HONEST RESIDUALS (declared — the mission's named next rungs, not faked):
  * EVIDENCE = the parse confidence (a COMPONENT of answer confidence), not a full recall-vs-alternatives
    balance; the substrate's other recall signals (rf readout magnitude / frac) are saturated on the
    tiny-demo, so the role-decode confidence is the graded signal available. A richer recall-margin evidence
    is the next rung.
  * NOT type-1/type-2 DISSOCIABLE: balance-of-evidence confidence is an encoding read (the de-risk's mapped
    boundary); the architecturally-dissociable comparator (`margin_abs`) is seed-fragile and remains the
    next rung. This is a genuine CONFIDENCE (decision-variable) read, not a separable second-order monitor.
  * NARROW DYNAMIC RANGE (largely RESOLVED by the nmda_norm read): the ABSOLUTE spike-margin's range at this
    operating point was small and mid-range-ambiguous — the noise floor that made the decision seed-fragile.
    The divisive-normalized NMDA-conductance read (now the default) tracks evidence monotonically and lands the
    threshold at a consistent boundary across seeds; the residual is that the NMDA margin's absolute magnitude
    is still modest (READ_REPS averaging denoises it) — the SNR is what changed, not the operating point.
  * CO-RESIDENT on the metacog-workspace bridge, ALONGSIDE the pragmatic organ on ONE shared `MergedSubstrate2`
    (pool #2, DEFAULT-ON 2026-08-13) — rides on the one-brain merge (burn-down #1), exactly as the
    affect/surprise/comprehension organs do; `BRAIN_ONEBRAIN_MERGE2=0` reverts to its own bridge.

FUNCTIONAL CORRELATE, NOT phenomenal: this measures + reports a metacognition CORRELATE (a confidence read
that tracks decision evidence). It makes NO claim of subjective experience.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import os

import numpy as np

from research.runners._second_order_metacog_monitor_derisk import (
    build_metacog_bridge,
    _run_trial,
    K_CLASSES,
)
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _restore_state, DRIVE_STEPS, FREE_STEPS,
)
from sim.backend import to_host

# ── the de-risk's balance operating point (its argparse defaults; the 6/6-GO regime) ─────────────────────────
BASE_PA = 300.0          # baseline drive to BOTH class assemblies
SIG_LO = 40.0            # min per-trial signal strength (hardest / lowest-evidence)
SIG_HI = 260.0           # max per-trial signal strength (easiest / highest-evidence)
READ_REPS = 8            # average the balance margin over N jittered reads (denoises the tiny single-trial margin)
READ_JITTER_PA = 30.0    # per-rep Gaussian drive jitter (samples the local evidence neighborhood; fixed-seeded)
READ_SEED = 4242         # fixed so a given evidence -> a DETERMINISTIC confidence decision (reproducible per turn)

# ── role-decode confidence -> evidence[0,1] normalization (STRETCH the observed recall band to the full range
# so the balance margin separates). RECALIBRATED 2026-08-27 (issue #181): the OLD band (0.35/0.52) was tuned
# against the self-ratio `confidence` field (`s[argmax]/max(s)`), which is 1.0 by construction on every
# non-degenerate decode -- it never varied, so the band's exact placement never mattered and the hedge never
# fired on real traffic (mean_role_confidence read 1.0 on every measured real turn, saturating above the OLD
# HI). `mean_role_confidence` now reads the genuine winner-vs-runner-up `margin` (`OneBrainComposer._margin`,
# the composer's own validated decisiveness signal -- see that method's docstring); this band is recalibrated
# against THAT signal's real measured distribution, on the ACTUAL production composer instance
# (`_build_tiny_demo`, the same builder `webapp/server.py` calls -- it enables `enable_attributed=True`, which
# adds an always-near-zero-margin `attribute` role chip to every real trace, real production's `roles` set is
# WIDER than a plain agent/action/patient/polarity composer) -- see `research/findings/raw/_metacog_confidence_recalib/`:
#   CONFIDENT (THROUGH THE REAL `/api/brain-chat` handler, all 5 real tiny-demo facts, intact store): mrc
#     0.504 .. 0.615 (`measure_real_confident.json`, `measure_real_build_noise_sweep.json`).
#   GENUINELY-UNCERTAIN (the SAME real composer instance + query, `_noise`-perturbed synaptic store -- the
#     identical legitimate synaptic-noise damage model 2026-06-18-emergent-graceful-degradation-derisk.md
#     validated for this composer -- at noise levels that still return an answer, i.e. a genuine weak/ambiguous
#     match, not an abstain): mrc ranges 0.149 (heavy noise) up through light-noise levels that correctly stay
#     confident (0.60 at sigma=0.3, barely perturbed); the clearly-degraded region (sigma>=1.1) reads 0.149..0.36.
# HI sits AT/BELOW the measured confident floor (0.504) so every real confident turn clips to evidence=1.0 (no
# regression); LO sits below the clearly-degraded region so a genuinely weak/ambiguous read reaches the organ's
# own low-evidence calibration zone (evidence<=~0.4) and hedges. A borderline (lightly-perturbed) read is
# allowed to land in the middle and go either way -- an honest ambiguous case, not gamed to a side.
ROLE_CONF_LO = 0.30      # a mean role-decode confidence at/below this -> evidence 0 (in the measured degraded region)
ROLE_CONF_HI = 0.50      # a mean role-decode confidence at/above this -> evidence 1 (at the measured confident floor)

# SCALE-INVARIANT decisiveness anchors (2026-09-02, board #94/#108 R3 -- the 100k recalibration). The read
# `mean_role_confidence` prefers for an LTM-sourced trace is now the composer's `margin_snr` (the winner's
# z-score above the candidate BULK, `RFPhasorComposer._cleanup_all_score_stats`), NOT the scale-VARIANT
# `margin_norm`. WHY (measured, research/findings/raw/_confidence_100k_recalib/): `margin_norm = (top-runner)/top`
# keys on the single runner-up = max over V-1 candidates, an order statistic that inflates as ~sqrt(2 ln V), so
# an EQUALLY-decisive clean recall reads margin_norm 0.497 at the 15k core but only 0.395 at the 100k bundle --
# below the confident floor, so `confident` could never read True at 100k (the #108 NO-GO). The winner-vs-BULK
# z-score is INVARIANT to codebook size (the SAME recall: winner_z 7.24 at 15k == 7.03 at 100k; the non-winner
# mean~0 / std~1/sqrt(D) are stable estimators of the noise floor). SNR_LO/SNR_HI map winner_z LINEARLY onto the
# SAME ROLE_CONF_LO/HI band, anchored to a 15k reference CLEAN vs DEGRADED recall so the shipped 15k operating
# point is reproduced at BOTH arms (clean z=7.237 <-> margin_norm 0.4966; degraded z=5.329 <-> margin_norm 0.3161,
# arms_15k_seed42.json) -- and the SAME anchors then self-consistently lift the 100k clean recall back above the
# confident floor. These are UNIVERSAL (one operating point for ALL vocab scales), NOT a per-bundle constant.
SNR_LO = 5.158           # 15k DEGRADED-recall reference winner_z -> maps to ROLE_CONF_LO
SNR_HI = 7.273           # 15k CLEAN-recall reference winner_z   -> maps to ROLE_CONF_HI

# CALIBRATION-AT-SCALE (2026-09-01, board #94): this band is NOT re-tuned per codebook size. The 15k-entity
# `wikidata_core_15k` LTM's genuine correct-recall `margin_norm` (peak-normalized, `RFPhasorComposer.
# _cleanup_all_score_stats`) measured 0.393..0.552 (mean 0.473, n=80, `research/findings/raw/
# _metacog_scale_recalib/measure_15k_ltm_margins.json`) -- already inside THIS SAME band, because `margin_norm`
# is a peak-relative ratio, not a raw cosine magnitude (which shrinks with a bigger candidate vocabulary). The
# actual scale bug was `mean_role_confidence` reading the WRONG (raw, unnormalized) field for an LTM-sourced
# trace, fixed in `mean_role_confidence`'s own docstring above -- see
# research/findings/2026-09-01-confidence-forthcomingness-margin-scale-recalibration.md.

# SPIKING RECALL-MARGIN calibration (2026-09-05, scaffold-retirement backlog rank 9). `margin_spiking`
# (`RFPhasorComposer._spiking_margin`) is the SAME normalized-decisiveness FORM as `margin`/`margin_norm`, off
# SPIKE COUNTS instead of a host score comparison -- strongly correlated with the host read (measured: Pearson
# r=0.964 over 25 real tiny-demo role reads, clean + 4 synaptic-noise levels, `research/findings/raw/
# _metacog_spiking_recall_margin_derisk/calibrate_margin_output.json`) but NOT numerically identical -- raw
# reuse of ROLE_CONF_LO/HI tracked well at the CLEAN/heavily-degraded extremes but misaligned the confident/
# hedge THRESHOLD crossing in the ambiguous middle band. The 6-seed de-risk (`research/findings/raw/
# _metacog_spiking_recall_margin_derisk/6seed_results.json`) measured, WITH this fit applied: 97.6% turn-level
# confident/hedge agreement with the host on UNAMBIGUOUS cases (host margin outside ROLE_CONF_LO/HI), 50% on
# genuinely AMBIGUOUS ones (a characterized precision residual, not resolved by a longer integration window --
# see the finding), 83.3% overall (n=60, 6 seeds x 10 conditions). A linear fit
# `margin_spiking ~= SPIKING_FIT_A*margin + SPIKING_FIT_B` over the 25 calibration points (Pearson r=0.964)
# inverted onto the ROLE_CONF band -- the SAME anchor-remap methodology SNR_LO/HI already uses for
# `margin_snr`, just fit by regression instead of two hand-picked reference arms because a genuinely different
# substrate (spike counts, not cosine scores) has no single "the" clean/degraded reference recall to anchor by
# hand. See research/findings/2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md.
SPIKING_FIT_A = 1.1080448437056127
SPIKING_FIT_B = -0.08949965933225146
SPIKING_MARGIN_LO = 0.24291379377943234   # a DEGRADED margin_spiking reference -> maps to ROLE_CONF_LO
SPIKING_MARGIN_HI = 0.4645227625205549    # a CONFIDENT margin_spiking reference -> maps to ROLE_CONF_HI

NORM_EPS = 1e-9          # divisive-normalization stabilizer for the nmda_norm confidence read


def metacog_enabled() -> bool:
    """Default-ON. `BRAIN_METACOG` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_METACOG")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def metacog_lesioned() -> bool:
    """`BRAIN_METACOG_LESION` in {1,true,yes,on} -> remove the evidence differential (load-bearing lesion)."""
    v = os.environ.get("BRAIN_METACOG_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def spiking_recall_margin_enabled() -> bool:
    """Default-OFF. `BRAIN_METACOG_SPIKING_MARGIN` in {1,true,on,yes} -> the recall composer (`OneBrainComposer`/
    `RFPhasorComposer`) populates each role chip's `margin_spiking` field (a genuine spiking-competition read,
    scaffold-retirement backlog rank 9) and `mean_role_confidence` prefers it over the host `margin`/`margin_norm`/
    `margin_snr` chain. Reads the SAME env var the composer constructors check (`RFPhasorComposer.
    spiking_recall_margin`) -- this accessor exists for callers (the de-risk runner, a lesion test) that want to
    confirm the flag's state without constructing a composer. See research/runners/rf_phasor_composer.py.

    PRODUCTION-FLIP VERIFIED NO-GO (2026-09-05): default-ON was tried + INTEGRATED-verified through the real
    webapp.server.brain_chat handler across the mandated 6 seeds -- genuinely load-bearing (6/6 lesion collapse)
    and content-preserving, but on 3 of 6 seeds a real degraded-recall turn read CONFIDENT under this evidence
    while the shipped host evidence correctly hedged the SAME turn (4 instances / 42 natural noise-sweep
    opportunities, 0 in the reverse direction) -- stays default-OFF. See
    research/findings/2026-09-05-metacog-spiking-margin-prodflip-verify-NOGO.md."""
    v = os.environ.get("BRAIN_METACOG_SPIKING_MARGIN", "")
    return v.strip().lower() in ("1", "true", "on", "yes")


CONFIDENCE_READS = ("nmda_norm", "balance")


def default_confidence_read() -> str:
    """Production default = `nmda_norm` (the divisive-normalized NMDA-conductance balance read: de-noises the
    confident/uncertain decision AND makes it invariant to the pool-#2 per-region init re-draw, GO
    `2026-08-13-metacog-robust-confidence-GO.md`). Escape `BRAIN_METACOG_READ=balance` -> the original absolute
    spike-rate margin (the pre-2026-08-13 shipped read, now at the noise floor / seed-fragile)."""
    v = os.environ.get("BRAIN_METACOG_READ")
    if v is None:
        return "nmda_norm"
    return "balance" if v.strip().lower() == "balance" else "nmda_norm"


def nmda_norm_margin(bridge, xp, idx, snap, evidence: float, lesion: bool = False) -> float:
    """The CANONICAL divisive-normalized NMDA-conductance confidence margin (the pool-#2-robust read):

        conf = |g_nmda(asm1) - g_nmda(asm0)| / (g_nmda(asm1) + g_nmda(asm0) + eps)

    the two class assemblies' late-window mean recurrent-NMDA conductance off `cp_conductance_g_nmda`, averaged
    over READ_REPS fixed-seed-jittered reads (denoises the modest single-trial margin). Drives the correct
    assembly with base+sig(evidence) and the other with base; `lesion` removes the evidence DIFFERENTIAL (both
    assemblies at base) so the NUMERATOR collapses -> margin ~0 regardless of evidence (load-bearing).

    Carandini & Heeger (2012) divisive normalization off CONDUCTANCES (the anti-cheat's sanctioned form), NOT a
    host rescale of the answer: numerator = the two competing accumulators' balance, denominator = their summed
    co-active NMDA drive (the normalization pool). Both terms are genuine spike-driven substrate state. Reused
    by `_metacog_robust_confidence_derisk.RobustMetacogProductionOrgan` (single source of truth)."""
    mem = idx["member_dev"]
    sig = SIG_LO + float(np.clip(evidence, 0.0, 1.0)) * (SIG_HI - SIG_LO)
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    rng = np.random.default_rng(READ_SEED)
    vals = []
    for _ in range(READ_REPS):
        j = float(rng.normal(0.0, READ_JITTER_PA)) if READ_JITTER_PA > 0 else 0.0
        if lesion:
            dp = [BASE_PA + j, BASE_PA + j]           # NO evidence differential -> numerator ~0
        else:
            dp = [BASE_PA + sig + j, BASE_PA + j]
        dp = [max(0.0, x) for x in dp]

        bridge.cp_external_input_current[:] = 0.0
        _restore_state(bridge, snap)
        bridge.cp_external_input_current[:] = 0.0

        def _set_drive():
            for k in range(K_CLASSES):
                bridge.cp_external_input_current[mem[k]] = xp.float32(float(dp[k]))

        for _ in range(DRIVE_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            _set_drive()
            bridge._run_one_simulation_step()

        g_acc = {0: 0.0, 1: 0.0}
        n_late = 0
        for t in range(FREE_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            _set_drive()
            bridge._run_one_simulation_step()
            if t >= late_start:
                for k in range(K_CLASSES):
                    g_acc[k] += float(to_host(bridge.cp_conductance_g_nmda[mem[k]].astype(xp.float64).mean()))
                n_late += 1
        n_late = float(max(1, n_late))
        g0 = g_acc[0] / n_late
        g1 = g_acc[1] / n_late
        vals.append(abs(g1 - g0) / (g0 + g1 + NORM_EPS))
    return float(np.mean(vals))


def evidence_from_role_conf(mean_role_conf: float | None) -> float | None:
    """Normalize the brain's mean role-decode confidence to an evidence scalar in [0,1] (stretch the observed
    recall band ROLE_CONF_LO..ROLE_CONF_HI). None (no decoded roles / no parse) -> None (out of scope)."""
    if mean_role_conf is None:
        return None
    span = max(1e-6, ROLE_CONF_HI - ROLE_CONF_LO)
    return float(np.clip((float(mean_role_conf) - ROLE_CONF_LO) / span, 0.0, 1.0))


def mean_role_confidence(activity: dict | None) -> float | None:
    """Extract the mean per-role DECODE-CONFIDENCE from the composer's per-turn activity trace (the brain's own
    spiking parse certainty for the answer it is giving).

    PREFERENCE ORDER (highest first): `margin_spiking` (a genuine spiking-competition read, opt-in, see below)
    > `margin_snr` > `margin_norm` > `margin` > the legacy `confidence`. All four host fields remain HOST
    ARITHMETIC over the matched-filter scores -- comparisons of numbers, not a read of any circuit's spiking.
    `margin_spiking` (2026-09-05, scaffold-retirement backlog rank 9) is the one non-host member of this list:
    `RFPhasorComposer._spiking_margin` reads a winner-vs-runner-up SPIKE-COUNT margin off the SAME Izhikevich
    cleanup bank `_spiking_cleanup`/`OneBrainComposer._spiking_select` already drive for the on-substrate
    winner-PICK. Default OFF (`BRAIN_METACOG_SPIKING_MARGIN` unset -> every chip's `margin_spiking` is None ->
    byte-identical fall-through to the existing host chain below). See
    research/findings/2026-09-05-metacog-spiking-recall-margin-derisk*.md.

    ROOT CAUSE (issue #181, 2026-08-27): this used to average each role chip's `confidence` field
    (`one_brain_composer._winner`: `s[argmax(s)] / max(s)`) -- but `s[argmax(s)]` IS `max(s)` BY CONSTRUCTION, so
    that field is 1.0 for every non-degenerate decode and 0.0 only when literally nothing scored above zero. It
    can never discriminate a confident recall from a genuinely uncertain one (measured: mean_role_confidence read
    1.0 on every real production turn, well above the metacog HIGH band, so the honesty hedge never fired).

    FIX: prefer each role chip's `margin` field when present -- `OneBrainComposer._margin(scores)`, the SAME
    normalized winner-vs-runner-up decisiveness read the composer's own `confidence_gate` familiarity gate uses
    (`(peak-runner_up)/peak`; multi-seed validated in 2026-06-18-emergent-graceful-degradation-derisk: ~0 on a
    noise-dominated/damaged read, ~0.5+ on an intact confident one). This is a genuine COMPETITION read: it is
    LOW exactly when the decode was ambiguous (a close runner-up candidate), HIGH exactly when one candidate
    dominates -- unlike the legacy `confidence` field, it actually varies with how uncertain the recall was.
    Falls back to the legacy `confidence` field for any chip/composer that does not populate `margin` (a safe,
    unchanged fallback -- e.g. `decoded_extra` host-composed chips, or a composer variant without the field).
    None (no decoded roles / no parse) -> None (out of scope).

    CALIBRATION-AT-SCALE FIX (2026-09-01, board #94's #184 follow-up): `margin_norm` -- when a role chip carries
    it -- is now preferred OVER the plain `margin` field. `RFPhasorComposer._cleanup_all_score_stats` (the LTM
    tier's own trace source, via `ShardedPhasorStore`) ALSO emits a field literally named `margin`, but it is the
    RAW, UNNORMALIZED `top_raw - runner_raw` cosine difference -- a DIFFERENT formula colliding under the SAME
    key `OneBrainComposer` uses for its peak-normalized ratio. Measured directly (80 real correct recalls, the
    shipped `wikidata_core_15k` LTM): the raw field sits in [0.155, 0.275], entirely below the metacog band's own
    LOW floor (`ROLE_CONF_LO=0.30`) regardless of true recall quality -- `evidence_from_role_conf` saturates at 0
    and `confident` can never read True on this store. `margin_norm` (added ADDITIVELY to
    `_cleanup_all_score_stats`, the IDENTICAL `(peak-runner)/peak` formula `OneBrainComposer._margin` already
    uses) sits in [0.393, 0.552] on the SAME 80 recalls -- squarely inside the EXISTING 0.30/0.50 band with NO
    band change needed, because it is now comparing like-with-like across composer types instead of a raw
    cosine-similarity magnitude (which shrinks with codebook size -- more candidate words inflate the runner-up's
    extreme value) against a magnitude-agnostic ratio the buffer composer was calibrated against.
    `OneBrainComposer`'s own role chips never populate `margin_norm`, so this preference change is BYTE-IDENTICAL
    for the tiny-demo buffer path (falls through to `margin`, unchanged) -- it only changes what is read for an
    LTM-sourced (`RFPhasorComposer`/`ShardedPhasorStore`) trace. See
    research/findings/2026-09-01-confidence-forthcomingness-margin-scale-recalibration.md.

    SCALE-INVARIANCE FIX (2026-09-02, board #94/#108 R3): `margin_norm` was INTENDED to be scale-invariant but is
    NOT -- its runner-up term is the max over V-1 candidates, an order statistic that inflates as ~sqrt(2 ln V),
    so an equally-decisive clean recall reads margin_norm 0.497 at the 15k core but 0.395 at the 100k bundle (below
    the confident floor -> the #108 100k NO-GO). So `margin_snr` -- the winner's z-score above the candidate BULK
    (`(top-mean_nonwin)/std_nonwin`, added to `_cleanup_all_score_stats`), which IS scale-invariant (winner_z 7.24
    at 15k == 7.03 at 100k for the identical recall) -- is now preferred OVER `margin_norm` when a chip carries it,
    mapped linearly onto the ROLE_CONF band via the 15k reference anchors SNR_LO/SNR_HI (reproduces the shipped 15k
    operating point at both the clean and degraded reference arms). `OneBrainComposer` buffer chips never populate
    `margin_snr` either, so the tiny-demo path is still byte-identical. See
    research/findings/2026-09-02-confidence-forthcoming-100k-recalibration-*.md."""
    if not activity:
        return None
    roles = activity.get("roles") or []
    vals = []
    for r in roles:
        sp = r.get("margin_spiking")
        snr = r.get("margin_snr")
        mn = r.get("margin_norm")
        m = r.get("margin")
        if sp is not None:
            # SPIKING recall-margin (2026-09-05, scaffold-retirement backlog rank 9, opt-in via
            # BRAIN_METACOG_SPIKING_MARGIN, default None -> unreached, byte-identical): a WTA winner-vs-runner-up
            # SPIKE-COUNT margin off the recall circuit's OWN Izhikevich cleanup competition
            # (`RFPhasorComposer._spiking_margin`), not host arithmetic over the matched-filter scores. TOP
            # priority when present -- when the flag is on, every role chip a given composer traces carries this
            # field, so a query's evidence is either ALL-spiking or ALL-host, never a silent per-role blend.
            # Mapped through SPIKING_MARGIN_LO/HI (a regression fit against the host `margin`, the SAME
            # anchor-remap methodology `margin_snr` uses via SNR_LO/HI) rather than reused raw: the two reads
            # correlate strongly (Pearson r~0.96) but are not numerically identical, and raw reuse of
            # ROLE_CONF_LO/HI tracked the CLEAN/heavily-degraded extremes while misaligning the confident/hedge
            # threshold crossing in the ambiguous middle band (measured, see the constants' docstring above).
            v = ROLE_CONF_LO + ((float(sp) - SPIKING_MARGIN_LO) / (SPIKING_MARGIN_HI - SPIKING_MARGIN_LO)) * (
                ROLE_CONF_HI - ROLE_CONF_LO)
            if v < 0.0:
                v = 0.0
        elif snr is not None:
            # SCALE-INVARIANT read (2026-09-02, board #94/#108 R3): map the winner-vs-bulk z-score linearly onto
            # the ROLE_CONF band via the 15k reference anchors (SNR_LO/SNR_HI). This reproduces the shipped 15k
            # operating point (winner_z there maps back to the old margin_norm at both the clean and degraded
            # reference arms, mean within ~4e-4 -> no 15k regression) while being invariant to codebook size, so
            # a clean 100k recall no longer falls below the confident floor. `margin_snr` is emitted only by the
            # LTM tier's `RFPhasorComposer` trace; `OneBrainComposer`'s own buffer chips never populate it, so the
            # tiny-demo buffer path falls through to `margin_norm`/`margin` UNCHANGED (byte-identical).
            v = ROLE_CONF_LO + ((float(snr) - SNR_LO) / (SNR_HI - SNR_LO)) * (ROLE_CONF_HI - ROLE_CONF_LO)
            if v < 0.0:
                v = 0.0
        else:
            v = mn if mn is not None else (m if m is not None else r.get("confidence"))
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return float(np.mean(vals))


class MetacogProductionOrgan:
    """A process-shared spiking balance-of-evidence confidence monitor. Built ONCE (lazily): the metacog
    workspace WTA (built NMDA-capable via build_metacog_bridge), plus a build-time calibration of the
    confident-vs-uncertain margin threshold from a synthetic high/low-evidence battery. Each read maps the
    answer's evidence scalar to a graded workspace drive, settles the WTA, and reads the confidence MARGIN.

    `confidence_read` selects the margin: `nmda_norm` (DEFAULT) = the divisive-normalized NMDA-conductance
    balance `|g_nmda(asm1)-g_nmda(asm0)|/(g_nmda(asm1)+g_nmda(asm0)+eps)` off cp_conductance_g_nmda (de-noised,
    pool-#2 re-draw-invariant, `2026-08-13-metacog-robust-confidence-GO.md`); `balance` = the original absolute
    spike-rate margin off cp_firing_states (escape `BRAIN_METACOG_READ=balance`; at the noise floor)."""

    def __init__(self, seed: int = 42, shared=None, confidence_read: str | None = None):
        self.seed = int(seed)
        # ONE-BRAIN MERGE pool #2 (default-ON per BRAIN_ONEBRAIN_MERGE2): when a MergedSubstrate2 is injected,
        # this organ's workspace/meta_schema slice lives on the SHARED spiking bridge it co-inhabits with the
        # pragmatic organ (one cp_membrane_potential_v) instead of its own bridge. See
        # research/runners/onebrain_merge_production2.py.
        self._shared = shared
        # the confidence read: nmda_norm (default) or balance (env escape). Explicit arg overrides the env.
        self.confidence_read = default_confidence_read() if confidence_read is None else confidence_read
        if self.confidence_read not in CONFIDENCE_READS:
            raise ValueError(f"unknown confidence_read={self.confidence_read!r}")
        self._built = False
        self.bridge = self.xp = self.idx = self.snap = None
        self.threshold = None
        self.calib = None

    def ensure_built(self):
        if self._built:
            return
        if self._shared is not None:
            # ONE-BRAIN MERGE pool #2: read this organ's slice of the SHARED bridge (built by MergedSubstrate2).
            # No training (frozen balance operating point); the threshold below self-calibrates on the shared slice.
            self._shared.ensure_built()
            self.bridge, self.xp = self._shared.bridge, self._shared.xp
            self.idx, self.snap = self._shared.metacog_idx(), self._shared.snap
        else:
            self.bridge, self.xp, self.idx, self.snap = build_metacog_bridge(
                seed=self.seed, confidence_read="balance")
        # CALIBRATE the confident/uncertain threshold from a synthetic high- vs low-evidence battery (the same
        # workspace read the production turn uses). Place the threshold in the gap, biased toward the low side so a
        # clearly-high-evidence answer reliably reads confident (no spurious hedge); else the class-mean midpoint.
        hi = [self._margin(e) for e in np.linspace(0.6, 1.0, 8)]
        lo = [self._margin(e) for e in np.linspace(0.0, 0.4, 8)]
        min_hi, max_lo = float(np.min(hi)), float(np.max(lo))
        mean_hi, mean_lo = float(np.mean(hi)), float(np.mean(lo))
        self.threshold = (0.5 * (min_hi + max_lo)) if min_hi > max_lo else (0.5 * (mean_hi + mean_lo))
        self.calib = {"mean_hi": mean_hi, "min_hi": min_hi, "mean_lo": mean_lo, "max_lo": max_lo,
                      "clean_gap": bool(min_hi > max_lo), "read_reps": READ_REPS,
                      "role_conf_lo": ROLE_CONF_LO, "role_conf_hi": ROLE_CONF_HI,
                      "confidence_read": self.confidence_read}
        self._built = True

    def _margin(self, evidence: float, lesion: bool = False) -> float:
        """The SPIKING confidence margin for an answer whose evidence is `evidence` in [0,1], per
        `self.confidence_read`. DEFAULT `nmda_norm`: the divisive-normalized NMDA-conductance balance
        (`nmda_norm_margin`, de-noised + pool-#2 re-draw-invariant). `balance`: the original absolute spike-rate
        margin off cp_firing_states. `lesion` removes the evidence differential -> the margin collapses (load-bearing)."""
        if self.confidence_read == "nmda_norm":
            return nmda_norm_margin(self.bridge, self.xp, self.idx, self.snap, evidence, lesion=lesion)
        return self._balance_margin(evidence, lesion=lesion)

    def _balance_margin(self, evidence: float, lesion: bool = False) -> float:
        """The original ABSOLUTE spike-rate balance margin |rate(asm1)-rate(asm0)| off cp_firing_states
        (`confidence_read='balance'` escape). Drives the correct assembly with base+sig(evidence) and the other
        with base; averages READ_REPS fixed-seed-jittered reads. `lesion` removes the evidence differential."""
        sig = SIG_LO + float(np.clip(evidence, 0.0, 1.0)) * (SIG_HI - SIG_LO)
        rng = np.random.default_rng(READ_SEED)
        vals = []
        for _ in range(READ_REPS):
            j = float(rng.normal(0.0, READ_JITTER_PA)) if READ_JITTER_PA > 0 else 0.0
            if lesion:
                dp = [BASE_PA + j, BASE_PA + j]           # NO evidence differential -> margin ~0
            else:
                dp = [BASE_PA + sig + j, BASE_PA + j]
            dp = [max(0.0, x) for x in dp]
            vals.append(float(_run_trial(self.bridge, self.xp, self.idx, self.snap, dp)["meta"]))
        return float(np.mean(vals))

    def judge(self, evidence: float, lesion: bool = False) -> dict:
        """Read the confidence of an answer whose evidence scalar is `evidence` in [0,1]. Returns the spiking balance
        margin, the calibrated threshold, and `confident` (margin >= threshold). A LOW margin -> the honest hedge."""
        self.ensure_built()
        margin = self._margin(evidence, lesion=lesion)
        return {"on": True, "lesioned": bool(lesion), "evidence": float(evidence),
                "balance": float(margin), "threshold": float(self.threshold),
                "confident": bool(margin >= self.threshold), "calib": self.calib}


_ORGAN: MetacogProductionOrgan | None = None


def get_organ(seed: int = 42) -> MetacogProductionOrgan:
    """The process-shared metacog organ (built once on first use). When the ONE-BRAIN MERGE pool-#2 flag is ON
    (`BRAIN_ONEBRAIN_MERGE2`, default per _MERGE2_DEFAULT_ON) the organ is backed by the process-shared
    MergedSubstrate2 it co-inhabits with the pragmatic organ (ONE spiking bridge); OFF -> its own bridge as today."""
    global _ORGAN
    if _ORGAN is None:
        # ONE-BRAIN SINGLE-POOL merge (opt-in, `BRAIN_ONEBRAIN_SINGLE_POOL`, default-OFF) WINS when on: all 4 core
        # organs co-inhabit ONE merge_organs pool. OFF -> the current pool-#2 pairwise path, byte-identical.
        from research.runners.onebrain_single_pool_production import single_pool_enabled, get_single_pool
        if single_pool_enabled():
            shared = get_single_pool(seed)
        else:
            from research.runners.onebrain_merge_production2 import merge2_enabled, get_merged_substrate2
            shared = get_merged_substrate2(seed) if merge2_enabled() else None
        _ORGAN = MetacogProductionOrgan(seed=seed, shared=shared)
    return _ORGAN


def hedge_prefix() -> str:
    """The honest functional hedge PREPENDED to a low-confidence answer. A FUNCTIONAL read of the spiking
    decision-margin — never a phenomenal claim, never a change to the answer's content."""
    return ("My decision-margin reads this as low-confidence, so take it as uncertain: ")
