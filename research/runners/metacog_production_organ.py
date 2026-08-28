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
    None (no decoded roles / no parse) -> None (out of scope)."""
    if not activity:
        return None
    roles = activity.get("roles") or []
    vals = []
    for r in roles:
        m = r.get("margin")
        v = m if m is not None else r.get("confidence")
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
        from research.runners.onebrain_merge_production2 import merge2_enabled, get_merged_substrate2
        shared = get_merged_substrate2(seed) if merge2_enabled() else None
        _ORGAN = MetacogProductionOrgan(seed=seed, shared=shared)
    return _ORGAN


def hedge_prefix() -> str:
    """The honest functional hedge PREPENDED to a low-confidence answer. A FUNCTIONAL read of the spiking
    decision-margin — never a phenomenal claim, never a change to the answer's content."""
    return ("My decision-margin reads this as low-confidence, so take it as uncertain: ")
