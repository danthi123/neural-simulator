"""SELF-MODEL / METACOGNITION — an honest FUNCTIONAL confidence read-out wired into the PRODUCTION turn
(Gate-B, E1, 2026-08-12).

The owner's honesty-boundary self-model: a brain that reads the CONFIDENCE of the answer it is about to
give and, when that confidence is LOW, honestly QUALIFIES it ("my decision-margin reads this as
low-confidence — take it as uncertain") instead of asserting a marginal recall as if it were certain. This
is a FUNCTIONAL metacognition read-out — never a phenomenal claim, never a fabricated fact.

It REUSES (does not reinvent) the adversarially-verified E1 faculty
(`research/runners/_second_order_metacog_monitor_derisk.py`, confidence_read='balance', 6/6 GO —
type2_auc 0.67–0.82, meta-d' > 0 on every seed): confidence = the ABSOLUTE MARGIN of the first-order
workspace WTA competition, `|rate(assembly_1) − rate(assembly_0)|`, read DIRECTLY from `cp_firing_states`
over the evidence window (Vickers balance-of-evidence / Kepecs distance-to-bound). No downstream pool, no
host confidence formula — the confidence IS a spike-rate margin.

HOW IT MAPS ONTO A TURN: the brain's OWN spiking recall produces a graded confidence signal for the answer
it gives — the mean role-decode confidence of the thematic-role parse (the parser's per-role assignment
certainty, read off the composer's trace). That evidence scalar is delivered as the graded drive to the
metacog workspace competition (the correct assembly gets `base + evidence`, the other `base`); the settled
WTA MARGIN off `cp_firing_states` IS the confidence. A calibrated threshold splits confident vs uncertain;
a low-confidence answer gets an honest hedge PREPENDED. The load-bearing SPIKING part is the balance margin
read; the evidence DERIVATION (role-decode confidence) is a declared host boundary (exactly as affect's
appraisal injection and the surprise organ's sensory encoding are declared boundaries with the spiking
read-back load-bearing).

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
  * NARROW DYNAMIC RANGE: the balance margin's absolute range on this operating point is small; the wire
    reliably separates clearly-high from clearly-low evidence, the mid-range is a boundary (averaged reads
    denoise it).
  * CO-RESIDENT on its own metacog-workspace bridge, ALONGSIDE the recall composer — rides on the one-brain
    merge (burn-down #1), exactly as the affect/surprise/comprehension organs do.

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
)

# ── the de-risk's balance operating point (its argparse defaults; the 6/6-GO regime) ─────────────────────────
BASE_PA = 300.0          # baseline drive to BOTH class assemblies
SIG_LO = 40.0            # min per-trial signal strength (hardest / lowest-evidence)
SIG_HI = 260.0           # max per-trial signal strength (easiest / highest-evidence)
READ_REPS = 8            # average the balance margin over N jittered reads (denoises the tiny single-trial margin)
READ_JITTER_PA = 30.0    # per-rep Gaussian drive jitter (samples the local evidence neighborhood; fixed-seeded)
READ_SEED = 4242         # fixed so a given evidence -> a DETERMINISTIC confidence decision (reproducible per turn)

# ── role-decode confidence -> evidence[0,1] normalization (STRETCH the observed recall band to the full range
# so the balance margin separates). Measured on the tiny-demo recalls (2026-08-12): mean role-decode confidence
# ran 0.400 ("dog chase cat", the lowest-confidence recall) .. 0.476 ("brain use spikes", the highest); these
# bounds place 0.400 -> evidence ~0.29 (uncertain -> hedge) and 0.476 -> evidence ~0.74 (confident -> no hedge). ─
ROLE_CONF_LO = 0.35      # a mean role-decode confidence at/below this -> evidence 0 (lowest confidence)
ROLE_CONF_HI = 0.52      # a mean role-decode confidence at/above this -> evidence 1 (highest confidence)


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


def evidence_from_role_conf(mean_role_conf: float | None) -> float | None:
    """Normalize the brain's mean role-decode confidence to an evidence scalar in [0,1] (stretch the observed
    recall band ROLE_CONF_LO..ROLE_CONF_HI). None (no decoded roles / no parse) -> None (out of scope)."""
    if mean_role_conf is None:
        return None
    span = max(1e-6, ROLE_CONF_HI - ROLE_CONF_LO)
    return float(np.clip((float(mean_role_conf) - ROLE_CONF_LO) / span, 0.0, 1.0))


def mean_role_confidence(activity: dict | None) -> float | None:
    """Extract the mean role-decode confidence from the composer's per-turn activity trace (the brain's own
    spiking parse certainty for the answer it is giving). None when no roles carry a confidence."""
    if not activity:
        return None
    roles = activity.get("roles") or []
    confs = [r.get("confidence") for r in roles if r.get("confidence") is not None]
    if not confs:
        return None
    return float(np.mean([float(c) for c in confs]))


class MetacogProductionOrgan:
    """A process-shared spiking balance-of-evidence confidence monitor. Built ONCE (lazily): the metacog
    workspace WTA (confidence_read='balance'), plus a build-time calibration of the confident-vs-uncertain
    margin threshold from a synthetic high/low-evidence battery. Each read maps the answer's evidence scalar to
    a graded workspace drive, settles the WTA, and reads the balance margin off cp_firing_states."""

    def __init__(self, seed: int = 42, shared=None):
        self.seed = int(seed)
        # ONE-BRAIN MERGE pool #2 (opt-in, default per BRAIN_ONEBRAIN_MERGE2): when a MergedSubstrate2 is injected,
        # this organ's workspace/meta_schema slice lives on the SHARED spiking bridge it co-inhabits with the
        # pragmatic organ (one cp_membrane_potential_v) instead of its own bridge. See
        # research/runners/onebrain_merge_production2.py.
        self._shared = shared
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
                      "role_conf_lo": ROLE_CONF_LO, "role_conf_hi": ROLE_CONF_HI}
        self._built = True

    def _margin(self, evidence: float, lesion: bool = False) -> float:
        """The SPIKING balance-of-evidence margin |rate(asm1)-rate(asm0)| for an answer whose evidence is `evidence`
        in [0,1]. Drives the correct assembly with base+sig(evidence) and the other with base; averages READ_REPS
        fixed-seed-jittered reads (denoises the small single-trial margin). `lesion` removes the evidence
        DIFFERENTIAL (both assemblies at base) so the margin collapses regardless of evidence -> load-bearing."""
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
