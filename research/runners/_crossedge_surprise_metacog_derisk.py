"""ONE-BRAIN CROSS-EDGE #2 (D2 surprise -> E1 metacog, ERROR -> CONFIDENCE) — a DE-RISK, not a production wire-in.

Ranked cross-edge #2 in `research/findings/2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md`
(the surprise->metacog coupling that finding confirms UN-BUILT). On the single-pool substrate that co-locates D2
surprise + E1 metacog (`onebrain_single_pool_production.get_single_pool` / the reconciled 4-organ family in
`_onebrain_twopool_merge_organread_verify._recon_descriptors`), a genuine LEARNED spiking synapse carries the D2
surprise circuit's expectation-violation firing onto the E1 metacog confidence read, so a VIOLATED prediction /
high surprise LOWERS the confidence margin the substrate reads off itself — NOT via a host if-statement.

THE BIOLOGY: unexpected uncertainty (a violated prediction) transiently RESETS / DOWN-WEIGHTS confidence in the
current internal model — the alternative interpretations gain, so the winner-vs-runner-up decisiveness the metacog
monitor reads collapses (Yu & Dayan 2005, "Uncertainty, neuromodulation, and attention", *Neuron* 46(4):681-692,
DOI 10.1016/j.neuron.2005.05.019, PMID 15944135 — ACh/NE signal unexpected uncertainty and reduce the weight of
top-down prediction, broadening the posterior; Carandini & Heeger 2012 divisive normalization — the SAME balance-
of-evidence read the metacog organ's own `nmda_norm` confidence uses). This edge makes the substrate's OWN spiking
prediction-error signal (the D2 surprise pool) the thing that flattens the metacog confidence margin, realizing the
"error -> lower confidence -> the reply hedges more" story at the synapse instead of in Python.

THE MECHANISM (emergence-compliant; NO sim/ edit; brain-based; ONE brain):
  * ONE shared spiking bridge holds D2 SURPRISE + E1 METACOG, built from the SINGLE-POOL RECONCILED family
    (`_recon_descriptors()` filtered to {surprise, metacog} — reuse-by-import, the SAME reconciliation the
    2026-09-02 single-pool organ-read GO validated: pool-1 hebbian-global + a per-region param-het mask on
    metacog + a gain-0 freeze of metacog's internal edges + name-keyed seams). worldmodel/pragmatic are dropped
    (this edge does not span them); surprise + metacog co-reside byte-identically to the full single pool for the
    two regions the edge touches. NOT a bespoke pool: the exact organs the design says the edge spans.
  * ONE plastic cross-edge `surprise -> metacog.workspace member[1]` (the RUNNER-UP first-order assembly),
    declared as a `CrossEdge` row (init_weight W0=0.05, the SOLE plastic synapse via
    `MergedPool.apply_cross_edge_freeze()`'s whitelist inversion). E_TO_E onto the competing assembly: when the
    prediction is violated the surprise pool EXCITES the runner-up, raising g_nmda(asm1) toward g_nmda(asm0) so
    the divisively-normalized winner-vs-runner-up margin d=(g0-g1)/(g0+g1+eps) — the metacog confidence — DROPS.
    (Raising the alternative is the Yu-Dayan "unexpected uncertainty broadens the posterior" arm, realized as a
    spike-driven synapse; the net functional read is a LOWER confidence margin, exactly the design's claim.)
  * DEFAULT mechanism (`--ablation plain`, the ORIGINAL, still-live form): the edge GROWS by the substrate's OWN
    standard rate-Hebbian rule, unconditionally, over TRAINING episodes that co-drive a CONTRADICT (mismatch)
    trial on the surprise circuit — cue block c (this seed's randomly-assigned recall probe) + patient_asserted
    block c'!=c (the false assertion), which the FIXED surprise wiring turns into "surprise" firing SPECIFICALLY
    in block c' — with a tonic teaching current directly into member[1] (MEMBER_TEACH_PA), so member[1] reliably
    co-fires with the block-c' slice of `surprise` during training, Hebbian-binding that slice's edges to
    member[1]. Same host-supervised co-drive class every cross-edge in this codebase uses
    (R1/R4/surprise->provenance/surprise->encode-decision).
  * TESTED-AND-REJECTED ablation (`--ablation gated`, banked 2026-09-02, NOT the default): a THIRD-FACTOR-GATED
    port of sibling C1's (`_crossedge_surprise_worldmodel_derisk`) mechanism, which passed 6/6 where this edge's
    plain form went 3/6 NO-GO on the cupy verify. EACH episode first READS that trial's own D2-surprise firing
    rate (`cp_firing_states[surprise]`, learning OFF); the Hebbian window opens only when the measured rate
    clears a threshold calibrated from THIS SEED's own CONFIRM-vs-CONTRADICT firing gap (`_calibrate_conf_gate`,
    GATE_FRAC=0.35 — identical boundary to C1's `_calibrate_gate`). HYPOTHESIS TESTED, NOT CONFIRMED: a
    same-runner, same-backend (numpy), same-6-seed controlled A/B (see
    `research/findings/2026-09-02-c2-metacog-error-gated-port-second-negative.md`) found this REGRESSES
    robustness (plain 6/6 GO -> gated 3/6 GO), not fixes it. Diagnosis: unlike C1's transition (which directly
    determines its own gate's future surprise input, closing a genuine feedback loop that makes the gate
    self-limiting), this edge's D2-surprise circuit is FIXED/open-loop — training the cross-edge has zero effect
    on `surprise`'s own firing, so the gate opens 80/80 episodes every seed (never discriminates) while the
    extra un-learned read passes perturb un-reset homeostatic state, adding noise without adding selectivity.
    Banked as a tested method, not a recommendation — do not re-port this class of fix onto an OPEN-LOOP error
    signal without first checking for a closed loop as C1 has.
  * ANTI-CHEAT (reused by import): `_assign_blocks` draws THIS seed's (cue_c, assert_cp) block pair from a
    seed-keyed RNG independent of every other seeded draw -- the edge must grow on WHICHEVER block was randomly
    assigned this seed's "surprise" role, not a memorized identity; the OTHER (never-mismatched) surprise blocks'
    edges into member[1] must stay near W0.

THE LOAD-BEARING TEST (the crux; via `onebrain_crossedge_gate.CrossEdgeGateSpec` + `run_gate`, reused UNMODIFIED):
  * condition "low" (CONTROL): a CONFIRM trial (cue + matching assertion -> surprise stays near-silent, its own
    validated CONFIRM-cancels behavior) is co-driven WITH the fixed high-evidence metacog member drive
    [base+sig, base] (asm0 the intended winner). The confidence margin reads HIGH (no surprise flattening).
  * condition "high": a CONTRADICT trial (cue + this seed's mismatched assertion -> surprise FIRES in the trained
    block) co-driven WITH the SAME member drive. The learned edge carries surprise's firing onto member[1],
    flattening the margin -> the confidence reads LOWER.
  * `run_gate` computes the generic emergence + interaction (vary/lesion) numbers from the declaration: the edge
    GREW (emergence), the confidence read DROPS from low->high with the edge intact (interaction, expected sign
    -1), and the drop VANISHES once the edge is lesioned (attributable_to ~1). BYTE-OFF: the no-cross-edge pool's
    base connectivity is byte-identical once the declared edge's own synapse slots are excluded -> flag-off, the
    metacog confidence reads exactly as today.

CONFIDENCE READ: the SIGNED divisive-normalized NMDA-conductance margin d=(g0-g1)/(g0+g1+eps) over the two
workspace member assemblies (higher = more confident in the correct answer asm0). This is the metacog organ's own
`nmda_norm` read (`metacog_production_organ.nmda_norm_margin`, Carandini-Heeger divisive normalization off
`cp_conductance_g_nmda`) with the SIGN kept rather than |.|: in this de-risk asm0 is the winner BY CONSTRUCTION
(the higher evidence drive), so the signed winner-dominance margin is the faithful "confidence in the correct
answer", monotone in member[1]'s activity (no |.|-fold sign ambiguity when surprise pushes the runner-up up). We
report both signed and |.| forms; the gate keys on the signed drop.

DE-RISK ONLY — no production wiring, no `sim/` edit, no default flip, additive. numpy CPU throughout (routes off
the GPU; the {surprise, metacog} slice of the single pool is a few hundred neurons). PARTIAL-pending the 6-seed
cupy verify (queued separately); a numpy smoke is a 1-2-seed indicator, never the GO.

Run:
  SIM_BACKEND=numpy python -m research.runners._crossedge_surprise_metacog_derisk --smoke
  SIM_BACKEND=numpy python -m research.runners._crossedge_surprise_metacog_derisk --calibrate --seed 7
  SIM_BACKEND=numpy python -m research.runners._crossedge_surprise_metacog_derisk \\
      --seeds 42,43,44,100,101,102 \\
      --out research/findings/raw/_crossedge_surprise_metacog_6seed.json
"""
from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend
from tools.lab import attributable_to, lever

from research.runners.onebrain_merge_framework import CrossEdge, merge_organs, ASSEMBLY_SIZE, K_CLASSES
from research.runners.onebrain_crossedge_gate import (
    CrossEdgeGateSpec, run_gate, verify_byte_off, cross_edge_masks,
)
from research.runners._onebrain_twopool_merge_organread_verify import _recon_descriptors
from research.runners._onebrain_integration_surprise_episodic_crossedge import (
    _assign_blocks, CUE_PA, PRE_STEPS,
)
from research.runners.metacog_production_organ import NORM_EPS

# ── the metacog confidence read's operating point (the production organ's own `nmda_norm` regime) ──
BASE_PA = 300.0          # baseline drive to BOTH first-order class assemblies (metacog_production_organ.BASE_PA)
SIG_LO = 40.0
SIG_HI = 260.0
READ_EVIDENCE = 0.9      # fixed HIGH-evidence answer: asm0 clearly wins at baseline (high control confidence,
                          #   leaving head-room for surprise to flatten it) -> sig = 40 + 0.9*220 = 238 pA
READ_REPS = 6            # average the confidence margin over N jittered reads (denoises the modest single margin)
READ_JITTER_PA = 30.0
READ_SEED = 4242
DRIVE_STEPS = 35         # _gnw_rung1_ignition_curve_derisk drive-window (the metacog read's settle window)
FREE_STEPS = 100

# ── the declarative edge + training protocol (calibrated on seed 7, non-canonical, BEFORE any canonical seed —
#    see --calibrate; frozen here as the pre-registered values the 6-seed run uses unchanged) ──
W0 = 0.05                # near-zero seed weight — must GROW, not be pre-wired (framework default)
GATE = "surprise_to_metacog_conf"
MEMBER_TEACH_PA = 340.0  # training-only teaching current into member[1] (order of BASE_PA — member assemblies fire
                          #   at ~BASE_PA; makes member[1] reliably co-fire with the CONTRADICT surprise slice)
TRAIN_STEPS = 60         # matches the surprise circuit's own validated CONTRADICT measurement window
N_EPISODES = 80
CROSS_EDGE_LR = 0.08
HMAX = 8.0               # hebbian_max_weight soft bound for the cross-edge (calibrated seed 7)

# ── ERROR-GATED THIRD-FACTOR PORT (2026-09-02, ported from the sibling C1 edge
#    `_crossedge_surprise_worldmodel_derisk._gated_update_step`/`_calibrate_gate`, which passed 6/6 where this
#    edge's plain rate-Hebbian form went 3/6 NO-GO) — the Hebbian window is no longer unconditionally open every
#    training episode; it OPENS only when THIS episode's own measured D2-surprise firing rate (a
#    `cp_firing_states[surprise]` read, learning OFF, never a host compare) clears a build-time threshold
#    calibrated from THIS SEED's own CONFIRM (expected, low-surprise) vs CONTRADICT (violated, high-surprise)
#    trials on its randomly-assigned block pair — exactly C1's boundary (`GATE_FRAC` into the expected->violated
#    gap). Pre(surprise firing) x post(member[1] co-fire) x gate(threshold-cleared) — a genuine three-factor
#    update, not a repeated presumed-always-surprising co-drive.
GATE_FRAC = 0.35         # identical to C1's GATE_FRAC — threshold = expected_hz + GATE_FRAC*(violated_hz-expected_hz)

INTACT_FLOOR = 0.020     # the "high" confidence must drop below "low" by at least this (signed margin units)
LESION_RATIO = 0.34      # R1/R4 convention: lesioned |delta| must be < this * intact |delta|

_CONDUCT = ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
            "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise",
            "cp_conductance_g_gabab_slow", "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise")


def _member_idx(bridge, k):
    """Absolute neuron indices of the metacog first-order class-assembly member[k] (a sub-slice of the pool's
    single 'workspace' region: ws[k*ASSEMBLY_SIZE:(k+1)*ASSEMBLY_SIZE], exactly as `_metacog_idx_fn` computes)."""
    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), np.int64)
    return ws[k * ASSEMBLY_SIZE:(k + 1) * ASSEMBLY_SIZE]


def _member1_idx_fn(bridge):
    return _member_idx(bridge, 1)


CROSS_EDGES = [
    CrossEdge(key=GATE, source_key="surprise", source_region="surprise",
              target_key="metacog", target_region="workspace",   # documentation; the sub-slice is via target_idx_fn
              target_idx_fn=_member1_idx_fn,                       # RUNNER-UP assembly member[1] (a sub-slice)
              init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True),
]


def _surprise_metacog_descriptors():
    """The {surprise, metacog} slice of the single-pool RECONCILED family (reuse-by-import of the single source of
    truth), so this de-risk's substrate == the shipped single pool for exactly the two organs the edge spans.

    TWO scoped adjustments to the reconciled descriptors, both required to declare a LEARNED cross-edge INTO a
    reconciled-frozen region — neither changes the metacog read (conduction is untouched; only WHICH synapse may
    learn changes, and the whitelist `apply_cross_edge_freeze` re-imposes the SAME freeze):
      (1) enable_hebbian_learning -> False GLOBALLY (both organs), so NOTHING drifts during the pool's build/settle;
          my train window re-enables it, and `apply_cross_edge_freeze()` (freeze_rest=True) then makes my edge the
          SOLE plastic synapse — the encode-decision runner's exact convention. (The reconciled family instead sets
          it True-global + a per-region gain-0 freeze on metacog; the net "only-my-edge-learns" state is identical.)
      (2) metacog freeze_regions -> () . The build's step-7 `_apply_gain0_freeze` REJECTS any edge with exactly one
          endpoint in a `freeze_regions` region as an "unintended cross-synapse" — which is precisely what a
          DECLARED surprise->workspace cross-edge is. Dropping freeze_regions skips that guard; the metacog internal
          edges are still gain-0 frozen, now by `apply_cross_edge_freeze`'s whitelist inversion instead (same
          result: metacog conducts, never learns)."""
    out = []
    for d in _recon_descriptors():
        if d.key == "surprise":
            out.append(replace(d, config={**d.config, "enable_hebbian_learning": False,
                                          "hebbian_rate_window": False}))
        elif d.key == "metacog":
            out.append(replace(d, config={**d.config, "enable_hebbian_learning": False}, freeze_regions=()))
    return out


def _build(seed, with_edge: bool):
    descs = _surprise_metacog_descriptors()
    return merge_organs(descs, seed=seed, config_descriptors=descs, wire=True,
                        cross_edges=(CROSS_EDGES if with_edge else None))


class SurpriseMetacogPool:
    """The integrated pool: [surprise, metacog] (single-pool reconciled family) + the DECLARATIVE learned
    surprise->metacog-member[1] cross-edge. Structure mirrors `_onebrain_surprise_episodic_encode_decision.
    EncodeGatePool` (the cleanest recent CrossEdgeGateSpec consumer) — subclass-free, direct build + primitives +
    train/read."""

    def __init__(self, seed, gated=True):
        self.seed = int(seed)
        self.gated = bool(gated)   # A/B ablation: True = the 2026-09-02 error-gated port; False = the ORIGINAL
                                    # committed unconditional rate-Hebbian train() (the pre-port mechanism), kept
                                    # live in this SAME runner so the comparison is reproducible from one file
                                    # instead of a throwaway duplicate.
        self.xp, _ = get_backend()
        self.pool = _build(seed, with_edge=True)
        self.b = self.bridge = self.pool.bridge
        rm = self.b.region_manager

        def idxr(nm):
            return np.asarray(rm.indices(nm), np.int64)

        self.ix = {nm: idxr(nm) for nm in ("cue", "patient_expected", "patient_asserted", "surprise")}
        self.ix["member0"] = _member_idx(self.b, 0)
        self.ix["member1"] = _member_idx(self.b, 1)
        meta = self.pool.meta["surprise"]
        self.blk = int(meta["blk"]); self.n_trained = int(meta["n_trained"])
        self.cue_c, self.assert_cp = _assign_blocks(seed, self.n_trained)   # THIS SEED's random block pair

        self.pool.apply_cross_edge_freeze()   # the declared edge is the SOLE plastic synapse (R1/R4 whitelist)
        self.masks = cross_edge_masks(self.b, CROSS_EDGES)

        self._noncross = ~np.zeros(np.asarray(to_host(self.b.cp_connections.data)).shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]

        for kk, vv in dict(hebbian_symmetric=True, hebbian_learning_rate=CROSS_EDGE_LR, hebbian_max_weight=HMAX,
                           hebbian_min_weight=0.0, hebbian_weight_decay=0.0).items():
            setattr(self.b.core_config, kk, vv)

        # settle to rest (no drive) so each read/train starts from the same substrate resting state
        self.b.cp_external_input_current[:] = 0.0
        self.b.core_config.enable_hebbian_learning = False
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()

    # ---- primitives (encode-decision house style) ----
    def _hard_reset(self):
        b, xp = self.b, self.xp
        b.cp_membrane_potential_v[:] = xp.asarray(self.rest_v)
        b.cp_recovery_variable_u[:] = xp.asarray(self.rest_u)
        for nm in _CONDUCT:
            a = getattr(b, nm, None)
            if a is not None:
                a[:] = 0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        if getattr(b, "cp_hebb_coactivity_trace", None) is not None:
            b.cp_hebb_coactivity_trace[:] = 0.0
        b.cp_external_input_current[:] = 0.0

    def _drive(self, pairs, steps, learn=False, read_nmda=False, pre_pairs=None, pre_steps=0):
        """Optional prediction pre-phase (cue alone; lets the slow GABA_B subtractive prediction settle before the
        assertion volley — the validated mismatch-negativity protocol), then the measured phase. When read_nmda,
        accumulate the LATE-window mean recurrent-NMDA conductance on member[0]/member[1] (the confidence read)."""
        b, xp = self.b, self.xp
        b.core_config.enable_hebbian_learning = False
        if pre_pairs is not None and pre_steps > 0:
            precur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
            for idx, pa in pre_pairs:
                precur[xp.asarray(idx)] = xp.float32(pa)
            for _ in range(pre_steps):
                b.cp_external_input_current[:] = precur
                b._run_one_simulation_step()
        b.core_config.enable_hebbian_learning = bool(learn)
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        late_start = steps - max(1, steps // 3)
        g_acc = {0: 0.0, 1: 0.0}
        n_late = 0
        m0 = xp.asarray(self.ix["member0"]); m1 = xp.asarray(self.ix["member1"])
        for t in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            if read_nmda and t >= late_start:
                g_acc[0] += float(to_host(b.cp_conductance_g_nmda[m0].astype(xp.float64).mean()))
                g_acc[1] += float(to_host(b.cp_conductance_g_nmda[m1].astype(xp.float64).mean()))
                n_late += 1
        b.cp_external_input_current[:] = 0.0
        b.core_config.enable_hebbian_learning = False
        if read_nmda:
            n_late = float(max(1, n_late))
            return g_acc[0] / n_late, g_acc[1] / n_late
        return None

    def _wmean(self):
        return float(np.asarray(to_host(self.b.cp_connections.data))[self.masks[GATE]].mean())

    # ---- ERROR-GATED THIRD-FACTOR PORT: the brain's OWN D2 error signal for a trial (learning OFF, a
    #      cp_firing_states[surprise] rate read) — mirrors C1's `_read_surprise` verbatim in shape. ----
    def _read_surprise_hz(self, surprise_pairs, steps=TRAIN_STEPS):
        b, xp = self.b, self.xp
        self._hard_reset()
        b.core_config.enable_hebbian_learning = False
        pre = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in self._cue_pre_pairs():
            pre[xp.asarray(idx)] = xp.float32(pa)
        for _ in range(PRE_STEPS):
            b.cp_external_input_current[:] = pre
            b._run_one_simulation_step()
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in surprise_pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        surp_idx = xp.asarray(self.ix["surprise"])
        count = 0
        for _ in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            count += int(to_host(b.cp_firing_states[surp_idx]).sum())
        b.cp_external_input_current[:] = 0.0
        dur_s = steps * self.b.core_config.dt_ms * 1e-3
        return count / max(len(self.ix["surprise"]), 1) / dur_s

    def _calibrate_conf_gate(self):
        """Build-time gate threshold (host-declared boundary, exactly C1's `_calibrate_gate`): CONFIRM
        (expected, low-surprise) vs CONTRADICT (violated, high-surprise) D2-surprise firing rate for THIS SEED's
        randomly-assigned block pair; threshold sits GATE_FRAC into the expected->violated gap."""
        exp_hz = self._read_surprise_hz(self._confirm_pairs())
        vio_hz = self._read_surprise_hz(self._contradict_pairs())
        thr = exp_hz + GATE_FRAC * max(vio_hz - exp_hz, 0.0)
        return thr, exp_hz, vio_hz

    # ---- the surprise trials for THIS seed's random block pair (same shape as the sibling edges' own) ----
    def _cue_idx(self, concept):
        return self.ix["cue"][concept * self.blk:(concept + 1) * self.blk]

    def _cue_pre_pairs(self):
        return [(self._cue_idx(self.cue_c), CUE_PA)]

    def _contradict_pairs(self):
        assert_idx = self.ix["patient_asserted"][self.assert_cp * self.blk:(self.assert_cp + 1) * self.blk]
        return [(self._cue_idx(self.cue_c), CUE_PA), (assert_idx, CUE_PA)]

    def _confirm_pairs(self):
        assert_idx = self.ix["patient_asserted"][self.cue_c * self.blk:(self.cue_c + 1) * self.blk]
        return [(self._cue_idx(self.cue_c), CUE_PA), (assert_idx, CUE_PA)]

    def _member_drive_pairs(self):
        """The fixed HIGH-evidence metacog member drive: asm0 the intended winner (base+sig), asm1 the runner-up
        (base). Concurrent with the surprise trial so a co-firing surprise slice reaches asm1 via the learned edge."""
        sig = SIG_LO + float(np.clip(READ_EVIDENCE, 0.0, 1.0)) * (SIG_HI - SIG_LO)
        return [(self.ix["member0"], BASE_PA + sig), (self.ix["member1"], BASE_PA)]

    # ---- emergence: grow the cross-edge from experience via an ERROR-GATED three-factor update (ported from
    #      sibling C1's `_gated_update_step`): each episode first READS this trial's own D2-surprise firing rate
    #      (learning OFF); the Hebbian window OPENS for the CONTRADICT + tonic member[1] co-drive (the
    #      host-supervised "prediction-error raises the alternative" teaching signal, declared not hidden) ONLY
    #      when that measured surprise clears the calibrated threshold — pre(surprise) x post(member[1]) x
    #      gate(threshold-cleared), not a repeated presumed-always-surprising co-drive. ----
    def train(self, n_episodes=N_EPISODES):
        if not self.gated:
            # ORIGINAL mechanism (pre-2026-09-02, banked for the A/B control): unconditional rate-Hebbian —
            # every episode's Hebbian window is open, no per-trial surprise read/gate.
            self.gate_calib = None
            traj = [dict(ep=0, w=round(self._wmean(), 4))]
            for ep in range(n_episodes):
                self._hard_reset()
                self._drive(self._contradict_pairs() + [(self.ix["member1"], MEMBER_TEACH_PA)], TRAIN_STEPS,
                            learn=True, pre_pairs=self._cue_pre_pairs(), pre_steps=PRE_STEPS)
                if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                    traj.append(dict(ep=ep + 1, w=round(self._wmean(), 4)))
            self.gate_opens = n_episodes
            self.gate_surp_trace = []
            self.b.core_config.enable_hebbian_learning = False
            self.other_block_after_train = _other_block_drift(self)
            return traj

        thr, exp_hz, vio_hz = self._calibrate_conf_gate()
        self.gate_calib = dict(expected_hz=round(exp_hz, 3), violated_hz=round(vio_hz, 3), threshold=round(thr, 3))
        traj = [dict(ep=0, w=round(self._wmean(), 4))]
        gate_opens = 0
        surp_trace = []
        for ep in range(n_episodes):
            surp_hz = self._read_surprise_hz(self._contradict_pairs())   # THIS episode's own D2 error signal
            gate_open = bool(surp_hz >= thr)
            surp_trace.append(round(surp_hz, 2))
            if gate_open:
                gate_opens += 1
                self._hard_reset()
                self._drive(self._contradict_pairs() + [(self.ix["member1"], MEMBER_TEACH_PA)], TRAIN_STEPS,
                            learn=True, pre_pairs=self._cue_pre_pairs(), pre_steps=PRE_STEPS)
            if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                traj.append(dict(ep=ep + 1, w=round(self._wmean(), 4)))
        self.gate_opens = gate_opens
        self.gate_surp_trace = surp_trace[:8]
        self.b.core_config.enable_hebbian_learning = False
        # ANTI-CHEAT snapshot taken HERE (post-train, PRE-lesion) — run_gate's interaction lesions the pool at its
        # end, so this must be captured before that or the check is vacuous.
        self.other_block_after_train = _other_block_drift(self)
        return traj

    # ---- the load-bearing read: the metacog confidence margin under surprise CONFIRM ('low') vs CONTRADICT
    #      ('high'), co-driven with the fixed high-evidence member drive. Returns the SIGNED divisive-normalized
    #      NMDA margin d=(g0-g1)/(g0+g1+eps) (higher = more confident in the correct answer asm0). ----
    def read_confidence(self, condition):
        if condition == "low":
            surprise_pairs = self._confirm_pairs()
        elif condition == "high":
            surprise_pairs = self._contradict_pairs()
        else:
            raise ValueError(condition)
        member_pairs = self._member_drive_pairs()
        rng = np.random.default_rng(READ_SEED)
        ds, dabs_, g0s, g1s = [], [], [], []
        for _ in range(READ_REPS):
            j = float(rng.normal(0.0, READ_JITTER_PA)) if READ_JITTER_PA > 0 else 0.0
            jittered = [(idx, max(0.0, pa + j)) for idx, pa in member_pairs]
            self._hard_reset()
            g0, g1 = self._drive(jittered + surprise_pairs, DRIVE_STEPS + FREE_STEPS,
                                 pre_pairs=self._cue_pre_pairs(), pre_steps=PRE_STEPS, read_nmda=True)
            denom = g0 + g1 + NORM_EPS
            ds.append((g0 - g1) / denom)
            dabs_.append(abs(g0 - g1) / denom)
            g0s.append(g0); g1s.append(g1)
        return {"conf_signed": float(np.mean(ds)), "conf_abs": float(np.mean(dabs_)),
                "g0": float(np.mean(g0s)), "g1": float(np.mean(g1s))}


GATE_SPEC = CrossEdgeGateSpec(
    name="SURPRISE_to_METACOG_CONFIDENCE",
    cross_edges=CROSS_EDGES,
    train_fn=lambda pool: pool.train(),
    read_fn=lambda pool, cond: pool.read_confidence(cond)["conf_signed"],
    init_weight=W0,
    correct_edges=(GATE,),
    selectivity_pairs=(),   # ONE-SIDED BY DESIGN — the edge targets the single runner-up assembly member[1]; no
                            # companion population for a weight-ratio comparison. Selectivity is demonstrated
                            # FUNCTIONALLY (the anti-cheat other-block check) + at the read (lesion-vanish), not
                            # as a weight ratio.
    grow_factor=5.0, drift_tol=1e-6,
    condition_order=("low", "high"),      # 'low' (CONFIRM, surprise near-silent -> high confidence) is the control
    control="low",
    expected={"high": {"sign": -1, "floor": INTACT_FLOOR}},   # surprise LOWERS the confidence -> negative delta
    lesion_ratio=LESION_RATIO, credit_signal="rate_hebbian",   # DEFAULT is plain; --ablation gated is a banked,
                                                                # tested-and-rejected alternative (see module docstring)
)


def _noedge_pool(seed):
    return _build(seed, with_edge=False)


def _other_block_drift(pool: SurpriseMetacogPool) -> float:
    """ANTI-CHEAT: the OTHER (never-mismatched) surprise blocks' edges into member[1] must stay near W0=0.05 — the
    edge must track THIS SEED's randomly-assigned surprise block, not every block indiscriminately."""
    coo = pool.b.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    data = np.asarray(to_host(pool.b.cp_connections.data))
    surprise_idx = pool.ix["surprise"]
    post = pool.ix["member1"]
    other_max = 0.0
    for c in range(pool.n_trained):
        if c == pool.assert_cp:
            continue
        block = surprise_idx[c * pool.blk:(c + 1) * pool.blk]
        m = np.isin(row, block) & np.isin(col, post)
        if m.any():
            other_max = max(other_max, float(data[m].mean()))
    return other_max


def run_seed(seed, gated=True):
    t0 = time.time()
    pool = SurpriseMetacogPool(seed, gated=gated)
    other_block_w0 = _other_block_drift(pool)          # BEFORE training (should already be ~W0)

    gate = run_gate(pool, GATE_SPEC)                    # trains + emergence + interaction (lesions in place at end)
    other_block_after = pool.other_block_after_train

    bridge_with = SurpriseMetacogPool(seed, gated=gated).b
    bridge_without = _noedge_pool(seed).bridge
    byte_off = verify_byte_off(bridge_with, bridge_without, GATE_SPEC)

    itn = gate["interaction"]
    c_low_i, c_high_i = itn["reads_intact"]["low"], itn["reads_intact"]["high"]
    c_low_l, c_high_l = itn["reads_lesion"]["low"], itn["reads_lesion"]["high"]

    go = bool(gate["emergence"]["PASS"] and gate["interaction"]["PASS"] and byte_off["PASS"]
              and other_block_after < 5.0 * W0)
    return {"seed": int(seed), "GO": go, "elapsed_s": round(time.time() - t0, 1), "gated": bool(gated),
            "emergence": gate["emergence"], "interaction": gate["interaction"], "byte_off": byte_off,
            "other_block_w0_before": round(other_block_w0, 4), "other_block_w0_after": round(other_block_after, 4),
            "gate_calib": pool.gate_calib, "gate_opens": pool.gate_opens, "gate_n_episodes": N_EPISODES,
            "gate_surp_trace": pool.gate_surp_trace,
            "confidence": {"low_intact": c_low_i, "high_intact": c_high_i,
                           "low_lesion": c_low_l, "high_lesion": c_high_l,
                           "drop_intact": c_high_i - c_low_i, "drop_lesion": c_high_l - c_low_l},
            "trajectory": gate["trajectory"]}


def calibrate(seed):
    """Non-canonical calibration run (seed 7 by convention — never one of the 6 canonical seeds): trains the edge,
    reads 'low'/'high' confidence BEFORE lesion, prints the numbers this module's constants were frozen from."""
    pool = SurpriseMetacogPool(seed)
    traj = pool.train()
    low = pool.read_confidence("low")
    high = pool.read_confidence("high")
    print(f"[calibrate seed={seed}] trained w={traj[-1]['w']:.4f} (from {traj[0]['w']:.4f})")
    print(f"  low  (CONFIRM):    conf_signed={low['conf_signed']:+.4f} conf_abs={low['conf_abs']:.4f} "
          f"g0={low['g0']:.4f} g1={low['g1']:.4f}")
    print(f"  high (CONTRADICT): conf_signed={high['conf_signed']:+.4f} conf_abs={high['conf_abs']:.4f} "
          f"g0={high['g0']:.4f} g1={high['g1']:.4f}")
    print(f"  drop (high-low) signed={high['conf_signed']-low['conf_signed']:+.4f}  (INTACT_FLOOR={INTACT_FLOOR})")
    return low, high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1-2 seed indicator")
    ap.add_argument("--calibrate", action="store_true", help="print low/high confidence for --seed, no gate run")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ablation", default="plain", choices=("gated", "plain"),
                    help="'plain' (DEFAULT) = the ORIGINAL committed unconditional rate-Hebbian train() — kept as "
                         "the default because the 2026-09-02 controlled numpy A/B (see the "
                         "*-second-negative finding) found 'gated' REGRESSES robustness (6/6->3/6 on the same 6 "
                         "seeds), so it is banked as a tested-and-rejected method, not preferred. "
                         "'gated' = the 2026-09-02 error-gated three-factor port (ported from sibling C1) — opt-in "
                         "for reproducing that comparison, NOT recommended for the next cupy verify.")
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.seed)
        return 0
    seeds = [42, 43] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    gated = (args.ablation == "gated")

    runs = []
    for s in seeds:
        r = run_seed(s, gated=gated)
        runs.append(r)
        c = r["confidence"]
        print(f"[seed {s}] GO={r['GO']} | grown={r['emergence']['grown'][GATE]:.3f} "
              f"nocorr={r['emergence']['no_corruption']} other_block(before/after)="
              f"{r['other_block_w0_before']:.3f}/{r['other_block_w0_after']:.3f} | "
              f"gate opens={r['gate_opens']}/{r['gate_n_episodes']} calib={r['gate_calib']} | "
              f"conf low={c['low_intact']:+.4f} high={c['high_intact']:+.4f} (drop={c['drop_intact']:+.4f}) | "
              f"lesion low={c['low_lesion']:+.4f} high={c['high_lesion']:+.4f} (drop={c['drop_lesion']:+.4f}) | "
              f"emg={r['emergence']['PASS']} int={r['interaction']['PASS']} byteoff={r['byte_off']['PASS']} "
              f"({r['elapsed_s']}s)", flush=True)
        # LEVER: the lesion must have actually MOVED the high-condition read, else the vanish claim is vacuous.
        lever(f"seed{s} metacog confidence high-condition intact->lesion",
              c["high_intact"], c["high_lesion"])

    n_go = sum(r["GO"] for r in runs)
    all_go = (n_go == len(runs)) and not args.smoke
    tag = "GO" if all_go else ("SMOKE-GO (1-2-seed indicator)" if args.smoke and n_go == len(runs) else
                               ("SMOKE-PARTIAL" if args.smoke else "NO-GO"))
    if gated:
        mech_desc = ("ERROR-GATED (the 2026-09-02 port from sibling C1's 6/6-GO third-factor gate): the edge "
                     "GROWS ONLY on episodes where THIS trial's own measured D2-surprise firing clears a "
                     "calibrated threshold")
    else:
        mech_desc = ("PLAIN rate-Hebbian (the original, DEFAULT mechanism — the error-gated ablation is a "
                     "tested-and-rejected alternative, see --ablation gated): the edge grows unconditionally "
                     "every training episode")
    verdict = (f"{tag} — a learned cross-edge D2-surprise -> E1-metacog confidence, {mech_desc}: {n_go}/{len(runs)} "
               f"seeds GROW the edge from the substrate's own rate-Hebbian rule and are LOAD-BEARING (a "
               f"genuinely high-surprise CONTRADICT trial LOWERS the metacog winner-vs-runner-up confidence "
               f"margin relative to a low-surprise CONFIRM trial, via the learned edge alone), the drop VANISHES "
               f"on lesion, and the pool is byte-identical-off. numpy CPU; NO sim/ edit; additive; no production "
               f"wiring. PARTIAL pending a 6-seed cupy verify.")

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("crossedge_surprise_metacog")
        # NOTE (2026-09-02 fix): "all_seeds_go" was NOT here in the version that produced the already-committed
        # cupy artifact (_crossedge_surprise_metacog_6seed.json, 3 preconditions only) -- it was a later addition
        # that conflated the OUTCOME (n_go == len(runs), i.e. the verdict itself) with a PRECONDITION (a validity
        # check the run's interpretability depends on). tools.verdict.Verdict.decide() forces UNDEFINED whenever
        # ANY registered precondition is unmet, so an outcome-as-precondition makes a genuine, honest NO-GO
        # unwritable (gates/verdict_preconditions correctly BLOCKED on this). Removed; the 3 checks below are the
        # real validity preconditions (did the read/build stay uncorrupted enough to trust the outcome at all).
        Vd.require("confidence_drops_intact", sum(r["confidence"]["drop_intact"] < 0 for r in runs),
                   expect=lambda x: x == len(runs),
                   note="the metacog confidence margin must actually DROP (high < low) with the edge intact on "
                        "every seed — else the 'error lowers confidence' claim is vacuous")
        Vd.require("byte_identical_off", sum(r["byte_off"]["PASS"] for r in runs), expect=lambda x: x == len(runs),
                   note="the no-edge pool's base connectivity is byte-identical (integration added ONLY the edge)")
        Vd.require("anti_cheat_other_blocks_stay_near_w0",
                   sum(r["other_block_w0_after"] < 5.0 * W0 for r in runs), expect=lambda x: x == len(runs),
                   note="non-participating surprise blocks' edges into member[1] must stay near the W0=0.05 seed")
        dec = Vd.decide(all_go or (args.smoke and n_go == len(runs)), verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "crossedge_surprise_metacog_derisk", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(seeds), "seeds": seeds, "ablation": args.ablation,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "gate_spec": {"name": GATE_SPEC.name, "correct_edges": GATE_SPEC.correct_edges,
                             "conditions": GATE_SPEC.condition_order, "control": GATE_SPEC.control,
                             "expected": GATE_SPEC.expected, "credit_signal": GATE_SPEC.credit_signal,
                             "read_evidence": READ_EVIDENCE, "member_teach_pa": MEMBER_TEACH_PA,
                             "cross_edges": [dict(key=ce.key, src=ce.source_region, tgt="workspace.member[1]")
                                             for ce in CROSS_EDGES]},
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[SURPRISE->METACOG] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
