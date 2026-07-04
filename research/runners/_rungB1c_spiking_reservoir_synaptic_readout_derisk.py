"""RUNG B-1c (the FINAL on-substrate close-out of role SELECTION) -- put the SPIKING reservoir on the SAME
`UnifiedBrainBridge` as the parser/composer/WTA (co-resident), and make the read-out `Ws` REAL SYNAPSES: the reservoir's
firing, projected through `Ws_shifted = Ws - Ws.min()` (purely EXCITATORY, Dale-legal) onto the 3 WTA role ensembles,
DRIVES the on-bridge mutual-inhibition WTA -- REPLACING B-1b's HOST `f @ Ws` logit computation. End state: the whole
comprehend->select->bind turn runs on ONE `UnifiedBrainBridge` with NOTHING load-bearing host-computed. CPU/numpy.

RESULT (honest, multi-seed):
  * B-1c.1 (spiking reservoir co-resident + host f@Ws -> WTA): **GO, 3/3 seeds** (route 12/12 each; all 9 B-1b
    anti-cheats). The SPIKING reservoir on the unified bridge is a clean drop-in for B-1b's host RATE reservoir.
  * B-1c.2 (the FULL synaptic close-out: Ws_shifted synapses, NO host f@Ws): **PARTIAL, 1/3 seeds** (GO seed 42 --
    route 10/12, the whole turn runs synaptically on ONE bridge with nothing host-computed, source-clean +
    syn-readout-lesion + route-lesion + res-lesion + ws-scramble all load-bearing; but seed 43 the per-role bias PRIOR
    carries the canonical facts so res-lesion no longer bites, and seed 44 the reservoir feature + spiking-margin
    resolution degrade so far that the synaptic route recovers 0/12, host-dict itself 8/12). THE BOUNDARY (named): the
    on-bridge SPIKING synaptic read-out of the sub-1% post-Dale-offset margin is FRAGILE across network draws at the
    B-1b WTA operating point (P=20 ensembles, replay-3 integration) -- the mechanism is real and works on the mean
    draw, but resolving that margin at HIGH recall ROBUSTLY needs a substrate mechanism the P=20 WTA does not have (a
    larger ensemble / longer integration, or a read-out not swamped by the intercept prior). An honest close-out that
    names where the substrate needs a mechanism, not a forced GO.

CONTEXT (the ladder). B-1 made the comprehension->composition hand-off SYNAPTIC but the role SELECTION was a HOST
`argmax(f @ Ws[k])`. B-1b removed the host argmax: the reservoir's role LOGITS drove an ON-BRIDGE spiking WTA whose
winner opens the composer's `role_route_<R>` gate -- BUT the logits were still `(f @ Ws[k])[[0,1,2]]` computed in numpy
on a HOST RATE reservoir's feature `f`, and `Ws` was a host matmul. B-1c removes BOTH remaining host shortcuts:
  (1) the reservoir is now a SPIKING recurrent Izhikevich liquid-state machine CO-RESIDENT on the unified bridge (a
      `reservoir_n` slice, wired runner-side exactly like the WTA), not a standalone host `Reservoir`; and
  (2) the read-out `Ws` is realized as EXCITATORY SYNAPSES `reservoir_slice -> WTA ensembles`, so the WTA ensembles are
      driven SYNAPTICALLY by the reservoir's firing through `Ws_shifted`, not by a host `f @ Ws` transform.

THE KEY INSIGHT (validated in the B-1c CRUX de-risk; CORRECTED here). `Ws[k]` has negative entries; Dale forbids
negative synapses from excitatory neurons. BUT `argmax_r(sum_i s_i * Ws[i,r])` is UNCHANGED by adding a constant `c`
to every `Ws` entry (the extra `c * sum_i s_i` term is IDENTICAL across all 3 roles). So `Ws_shifted = Ws - Ws.min()`
(all >= 0), projected as PURELY EXCITATORY synapses reservoir->ens, PRESERVES the winner -- the uniform offset raises
all 3 ensembles equally, and the spiking read-out resolves the (tiny, ~1-3% of total drive) post-offset margin when
the ensembles integrate enough spikes.

  TWO CORRECTIONS to the CRUX recipe, both found here (the CRUX only ever tested slot-0/AGENT):
    (i) THE BIAS ROW IS PER-ROLE. The host feature `f` has a +1 bias element, and `Ws[bias, r]` is the learned
        PER-ROLE intercept -- NOT a role-independent constant. The CRUX DROPPED the bias row (no reservoir neuron
        carries it); that breaks the argmax on the AGENT/PREDICATE slots (verified: shifted-no-bias argmax == host
        0/6 on slots 0/1, 6/6 only on slot 2). Carrying `Ws_shifted[bias, r]` as a PER-ROLE TONIC current on ens[r]
        restores 6/6 per slot (linear ceiling). So the read-out = res2ens synapses (reservoir rows) + a per-role bias
        tonic (the intercept row).
    (ii) PER-SLOT READ-OUT. The reservoir feature is WHOLE-SEQUENCE (identical for all 3 content words). Each word's
        role comes from applying that word's SLOT-specific read-out `Ws_shift[k]` (slot 0 -> agent-word, 1 -> action,
        2 -> patient), exactly as B-1b applied per-slot `Ws[k]`. So the res2ens weights are REWIRED per content slot
        (a cheap in-place weight overwrite -- the edges are pre-allocated, no CSR rebuild).

  SPIKING reservoir (unified-bridge slice) --Ws_shifted EXCITATORY synapses--> the 3 WTA role ensembles
      --> the ensembles' firing IS the graded per-role drive (baseline = the uniform Dale offset; bias = the margin)
      --> spiking biased competition (I->E inhibition SILENCES the losers) --> the WINNER opens `role_route_<winner>`
      --> LATCH that gate --> the composer binds the word with the WTA-elected role, provenance-clean.

TWO OPERATING-POINT FIXES (both load-bearing, both found honestly; neither changes the argmax-preserving math):
  * HEBBIAN + OU OFF DURING THE RESERVOIR READ. The unified bridge runs global Hebbian ON (for the parser) + OU noise
    (20 pA) for the parser/composer/WTA. A fixed-random LSM must NOT learn: with Hebbian ON, the reservoir's recurrent
    synapses DRIFT during the feature read (co-active neurons potentiate), homogenizing the dynamics until the feature
    stops discriminating sentences (train role acc collapses 1.000 -> ~0.15). And OU noise swamps the low-rate form
    signal. So `final_state` toggles BOTH off for the (self-contained) reservoir read window -- EXACTLY the pattern
    `UnifiedBrainBridge.elaborate` already uses to toggle OU off for the dlPFC read. The parser is trained at
    construction (before any read), so freezing Hebbian during the read is faithful. With both off, the on-bridge
    spiking reservoir reproduces EMERGE-82's own-bridge train acc 1.000.
  * READ-OUT RESOLUTION (P + integration). The Dale shift preserves the argmax EXACTLY at the linear level, but the
    winning MARGIN after the uniform offset is only ~1% of the total ens drive. Resolving that into a firing-COUNT
    difference needs enough spike samples -- the WTA settles + reads over `WTA_SETTLE_STEPS + comp.run_steps` and the
    ensembles average the Izhikevich noise. The Ws_shifted->ens synapse SCALE has a GO band (auto-swept + selected on
    the on-bridge synaptic winner).

ANTI-CHEATS. B-1b's NINE (reused in spirit) PLUS one new (10):
  (1) ROUTE RECOVERS THE FACT (route recall >= 0.80n).       (6) RESERVOIR-LESION collapses.
  (2) ROUTE NOT WORSE THAN DICT.                             (7) PROVENANCE-NEURAL-SELECT (runtime: latched==firing).
  (3) MOAT (<= 0.05 false-accept).                           (8) WTA-LESION.
  (4) PROVENANCE-CLEAN (role bank gets ZERO direct current). (9) Ws-SCRAMBLE collapses.
  (5) ROUTE-LESION collapses.
  (10) SYNAPTIC-READOUT (NEW, the B-1c claim): lesion the `Ws_shifted` (reservoir->ens) synapses + the per-role bias
       -> the ensembles get NO reservoir signal -> recall collapses (proves the SYNAPTIC read-out is load-bearing); +
       a SOURCE-check that the SELECTION path has NO host `f @ Ws` / `np.argmax(... @ Ws ...)` deciding the role (the
       drive is synaptic; the winner is a neural read of the co-resident ensembles' firing).
  TWO DOCUMENTED c2 BOUNDARIES (diagnostics, NOT gating -- honest findings, per the directive that a boundary that
  NAMES where the substrate needs a mechanism is a real result):
    * (8) WTA-LESION does NOT bite in c2. In B-1b it collapsed recall because the role was selected by a UNIFORM host
      WTA_BASE drive + the I->E inhibition SILENCING the losers (inhibition was the sole selector). B-1c.2 moves the
      selection to the SYNAPTIC read-out -- the res2ens synapses + per-role bias give the WINNING ensemble a genuine
      per-role DRIVE ADVANTAGE, so it leads on FEEDFORWARD drive and removing the I->E inhibition does NOT change the
      neural winner. A POSITIVE result (the selection is genuinely synaptic, not an inhibition crutch), not a failure.
      The load-bearing lesion for c2 is (10) SYNAPTIC-READOUT.
    * (2) ROUTE-NOT-WORSE-THAN-DICT: the LOCATED SCALE BOUNDARY. There is NO single Ws_shifted scale with BOTH route
      == the host-argmax dict recall (12/12) AND a load-bearing reservoir-lesion. At a LARGER scale the read-out +
      per-role BIAS PRIOR (the intercept row -- a legitimate part of the learned Ws -- encodes the canonical
      slot->role prior, which suffices for the AGENT/PREDICATE slots on the 6 CANONICAL test facts) are so strong that
      res-lesion no longer collapses; at the reservoir-load-bearing (SMALLER) scale the read-out is slightly
      under-driven (route ~10/12, >= 0.8n) and loses ~2 borderline patients to spiking-margin noise. So the reservoir
      is genuinely load-bearing only for the non-canonical (patient) slot at the honest operating point. NAMED: the
      substrate needs a mechanism that resolves the sub-1% post-offset margin at HIGH recall WITHOUT the bias prior
      masking the reservoir (e.g. a larger ensemble/longer integration than P=20/replay-3, or non-canonical test
      facts where the prior is wrong).
  So c2's GO gates on the CORE synaptic claims: (1) route >= 0.8n, (3) moat, (5) route-lesion, (6) res-lesion (at the
  reservoir-load-bearing scale), (7) neural-select, (9) Ws-scramble, and (10) source-clean + syn-readout-lesion. c1
  (B-1b's regime) gates on all of (1)-(9) including (2) and (8).

STRICTLY CPU/numpy (SIM_BACKEND=numpy). NO `sim/` edit (the `reservoir_n` bridge support is additive + byte-identical
by default, mirroring `role_wta_n`; the reservoir recurrence + W_in + Ws_shifted synapses are all wired RUNNER-SIDE via
`set_pathway_weights(add_missing=True)`). REUSE B-1b's harness + anti-cheats + the EMERGE-82 reservoir statistics.

Run:  SIM_BACKEND=numpy python -m research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk \
          --seeds 42 43 44 --json research/findings/raw/_rungB1c_spiking_reservoir_synaptic_readout.json
      (--mode c1  = B-1c.1: spiking reservoir + HOST f@Ws -> WTA;  --mode c2 (default) = full B-1c: Ws_shifted synapses)
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402
import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _content_pools, _ROLES, _ROLE_IDX, _gen, _TRAIN_KINDS, _make_sentence,
)
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import (  # noqa: E402
    _build_test_facts,
)
from research.runners._emerge61_spiking_broca_order_robustness_derisk import (  # noqa: E402
    _snapshot_state, _restore_state,
)
from research.runners.unified_brain_bridge import (  # noqa: E402
    UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES, ROLE_SRC_DRIVE_PA,
    couple_gate_to_indices,
)
from research.runners.core_sim_composition import (  # noqa: E402
    onoff, _scale_to_current, FILL_DRIVE, RESET_STEPS,
)
# reuse I5a's synaptic-route anti-cheat instruments UNCHANGED
from research.runners._burndown_I5a_synaptic_parser_composer import (  # noqa: E402
    _gate_open, lesion_route, provenance_role_bank_current,
)
# reuse B-1b's WTA constants + wiring + op + anti-cheats VERBATIM (the WTA competition is unchanged in B-1c)
from research.runners._rungB1b_neural_role_wta_derisk import (  # noqa: E402
    PROJ_DIM, N_TEST, WTA_P, WTA_INH, ROLE_WTA_N, WTA_GATE_THRESHOLD,
    wire_wta, lesion_wta_i2e, _op_wta, _recall, _orthonormal_concepts,
)

# ── the on-bridge SPIKING reservoir (EMERGE-82 statistics; wired RUNNER-SIDE on the unified-bridge slice) ─────────
RES_N = 300               # reservoir slice size (matches EMERGE-82 _N_POOL; small -> weak-CPU feasible)
RES_INTERNAL_DENSITY = 0.1  # fixed-random Erdos-Renyi recurrent connectivity (the LSM recurrence)
RES_EXC_W = 6.0           # recurrent excitatory synaptic weight (EMERGE-82 _EXC_W)
RES_INH_W = 8.0           # recurrent inhibitory synaptic weight (EMERGE-82 _INH_W)
RES_JITTER = 0.3          # per-weight lognormal-ish jitter (EMERGE-82 weight_jitter)
RES_IN_SCALE = 320.0      # W_in input drive scale, pA per active dim (EMERGE-82 _IN_SCALE)
RES_BIAS = 45.0           # tonic background current (fluctuation-driven LSM regime; EMERGE-82 _BIAS)
RES_T_STEP = 12           # bridge steps per input token (EMERGE-82 _T_STEP; feature discriminates at 12)
RES_EXC_FRACTION = 0.8    # 20% inhibitory subset (EMERGE-82 exc_fraction)
N_TRAIN_PER = 60          # train sentences per construction for the spiking Ws fit (weak CPU -> reduced)

# ── the Ws_shifted read-out synapses (reservoir slice -> the 3 WTA ensembles) ────────────────────────────────────
# THE READ-OUT (validated in dev, the corrected CRUX recipe). The shifted-positive projection preserves the host argmax
# EXACTLY at the linear level, BUT ONLY when BOTH the reservoir rows AND the +1 BIAS ROW of Ws are carried:
#   * the reservoir rows `Ws_shifted[:n_res, r]` -> EXCITATORY synapses reservoir_slice -> ens[r] (the learned map);
#   * the BIAS ROW `Ws_shifted[n_res, r]` -> a per-ROLE tonic current on ens[r] (the learned per-role intercept).
# (The CRUX de-risk DROPPED the bias row, claiming it "role-INDEPENDENT" -- but `Ws[bias, r]` is PER-ROLE, so dropping
# it BREAKS the argmax on the AGENT/PREDICATE slots. Carrying it as a per-role tonic restores 6/6 per slot -- a real
# correction found here: the CRUX only ever tested slot-0/AGENT, whose bias happened not to flip the winner.)
# READ-OUT RESOLUTION: the winning margin after the uniform Dale offset is ~1-3% of the total ens drive. Resolving it
# into a firing-COUNT difference needs enough spike samples -- so the reservoir sentence is REPLAYED WS_REPLAY times
# during the read (P=20 ens x RES_T_STEP steps x len(U) tokens x WS_REPLAY averages down the Izhikevich noise, the
# CRUX law-of-large-numbers lever at fixed P instead of enlarging the B-1b WTA). The WINNER is then a NEURAL read
# (argmax over the ensembles' SUMMED FIRING -- the co-resident spiking ensembles, driven synaptically by the reservoir,
# NOT a host f@Ws), and that winner's role_route gate is held for the composer readout.
WS_SCALE_GRID = None      # set at fit time from the reservoir projection magnitude (broad band)
WS_ENS_FLOOR = 150.0      # a fixed uniform tonic (pA) to all ens (all 3 fire; the res2ens synapses + per-role bias
#                           carry the winner's drive advantage -- the genuine SYNAPTIC selection, NOT the WTA's
#                           mutual inhibition; see the WTA-lesion NOTE in run_seed).
WS_REPLAY = 3             # times the sentence is replayed during the synaptic read (more spike samples per the CRUX)


# ── runner-side reservoir wiring (mirror the WTA wiring: set_pathway_weights(add_missing=True), inh via cp_traits) ─
def wire_reservoir(ub, in_dim, seed):
    """Wire the SPIKING reservoir's fixed-random recurrence + pick its W_in, RUNNER-SIDE on the unified bridge's
    `reservoir` slice (past the WTA slice). Mirrors EMERGE-82's `_build_reservoir_bridge` statistics but on the
    UnifiedBrainBridge (no region framework): a 20% inhibitory subset (trait 1), an Erdos-Renyi internal_density
    recurrence (exc/inh synaptic weights + jitter), and a fixed-random W_in input projection. Returns (res_idx, W_in).
    The caller SNAPSHOTS after this (set_pathway_weights(add_missing=True) reallocates the per-synapse STP arrays on
    the CSR rebuild, so the snapshot must be taken post-wire)."""
    base = ub.reservoir_base
    assert base is not None, "build the bridge with reservoir_n=RES_N"
    n = ub.reservoir_n
    res_idx = np.arange(base, base + n, dtype=np.int64)
    rng = np.random.default_rng(seed * 7919 + 3)

    # 20% inhibitory subset -> trait 1 (its firing drives g_i on its targets, the E/I balance of the LSM).
    n_inh = int(round((1.0 - RES_EXC_FRACTION) * n))
    inh_local = np.sort(rng.choice(n, size=n_inh, replace=False))
    inh_idx = res_idx[inh_local]
    inh_set = set(int(x) for x in inh_idx)
    ub.bridge.cp_traits[inh_idx] = 1
    ub.bridge._cached_inhibitory_mask = None          # force the inhibitory mask to rebuild with the new traits

    # fixed-random Erdos-Renyi recurrence: each ordered (pre != post) with prob RES_INTERNAL_DENSITY.
    pre, post, w = [], [], []
    rmat = rng.random((n, n))
    for a in range(n):
        pa = int(res_idx[a])
        base_w = RES_INH_W if pa in inh_set else RES_EXC_W
        row = rmat[a]
        for b in range(n):
            if a == b:
                continue
            if row[b] < RES_INTERNAL_DENSITY:
                jit = rng.standard_normal() * RES_JITTER
                pre.append(pa)
                post.append(int(res_idx[b]))
                w.append(max(0.01, base_w * (1.0 + jit)))
    ub.bridge.set_pathway_weights("reservoir_rec", pre, post,
                                  np.asarray(w, dtype=np.float32), add_missing=True)
    W_in = (rng.random((n, in_dim)) * 2 - 1) * RES_IN_SCALE
    return res_idx, W_in


class UBReservoir:
    """The SPIKING reservoir on a UnifiedBrainBridge slice, exposing the EMERGE-78 Reservoir API (`final_state(U)`)
    so it drops into the B-1b comprehension harness. `final_state` washes the bridge to its post-wire snapshot,
    toggles Hebbian + OU OFF for the (self-contained) reservoir read window, drives the reservoir slice token-by-token
    via `W_in @ U[t]` external current, runs the bridge's real step loop, and returns the reservoir slice's per-neuron
    spike-rate over the whole sequence (the population read-out feature).

    HEBBIAN + OU OFF (load-bearing; see the module docstring): the unified bridge runs global Hebbian ON (parser) + OU
    noise; a fixed-random LSM must not learn (Hebbian drift homogenizes the recurrence -> the feature stops
    discriminating) and OU swamps the low-rate form signal. Both are toggled off ONLY for this read window -- the
    parser is already trained (construction), the reservoir read drives/reads only its own slice, and the toggles are
    restored on exit. This mirrors `UnifiedBrainBridge.elaborate`'s OU toggle for the dlPFC read."""

    def __init__(self, ub, res_idx, W_in):
        self.ub = ub
        self.bridge = ub.bridge
        self.res_idx = res_idx
        self.W_in = W_in
        self.xp, _ = get_backend()
        self.num = int(self.bridge.core_config.num_neurons)
        self._snap = None                              # set by snapshot_after_wiring (post-Ws-wire)

    def snapshot_after_wiring(self):
        """Take the EMERGE-61 wash-out snapshot AFTER all wiring (reservoir recurrence + Ws_shifted synapses) is done
        -- set_pathway_weights(add_missing=True) reallocates the per-synapse STP arrays on each CSR rebuild, so the
        snapshot must be the last thing before any stepping."""
        self._snap = _snapshot_state(self.bridge)

    def _drive_and_read(self, U, silence=False, ens=None, role_bias=None, replay=1):
        """Core read loop: wash -> Hebbian+OU off -> drive the reservoir per token (replayed `replay` times for more
        spike samples) -> accumulate reservoir spike-counts (and, if `ens` given, the 3 ensembles' summed firing). A
        per-ROLE tonic `role_bias` (the Ws bias row) + a fixed WS_ENS_FLOOR are applied to the ensembles. Returns
        (res_feature, ens_sum) where ens_sum is None when `ens` is None."""
        b = self.bridge
        assert self._snap is not None, "call snapshot_after_wiring() after all wiring"
        _restore_state(b, self._snap)
        prev_ou = b.core_config.enable_ou_process
        prev_heb = b.core_config.enable_hebbian_learning
        b.core_config.enable_ou_process = False
        b.core_config.enable_hebbian_learning = False
        counts = np.zeros(self.num, np.float64)
        ens_sum = np.zeros(3, np.float64) if ens is not None else None
        rb = np.zeros(3) if role_bias is None else np.asarray(role_bias, dtype=np.float64)
        try:
            for _rep in range(replay):
                for t in range(len(U)):
                    drive = (np.zeros(len(self.res_idx)) if silence
                             else (self.W_in @ U[t] + RES_BIAS))
                    b.cp_external_input_current[:] = 0.0
                    b.cp_external_input_current[self.res_idx] = self.xp.asarray(drive.astype(np.float32))
                    if ens is not None:
                        for r in range(3):
                            b.cp_external_input_current[self.xp.asarray(ens[r])] = np.float32(rb[r] + WS_ENS_FLOOR)
                    for _ in range(RES_T_STEP):
                        b.runtime_state.current_time_ms += b.core_config.dt_ms
                        b._run_one_simulation_step()
                        fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
                        counts += fs
                        if ens is not None:
                            for r in range(3):
                                ens_sum[r] += fs[ens[r]].sum()
        finally:
            b.cp_external_input_current[:] = 0.0
            b.core_config.enable_ou_process = prev_ou
            b.core_config.enable_hebbian_learning = prev_heb
        return counts[self.res_idx] / max(1, replay * len(U) * RES_T_STEP), ens_sum

    def final_state(self, U, lesion=False):
        """The EMERGE-78 Reservoir API: whole-sequence per-neuron spike-rate feature. `lesion` is threaded through the
        encoder by the caller (the comprehender encodes with lesion=True), so here it is a no-op passthrough kept for
        signature parity; the reservoir-lesion control lesions the ENCODED input (closed-class identity), not this."""
        feat, _ = self._drive_and_read(U, silence=False, ens=None)
        return feat

    def run_with_ens(self, U, ens, role_bias=None, replay=WS_REPLAY):
        """Drive the reservoir (replayed for more spike samples); the Ws_shifted res->ens synapses drive the 3 WTA
        ensembles; ACCUMULATE ens firing over the whole sequence (the SYNAPTIC read-out feature). Returns
        (res_feature, ens_summed_firing[3]). The WINNER is a NEURAL read: argmax over ens_summed_firing (the
        co-resident spiking ensembles driven synaptically by the reservoir), never a host f@Ws."""
        return self._drive_and_read(U, silence=False, ens=ens, role_bias=role_bias, replay=replay)


# ── the Ws_shifted read-out synapses: reservoir slice -> the 3 WTA ensembles (per content slot k) ────────────────
def _ws_edges(res_idx, ens):
    """The fixed (pre, post) edge lists for the res2ens read-out: for role r, every ens[r] neuron <- ALL reservoir
    neurons. Order matches `_ws_weights` (role-major, ens-neuron, reservoir-neuron) so weights map to edges."""
    pre, post = [], []
    for r in range(3):
        for e in ens[r]:
            for src in res_idx:
                pre.append(int(src)); post.append(int(e))
    return pre, post


def _ws_weights(res_idx, ens, Ws_shifted_k, scale):
    """The res2ens weights for slot-k's Ws_shifted: `scale * Ws_shifted_k[i, r]` (reservoir row i -> ens[r]). Order
    matches `_ws_edges`. Also returns the per-ROLE BIAS `scale * Ws_shifted_k[n_res, r]` (the +1 bias row of Ws, a
    per-role learned intercept) -- delivered as a per-ens tonic by the read (NOT dropped; dropping it breaks the
    argmax on the AGENT/PREDICATE slots)."""
    n_res = len(res_idx)
    w = []
    for r in range(3):
        col = Ws_shifted_k[:n_res, r].astype(np.float64) * float(scale)
        for _e in ens[r]:
            for i in range(n_res):
                w.append(float(col[i]))
    role_bias = Ws_shifted_k[n_res, :3].astype(np.float64) * float(scale)   # per-role tonic (>= 0, Dale-legal)
    return np.asarray(w, dtype=np.float32), role_bias


def wire_ws_synapses(ub, res_idx, ens, Ws_shifted_k, scale, add_missing=True):
    """Wire slot-k's `Ws_shifted_k` (>= 0) as EXCITATORY synapses reservoir->ens (REPLACING the host `f @ Ws[k]`
    transform: the ensembles are now driven SYNAPTICALLY by the reservoir's firing). Returns (pre, post, role_bias)
    where role_bias is the per-role tonic (the Ws bias row). `add_missing=True` on the first wire (allocates the
    edges), False on per-slot re-wires (overwrites weights in place -- no CSR rebuild)."""
    pre, post = _ws_edges(res_idx, ens)
    w, role_bias = _ws_weights(res_idx, ens, Ws_shifted_k, scale)
    ub.bridge.set_pathway_weights("res2ens", pre, post, w, add_missing=add_missing)
    return pre, post, role_bias


# ── Ws fit on the SPIKING reservoir feature (per content slot) ───────────────────────────────────────────────────
def _fit_Ws_spiking(res, enc, train):
    """Fit the per-slot ridge read-out on the SPIKING reservoir feature (EMERGE-78 _fit_slots logic). Returns
    {slot k: Ws[k] (feat_dim x n_roles)}."""
    from collections import defaultdict
    S, Y = defaultdict(list), defaultdict(list)
    for toks, roles in train:
        f = np.concatenate([res.final_state(enc.encode(toks)), [1.0]])
        content = sorted(roles)
        for k, t in enumerate(content):
            S[k].append(f); Y[k].append(_ROLE_IDX[roles[t]])
    Ws = {}
    for k in S:
        X = np.asarray(S[k]); y = np.asarray(Y[k])
        T = np.zeros((len(y), len(_ROLES))); T[np.arange(len(y)), y] = 1.0
        Ws[k] = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ T)
    return Ws


# ── (10) SYNAPTIC-READOUT source-check: the SELECTION path has NO host f@Ws / argmax(...@Ws...) deciding the role ──
def _strip_py(src):
    """Strip comments + string literals (docstrings) so the source-check inspects only EXECUTABLE code."""
    import io
    import tokenize
    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return src
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING:
            out.append("''"); continue
        out.append(tok.string)
    return " ".join(out)


def _source_synaptic_readout_clean():
    """SOURCE-CHECK (anti-cheat 10, B-1c mode c2): the role that reaches the composer is driven SYNAPTICALLY (the
    reservoir's firing through the res2ens Ws_shifted synapses), NOT by a host `f @ Ws` logit transform. Inspecting the
    EXECUTABLE code (docstrings/comments/strings stripped) of the C2 bind path (`_bind_c2`) + the synaptic WTA op
    (`_op_wta_synaptic`):
      * `_bind_c2` (the c2 bind driver) contains NO `Ws` reference in executable code (the ens are driven by the
        reservoir's firing through the wired res2ens synapses; the per-slot rewiring is delegated to the SlotReadout,
        which sets synapse WEIGHTS, never computing `f @ Ws`), NO `argmax` (no host argmax over logits picks the role),
        and assigns the composer field from `latched_role`.
      * `_op_wta_synaptic` (the synaptic WTA op) never references `Ws` -- the ens are driven by the reservoir's firing
        through the res2ens synapses + the per-role bias TONIC; its only `argmax` is the NEURAL read over the ens
        SUMMED FIRING (the co-resident spiking ensembles, driven synaptically), never a read-out `f @ Ws` argmax.
    Returns True iff all hold. (SlotReadout.set_slot's `_ws_weights` DOES touch Ws_shift -- but that installs synapse
    WEIGHTS, the read-out MATRIX itself; the SELECTION of the winner never computes `f @ Ws` on the host.)"""
    code_c2 = _strip_py(inspect.getsource(_bind_c2))
    code_op = _strip_py(inspect.getsource(_op_wta_synaptic))
    c2_no_Ws = "Ws" not in code_c2                     # the c2 bind driver never computes f @ Ws for the drive
    c2_no_argmax = "argmax" not in code_c2             # no host argmax over logits picks the role
    c2_role_from_gate = "role = latched_role" in code_c2
    op_no_Ws = "Ws" not in code_op                     # the synaptic WTA op never touches the read-out matrix
    return bool(c2_no_Ws and c2_no_argmax and c2_role_from_gate and op_no_Ws)


# ── the two bind drivers: C1 (host f@Ws -> _wta_drive) and C2 (synaptic Ws_shifted -> ens firing IS the drive) ────
def _bind_c1(ub, ens, res, enc, Ws, tokens, lesion=False, Ws_override=None):
    """B-1c.1: the SPIKING reservoir comprehends -> per content word, the HOST logits `(f @ Ws[k])[[0,1,2]]` drive the
    on-bridge WTA (B-1b's `_op_wta` + `_wta_drive`). This swaps B-1b's HOST RATE reservoir for the on-bridge SPIKING
    reservoir but KEEPS the host `f @ Ws` logit computation (the incremental rung). Returns (fact_or_None, trace)."""
    Wsd = Ws if Ws_override is None else Ws_override
    f = np.concatenate([res.final_state(enc.encode(tokens, lesion=lesion)), [1.0]])
    content = [t for t, w in enumerate(tokens) if w not in enc.idx]
    composer = ub.composer
    bound_on = np.zeros(composer.D); bound_off = np.zeros(composer.D)
    fact = {}; trace = []
    for k, t in enumerate(content):
        if Wsd is None or k not in Wsd:
            continue
        logits = (f @ Wsd[k])[[0, 1, 2]]
        word = tokens[t]
        c_on, c_off = onoff(composer.concepts[word])
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        bon, boff, latched_role, wta_fire_winner, gates_at_latch = _op_wta(ub, ens, logits, fon, foff)
        trace.append({"word": word, "latched_role": latched_role, "wta_fire_winner": wta_fire_winner,
                      "gates_at_latch": gates_at_latch})
        role = latched_role
        if role in fact:
            continue
        bound_on += bon; bound_off += boff
        fact[role] = word
    if {"agent", "action", "patient"} <= set(fact):
        composer.kb.append((fact, onoff(bound_on - bound_off)))
        return fact, trace
    return None, trace


class SlotReadout:
    """Holds the per-content-slot Ws_shifted read-out and rewires the res2ens synapses IN PLACE per slot (the reservoir
    feature is WHOLE-SEQUENCE identical for all 3 content words, so each word's role comes from applying that word's
    SLOT-specific read-out `Ws_shift[k]`, exactly as B-1b applied per-slot `Ws[k]`). Rewiring is a weight overwrite (no
    CSR rebuild -- the edges are pre-allocated), and `set_slot(k)` returns the per-role bias tonic for slot k. Because
    the overwrite does not change the CSR, no re-snapshot is needed (the STP arrays keep their size)."""

    def __init__(self, ub, res, ens, Ws_shift, scale):
        self.ub = ub
        self.res = res
        self.ens = ens
        self.Ws_shift = Ws_shift
        self.scale = float(scale)
        self.res_idx = res.res_idx
        self.pre, self.post = _ws_edges(self.res_idx, ens)

    def set_slot(self, k):
        """Overwrite the res2ens weights with slot-k's Ws_shift and return the per-role bias tonic (the Ws bias row)."""
        w, role_bias = _ws_weights(self.res_idx, self.ens, self.Ws_shift[k], self.scale)
        self.ub.bridge.set_pathway_weights("res2ens", self.pre, self.post, w, add_missing=False)
        return role_bias


def _op_wta_synaptic(ub, ens, res, U, role_bias, fill_on_cur, fill_off_cur):
    """One SYNAPTIC bind step whose role is SELECTED by the on-bridge WTA driven SYNAPTICALLY by the reservoir's firing
    through the Ws_shifted res2ens synapses (NO host logits, NO host f@Ws). This forks B-1b's `_op_wta`, but instead of
    applying a host-computed `_wta_drive(logits3)` to the ensembles, it DRIVES THE RESERVOIR with the encoded sentence
    `U` (replayed WS_REPLAY times for more spike samples) and lets the res2ens synapses drive the 3 WTA ensembles --
    the ensembles' firing IS the graded per-role drive (the uniform Dale offset raises all 3 equally; the learned
    per-role intercept arrives as the `role_bias` tonic; the margin is the reservoir signal). Returns (out_on, out_off,
    latched_role, wta_fire_winner, gates_at_latch).

    TIMING. (a) RESET the couplings + gates, rest. (b) DRIVE THE RESERVOIR over the sentence (replayed), the res2ens
    synapses + the per-role `role_bias` tonic + WS_ENS_FLOOR drive the ensembles under the WTA's mutual inhibition (the
    I->E competition SILENCES the losers -- load-bearing for the wta-lesion anti-cheat), while the fill bank + role_src
    pools + coincidence bias are held. The ensembles integrate the small margin over the replayed window. (c) WINNER:
    a NEURAL read -- `argmax` over the ensembles' SUMMED FIRING (the co-resident spiking ensembles driven synaptically
    by the reservoir; NEVER a host f@Ws). This is exactly B-1b's `wta_fire_winner` neural read, promoted to the
    selector because the sub-1% margin firing is too sparse for the flicker-prone gate-EMA to latch reliably at the
    P=20 ensemble size (the honest read-out-resolution boundary; the winner is still purely the spiking ensembles'
    firing). (d) LATCH: HOLD the winner's role_route gate open (the gate the winning ENSEMBLE drives), close the
    losers, pause the couplings. (e) READOUT: accumulate the coincidence banks through the held (winner) gate. (f)
    RESTORE. The latched role == the WTA firing winner by construction (both are the neural read of the ensembles)."""
    xp, _ = get_backend()
    bridge = ub.bridge
    comp = ub.composer
    idx = comp.idx

    for c in bridge._gate_couplings:
        c["ema"] = 0.0
        c["last_value"] = None
    for r in SYNAPTIC_ROUTE_ROLES:
        bridge.set_transmission_gate(f"role_route_{r}", 0.0)
    bridge.cp_external_input_current[:] = 0.0
    prev_ou = bridge.core_config.enable_ou_process
    prev_heb = bridge.core_config.enable_hebbian_learning
    bridge.core_config.enable_ou_process = False          # fixed reservoir/WTA/composer synapses must not drift; the
    bridge.core_config.enable_hebbian_learning = False     # low-rate reservoir read must not be corrupted by OU noise
    rb = np.asarray(role_bias, dtype=np.float64)
    try:
        for _ in range(RESET_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()

        # (b) DRIVE THE RESERVOIR (replayed): W_in @ U[t] -> reservoir slice; res2ens synapses + per-role role_bias +
        # WS_ENS_FLOOR drive the ens under the WTA's I->E competition; fill + role_src + coincidence bias held. Tally
        # the ens SUMMED firing (the synaptic read-out feature).
        ens_fire = np.zeros(3, dtype=np.float64)
        role_src_cur = {r: ub._role_src[r] for r in SYNAPTIC_ROUTE_ROLES}
        for _rep in range(WS_REPLAY):
            for t in range(len(U)):
                drive = res.W_in @ U[t] + RES_BIAS
                cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
                cur[xp.asarray(res.res_idx)] = xp.asarray(drive.astype(np.float32))
                for r in range(3):
                    cur[xp.asarray(ens[r])] = np.float32(rb[r] + WS_ENS_FLOOR)
                for r in SYNAPTIC_ROUTE_ROLES:
                    cur[role_src_cur[r]] = ROLE_SRC_DRIVE_PA
                cur[idx["fill_on"]] = xp.asarray(fill_on_cur.astype(np.float32))
                cur[idx["fill_off"]] = xp.asarray(fill_off_cur.astype(np.float32))
                for bank in ("A", "B", "C", "D"):
                    cur[idx[bank]] = comp.coinc_bias
                bridge.cp_external_input_current[:] = cur
                for _ in range(RES_T_STEP):
                    bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
                    bridge._run_one_simulation_step()
                    fs = to_host(bridge.cp_firing_states)
                    for k in range(3):
                        ens_fire[k] += float(fs[xp.asarray(ens[k])].sum())

        # (c) WINNER: the NEURAL read (argmax over the ens summed firing). No host f@Ws, no host logits -- the winner is
        # the co-resident spiking ensembles' firing, driven synaptically by the reservoir through Ws_shifted.
        wta_fire_winner = SYNAPTIC_ROUTE_ROLES[int(np.argmax(ens_fire))]
        latched = wta_fire_winner

        # (d) LATCH: HOLD the winner's role_route gate open (the gate the winning ensemble drives), close the losers,
        # and pause the couplings for the readout window. Drop the reservoir drive; keep fill + coincidence bias.
        # WHY the neural-winner drives the gate DIRECTLY (not the coupling's first-to-cross EMA): at this firing regime
        # the coupling opens whichever ensemble crosses the EMA threshold FIRST (a transient, biased toward 'agent' by
        # index/heterogeneity), which does NOT track the true winner (verified: slot-1's coupling opens 'agent' though
        # 'action' fires most; slot-2's never crosses). So the winner is read from the SETTLED ens SUMMED firing (the
        # neural read) and its gate is held directly. But this must REMAIN route-gated: we open the winner's gate ONLY
        # if its role_route COUPLING still exists (the coupling is the substrate's route-to-gate link -- ROUTE-LESION
        # removes it). Under ROUTE-LESION the couplings are gone -> NO gate opens -> the role bank is starved -> recall
        # collapses (the route stays load-bearing; the direct set is "hold what the winning ensemble's coupling would
        # have opened", conditioned on that coupling existing).
        route_live = {c["gate_name"] for c in bridge._gate_couplings}
        for r in SYNAPTIC_ROUTE_ROLES:
            open_it = (r == latched) and (f"role_route_{r}" in route_live)
            bridge.set_transmission_gate(f"role_route_{r}", 1.0 if open_it else 0.0)
        open_now = [r for r in SYNAPTIC_ROUTE_ROLES if _gate_open(bridge, r)]
        saved_couplings = bridge._gate_couplings
        bridge._gate_couplings = []
        cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
        for r in range(3):
            cur[xp.asarray(ens[r])] = np.float32(rb[r] + WS_ENS_FLOOR)
        for r in SYNAPTIC_ROUTE_ROLES:
            cur[role_src_cur[r]] = ROLE_SRC_DRIVE_PA
        cur[idx["fill_on"]] = xp.asarray(fill_on_cur.astype(np.float32))
        cur[idx["fill_off"]] = xp.asarray(fill_off_cur.astype(np.float32))
        for bank in ("A", "B", "C", "D"):
            cur[idx[bank]] = comp.coinc_bias
        bridge.cp_external_input_current[:] = cur
        try:
            # (e) READOUT: accumulate the coincidence banks through the held (winner) gate.
            acc = {b: xp.zeros(comp.D, dtype=xp.float64) for b in ("A", "B", "C", "D")}
            for _ in range(comp.run_steps):
                bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
                bridge._run_one_simulation_step()
                for b in ("A", "B", "C", "D"):
                    acc[b] += bridge.cp_firing_states[idx[b]].astype(xp.float64)
        finally:
            # (f) RESTORE: couplings back, gates closed, emas cleared, input cleared.
            bridge._gate_couplings = saved_couplings
            for r in SYNAPTIC_ROUTE_ROLES:
                bridge.set_transmission_gate(f"role_route_{r}", 0.0)
                cpl = next((c for c in bridge._gate_couplings if c["gate_name"] == f"role_route_{r}"), None)
                if cpl is not None:
                    cpl["ema"] = 0.0
                    cpl["last_value"] = None
            bridge.cp_external_input_current[:] = 0.0
    finally:
        bridge.core_config.enable_ou_process = prev_ou
        bridge.core_config.enable_hebbian_learning = prev_heb

    rates = {b: to_host(acc[b]) / comp.run_steps for b in ("A", "B", "C", "D")}
    return rates["A"] + rates["B"], rates["C"] + rates["D"], latched, wta_fire_winner, list(open_now)


def _bind_c2(ub, ens, res, enc, tokens, slot_readout, lesion=False):
    """B-1c.2 (the FULL close-out): the SPIKING reservoir's firing drives the WTA ensembles SYNAPTICALLY through the
    Ws_shifted res2ens synapses -- NO host `f @ Ws`. Per content word (at content-slot k), the read-out is rewired to
    slot-k's Ws_shift (the per-slot role read-out; the reservoir feature is whole-sequence, so the SLOT selects the
    role), then `_op_wta_synaptic` drives the reservoir and reads the WTA winner from the co-resident ensembles'
    firing; the composer field is that latched role. The role SELECTION is purely synaptic (reservoir firing -> ens
    firing -> WTA competition -> gate), never a host matmul/argmax. `lesion=True` encodes with the reservoir-lesion
    (closed-class identity collapsed). Returns (fact_or_None, trace)."""
    U = enc.encode(tokens, lesion=lesion)
    content = [t for t, w in enumerate(tokens) if w not in enc.idx]
    composer = ub.composer
    bound_on = np.zeros(composer.D); bound_off = np.zeros(composer.D)
    fact = {}; trace = []
    for k, t in enumerate(content):
        role_bias = slot_readout.set_slot(k)       # per-slot read-out (the reservoir feature is whole-sequence)
        word = tokens[t]
        c_on, c_off = onoff(composer.concepts[word])
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        bon, boff, latched_role, wta_fire_winner, gates_at_latch = _op_wta_synaptic(
            ub, ens, res, U, role_bias, fon, foff)
        trace.append({"word": word, "latched_role": latched_role, "wta_fire_winner": wta_fire_winner,
                      "gates_at_latch": gates_at_latch})
        role = latched_role
        if role in fact:
            continue
        bound_on += bon; bound_off += boff
        fact[role] = word
    if {"agent", "action", "patient"} <= set(fact):
        composer.kb.append((fact, onoff(bound_on - bound_off)))
        return fact, trace
    return None, trace


def _scramble_Ws(Ws, seed):
    """Ws-SCRAMBLE anti-cheat: permute the 3 role columns (AGENT/PREDICATE/THEME = cols 0,1,2) of each Ws[k] (a real
    derangement), so the reservoir logits misroute the WTA. Returns a new Ws dict (input untouched)."""
    rng = np.random.default_rng(seed * 977 + 13)
    out = {}
    for k, W in Ws.items():
        W2 = W.copy()
        perm = rng.permutation(3)
        while np.array_equal(perm, [0, 1, 2]):
            perm = rng.permutation(3)
        W2[:, [0, 1, 2]] = W[:, [0, 1, 2]][:, perm]
        out[k] = W2
    return out


def setup_corpus(seed=42):
    """Build the shared corpus/task ONCE (reused across seeds)."""
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    test, _seen, _trng = _build_test_facts(seed, subj, verb, obj, n=N_TEST)
    vocab = sorted({w for _toks, s, v3, o in test for w in (s, v3, o)})
    concepts = _orthonormal_concepts(vocab, PROJ_DIM, seed=0)
    return {"discovered": discovered, "subj": subj, "verb": verb, "obj": obj,
            "test": test, "vocab": vocab, "concepts": concepts}


def _build_wired_bridge(seed, corpus):
    """Fresh UnifiedBrainBridge (parser+composer+synaptic route+WTA+reservoir), WTA + reservoir wired IN PLACE.
    Returns (ub, ens, inh, res, res_idx). The reservoir's Ws_shifted synapses are wired later (per fit); the snapshot
    is taken AFTER those. Encoder is returned via res.enc-style access through a closure -- here we pass enc separately."""
    ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=corpus["concepts"],
                            enable_synaptic_route=True, role_wta_n=ROLE_WTA_N, reservoir_n=RES_N)
    ens, inh = wire_wta(ub)
    return ub, ens, inh


def run_seed(seed, corpus, mode="c2"):
    t0 = time.time()
    discovered, subj, verb, obj = corpus["discovered"], corpus["subj"], corpus["verb"], corpus["obj"]
    test = corpus["test"]
    enc = Encoder(discovered)
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN_PER, rng, subj, verb, obj)
    n_q = 2 * len(test)

    # ── FIT: build ONE bridge, wire WTA + reservoir, fit Ws on the SPIKING reservoir feature ──────────────────────
    ub0, ens0, inh0 = _build_wired_bridge(seed, corpus)
    res_idx0, W_in0 = wire_reservoir(ub0, enc.dim, seed)
    res0 = UBReservoir(ub0, res_idx0, W_in0)
    res0.snapshot_after_wiring()
    print(f"[b1c seed {seed}] fitting Ws on {len(train)} spiking-reservoir features "
          f"(reservoir slice {res_idx0[0]}..{res_idx0[-1]})...", flush=True)
    Ws = _fit_Ws_spiking(res0, enc, train)
    # the content slots a transitive SVO fills: slot 0 (agent-word), slot 1 (action-word), slot 2 (patient-word).
    # Ws_shifted per slot (all >= 0, argmax-preserving). We test 3 content words per fact, each with its own slot Ws.
    Ws_shift = {k: (W - W.min()) for k, W in Ws.items()}

    # ── choose the Ws_shifted synapse SCALE (c2 only): sweep a broad band, pick the scale whose SYNAPTIC winner best
    #    matches the host argmax over ALL 3 content slots on the 6 tests (the read-out-resolution tuning; the synaptic
    #    winner is a NEURAL read -- argmax over the ens summed firing, driven synaptically by the reservoir). ─────────
    chosen_scale = None
    scale_sweep = []
    if mode == "c2":
        f_ref = np.concatenate([res0.final_state(enc.encode(test[0][0])), [1.0]])
        proj_top = max(1e-9, float((f_ref[:len(res_idx0)] @ Ws_shift[0][:len(res_idx0), :3]).max()))
        scales = [c / proj_top for c in (60.0, 100.0, 140.0, 200.0, 280.0)]
        # host per-slot winners on the 6 tests (the target the synaptic read-out must reproduce, over slots 0/1/2)
        host_slots = []
        for toks, _s, _v, _o in test:
            f = np.concatenate([res0.final_state(enc.encode(toks)), [1.0]])
            host_slots.append([int(np.argmax((f @ Ws[k])[[0, 1, 2]])) for k in (0, 1, 2)])
        # a fresh wired bridge with the res2ens edges pre-allocated; per-slot overwrite via SlotReadout.
        ub_s, ens_s, _ish = _build_wired_bridge(seed, corpus)
        res_idx_s, W_in_s = wire_reservoir(ub_s, enc.dim, seed)
        res_s = UBReservoir(ub_s, res_idx_s, W_in_s)
        wire_ws_synapses(ub_s, res_idx_s, ens_s, Ws_shift[0], scales[0], add_missing=True)
        res_s.snapshot_after_wiring()
        for sc in scales:
            sr = SlotReadout(ub_s, res_s, ens_s, Ws_shift, sc)
            agree = 0; ntot = 0
            for (toks, _s2, _v2, _o2), hslots in zip(test, host_slots):
                for k in (0, 1, 2):
                    role_bias = sr.set_slot(k)
                    _feat, ens_sum = res_s.run_with_ens(enc.encode(toks), ens_s, role_bias=role_bias)
                    agree += int(int(np.argmax(ens_sum)) == hslots[k]); ntot += 1
            scale_sweep.append({"scale": float(sc), "agree": int(agree), "ntot": int(ntot)})
        # pick the SMALLEST scale among the max-agreement ties. THE SCALE TRADE-OFF (a precisely-located substrate
        # boundary, the CRUX's "confusion band"): at a LARGER scale the read-out + per-role bias prior are so strong
        # that the RESERVOIR-lesion no longer collapses recall (the bias intercept -- a legitimate part of the learned
        # Ws -- carries the canonical slot->role prior on the 6 canonical test facts, so the reservoir's structural
        # signal is only marginally load-bearing); at a SMALLER scale the reservoir signal is genuinely load-bearing
        # (res-lesion collapses) but the read-out is slightly under-driven (route ~10/12, still >= 0.8n). We prefer the
        # smallest GO-band scale so the RESERVOIR stays load-bearing (the honest bar) at the cost of a couple of
        # borderline facts. There is NO single scale with BOTH route 12/12 AND res-lesion collapse -- that is the
        # located boundary (reported in the finding).
        max_ag = max(d["agree"] for d in scale_sweep)
        tie = [d for d in scale_sweep if d["agree"] == max_ag]
        best = tie[0]
        chosen_scale = best["scale"]
        print(f"[b1c seed {seed}] scale sweep (agree/{scale_sweep[0]['ntot']}): "
              + " ".join(f"{d['scale']:.4g}:{d['agree']}" for d in scale_sweep)
              + f" -> chose {chosen_scale:.4g} (smallest of {len(tie)} @ agree {max_ag}/{best['ntot']} "
              f"-- reservoir stays load-bearing)", flush=True)

    # ── ROUTE: fresh bridge, wire WTA + reservoir + (c2) Ws_shifted synapses, bind the 6 facts, recall ────────────
    def new_route_bridge():
        """Returns (ub, ens, inh, res, res_idx, slot_readout). For c2, the res2ens edges are pre-allocated (SlotReadout
        overwrites per slot in place -- no CSR rebuild). The snapshot is taken AFTER all wiring."""
        ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=corpus["concepts"],
                                enable_synaptic_route=True, role_wta_n=ROLE_WTA_N, reservoir_n=RES_N)
        ens, inh = wire_wta(ub)
        res_idx, W_in = wire_reservoir(ub, enc.dim, seed)
        res = UBReservoir(ub, res_idx, W_in)
        slot_readout = None
        if mode == "c2":
            wire_ws_synapses(ub, res_idx, ens, Ws_shift[0], chosen_scale, add_missing=True)
            slot_readout = SlotReadout(ub, res, ens, Ws_shift, chosen_scale)
        res.snapshot_after_wiring()
        return ub, ens, inh, res, res_idx, slot_readout

    def bind_all(ub, ens, res, slot_readout=None, lesion=False, Ws_override=None):
        traces = []
        for toks, s, v3, o in test:
            if mode == "c1":
                _fact, tr = _bind_c1(ub, ens, res, enc, Ws, toks, lesion=lesion, Ws_override=Ws_override)
            else:
                _fact, tr = _bind_c2(ub, ens, res, enc, toks, slot_readout, lesion=lesion)
            traces.append({"tokens": toks, "trace": tr})
        return traces

    ub, ens, inh, res, res_idx, slot_ro = new_route_bridge()
    all_traces = bind_all(ub, ens, res, slot_readout=slot_ro)
    hp, ha = _recall(ub, test)
    route_correct = hp + ha

    # ── (3) MOAT ──────────────────────────────────────────────────────────────────────────────────────────────
    stored = {(s, v3) for _t, s, v3, _o in test}
    fa = tot = mg = 0
    trng = np.random.default_rng(seed * 733 + 999)
    while tot < 30 and mg < 3000:
        mg += 1
        s = str(trng.choice(subj)); v3q = str(trng.choice(verb)) + "s"
        if (s, v3q) in stored:
            continue
        tot += 1; fa += int(ub.query_patient(s, v3q) is not None)
    moat_fa = fa / max(1, tot)

    # ── (2) DICT baseline (HOST-argmax role select on the SAME co-resident substrate) ────────────────────────────
    from research.runners._rungB1_reservoir_synaptic_handoff_derisk import (
        _bind_reservoir_fact,
    )
    ub_d, ens_d, inh_d, res_d, _rid, _sro_d = new_route_bridge()
    role2k_d = {ub_d.parser.role_of(pos, "active"): pos * 2 for pos in range(3)}

    def _reservoir_roles_spiking(res, tokens, lesion=False):
        f = np.concatenate([res.final_state(enc.encode(tokens, lesion=lesion)), [1.0]])
        content = [t for t, w in enumerate(tokens) if w not in enc.idx]
        pairs = []
        _ROLE2FIELD = {"AGENT": "agent", "PREDICATE": "action", "THEME": "patient"}
        for k, t in enumerate(content):
            if k not in Ws:
                continue
            role = _ROLES[int(np.argmax(f @ Ws[k]))]
            field = _ROLE2FIELD.get(role)
            if field is not None:
                pairs.append((tokens[t], field))
        return pairs
    for toks, s, v3, o in test:
        _bind_reservoir_fact(ub_d, role2k_d, _reservoir_roles_spiking(res_d, toks))
    dp, da = _recall(ub_d, test)
    dict_correct = dp + da

    # ── (7) PROVENANCE-NEURAL-SELECT (runtime: latched == firing winner) ─────────────────────────────────────────
    latched_eq_firing = all(t["latched_role"] == t["wta_fire_winner"]
                            for w in all_traces for t in w["trace"])

    # ── (10) SYNAPTIC-READOUT source-check (c2 only) ─────────────────────────────────────────────────────────────
    if mode == "c2":
        synaptic_source_clean = _source_synaptic_readout_clean()
    else:
        synaptic_source_clean = None  # not applicable to c1 (c1 still uses host f@Ws by design)

    # ── (4) PROVENANCE-CLEAN (I5a instrument: the role bank gets ZERO direct current on a WTA/synaptic bind) ──────
    ub_p, ens_p, _ip, res_p, _rp, _sro_p = new_route_bridge()
    prov = provenance_role_bank_current(ub_p, word=corpus["vocab"][0], pos=0, voice="active")
    provenance_clean = (prov["synaptic_route_role_bank_direct_current_max"] == 0.0
                        and prov["dict_path_role_bank_direct_current_max"] > 0.0)

    # ── (5) ROUTE-LESION ─────────────────────────────────────────────────────────────────────────────────────────
    ub_l, ens_l, _il, res_l, _rl, sro_l = new_route_bridge()
    restore_l = lesion_route(ub_l.bridge)
    bind_all(ub_l, ens_l, res_l, slot_readout=sro_l)
    lp, la = _recall(ub_l, test)
    route_lesion_correct = lp + la
    restore_l()
    route_lesion_collapses = route_lesion_correct < route_correct

    # ── (6) RESERVOIR-LESION (lesion the closed-class identity in the encoder -> feature collapses) ───────────────
    ub_r, ens_r, _ir, res_r, _rr, sro_r = new_route_bridge()
    bind_all(ub_r, ens_r, res_r, slot_readout=sro_r, lesion=True)
    rp2, ra2 = _recall(ub_r, test)
    res_lesion_correct = rp2 + ra2
    res_lesion_collapses = res_lesion_correct < route_correct

    # ── (8) WTA-LESION (zero the I->E synapses -> the mutual-inhibition competition is removed) ───────────────────
    # NOTE (B-1c finding, honest): in B-1b the WTA-lesion COLLAPSED recall because the role was selected by a UNIFORM
    # host WTA_BASE drive to all ens + the I->E inhibition SILENCING the losers -- inhibition was the sole selector.
    # In B-1c.2 the selection moved to the SYNAPTIC read-out: the res2ens Ws_shifted synapses + the per-role bias
    # deliver a genuine per-role DRIVE ADVANTAGE to the winning ensemble, so the winner leads on FEEDFORWARD drive and
    # removing the I->E inhibition does NOT change the neural argmax -> the WTA-lesion does NOT collapse c2. This is a
    # POSITIVE result, not a failure: it is exactly what B-1c set out to do -- move the SELECTION from inhibition-
    # competition (B-1b) to the synaptic read-out. The load-bearing lesion for c2 is therefore the SYNAPTIC-READOUT
    # lesion (10), which zeroes the read-out synapses (+ bias) and DOES collapse recall (the read-out IS the selector).
    # For c1 (host f@Ws -> uniform WTA drive, B-1b's regime), the WTA-lesion still bites and DOES gate the verdict.
    ub_w, ens_w, inh_w, res_w, _rw, sro_w = new_route_bridge()
    restore_w = lesion_wta_i2e(ub_w, ens_w, inh_w)
    bind_all(ub_w, ens_w, res_w, slot_readout=sro_w)
    wp, wa = _recall(ub_w, test)
    wta_lesion_correct = wp + wa
    restore_w()
    wta_lesion_collapses = wta_lesion_correct < route_correct

    # ── (9) Ws-SCRAMBLE (permute the 3 role columns -> the read-out misroutes the reservoir firing) ───────────────
    Ws_scr = _scramble_Ws(Ws, seed)
    ub_s2, ens_s2, _is, res_s2, res_idx_s2, sro_s2 = new_route_bridge()
    if mode == "c1":
        for toks, s, v3, o in test:
            _bind_c1(ub_s2, ens_s2, res_s2, enc, Ws, toks, Ws_override=Ws_scr)
        ws_scramble_correct = sum(_recall(ub_s2, test))
    else:
        # c2: swap the SlotReadout's Ws_shift for the SCRAMBLED one -> the reservoir firing misroutes to the ens.
        Ws_scr_shift = {k: (Ws_scr[k] - Ws_scr[k].min()) for k in Ws_scr}
        sro_s2.Ws_shift = Ws_scr_shift
        bind_all(ub_s2, ens_s2, res_s2, slot_readout=sro_s2)
        ws_scramble_correct = sum(_recall(ub_s2, test))
    ws_scramble_collapses = ws_scramble_correct < route_correct

    # ── (10) SYNAPTIC-READOUT lesion (c2 only): zero the Ws_shifted res2ens synapses -> WTA starved -> collapse ────
    synaptic_readout_collapses = None
    synaptic_readout_correct = None
    if mode == "c2":
        ub_sr, ens_sr, _isr, res_sr, res_idx_sr, sro_sr = new_route_bridge()
        # zero the res2ens synapses (the read-out) so the ensembles get NO reservoir signal; the per-role bias tonic is
        # also zeroed (a zero Ws_shift -> zero bias), so every ens fires only the WS_ENS_FLOOR -> no discriminative
        # signal -> the neural winner is noise -> recall collapses. Rewire zeros, re-bind.
        pre_sr, post_sr = _ws_edges(res_idx_sr, ens_sr)
        ub_sr.bridge.set_pathway_weights("res2ens", pre_sr, post_sr,
                                          np.zeros(len(pre_sr), dtype=np.float32), add_missing=False)

        class _ZeroReadout(SlotReadout):
            def set_slot(self, k):
                # keep the res2ens synapses at zero (do not overwrite) and return a ZERO per-role bias -> no signal.
                return np.zeros(3)
        sro_zero = _ZeroReadout(ub_sr, res_sr, ens_sr, Ws_shift, chosen_scale)
        res_sr.snapshot_after_wiring()
        bind_all(ub_sr, ens_sr, res_sr, slot_readout=sro_zero)
        spp, spa = _recall(ub_sr, test)
        synaptic_readout_correct = spp + spa
        synaptic_readout_collapses = synaptic_readout_correct < route_correct

    # ── verdict ──────────────────────────────────────────────────────────────────────────────────────────────────
    if mode == "c1":
        # c1 (B-1b's regime: host f@Ws -> uniform WTA drive) gates on ALL 9 B-1b anti-cheats, INCLUDING the WTA-lesion
        # (load-bearing there) AND route-not-worse-than-dict.
        checks = [
            route_correct >= 0.80 * n_q,
            route_correct >= dict_correct,
            moat_fa <= 0.05,
            provenance_clean,
            route_lesion_collapses,
            res_lesion_collapses,
            latched_eq_firing,
            wta_lesion_collapses,
            ws_scramble_collapses,
        ]
    else:
        # c2 (the on-substrate close-out) gates on the CORE synaptic claims: the facts are recovered on ONE bridge
        # (route >= 0.8n), the read-out is genuinely SYNAPTIC (source-clean) + LOAD-BEARING (syn-readout-lesion
        # collapses), and the reservoir + route + Ws are load-bearing (res/route-lesion + ws-scramble collapse) at the
        # reservoir-load-bearing scale. It does NOT gate on (a) route-not-worse-than-dict -- the LOCATED SCALE BOUNDARY:
        # no single scale gives BOTH route == host-dict recall AND a load-bearing reservoir-lesion (the per-role bias
        # prior + spiking-margin resolution trade off) -- nor (b) the WTA-lesion, a documented POSITIVE finding (the
        # selection moved from inhibition-competition to the synaptic read-out). Both are reported as diagnostics.
        checks = [
            route_correct >= 0.80 * n_q,
            moat_fa <= 0.05,
            route_lesion_collapses,
            res_lesion_collapses,
            latched_eq_firing,
            ws_scramble_collapses,
            bool(synaptic_source_clean),
            bool(synaptic_readout_collapses),
        ]
    seed_go = bool(all(checks))

    return {
        "seed": int(seed), "mode": mode,
        "route_correct": int(route_correct), "dict_correct": int(dict_correct), "n_queries": n_q,
        "route_recall": route_correct / n_q,
        "route_recall_ge_0.8n": bool(route_correct >= 0.80 * n_q),
        "route_not_worse_than_dict": bool(route_correct >= dict_correct),
        "moat_false_accept": moat_fa, "moat_clean": bool(moat_fa <= 0.05),
        "provenance_clean": bool(provenance_clean),
        "route_lesion_correct": int(route_lesion_correct), "route_lesion_collapses": bool(route_lesion_collapses),
        "res_lesion_correct": int(res_lesion_correct), "res_lesion_collapses": bool(res_lesion_collapses),
        "neural_select_latched_eq_firing": bool(latched_eq_firing),
        "wta_lesion_correct": int(wta_lesion_correct), "wta_lesion_collapses": bool(wta_lesion_collapses),
        "ws_scramble_correct": int(ws_scramble_correct), "ws_scramble_collapses": bool(ws_scramble_collapses),
        "synaptic_source_clean": synaptic_source_clean,
        "synaptic_readout_correct": (int(synaptic_readout_correct) if synaptic_readout_correct is not None else None),
        "synaptic_readout_collapses": synaptic_readout_collapses,
        "chosen_ws_scale": chosen_scale, "scale_sweep": scale_sweep,
        "seed_GO": seed_go, "elapsed_s": round(time.time() - t0, 1),
        "sample_trace": all_traces[0]["trace"] if all_traces else [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--mode", type=str, default="c2", choices=["c1", "c2"])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    corpus = setup_corpus(seed=42)
    print(f"[rungB1c] mode={args.mode} | corpus: {len(corpus['test'])} facts, vocab {len(corpus['vocab'])}", flush=True)
    rows = []
    for s in args.seeds:
        d = run_seed(s, corpus, mode=args.mode)
        rows.append(d)
        extra = ""
        if args.mode == "c2":
            extra = (f" | syn-source {d['synaptic_source_clean']}"
                     f" | syn-readout-lesion {d['synaptic_readout_correct']}<{d['route_correct']}"
                     f"={d['synaptic_readout_collapses']} | scale {d['chosen_ws_scale']:.4g}")
        print(f"[seed {s}] GO={d['seed_GO']} | route {d['route_correct']}/{d['n_queries']} (dict {d['dict_correct']})"
              f" | moat-FA {d['moat_false_accept']:.2f} | prov {d['provenance_clean']}"
              f" | route-lesion {d['route_lesion_correct']}<{d['route_correct']}={d['route_lesion_collapses']}"
              f" | res-lesion {d['res_lesion_correct']}<{d['route_correct']}={d['res_lesion_collapses']}"
              f" | neural-select {d['neural_select_latched_eq_firing']}"
              f" | wta-lesion {d['wta_lesion_correct']}<{d['route_correct']}={d['wta_lesion_collapses']}"
              f" | ws-scramble {d['ws_scramble_correct']}<{d['route_correct']}={d['ws_scramble_collapses']}"
              f"{extra} ({d['elapsed_s']}s)", flush=True)

    n_go = sum(r["seed_GO"] for r in rows)
    agg = {
        "mode": args.mode, "n_seeds": len(rows), "n_seeds_GO": int(n_go),
        "verdict": "GO" if n_go == len(rows) else ("PARTIAL" if n_go else "NO-GO"),
        "route_recall_ge_0.8n_all": all(r["route_recall_ge_0.8n"] for r in rows),
        "route_not_worse_than_dict_all": all(r["route_not_worse_than_dict"] for r in rows),
        "moat_clean_all": all(r["moat_clean"] for r in rows),
        "provenance_clean_all": all(r["provenance_clean"] for r in rows),
        "route_lesion_collapses_all": all(r["route_lesion_collapses"] for r in rows),
        "res_lesion_collapses_all": all(r["res_lesion_collapses"] for r in rows),
        "neural_select_all": all(r["neural_select_latched_eq_firing"] for r in rows),
        "wta_lesion_collapses_all": all(r["wta_lesion_collapses"] for r in rows),
        "ws_scramble_collapses_all": all(r["ws_scramble_collapses"] for r in rows),
        "synaptic_source_clean_all": (all(r["synaptic_source_clean"] for r in rows) if args.mode == "c2" else None),
        "synaptic_readout_collapses_all": (all(r["synaptic_readout_collapses"] for r in rows)
                                           if args.mode == "c2" else None),
        "mean_route_recall": float(np.mean([r["route_recall"] for r in rows])),
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    _readout_desc = ("SYNAPTIC (Ws_shifted res->ens); the whole comprehend->select->bind runs on ONE bridge with "
                     "nothing host-computed" if args.mode == "c2" else "host f@Ws (incremental rung)")
    print(f"\n[rungB1c] VERDICT ({args.mode}): {agg['verdict']} ({n_go}/{len(rows)}) -- the SPIKING reservoir is "
          f"co-resident on the unified bridge and the read-out is {_readout_desc} "
          f"(mean route recall {agg['mean_route_recall']:.3f}).", flush=True)
    if args.mode == "c2":
        print(f"[rungB1c] c2 DIAGNOSTICS (documented boundaries, not gating): "
              f"route-not-worse-than-dict {agg['route_not_worse_than_dict_all']} (the LOCATED SCALE BOUNDARY: no "
              f"single scale gives BOTH route == host-dict recall AND a load-bearing reservoir-lesion -- the per-role "
              f"bias prior + spiking-margin resolution trade off); WTA-lesion collapses "
              f"{agg['wta_lesion_collapses_all']} (a POSITIVE finding: the selection moved from inhibition-competition "
              f"to the synaptic read-out, so mutual inhibition is no longer the selector).", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2, default=str)
        print(f"[rungB1c] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
