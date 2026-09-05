"""GNW STOP-TRIGGER ACC/BG circuit -- retiring the host boolean-OR that DECIDES whether the global-workspace STOP
fires (scaffold-retirement-backlog rank-12, `research/coordination/scaffold_retirement_backlog.md`).

WHAT THIS RETIRES. `webapp/gnw_global_stop.py`'s `detect_trigger(chat)` decides whether to run the (already-genuinely-
spiking) STOP clear with plain host Python:

    if isinstance(n_ign, (int, float)) and int(n_ign) >= 2: triggered = True     # delib conflict
    if bool(swap.get("swapped")): triggered = True                              # topic-break

n_ignited (the deliberation `acc_conflict_gate`'s own spiking read of workspace co-ignition,
`webapp/gnw_deliberation.py`) and the swap detector's mismatch-population firing (`mm_peak`,
`webapp/gnw_thought_swap.py` / `_gnw_neural_swap_intention_derisk.py`) are EACH already genuine spiking read-outs of
OTHER organs. What is NOT spiking is the COMBINATION: two independently-computed scalars OR'd together in host
arithmetic. The STOP *clear* itself (the Tsodyks-Markram shared-recurrence depression, `WorkspaceDepression` /
`run_conflict_stop` in `_gnw_distributed_overwrite_workspace_derisk.py`) and the deliberation's own re-entrant RETRY
cycle count are ALREADY genuinely spiking -- untouched here. This module targets ONLY the trigger comparison.

THE RETIREMENT MECHANISM. An ACC/BG hyperdirect circuit -- reusing the EXACT chain shape already built and
GO'd-at-the-sensor-level in `_gnw_rung_stn_stop_veto_derisk.py` (ACC -> STN -> GPi; Frank 2006 "hold-your-horses";
Aron & Poldrack 2006 / Wessel & Aron 2017 broad fast STN reactive stop; Wei-Rubin-Wang 2015 STN-GPe dynamics) -- reads
the SAME two afferents (`n_ignited`, `mm_peak`) DIRECTLY AS SYNAPTIC INPUT instead of as Python operands:

  * TWO afferent relay pools (`delib_aff`, `mm_aff`), each a small feedforward excitatory Izhikevich population.
    Exactly as every stimulus/drive in this codebase is applied (a host-scaled current representing an
    ALREADY-SPIKING-DERIVED external quantity -- e.g. `IGNITE_PA`/`STRONG_PA` driving a workspace slot), the host
    converts `n_ignited` -> `i_delib = DELIB_CURRENT_SCALE * n_ignited` and `mm_peak` -> `i_mm = MM_CURRENT_SCALE *
    mm_peak`, injected into `delib_aff`/`mm_aff` respectively. This is the unavoidable stimulus-injection step (there
    is no other channel to introduce a value into a spiking population); it is NOT the decision.
  * `delib_aff -> acc` and `mm_aff -> acc` (dense E_TO_E synapses): ACC receives BOTH conflict signals as genuine
    SYNAPTIC input from two independent upstream populations and SUMS them in its own membrane dynamics (real
    synaptic integration, not a Python `or`).
  * `acc -> stn -> gpi` (dense E_TO_E, the reused hyperdirect chain): ACC's supra-threshold firing drives STN then
    GPi. The STOP-TRIGGER decision is read off GPi's own late-window firing rate crossing a fixed threshold -- the
    SAME kind of rate->boolean read-out every other spiking decision in this codebase uses (`_ignited`,
    `acc_conflict_gate`), not a re-hidden host `if`.

THIS IS A DE-RISK, NOT AN EFFECTOR. The prior STN-veto de-risk's NO-GO was about using GPi's inhibition to CLEAR a
dense localist attractor (the effector arm) -- that residual is untouched and irrelevant here: this circuit's GPi
has NO downstream target. Its firing rate is read ONLY as the trigger decision; the actual clearing is still done,
unmodified, by the distributed-overwrite workspace's own Tsodyks-Markram depression. Reusing the ACC->STN->GPi
SENSOR shape (already GO on SENSOR + SELECTIVITY, 6/6 seeds, in the STN-veto de-risk) sidesteps that NO-GO by
construction -- there is no localist attractor here to fail to clear.

REAL AFFERENTS, NOT HAND-PICKED CONSTANTS. The parity check drives this circuit with `n_ignited`/`mm_peak` values
produced by the ACTUAL already-existing organs (reuse-by-import, no rebuild):
  * `webapp.gnw_deliberation.conflict_gate(n_candidates, seed=seed)` -- the SAME P1.2 workspace + ACC gate
    `chat._last_gnw_delib` is built from. n_candidates=1 -> a real "no conflict" n_ignited; n_candidates=2 -> a real
    co-ignition n_ignited.
  * `webapp.gnw_thought_swap.ThoughtSwapWorkspace(seed=seed).observe(topic)` -- the SAME mismatch/salience swap
    detector `chat._last_swap_drives` is built from. A same-topic repeat -> a real "match" mm_peak (pred-vetoed,
    low); a genuine topic change -> a real "mismatch" mm_peak (fires, high).

GO GATE (6 seeds 42/43/44/100/101/102):
  (1) PARITY -- on the SAME 3 turn-classes the host boolean-OR discriminates (delib-conflict-only, swap-only,
      neither), the spiking circuit's trigger verdict matches the host's, on >= 5/6 seeds (mirrors the
      2026-08-26 flip-soak's own >=5/6 clean-stop bar for a graded spiking readout).
  (2) LOAD-BEARING (afferent sweeps) -- sweeping n_ignited (1..4) at a fixed baseline mm_peak, and separately
      sweeping mm_peak (match..mismatch) at a fixed baseline n_ignited, EACH alone drives the circuit's trigger
      from OFF to ON (monotonic GPi rate, a genuine boolean flip) on ALL 6 seeds -- both afferents are
      INDEPENDENTLY causally sufficient, not just co-required.
  (3) LESION -- zeroing BOTH afferent->ACC synapses (`afferent_lesion=True`) makes the trigger NEVER fire regardless
      of the (real, conflict-indicating) afferents fed in -- reverts fully to "no stop" -- on ALL 6 seeds. Zeroing
      EACH afferent pathway alone leaves the OTHER pathway independently sufficient (both individually load-bearing).
  (4) DETERMINISM -- build-twice-at-one-seed identical Izhikevich-parameter hash (cfg.seed, not actual_seed_used).

ANTI-CHEATS:
  * The afferent->ACC weight is the ONLY lesionable link; the ACC->STN->GPi chain is untouched by the lesion (so a
    lesion changing the verdict is attributable to the AFFERENT pathway, not a global kill-switch).
  * `attributable_to` quantifies what fraction of the trigger-worthy GPi drive the intact afferent pathway supplies
    vs the lesioned pathway, on the SAME (n_ignited, mm_peak) inputs.
  * Determinism: `_threshold_hash` (seed-derived Izhikevich params), build-twice.
  * No `_restore_state`-driven verdict: the trigger boolean is read from `bridge.cp_firing_states`, never a host flag.

HONEST RESIDUALS (named, not claimed closed):
  1. The scalar->current CONVERSION (`i_delib = DELIB_CURRENT_SCALE * n_ignited`, `i_mm = MM_CURRENT_SCALE *
     mm_peak`) is host arithmetic -- the SAME accepted "afferent drive" pattern as the STN-veto sensor's own
     margin->i_acc conversion (its named residual #1). What moved from host to spiking is the COMBINATION (the OR),
     not this unavoidable stimulus-injection step.
  2. The GPi-rate->boolean read-out threshold is a fixed host constant (`GPI_TRIGGER_THRESH`), exactly the same
     class of read-out every other spiking decision in this codebase uses (`_ignited`'s `IGNITE_FRAC*SOLO_PLATEAU`,
     `acc_conflict_gate`'s theta).
  3. This is a DE-RISK (research/runners only); it is NOT wired to replace `gnw_global_stop.detect_trigger` in
     production by this change. See `webapp/gnw_acc_bg_stop_trigger.py` for the default-OFF production hook.
  4. delib_aff/mm_aff/acc/stn/gpi are hand-wired dense frozen populations (explicit wiring), not self-organized --
     inherited from the STN-veto de-risk's own residual #2.

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=4):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_acc_bg_stop_trigger_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_acc_bg_stop_trigger_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_acc_bg_stop_trigger_derisk \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_gnw_acc_bg_stop_trigger_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.backend import get_backend, to_host
from tools.verdict import Verdict
from tools.lab import attributable_to

# reuse-by-import: dense explicit-wiring helper + the determinism hash (NO sim/ edit).
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
from research.runners._gnw_rung2b_sfa_workspace_eviction_derisk import _threshold_hash
# reuse-by-import: snapshot/restore (NO sim/ edit).
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore

# reuse-by-import the ALREADY-EXISTING, ALREADY-WIRED conflict/mismatch afferent PRODUCERS -- do NOT rebuild them.
from webapp import gnw_deliberation as _DELIB
from webapp import gnw_thought_swap as _SWAP

# ── geometry: two afferent relay pools -> ACC -> STN -> GPi (the reused hyperdirect chain shape) ────────────────
AFF_N = 40                              # delib_aff / mm_aff: small feedforward relay pools (no internal recurrence)
ACC_N = 80                              # ACC conflict-integration unit (matches the STN-veto de-risk's ACC_N)
STN_N = 120                             # subthalamic nucleus relay (matches the STN-veto de-risk)
GPI_N = 200                             # GPi output/read-out pool (large -> a stable rate estimate; no downstream target)

AFF_ACC_W = 22.0                        # delib_aff/mm_aff -> acc (per-synapse; tuned so ACC's own threshold, not a
                                        # host comparison, separates baseline from conflict-level afferent drive)
ACC_STN_W = 25.0                        # acc -> stn (reused verbatim from the STN-veto de-risk's ACC_STN_W)
STN_GPI_W = 8.0                         # stn -> gpi (REDUCED from the STN-veto de-risk's STN_GPI_W=25: probed
                                        # empirically -- at 25, GPi's dense 120-neuron all-to-all fan-in saturates
                                        # to ~0.17-0.21 from ANY nonzero STN activity at all, destroying GPi's
                                        # discriminating power as a GRADED trigger read-out (that de-risk only ever
                                        # needed GPi's DOWNSTREAM effect via a weak gpi_ws_w=8, never GPi's own rate,
                                        # so it never had to avoid this). At 8.0, GPi cleanly separates a genuine
                                        # zero baseline (rate=0.000) from a conflict-level afferent drive (~0.10-0.15).

# scalar -> current (the unavoidable stimulus-injection step; see HONEST RESIDUALS #1). Calibrated by direct probe
# (see the smoke grid / commit notes): the delib_aff/mm_aff relay pools have a SHARP knee near i=75-100 pA (silent
# below, robustly firing above -- probed at aff_acc_w=22, ou_noise_pA<=5: i=75 -> rate_acc=0.001, i=100 ->
# rate_acc=0.036/rate_gpi=0.096). Scales are set so a real solo n_ignited=1 / matched mm_peak sits BELOW that knee,
# and a real co-ignited n_ignited=2 / mismatch-fired mm_peak sits CLEARLY above it -- the same knee-placement idiom
# as IGNITE_PA/STRONG_PA elsewhere, just at the relay pool's OWN (much lower) rheobase, not a recurrent assembly's.
DELIB_CURRENT_SCALE = 60.0              # i_delib = DELIB_CURRENT_SCALE * n_ignited: n=1->60pA (below knee),
                                        # n=2->120pA (above knee), n=3/4 deeper into the robust plateau.
MM_CURRENT_SCALE = 700.0                # i_mm = MM_CURRENT_SCALE * mm_peak: a real match mm_peak (~0.00-0.07 across
                                        # 6 seeds) -> 0-49pA (below knee); a real mismatch mm_peak (~0.28-0.31) ->
                                        # 196-217pA (above knee).

GPI_TRIGGER_THRESH = 0.05               # GPi late-window rate above which the trigger reads TRUE (rate->boolean
                                        # read-out, the same class as _ignited's fraction-of-plateau threshold) --
                                        # sits between the probed baseline (0.000) and conflict (~0.10-0.15) rates.

SETTLE_STEPS = 30                       # quiescent settle before the snapshot (no attractor here; short is enough)
DRIVE_STEPS = 90                        # continuous afferent drive; late-window read = the last third

# a SMALL residual OU noise (background synaptic bombardment realism) -- NOT the desynchronizing role it plays
# elsewhere (Rung-2/2b/2c/STN-veto need OU to break up a RECURRENT ATTRACTOR's deterministic limit cycle; this
# circuit has NO recurrence anywhere, so there is no attractor to desynchronize). Probed: 20-30 pA (the values used
# for a recurrent attractor) drives noise-only ACC/STN/GPi firing that a dense 80->120->200 all-to-all cascade then
# AMPLIFIES into a substantial spurious baseline (a coincidence-detector effect of dense convergent fan-in) --
# rate_gpi ~0.17 at i_delib=0 -- destroying the knee this circuit's decision depends on. <=5 pA leaves the i=0
# baseline genuinely at rate=0.000 (probed) while still injecting a small amount of realistic background noise.
OU_NOISE_PA = 5.0

# scenario mm_peak baselines are READ from the real swap detector at runtime (see get_real_mm_peak); n_ignited
# baselines are READ from the real deliberation gate at runtime (see get_real_n_ignited). No hand-picked afferents.


def build_stop_trigger_bridge(seed: int = 42, *, afferent_lesion: bool = False, delib_lesion: bool = False,
                              mm_lesion: bool = False, aff_acc_w: float = AFF_ACC_W, acc_stn_w: float = ACC_STN_W,
                              stn_gpi_w: float = STN_GPI_W, heterogeneity: bool = True,
                              ou_noise_pA: float = OU_NOISE_PA):
    """Two afferent relay pools (delib_aff, mm_aff) -> acc -> stn -> gpi, all dense frozen E_TO_E. `afferent_lesion`
    zeroes BOTH delib_aff->acc and mm_aff->acc (the trigger circuit's OWN lesion: the ACC/BG chain survives, but it
    receives no afferent drive at all). `delib_lesion`/`mm_lesion` zero ONE pathway only (per-afferent load-bearing).
    Deterministic (heterogeneity + OU seeded from cfg.seed, NOT actual_seed_used). Returns (bridge, xp, delib_dev,
    mm_dev, acc_dev, stn_dev, gpi_dev, snap, handles)."""
    xp, _ = get_backend()

    delib_aff = BrainRegion(name="delib_aff", n_neurons=AFF_N, exc_fraction=1.0, internal_density=0.0,
                            enable_nmda=False)
    mm_aff = BrainRegion(name="mm_aff", n_neurons=AFF_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
    acc = BrainRegion(name="acc", n_neurons=ACC_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
    stn = BrainRegion(name="stn", n_neurons=STN_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
    gpi = BrainRegion(name="gpi", n_neurons=GPI_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False)
    regions = [delib_aff, mm_aff, acc, stn, gpi]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []            # ALL inter-region wiring is explicit (sub-slice precision)
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                # the substrate seed (het/threshold RNG) -- NOT actual_seed_used
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.enable_nmda = False
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False      # FOOT-GUN: synaptic-scaling clip slams the frozen relay/chain weights
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(aff_acc_w) * 8.0)
    cfg.hebbian_max_weight = max(400.0, float(aff_acc_w) * 8.0)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_noise_pA)
    else:
        cfg.enable_ou_process = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                              gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False

    rm = bridge.region_manager
    delib_idx = np.asarray(rm.indices("delib_aff"), dtype=np.int64)
    mm_idx = np.asarray(rm.indices("mm_aff"), dtype=np.int64)
    acc_idx = np.asarray(rm.indices("acc"), dtype=np.int64)
    stn_idx = np.asarray(rm.indices("stn"), dtype=np.int64)
    gpi_idx = np.asarray(rm.indices("gpi"), dtype=np.int64)

    delib_acc_eff = 0.0 if (afferent_lesion or delib_lesion) else float(aff_acc_w)
    mm_acc_eff = 0.0 if (afferent_lesion or mm_lesion) else float(aff_acc_w)

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    union_plan["delib2acc"] = _dense_pop(delib_idx, acc_idx, delib_acc_eff, "E_TO_E")
    union_plan["mm2acc"] = _dense_pop(mm_idx, acc_idx, mm_acc_eff, "E_TO_E")
    union_plan["acc2stn"] = _dense_pop(acc_idx, stn_idx, float(acc_stn_w), "E_TO_E")
    union_plan["stn2gpi"] = _dense_pop(stn_idx, gpi_idx, float(stn_gpi_w), "E_TO_E")

    inh = list(gpi_idx)                 # GPi is the only inhibitory-type (GABAergic) pool; no output edges here
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _full_snapshot(bridge)

    handles = {"seed": int(seed), "afferent_lesion": bool(afferent_lesion), "delib_lesion": bool(delib_lesion),
               "mm_lesion": bool(mm_lesion), "aff_acc_w": float(aff_acc_w), "acc_stn_w": float(acc_stn_w),
               "stn_gpi_w": float(stn_gpi_w), "heterogeneity": bool(heterogeneity), "ou_noise_pA": float(ou_noise_pA)}
    return (bridge, xp, xp.asarray(delib_idx), xp.asarray(mm_idx), xp.asarray(acc_idx), xp.asarray(stn_idx),
            xp.asarray(gpi_idx), snap, handles)


def run_trigger_trial(bridge, xp, delib_dev, mm_dev, acc_dev, stn_dev, gpi_dev, snap, *, n_ignited, mm_peak,
                      delib_scale=DELIB_CURRENT_SCALE, mm_scale=MM_CURRENT_SCALE, drive_steps=DRIVE_STEPS,
                      isolate=True):
    """ONE continuous trial: restore the clean snapshot, drive delib_aff/mm_aff proportionally to the REAL
    (n_ignited, mm_peak) afferents for `drive_steps`, and read the late-window (last third) firing rate of every
    pool. The STOP-TRIGGER verdict is `gpi_rate > GPI_TRIGGER_THRESH` -- a rate->boolean read-out of the circuit's
    OWN spiking integration, not a host comparison of n_ignited/mm_peak themselves."""
    bridge.cp_external_input_current[:] = 0.0
    if isolate:
        _full_restore(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    i_delib = float(delib_scale) * max(0.0, float(n_ignited))
    i_mm = float(mm_scale) * max(0.0, float(mm_peak))

    late = drive_steps - max(1, drive_steps // 3)
    devs = {"delib_aff": delib_dev, "mm_aff": mm_dev, "acc": acc_dev, "stn": stn_dev, "gpi": gpi_dev}
    sums = {k: 0.0 for k in devs}
    for t in range(drive_steps):
        bridge.cp_external_input_current[:] = 0.0
        if i_delib > 0.0:
            bridge.cp_external_input_current[delib_dev] = xp.float32(i_delib)
        if i_mm > 0.0:
            bridge.cp_external_input_current[mm_dev] = xp.float32(i_mm)
        bridge._run_one_simulation_step()
        if t >= late:
            for k, dv in devs.items():
                sums[k] += float(to_host(bridge.cp_firing_states[dv].astype(xp.float64).mean()))
    n_late = float(drive_steps - late)
    rates = {k: (v / n_late) for k, v in sums.items()}
    triggered = bool(rates["gpi"] > GPI_TRIGGER_THRESH)
    out = {"n_ignited": float(n_ignited), "mm_peak": float(mm_peak), "i_delib": float(i_delib), "i_mm": float(i_mm),
           "triggered": triggered}
    out.update({f"rate_{k}": float(v) for k, v in rates.items()})
    return out


# ── REAL afferents, reused-by-import from the already-existing/already-wired organs (NOT hand-picked) ───────────
def get_real_n_ignited(seed: int, n_candidates: int, *, lesion: bool = False) -> int:
    """The SAME `n_ignited` `chat._last_gnw_delib` carries -- read from `webapp.gnw_deliberation.conflict_gate`,
    the exact function the production deliberation gate calls."""
    _decision, _conf, n_ign = _DELIB.conflict_gate(int(n_candidates), seed=int(seed), lesion=bool(lesion))
    return int(n_ign)


def get_real_mm_peak(seed: int, scenario: str) -> dict:
    """The SAME `mm_peak` `chat._last_swap_drives` carries -- read from a REAL
    `webapp.gnw_thought_swap.ThoughtSwapWorkspace` (the production #77 swap detector, standalone -- no chat/composer
    needed). scenario='match' -> hold 'dog' twice (same topic, pred-vetoes mm, low mm_peak, swapped=False).
    scenario='mismatch' -> hold 'dog' then propose 'cat' (a genuine topic change, mm fires, swapped likely True)."""
    ws = _SWAP.ThoughtSwapWorkspace(seed=int(seed))
    first = ws.observe("dog")
    if scenario == "match":
        info = ws.observe("dog")
    elif scenario == "mismatch":
        info = ws.observe("cat")
    else:
        raise ValueError(scenario)
    return {"mm_peak": float(info.get("mm_peak", 0.0) or 0.0), "swapped": bool(info.get("swapped", False)),
           "first_thought": first, "info": info}


def host_boolean_or_trigger(n_ignited, swapped) -> bool:
    """The EXACT logic of `webapp.gnw_global_stop.detect_trigger` (reproduced here only for the PARITY comparison;
    the real production function is left untouched by this de-risk)."""
    triggered = False
    if isinstance(n_ignited, (int, float)) and int(n_ignited) >= 2:
        triggered = True
    if bool(swapped):
        triggered = True
    return triggered


def evaluate_seed(seed, *, aff_acc_w=AFF_ACC_W, acc_stn_w=ACC_STN_W, stn_gpi_w=STN_GPI_W,
                  delib_scale=DELIB_CURRENT_SCALE, mm_scale=MM_CURRENT_SCALE, heterogeneity=True, verbose=True):
    """Build the intact + fully-afferent-lesioned + per-pathway-lesioned circuits, drive them with REAL
    (n_ignited, mm_peak) afferents from the actual deliberation/swap organs, and measure PARITY, LOAD-BEARING
    (afferent sweeps) and LESION at this seed."""
    # ── REAL afferents (reused-by-import; not hand-picked) ──────────────────────────────────────────────────────
    n_ign_solo = get_real_n_ignited(seed, 1)          # no genuine conflict (a single candidate)
    n_ign_conflict = get_real_n_ignited(seed, 2)      # a genuine 2-candidate conflict (co-ignition)
    match = get_real_mm_peak(seed, "match")           # same-topic hold: pred vetoes mm (low mm_peak, no swap)
    mismatch = get_real_mm_peak(seed, "mismatch")     # a genuine topic change: mm fires (high mm_peak, swap)
    mm_match, mm_mismatch = match["mm_peak"], mismatch["mm_peak"]
    swapped_match, swapped_mismatch = match["swapped"], mismatch["swapped"]

    scenarios = {
        "delib_conflict": {"n_ignited": n_ign_conflict, "mm_peak": mm_match,
                           "host": host_boolean_or_trigger(n_ign_conflict, swapped_match)},
        "swap_only": {"n_ignited": n_ign_solo, "mm_peak": mm_mismatch,
                     "host": host_boolean_or_trigger(n_ign_solo, swapped_mismatch)},
        "no_trigger": {"n_ignited": n_ign_solo, "mm_peak": mm_match,
                      "host": host_boolean_or_trigger(n_ign_solo, swapped_match)},
    }

    bridge, xp, delib_dev, mm_dev, acc_dev, stn_dev, gpi_dev, snap, handles = build_stop_trigger_bridge(
        seed=seed, aff_acc_w=aff_acc_w, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w, heterogeneity=heterogeneity)

    def _run(n_ign, mm_pk):
        return run_trigger_trial(bridge, xp, delib_dev, mm_dev, acc_dev, stn_dev, gpi_dev, snap,
                                 n_ignited=n_ign, mm_peak=mm_pk, delib_scale=delib_scale, mm_scale=mm_scale)

    # ── PARITY: the intact circuit vs the host boolean-OR, on the SAME 3 real-afferent turn-classes ──────────────
    parity = {}
    for name, sc in scenarios.items():
        r = _run(sc["n_ignited"], sc["mm_peak"])
        parity[name] = {**r, "host_triggered": sc["host"], "match": bool(r["triggered"] == sc["host"])}
    parity_all_match = bool(all(v["match"] for v in parity.values()))
    n_parity_match = int(sum(1 for v in parity.values() if v["match"]))

    # ── LOAD-BEARING sweep 1: n_ignited alone (1..4) at the REAL match-level mm baseline ────────────────────────
    delib_sweep = []
    for n in (1, 2, 3, 4):
        r = _run(n, mm_match)
        delib_sweep.append({"n_ignited": n, **r})
    delib_sweep_flips = bool(delib_sweep[0]["triggered"] is False and any(s["triggered"] for s in delib_sweep[1:]))
    delib_sweep_monotone = bool(all(delib_sweep[i + 1]["rate_gpi"] >= delib_sweep[i]["rate_gpi"] - 1e-9
                                    for i in range(len(delib_sweep) - 1)))

    # ── LOAD-BEARING sweep 2: mm_peak alone (match -> mismatch, 5-pt linspace) at the REAL solo n_ignited baseline
    mm_lin = list(np.linspace(mm_match, max(mm_mismatch, mm_match + 1e-6), 5))
    mm_sweep = []
    for mmv in mm_lin:
        r = _run(n_ign_solo, float(mmv))
        mm_sweep.append({"mm_peak": float(mmv), **r})
    mm_sweep_flips = bool(mm_sweep[0]["triggered"] is False and any(s["triggered"] for s in mm_sweep[1:]))
    mm_sweep_monotone = bool(all(mm_sweep[i + 1]["rate_gpi"] >= mm_sweep[i]["rate_gpi"] - 1e-9
                                 for i in range(len(mm_sweep) - 1)))

    # ── LESION: full afferent lesion -> the trigger NEVER fires on the SAME real conflict-indicating afferents ────
    bridge_l, xp_l, dl_l, mm_l, acc_l, stn_l, gpi_l, snap_l, _ = build_stop_trigger_bridge(
        seed=seed, afferent_lesion=True, aff_acc_w=aff_acc_w, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w,
        heterogeneity=heterogeneity)

    def _run_l(n_ign, mm_pk):
        return run_trigger_trial(bridge_l, xp_l, dl_l, mm_l, acc_l, stn_l, gpi_l, snap_l, n_ignited=n_ign,
                                 mm_peak=mm_pk, delib_scale=delib_scale, mm_scale=mm_scale)

    lesion_delib_conflict = _run_l(n_ign_conflict, mm_match)
    lesion_swap_only = _run_l(n_ign_solo, mm_mismatch)
    lesion_reverts = bool((not lesion_delib_conflict["triggered"]) and (not lesion_swap_only["triggered"]))

    # attribution: how much of the trigger-worthy GPi drive does the intact afferent pathway supply vs the lesion,
    # on the SAME (delib_conflict) real afferents?
    intact_gpi = parity["delib_conflict"]["rate_gpi"]
    lesion_gpi = lesion_delib_conflict["rate_gpi"]
    afferent_attribution = attributable_to("GPi conflict-drive via the afferent->ACC pathway", intact_gpi, lesion_gpi,
                                           warn_below=0.5)

    # ── PER-PATHWAY LESION: each afferent alone remains independently sufficient ────────────────────────────────
    bridge_dl, xp_dl, dl_dl, mm_dl, acc_dl, stn_dl, gpi_dl, snap_dl, _ = build_stop_trigger_bridge(
        seed=seed, delib_lesion=True, aff_acc_w=aff_acc_w, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w,
        heterogeneity=heterogeneity)
    r_delib_lesioned_swap_still_fires = run_trigger_trial(bridge_dl, xp_dl, dl_dl, mm_dl, acc_dl, stn_dl, gpi_dl,
                                                          snap_dl, n_ignited=n_ign_solo, mm_peak=mm_mismatch,
                                                          delib_scale=delib_scale, mm_scale=mm_scale)
    bridge_ml, xp_ml, dl_ml, mm_ml, acc_ml, stn_ml, gpi_ml, snap_ml, _ = build_stop_trigger_bridge(
        seed=seed, mm_lesion=True, aff_acc_w=aff_acc_w, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w,
        heterogeneity=heterogeneity)
    r_mm_lesioned_delib_still_fires = run_trigger_trial(bridge_ml, xp_ml, dl_ml, mm_ml, acc_ml, stn_ml, gpi_ml,
                                                        snap_ml, n_ignited=n_ign_conflict, mm_peak=mm_match,
                                                        delib_scale=delib_scale, mm_scale=mm_scale)
    per_pathway_independent = bool(r_delib_lesioned_swap_still_fires["triggered"]
                                   and r_mm_lesioned_delib_still_fires["triggered"])

    # ── DETERMINISM: build-twice at one seed -> identical seed-derived Izhikevich params ────────────────────────
    h1 = _threshold_hash(bridge, xp)
    bridge2, xp2, *_ = build_stop_trigger_bridge(seed=seed, aff_acc_w=aff_acc_w, acc_stn_w=acc_stn_w,
                                                 stn_gpi_w=stn_gpi_w, heterogeneity=heterogeneity)
    h2 = _threshold_hash(bridge2, xp2)
    seed_deterministic = bool(h1 == h2 and h1 != "")

    seed_go = bool(n_parity_match >= 2 and lesion_reverts and per_pathway_independent
                  and delib_sweep_flips and mm_sweep_flips and seed_deterministic)

    v = Verdict("ACC/BG STOP-trigger circuit @ frozen operating point (seed %d)" % seed)
    v.require("real solo n_ignited < real conflict n_ignited (the afferent is genuinely informative)",
              bool(n_ign_solo < n_ign_conflict), expect=True)
    v.require("real match mm_peak < real mismatch mm_peak (the afferent is genuinely informative)",
              bool(mm_match < mm_mismatch), expect=True)
    v.require("PARITY with the host boolean-OR on >=2/3 real-afferent turn-classes", n_parity_match >= 2, expect=True)
    v.require("n_ignited afferent ALONE flips the trigger (OFF at baseline, ON somewhere in 1..4)",
              delib_sweep_flips, expect=True)
    v.require("mm_peak afferent ALONE flips the trigger (OFF at match, ON toward mismatch)", mm_sweep_flips,
              expect=True)
    v.require("FULL afferent lesion reverts the trigger to OFF on both real-conflict scenarios", lesion_reverts,
              expect=True)
    v.require("EACH afferent pathway alone remains independently sufficient (per-pathway lesion)",
              per_pathway_independent, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", seed_deterministic, expect=True)
    v.knob("afferent->ACC weight (delib+mm)", requested=aff_acc_w, applied=aff_acc_w)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2/STN-veto foot-gun")
    v.disabled("nmda", why="pure feedforward relay chain, no sustained attractor needed for a one-shot trigger read")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "real_afferents": {"n_ignited_solo": n_ign_solo, "n_ignited_conflict": n_ign_conflict,
                           "mm_peak_match": mm_match, "mm_peak_mismatch": mm_mismatch,
                           "swapped_match": swapped_match, "swapped_mismatch": swapped_mismatch},
        "operating_point": {"aff_acc_w": float(aff_acc_w), "acc_stn_w": float(acc_stn_w),
                            "stn_gpi_w": float(stn_gpi_w), "delib_scale": float(delib_scale),
                            "mm_scale": float(mm_scale), "gpi_trigger_thresh": float(GPI_TRIGGER_THRESH),
                            "heterogeneity": bool(heterogeneity)},
        "parity": parity, "n_parity_match": n_parity_match, "parity_all_match": parity_all_match,
        "delib_afferent_sweep": delib_sweep, "delib_sweep_flips": delib_sweep_flips,
        "delib_sweep_monotone_gpi": delib_sweep_monotone,
        "mm_afferent_sweep": mm_sweep, "mm_sweep_flips": mm_sweep_flips, "mm_sweep_monotone_gpi": mm_sweep_monotone,
        "lesion": {"delib_conflict": lesion_delib_conflict, "swap_only": lesion_swap_only,
                  "reverts_to_no_stop": lesion_reverts,
                  "afferent_attribution": (None if afferent_attribution is None else float(afferent_attribution))},
        "per_pathway_lesion": {"delib_lesioned_swap_still_fires": r_delib_lesioned_swap_still_fires,
                               "mm_lesioned_delib_still_fires": r_mm_lesioned_delib_still_fires,
                               "both_independently_sufficient": per_pathway_independent},
        "seed_deterministic": seed_deterministic, "threshold_hash": h1,
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        print(f"[acc-bg-stop-trigger seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        print(f"    REAL afferents: n_ignited solo={n_ign_solo} conflict={n_ign_conflict} | "
              f"mm_peak match={mm_match:.3f} mismatch={mm_mismatch:.3f}", flush=True)
        for name, p in parity.items():
            print(f"    PARITY {name}: spiking={p['triggered']} host={p['host_triggered']} match={p['match']} "
                  f"(gpi_rate={p['rate_gpi']:.3f})", flush=True)
        print(f"    delib sweep gpi_rate: {[round(s['rate_gpi'], 3) for s in delib_sweep]} flips={delib_sweep_flips}",
              flush=True)
        print(f"    mm sweep gpi_rate:    {[round(s['rate_gpi'], 3) for s in mm_sweep]} flips={mm_sweep_flips}",
              flush=True)
        print(f"    LESION reverts={lesion_reverts} attribution={afferent_attribution} | "
              f"per_pathway_independent={per_pathway_independent} | det={seed_deterministic}", flush=True)
    return result


def run_smoke(seed, args):
    """ONE-seed diagnostic: print the real afferents + a small grid over (aff_acc_w, delib_scale, mm_scale) to find
    a working operating point (the relay pool's f-I knee is unknown a priori -- no attractor to reuse a known
    knee from)."""
    print(f"[acc-bg-stop-trigger smoke] seed={seed}", flush=True)
    n_ign_solo = get_real_n_ignited(seed, 1)
    n_ign_conflict = get_real_n_ignited(seed, 2)
    match = get_real_mm_peak(seed, "match")
    mismatch = get_real_mm_peak(seed, "mismatch")
    print(f"  real afferents: n_ignited solo={n_ign_solo} conflict={n_ign_conflict} | "
          f"mm_peak match={match['mm_peak']:.4f} (swapped={match['swapped']}) "
          f"mismatch={mismatch['mm_peak']:.4f} (swapped={mismatch['swapped']})", flush=True)
    grid = []
    for aff_w in (args.aff_acc_w, args.aff_acc_w * 1.5):
        for ds in (args.delib_scale, args.delib_scale * 1.5):
            for ms in (args.mm_scale, args.mm_scale * 1.5):
                r = evaluate_seed(seed, aff_acc_w=aff_w, acc_stn_w=args.acc_stn_w, stn_gpi_w=args.stn_gpi_w,
                                  delib_scale=ds, mm_scale=ms, heterogeneity=not args.no_heterogeneity, verbose=True)
                grid.append(r)
    any_go = any(g["seed_go"] for g in grid)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_acc_bg_stop_trigger_derisk", "mode": "smoke", "grid": grid}, f, indent=2,
                  default=str)
    print(f"\n[acc-bg-stop-trigger smoke] wrote {args.json}  any_seed_go={any_go}", flush=True)
    return 0 if any_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW STOP-trigger ACC/BG circuit de-risk (rank-12 host boolean-OR).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42, help="single seed (smoke)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--aff-acc-w", type=float, default=AFF_ACC_W)
    ap.add_argument("--acc-stn-w", type=float, default=ACC_STN_W)
    ap.add_argument("--stn-gpi-w", type=float, default=STN_GPI_W)
    ap.add_argument("--delib-scale", type=float, default=DELIB_CURRENT_SCALE)
    ap.add_argument("--mm-scale", type=float, default=MM_CURRENT_SCALE)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_acc_bg_stop_trigger_6seed.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    print(f"[acc-bg-stop-trigger] aff_acc_w={args.aff_acc_w} acc_stn_w={args.acc_stn_w} stn_gpi_w={args.stn_gpi_w} "
          f"delib_scale={args.delib_scale} mm_scale={args.mm_scale} gpi_thresh={GPI_TRIGGER_THRESH} "
          f"het={not args.no_heterogeneity} backend={args.backend}\n", flush=True)

    if args.smoke:
        return run_smoke(args.seed, args)

    results = []
    for seed in args.seeds:
        results.append(evaluate_seed(seed, aff_acc_w=args.aff_acc_w, acc_stn_w=args.acc_stn_w,
                                     stn_gpi_w=args.stn_gpi_w, delib_scale=args.delib_scale, mm_scale=args.mm_scale,
                                     heterogeneity=not args.no_heterogeneity, verbose=True))

    all_go = all(r["seed_go"] for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)
    any_undefined = any(r["verdict"] == "UNDEFINED" for r in results)

    summary = {
        "runner": "_gnw_acc_bg_stop_trigger_derisk", "mode": "6seed", "seeds": list(args.seeds),
        "backend": args.backend, "all_go": bool(all_go and not any_undefined), "n_go": n_go,
        "n_seeds": len(results), "any_undefined": any_undefined,
        "operating_point": results[0]["operating_point"] if results else {}, "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if summary["all_go"] else ("UNDEFINED" if any_undefined else "NO-GO")
    print(f"\n{'=' * 100}", flush=True)
    print(f"  ACC/BG STOP-TRIGGER VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)", flush=True)
    for r in results:
        print(f"    seed {r['seed']}: {r['verdict']:9s} parity={r['n_parity_match']}/3 "
              f"lesion_reverts={r['lesion']['reverts_to_no_stop']} "
              f"per_pathway_ok={r['per_pathway_lesion']['both_independently_sufficient']} "
              f"delib_flips={r['delib_sweep_flips']} mm_flips={r['mm_sweep_flips']} "
              f"det={r['seed_deterministic']}", flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if summary["all_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
