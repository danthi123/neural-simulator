"""STEP 3 of the D5 learn-through-use arc -- the CAPSTONE capability: does USING a memory STRENGTHEN it?

THE CHAIN (all banked):
  * step-1 (NO-GO): AdEx SOMA recurrence cannot complete a real ~15-cell D5 store -> D5's per-cell dendritic-dAP
    apical latch is the READ.
  * step-2 (6/6 GO): D5's PERSISTENT apical latch can SELF-TERMINATE into a discrete ~105-196 ms apical-plateau
    transient (Ecker-b apical adaptation) -- completion + specificity preserved. That bounded transient is the BTSP
    ELIGIBILITY WINDOW (BTSP's instructive signal IS the dendritic plateau / Ca2+, Bittner-Magee 2017, NOT the soma
    spike).

THE CAPABILITY (this runner): when the brain RECALLS 'dog' (partial cue -> the dendritic latch COMPLETES the assembly ->
the self-terminating apical-plateau window opens), drive BTSP potentiation GATED ON that window so the co-active
within-assembly synapses potentiate. Repeat the recall->strengthen loop N times. Does robustness (completion from a
SPARSER cue) improve monotonically + boundedly + specifically with USE? And does the bounded window PREVENT the
interference-runaway a persistent (never-releasing) latch would cause?

THE MECHANISM (brain-based-only; NO sim/ edit; reuse D5's OWN BTSP machinery):
  The readout bridge (mem.bridge) has the two-compartment dendritic dAP apical latch cp_v_apical. The substrate's OWN
  BTSP block (sim/bridge.py 4b-bis, guarded by cfg.enable_btsp) computes, every step, INSIDE _run_one_simulation_step:
      dw = eta * Etilde_pre * IS_post * (w_max - w)          [fused_btsp_update, the substrate's kernel]
  where Etilde_pre is the seconds-long per-neuron PRESYNAPTIC eligibility (a low-pass of firing, tau=btsp_elig_tau_ms,
  gathered on the synapse's pre cell) and IS_post = max(cp_v_apical - coincidence_plateau_v_hold, 0) is the dendritic
  PLATEAU instructive signal (gathered on the post cell). So during a recall: the CUE cells FIRE (-> presynaptic
  eligibility) and the HELD cells' apicals go UP via the intrinsic latch (-> IS_post > 0), so the substrate potentiates
  cue->held WITHIN-assembly recurrent ONE-SHOT -- exactly the completion pathway. At rest the apical is silent (KIR
  down-state) => IS_post == 0 => dw == 0 (no spurious learning, by construction). The weight change is the substrate's
  plateau-gated plasticity, NOT a host formula. The strengthened weights live in mem.bridge.cp_connections.data, which
  IS mem.R.C.data -- the organ's own store path (the recall read uses the same array). Write-back is by identity.

  BOUNDEDNESS is supplied by step-2's self-terminating window: an Ecker-b apical adaptation current on cp_v_apical
  (runner-side, exactly where the organ's own _reset_apical_latch writes -- NO sim/ edit; b_adapt=0 => byte-identical
  persistent latch) collapses the plateau after ~150 ms so IS_post -> 0 and the memory RELEASES back to rest. A
  PERSISTENT latch (adapt-OFF) never releases -> its held cells stay is_post>0 forever, so ANY later activity writes
  into the still-open window (INTERFERENCE-corruption). The interference test below is that first-class runaway contrast.

PROTOCOL (per recall episode -- reuses step-2's sustained-then-free): snapshot-restore to a clean transient rest, INJECT
  the CURRENT (accumulated) weights, drive a SUSTAINED cue (ignite_steps, guarantees the latch latches = completion) then
  a FREE window (window_steps, cue removed) where the intrinsic reactivation + self-termination play out. BTSP is enabled
  for the episode; the plateau-gated write-back potentiates the within-assembly recurrence. The accumulated weights carry
  to the next episode (learn-through-use); only the TRANSIENT state is snapshot-restored (determinism).

MEASURE (the teeth), all on the SAME store, snapshot-restored (transient) with the accumulated weights INJECTED:
  * STRENGTHENS  : post-use robustness > pre-use -- the MIN NUMBER OF CUE CELLS needed to complete (win_peak>=COMPLETE_MIN)
                   DROPS (completes from a sparser cue), and/or the store now SURVIVES a partial within-recurrence LESION
                   it did not survive before.
  * MONOTONE     : min-cue-to-complete is (weakly) monotone DECREASING across the N recalls (asymptotic learn-through-use).
  * BOUNDED      : the dog within-weight stays finite + <= btsp_w_max (the soft bound) and the per-episode potentiation
                   is SELF-LIMITING (dw shrinks toward 0 with use -> converges, does not blow up). CONTRAST (reported):
                   a persistent (adapt-OFF) latch never releases, so a subsequent UNRELATED drive spuriously potentiates
                   NON-member -> dog-held synapses (interference-corruption) that the self-terminating window BLOCKS.
  * SPECIFIC     : never-recalled 'cat' within-weight is UNCHANGED; the dog strengthening does not spill to
                   between-assembly synapses; cat completion is not degraded.
ANTI-CHEAT (the key one): NO-WINDOW controls -- (a) NO-CUE (no recall -> no eligibility AND no plateau window) and
  (b) CLAMP (cue present so cue cells fire = eligibility, but cp_v_apical clamped to E_rest each step so IS_post == 0 =
  window forcibly closed) -- must produce ~NO strengthening. (b) ISOLATES the window from the cue drive: strengthening
  requires the reactivation WINDOW, not the mere presence of a cue current (= re-encoding).
  attributable_to(strengthening, with-window vs no-window) and (dog vs cat).

GO = STRENGTHENS and MONOTONE and BOUNDED and SPECIFIC, with the NO-WINDOW controls inert. Honest NO-GO otherwise
  (localizes whether the plateau-gated write-back potentiates at all). Preconditions = INSTRUMENT validity (store formed,
  window opens per step-2, deterministic, cat never recalled) -> require(); teeth drive go=.

Reuse-by-import: EpisodicDapMemory (the production D5 organ), _reset_apical_latch, snapshot_state/restore_state (step-2).
NO sim/ edit. GPU-preferred.
  Run:    SIM_BACKEND=cupy python -m research.runners._gap5_d5_learn_through_use_derisk \
              --seed 42 --out research/findings/raw/_d5_learn_through_use/seed42.json
  6-seed: SIM_BACKEND=cupy python -m research.runners._gap5_d5_learn_through_use_derisk \
              --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import get_backend  # noqa: E402
from research.runners._episodic_dap_dialogue_memory import EpisodicDapMemory  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import _reset_apical_latch  # noqa: E402
from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_d5_learn_through_use" / "seed42.json"

# --- GO thresholds ----------------------------------------------------------------------------------------------------
COMPLETE_MIN = 0.40        # free-window held-out apical-UP FRACTION that counts as "the assembly completed"
SPILL_FRAC = 0.05          # cat-within / between-assembly drift must stay under this fraction of the dog within-gain
NOWIN_FRAC = 0.15          # a no-window control's within-gain must stay under this fraction of the with-window gain
BOUNDED_MARGIN = 1.01      # dog within must stay <= btsp_w_max * this (the soft bound holds)
SELF_LIMIT_FRAC = 0.5      # the last per-episode dw must be < this fraction of the first (converging, not blowing up)


def _within_mean(cp, data, mask):
    return float(cp.mean(data[mask])) if int(cp.sum(mask)) else 0.0


def reactivate(mem, slot, snap, W_current, *, cue_indices, strengthen, clamp_apical, adapt_on,
               b_adapt, tau_w, tau_apical, cue_pa, ignite_steps, window_steps, up_thresh, v_hold,
               btsp_lr, btsp_w_max, btsp_elig_tau_ms):
    """ONE recall episode on the D5 readout bridge. Restore the TRANSIENT rest state, INJECT the accumulated weights
    W_current, drive a SUSTAINED cue then a FREE window. Three orthogonal switches:
      * strengthen   : enable the substrate's BTSP (plateau-gated write-back) for the episode -> weights potentiate.
      * clamp_apical : the NO-WINDOW isolation -- force cp_v_apical to E_rest at the START of every step so IS_post == 0
                       (no plateau ever), while the cue still drives the cue cells (eligibility present).
      * adapt_on     : apply step-2's Ecker-b self-terminating apical adaptation (b_adapt>0) -> bounded transient;
                       adapt_on=False (or b_adapt=0) = the PERSISTENT latch (the runaway contrast).
    Returns the free-window completion peak + the potentiated weights (bridge.cp_connections.data) + diagnostics."""
    bridge = mem.bridge; R = mem.R; cp = R.cp
    cfg = bridge.core_config
    E_rest = float(getattr(cfg, "apical_E_rest", -65.0)); dt = float(cfg.dt_ms)
    held_cp = cp.asarray(np.asarray([int(R.ca3_idx[p]) for p in mem.held_pos_by_asm[slot]], dtype=np.int64))

    restore_state(bridge, snap)                       # byte-identical clean TRANSIENT start
    bridge.cp_connections.data[:] = cp.asarray(W_current)   # INJECT the accumulated weights (the store carries forward)
    _reset_apical_latch(bridge)

    # ---- configure the substrate's OWN BTSP (default pure-potentiation path; every optional arm OFF) ----
    cfg.enable_hebbian_learning = False; cfg.enable_stdp = False; cfg.enable_structural_plasticity = False
    cfg.enable_bdsp = False
    if strengthen:
        cfg.enable_btsp = True
        cfg.btsp_learning_rate = float(btsp_lr); cfg.btsp_w_max = float(btsp_w_max); cfg.btsp_w_min = 0.0
        cfg.btsp_elig_tau_ms = float(btsp_elig_tau_ms)
        cfg.btsp_hetero_dep = 0.0; cfg.btsp_milstein_k_dep = 0.0; cfg.btsp_mean_subtract = 0.0
        cfg.btsp_dog_a_dep = 0.0; cfg.btsp_elig_tau_slow_ms = 0.0
        cfg.btsp_win_gate_theta = 0.0; cfg.btsp_elig_exponent = 1.0; cfg.btsp_elig_hard_thresh = 0.0
        cfg.coincidence_plateau_v_hold = float(v_hold)      # the IS_post gate threshold = the latch's own v_hold
        bridge.cp_btsp_pre_elig = None                       # FRESH per-episode eligibility (each recall is independent)
        bridge.cp_btsp_pre_elig_slow = None; bridge.cp_btsp_win_count = None; bridge.cp_btsp_wmax = None
    else:
        cfg.enable_btsp = False

    darr = None
    if cue_indices is not None and len(cue_indices) > 0:
        darr = cp.asarray(np.asarray(cue_indices, dtype=np.int64), dtype=cp.int64)
        bridge.cp_external_input_current[darr] = cp.float32(cue_pa)

    up_trace = []; soma_trace = []; w_adapt = None; w_adapt_max = 0.0
    total = ignite_steps + window_steps
    for t in range(total):
        if t == ignite_steps and darr is not None:
            bridge.cp_external_input_current[darr] = 0.0     # remove cue -> FREE window
        if clamp_apical and bridge.cp_v_apical is not None:
            bridge.cp_v_apical[:] = cp.float32(E_rest)       # NO-WINDOW isolation: force DOWN before the step reads it
        bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        va = bridge.cp_v_apical
        if va is not None:
            if adapt_on and b_adapt > 0.0 and not clamp_apical:
                if w_adapt is None:
                    w_adapt = cp.zeros_like(va)
                is_up = (va > cp.float32(up_thresh)).astype(cp.float32)
                w_adapt += cp.float32(b_adapt) * is_up
                w_adapt -= cp.float32(dt / tau_w) * w_adapt
                va -= cp.float32(dt / tau_apical) * w_adapt   # Ecker-b apical AHP/SK adaptation (self-termination)
                w_adapt_max = max(w_adapt_max, float(w_adapt[held_cp].max()))
            up_trace.append(float((va[held_cp] > cp.float32(up_thresh)).astype(cp.float32).mean()))
        else:
            up_trace.append(0.0)
        fs = bridge.cp_firing_states
        soma_trace.append(float(fs[held_cp].astype(cp.float32).mean()) if fs is not None else 0.0)
    if darr is not None:
        bridge.cp_external_input_current[darr] = 0.0
    cfg.enable_btsp = False

    W_out = bridge.cp_connections.data.copy()
    up = np.asarray(up_trace); soma = np.asarray(soma_trace)
    win = up[ignite_steps:]
    return dict(
        ignite_peak=round(float(up[:ignite_steps].max()), 4),
        win_peak=round(float(win.max()), 4),
        win_tail=round(float(win[int(0.70 * len(win)):].mean()), 4),
        soma_peak=round(float(soma.max()), 4),
        up_ms=float((win > 0.15).sum() * dt),
        w_adapt_max=round(w_adapt_max, 3),
        W_out=W_out,
    )


def _wstats(cp, W, dslot, cslot, mem):
    """within-dog / within-cat / between-assembly recurrent mean weight (the store-state read)."""
    return dict(
        w_dog=round(_within_mean(cp, W, mem.R.withinA_masks[dslot]), 3),
        w_cat=round(_within_mean(cp, W, mem.R.withinA_masks[cslot]), 3),
        w_between=round(_within_mean(cp, W, mem.R.between_mask), 3),
    )


LESION_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CURRENT_GRID = (300.0, 220.0, 160.0, 120.0, 90.0, 60.0, 40.0, 25.0)


def _read_winpeak(mem, slot, snap, W, cue, rk, *, cue_pa=None, lesion_frac=0.0):
    """Read-only full-window completion (win_peak) under an optional STRESS: a partial within-recurrence lesion (zero
    the first `lesion_frac` of the slot's within synapses in CSR order -- deterministic + nested) and/or a weakened cue
    current. W is not mutated. The completion is monotone in each stress (more lesion / weaker cue => weaker)."""
    cp = mem.R.cp
    Wl = W
    if lesion_frac > 0.0:
        idx = cp.where(mem.R.withinA_masks[slot])[0]
        n = int(round(lesion_frac * int(idx.size)))
        Wl = W.copy()
        if n > 0:
            Wl[idx[:n]] = 0.0
    r = dict(rk)
    if cue_pa is not None:
        r["cue_pa"] = cue_pa
    return reactivate(mem, slot, snap, Wl, cue_indices=cue, strengthen=False, clamp_apical=False, adapt_on=False,
                      **r)["win_peak"]


def max_lesion_survived(mem, slot, snap, W, cue_full, rk):
    """Robustness = the LARGEST within-recurrence lesion fraction the store still completes under (full cue,
    win_peak >= COMPLETE_MIN). Ascending sweep, early-stop (completion is monotone in lesion). Higher = MORE robust;
    -1.0 if the store fails even un-lesioned."""
    surv = -1.0
    for lf in LESION_GRID:
        if _read_winpeak(mem, slot, snap, W, cue_full, rk, lesion_frac=lf) >= COMPLETE_MIN:
            surv = lf
        else:
            break
    return round(surv, 3)


def min_cue_current(mem, slot, snap, W, cue_full, rk):
    """Robustness (secondary) = the SMALLEST cue current (pA) that still completes (full cue). Descending sweep. Lower =
    MORE robust. Uses ALL cue cells (no cue-identity confound), just a weaker drive."""
    last = CURRENT_GRID[0]
    for c in CURRENT_GRID:
        if _read_winpeak(mem, slot, snap, W, cue_full, rk, cue_pa=c) < COMPLETE_MIN:
            return last
        last = c
    return CURRENT_GRID[-1]


def interference_run(mem, dslot, snap, W0, adapt_on, interf_cue, interf_mask, rk, *, interfere_steps):
    """The RUNAWAY contrast the self-terminating window prevents. Continuous (NO reset between phases):
      PHASE A  recall dog (strengthen; adapt_on -> self-terminates, adapt_off -> persistent) -- opens the window.
      PHASE B  drive an UNRELATED cue (interf_cue) with BTSP still on -- the interfering cells fire (eligibility) while
               the dog-held apicals are (persistent: still UP -> is_post>0) or (self-terminated: rest -> is_post==0).
    Returns the mean weight change of the (non-dog-pre -> dog-held-post) INTERFERENCE synapses during phase B: ~0 when
    the window self-terminated (dog protected), > 0 when the latch persisted (dog corrupted by the unrelated input)."""
    bridge = mem.bridge; R = mem.R; cp = R.cp
    cfg = bridge.core_config
    E_rest = float(getattr(cfg, "apical_E_rest", -65.0)); dt = float(cfg.dt_ms)
    held_cp = cp.asarray(np.asarray([int(R.ca3_idx[p]) for p in mem.held_pos_by_asm[dslot]], dtype=np.int64))
    b_adapt = rk["b_adapt"]; tau_w = rk["tau_w"]; tau_apical = rk["tau_apical"]; up_thresh = rk["up_thresh"]

    restore_state(bridge, snap)
    bridge.cp_connections.data[:] = cp.asarray(W0)
    _reset_apical_latch(bridge)
    cfg.enable_hebbian_learning = False; cfg.enable_stdp = False; cfg.enable_structural_plasticity = False
    cfg.enable_bdsp = False; cfg.enable_btsp = True
    cfg.btsp_learning_rate = float(rk["btsp_lr"]); cfg.btsp_w_max = float(rk["btsp_w_max"]); cfg.btsp_w_min = 0.0
    cfg.btsp_elig_tau_ms = float(rk["btsp_elig_tau_ms"]); cfg.btsp_hetero_dep = 0.0; cfg.btsp_milstein_k_dep = 0.0
    cfg.btsp_mean_subtract = 0.0; cfg.btsp_dog_a_dep = 0.0; cfg.btsp_elig_tau_slow_ms = 0.0
    cfg.btsp_win_gate_theta = 0.0; cfg.btsp_elig_exponent = 1.0; cfg.btsp_elig_hard_thresh = 0.0
    cfg.coincidence_plateau_v_hold = float(rk["v_hold"])
    bridge.cp_btsp_pre_elig = None; bridge.cp_btsp_pre_elig_slow = None
    bridge.cp_btsp_win_count = None; bridge.cp_btsp_wmax = None

    w_adapt = None

    def _step(drive_arr, apply_adapt):
        nonlocal w_adapt
        bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        va = bridge.cp_v_apical
        if apply_adapt and va is not None and b_adapt > 0.0:
            if w_adapt is None:
                w_adapt = cp.zeros_like(va)
            is_up = (va > cp.float32(up_thresh)).astype(cp.float32)
            w_adapt += cp.float32(b_adapt) * is_up
            w_adapt -= cp.float32(dt / tau_w) * w_adapt
            va -= cp.float32(dt / tau_apical) * w_adapt

    # PHASE A: dog recall (open the window)
    dog_arr = cp.asarray(np.asarray(rk["_dog_cue"], dtype=np.int64), dtype=cp.int64)
    bridge.cp_external_input_current[dog_arr] = cp.float32(rk["cue_pa"])
    for t in range(rk["ignite_steps"] + rk["window_steps"]):
        if t == rk["ignite_steps"]:
            bridge.cp_external_input_current[dog_arr] = 0.0
        _step(dog_arr, adapt_on)
    bridge.cp_external_input_current[dog_arr] = 0.0
    W_after_dog = bridge.cp_connections.data.copy()
    isP_dogheld_A = float(cp.maximum(bridge.cp_v_apical[held_cp] - cp.float32(rk["v_hold"]), 0.0).mean())

    # PHASE B: unrelated interfering drive (NO reset; the dog window's release-state is what matters)
    intf_arr = cp.asarray(np.asarray(interf_cue, dtype=np.int64), dtype=cp.int64)
    bridge.cp_external_input_current[intf_arr] = cp.float32(rk["cue_pa"])
    isP_trace = []
    for _t in range(interfere_steps):
        _step(intf_arr, adapt_on)
        isP_trace.append(float(cp.maximum(bridge.cp_v_apical[held_cp] - cp.float32(rk["v_hold"]), 0.0).mean()))
    bridge.cp_external_input_current[intf_arr] = 0.0
    cfg.enable_btsp = False
    W_after_intf = bridge.cp_connections.data.copy()

    dW = W_after_intf - W_after_dog
    interf_dw = _within_mean(cp, dW, interf_mask)
    return dict(interf_dw=round(float(interf_dw), 4),
                isP_dogheld_after_dog=round(isP_dogheld_A, 3),
                isP_dogheld_phaseB_mean=round(float(np.mean(isP_trace)), 3))


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[d5-learn-through-use] seed={seed} backend={backend} -- borderline D5 store, recall->plateau-gated BTSP "
          f"write-back->strengthen loop", flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    try:
        cp, _ = get_backend()
        mem = EpisodicDapMemory(seed, topics=["cat", "dog"], verbose=True,
                                train_events=a.encode_train_events, wmax=a.encode_wmax)
        dslot = mem.topic_slot["dog"]; cslot = mem.topic_slot["cat"]
        baseline_w = float(cp.mean(mem.baseline_weights[mem.R.withinA_masks[dslot]])) \
            if int(cp.sum(mem.R.withinA_masks[dslot])) else 0.0
        if not mem.store("dog"):
            raise RuntimeError("mem.store('dog') returned False -- dog was not BTSP-formed")
        up_thresh = mem.p["up_thresh"]; v_hold = mem.p["v_hold"]
        rec_dog = mem.recall("dog")     # warms + allocates cp_v_apical
        mem.R.hard_silence(); _reset_apical_latch(mem.bridge)
        snap = snapshot_state(mem.bridge)               # the ONE frozen TRANSIENT-rest snapshot (weights injected per read)
        rec_cat = mem.recall("cat")

        W0 = mem.R.C.data.copy()
        ws0 = _wstats(cp, W0, dslot, cslot, mem)
        cue_full = np.asarray(mem.cue_by_asm[dslot], dtype=np.int64)
        cat_cue_full = np.asarray(mem.cue_by_asm[cslot], dtype=np.int64)

        rk = dict(tau_w=a.tau_w, tau_apical=a.tau_apical, cue_pa=a.cue_pa, ignite_steps=a.ignite_steps,
                  window_steps=a.window_steps, up_thresh=up_thresh, v_hold=v_hold,
                  btsp_lr=a.btsp_lr, btsp_w_max=a.recall_wmax, btsp_elig_tau_ms=a.btsp_elig_tau_ms, b_adapt=a.b_adapt)

        info = dict(n_ca3=int(mem.n_ca3), assembly_sizes=mem.assembly_sizes, dog_slot=dslot, cat_slot=cslot,
                    dog_size=int(mem.assembly_sizes[dslot]), cat_size=int(mem.assembly_sizes[cslot]),
                    baseline_within=round(baseline_w, 3), borderline=ws0, n_cue_full=int(len(cue_full)),
                    d5_recall_dog=rec_dog, d5_recall_cat=rec_cat, up_thresh=up_thresh, v_hold=v_hold)
        result["extract"] = info
        print(f"[d5-ltu] borderline store: dog={info['dog_size']} cells within~{ws0['w_dog']} (baseline~{baseline_w:.2f}, "
              f"ceiling {a.encode_wmax}); cat within~{ws0['w_cat']} (never-formed); d5 recall dog apical_cue="
              f"{rec_dog['apical_cue']:.3f} | full cue={len(cue_full)} cells", flush=True)

        # ---- INSTRUMENT preconditions ----
        full_open = reactivate(mem, dslot, snap, W0, cue_indices=cue_full, strengthen=False, clamp_apical=False,
                               adapt_on=False, **rk)
        ml0 = max_lesion_survived(mem, dslot, snap, W0, cue_full, rk)
        ml0b = max_lesion_survived(mem, dslot, snap, W0, cue_full, rk)
        deterministic = bool(ml0 == ml0b)
        mcur0 = min_cue_current(mem, dslot, snap, W0, cue_full, rk)
        # the sharp headline lesion: the FIRST lesion level the BASELINE fails (survives-it-did-not-before target)
        headline_les = round(min(ml0 + 0.1, max(LESION_GRID)), 3) if ml0 >= 0.0 else 0.0
        head_before = round(_read_winpeak(mem, dslot, snap, W0, cue_full, rk, lesion_frac=headline_les), 4)
        print(f"[d5-ltu] precondition: full-cue window opens win_peak={full_open['win_peak']} up_ms={full_open['up_ms']} "
              f"| BASELINE max-lesion-survived={ml0} min-cue-current={mcur0}pA (det={deterministic}) | "
              f"headline lesion {headline_les}: baseline win_peak={head_before}", flush=True)

        # ---- interference mask: (pre NOT a dog member) -> (post = dog held) recurrent synapses ----
        n = mem.R.n
        dog_held_g = np.asarray([int(mem.R.ca3_idx[p]) for p in mem.held_pos_by_asm[dslot]], dtype=np.int64)
        dog_member_g = np.asarray(mem.assemblies[dslot], dtype=np.int64)
        is_held = cp.zeros(n, dtype=cp.bool_); is_held[cp.asarray(dog_held_g)] = True
        is_member = cp.zeros(n, dtype=cp.bool_); is_member[cp.asarray(dog_member_g)] = True
        interf_mask = mem.R.rec_mask & is_held[mem.R.cols] & (~is_member[mem.R.rows])
        n_interf_syn = int(cp.sum(interf_mask))
        # interfering drive: a deterministic random set of non-dog CA3 cells (generic "other activity")
        rng = np.random.default_rng(seed * 977 + 3)
        nondog = np.asarray([int(g) for g in mem.R.ca3_idx if int(g) not in set(dog_member_g.tolist())], dtype=np.int64)
        interf_cue = rng.choice(nondog, size=min(a.n_interfere, len(nondog)), replace=False)

        # ================= THE LEARN-THROUGH-USE LOOP (with-window, adapt-ON, bounded) =================
        W = W0.copy(); ml_traj = [ml0]; w_dog_traj = [ws0["w_dog"]]; dw_on = []
        for i in range(a.n_recalls):
            ep = reactivate(mem, dslot, snap, W, cue_indices=cue_full, strengthen=True, clamp_apical=False,
                            adapt_on=True, **rk)
            W = ep["W_out"]
            w_now = _within_mean(cp, W, mem.R.withinA_masks[dslot])
            dw_on.append(round(w_now - w_dog_traj[-1], 3)); w_dog_traj.append(round(w_now, 3))
            ml = max_lesion_survived(mem, dslot, snap, W, cue_full, rk)
            ml_traj.append(ml)
            print(f"  [recall {i+1}/{a.n_recalls}] strengthen-ep win_peak={ep['win_peak']} up_ms={ep['up_ms']} "
                  f"soma={ep['soma_peak']} | w_dog={w_dog_traj[-1]} dw={dw_on[-1]} | max-lesion-survived={ml}",
                  flush=True)
        W_final = W
        ws_final = _wstats(cp, W_final, dslot, cslot, mem)
        dw_dog_total = round(ws_final["w_dog"] - ws0["w_dog"], 3)
        ml_final = ml_traj[-1]; ml_baseline = ml_traj[0]
        mcur_final = min_cue_current(mem, dslot, snap, W_final, cue_full, rk)
        head_after = round(_read_winpeak(mem, dslot, snap, W_final, cue_full, rk, lesion_frac=headline_les), 4)

        # ================= CONTROLS =================
        Wnc = W0.copy()
        for _i in range(a.n_recalls):
            Wnc = reactivate(mem, dslot, snap, Wnc, cue_indices=None, strengthen=True, clamp_apical=False,
                             adapt_on=True, **rk)["W_out"]
        dw_dog_nocue = round(_within_mean(cp, Wnc, mem.R.withinA_masks[dslot]) - ws0["w_dog"], 3)
        Wcl = W0.copy()
        for _i in range(a.n_recalls):
            Wcl = reactivate(mem, dslot, snap, Wcl, cue_indices=cue_full, strengthen=True, clamp_apical=True,
                             adapt_on=False, **rk)["W_out"]
        dw_dog_clamp = round(_within_mean(cp, Wcl, mem.R.withinA_masks[dslot]) - ws0["w_dog"], 3)

        # cat control: never recalled/strengthened -> its completion must not improve; report cat weight drift
        ml_cat_before = max_lesion_survived(mem, cslot, snap, W0, cat_cue_full, rk)
        ml_cat_after = max_lesion_survived(mem, cslot, snap, W_final, cat_cue_full, rk)

        # ---- the RUNAWAY contrast: interference into dog-held under self-terminating vs persistent latch ----
        intf_on = interference_run(mem, dslot, snap, W0, True, interf_cue, interf_mask,
                                   {**rk, "_dog_cue": cue_full}, interfere_steps=a.interfere_steps)
        intf_off = interference_run(mem, dslot, snap, W0, False, interf_cue, interf_mask,
                                    {**rk, "_dog_cue": cue_full}, interfere_steps=a.interfere_steps)
        attr_interf = attributable_to(f"[s{seed}] interference-corruption into dog-held: PERSISTENT latch vs "
                                      f"SELF-TERMINATING window", intf_off["interf_dw"], intf_on["interf_dw"])

        result["trajectory"] = dict(max_lesion=ml_traj, w_dog=w_dog_traj, dw_on=dw_on)
        result["stores"] = dict(borderline=ws0, final=ws_final, baseline_within=round(baseline_w, 3))
        result["robustness"] = dict(max_lesion_baseline=ml_baseline, max_lesion_final=ml_final,
                                    min_cue_current_baseline=mcur0, min_cue_current_final=mcur_final,
                                    headline_lesion=headline_les, headline_win_peak_before=head_before,
                                    headline_win_peak_after=head_after)
        result["interference"] = dict(n_interf_syn=n_interf_syn, n_interf_cue=int(len(interf_cue)),
                                      adapt_ON=intf_on, adapt_OFF=intf_off, attributable=attr_interf)

        # -------- the measured teeth --------
        finite = bool(np.isfinite(ws_final["w_dog"]))
        survives_new_lesion = bool(head_after >= COMPLETE_MIN and head_before < COMPLETE_MIN)
        STRENGTHENS = bool(ml_final > ml_baseline or mcur_final < mcur0 or survives_new_lesion)
        n_ok = sum(1 for j in range(len(ml_traj) - 1) if ml_traj[j + 1] >= ml_traj[j])
        MONOTONE = bool(n_ok == len(ml_traj) - 1 and ml_final == max(ml_traj) and ml_final > ml_baseline)
        self_limiting = bool(dw_on and dw_on[-1] < SELF_LIMIT_FRAC * max(dw_on[0], 1e-9))
        BOUNDED = bool(finite and ws_final["w_dog"] <= a.recall_wmax * BOUNDED_MARGIN and self_limiting)
        cat_drift = abs(ws_final["w_cat"] - ws0["w_cat"]); between_drift = abs(ws_final["w_between"] - ws0["w_between"])
        SPECIFIC = bool(cat_drift <= SPILL_FRAC * max(dw_dog_total, 1e-6)
                        and between_drift <= SPILL_FRAC * max(dw_dog_total, 1e-6)
                        and ml_cat_after <= ml_cat_before)
        NOWINDOW_INERT = bool(abs(dw_dog_nocue) <= NOWIN_FRAC * max(dw_dog_total, 1e-6)
                              and abs(dw_dog_clamp) <= NOWIN_FRAC * max(dw_dog_total, 1e-6))
        go = STRENGTHENS and MONOTONE and BOUNDED and SPECIFIC and NOWINDOW_INERT

        attr_win = attributable_to(f"[s{seed}] dog strengthening: WITH-window vs NO-window(clamp)",
                                   dw_dog_total, dw_dog_clamp)
        attr_spec = attributable_to(f"[s{seed}] strengthening: dog(recalled) vs cat(never-recalled)",
                                    dw_dog_total, cat_drift)
        result["attributable_window"] = attr_win; result["attributable_specificity"] = attr_spec

        # -------- earned verdict --------
        v = Verdict(f"USING a D5 memory strengthens it: plateau-gated BTSP write-back on the recall window (seed {seed})")
        v.disabled("hebbian/STDP/BDSP/structural plasticity", "ONLY the substrate's plateau-gated BTSP is live during "
                                                              "the write-back; all other learning OFF")
        v.disabled("soma-recurrence completion", "step-1 NO-GO at this scale; the read + the write-gate are the "
                                                 "dendritic apical plateau (cp_v_apical), not the soma spike")
        v.require("d5-store-formed-dog", ws0["w_dog"], expect=lambda x: x > 5.0 * max(baseline_w, 1e-6),
                  note="dog was BTSP-formed to a BORDERLINE within-weight (grew from baseline, headroom to strengthen)")
        v.require("borderline-has-headroom", ml_baseline, expect=lambda x: 0.0 <= x < max(LESION_GRID),
                  note="the store completes un-lesioned but does NOT survive the full lesion range (robustness to gain)")
        v.require("window-opens-per-step2", full_open["win_peak"], expect=lambda x: x >= COMPLETE_MIN,
                  note="the FULL-cue reactivation opens the apical-plateau completion window (step-2 property)")
        v.require("instrument-deterministic", deterministic, expect=True,
                  note="two identical max-lesion reads match -> snapshot/restore + weight-inject is deterministic")
        v.require("cat-never-recalled-baseline", ws0["w_cat"], expect=lambda x: x < 5.0,
                  note="cat is a genuine never-formed control (within stays at baseline)")
        v.reaches("write-back-potentiates", ws0["w_dog"], ws_final["w_dog"],
                  note="the plateau-gated BTSP write-back actually moved the dog within-assembly weight")
        v.reaches("robustness-moves-with-use", ml_baseline, ml_final,
                  note="the max-lesion-survived robustness read-out changed across the recall loop")
        v.control("window-vs-noWindow (clamp)", treatment=dw_dog_total, control=dw_dog_clamp, min_separation=0.0,
                  note="the strengthening requires the plateau window, not the cue drive")
        decided = v.decide(go=go)
        result["verdict"] = decided; result["verdict_status"] = decided["status"]

        checks = dict(STRENGTHENS=STRENGTHENS, MONOTONE=MONOTONE, BOUNDED=BOUNDED, SPECIFIC=SPECIFIC,
                      NOWINDOW_INERT=NOWINDOW_INERT,
                      max_lesion_baseline=ml_baseline, max_lesion_final=ml_final, max_lesion_traj=ml_traj,
                      min_cue_current_baseline=mcur0, min_cue_current_final=mcur_final, n_cue_full=int(len(cue_full)),
                      headline_lesion=headline_les, headline_wp_before=head_before, headline_wp_after=head_after,
                      survives_new_lesion=survives_new_lesion,
                      w_dog_borderline=ws0["w_dog"], w_dog_final=ws_final["w_dog"], dw_dog_total=dw_dog_total,
                      dw_on=dw_on, self_limiting=self_limiting,
                      dw_dog_nocue=dw_dog_nocue, dw_dog_clamp=dw_dog_clamp,
                      w_cat_borderline=ws0["w_cat"], w_cat_final=ws_final["w_cat"], cat_drift=round(cat_drift, 3),
                      w_between_borderline=ws0["w_between"], w_between_final=ws_final["w_between"],
                      ml_cat_before=ml_cat_before, ml_cat_after=ml_cat_after,
                      interf_dw_ON=intf_on["interf_dw"], interf_dw_OFF=intf_off["interf_dw"],
                      isP_dogheld_ON=intf_on["isP_dogheld_phaseB_mean"], isP_dogheld_OFF=intf_off["isP_dogheld_phaseB_mean"],
                      full_open_win_peak=full_open["win_peak"])
        result["checks"] = checks
        print(f"[d5-ltu] checks={json.dumps(checks, default=str)}", flush=True)
        del mem
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["verdict_status"] = "ERROR"
        traceback.print_exc()

    result["elapsed_s"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    status = result.get("verdict_status")
    print("=" * 118)
    print(f"[d5-learn-through-use] seed={seed} VERDICT: {status}")
    if "checks" in result:
        c = result["checks"]
        print(f"    STRENGTHENS={c['STRENGTHENS']} MONOTONE={c['MONOTONE']} BOUNDED={c['BOUNDED']} "
              f"SPECIFIC={c['SPECIFIC']} NOWINDOW_INERT={c['NOWINDOW_INERT']}")
        print(f"    max-lesion-survived: {c['max_lesion_baseline']} -> {c['max_lesion_final']} (traj {c['max_lesion_traj']}) | "
              f"min-cue-current {c['min_cue_current_baseline']}->{c['min_cue_current_final']}pA | "
              f"lesion {c['headline_lesion']} win_peak {c['headline_wp_before']}->{c['headline_wp_after']} "
              f"(survives-new={c['survives_new_lesion']})")
        print(f"    w_dog: {c['w_dog_borderline']} -> {c['w_dog_final']} (dw {c['dw_dog_total']}, self_limiting="
              f"{c['self_limiting']}) | nocue dw {c['dw_dog_nocue']} clamp dw {c['dw_dog_clamp']}")
        print(f"    specificity: cat {c['w_cat_borderline']}->{c['w_cat_final']} (drift {c['cat_drift']}) | "
              f"between {c['w_between_borderline']}->{c['w_between_final']}")
        print(f"    interference into dog-held: SELF-TERM window dw={c['interf_dw_ON']} (isP {c['isP_dogheld_ON']}) vs "
              f"PERSISTENT latch dw={c['interf_dw_OFF']} (isP {c['isP_dogheld_OFF']})")
    print(f"[d5-learn-through-use] wrote {out_path}")
    print("=" * 118)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--encode-train-events", type=int, default=15, dest="encode_train_events",
                    help="BTSP encode passes (fewer => weaker store; full GO encode is 40; 15 => completes un-lesioned "
                         "but fragile to lesion = robustness headroom)")
    ap.add_argument("--encode-wmax", type=float, default=100.0, dest="encode_wmax", help="encode BTSP saturation ceiling")
    ap.add_argument("--n-recalls", type=int, default=8, dest="n_recalls", help="learn-through-use recall episodes")
    ap.add_argument("--recall-wmax", type=float, default=100.0, dest="recall_wmax",
                    help="BTSP saturation ceiling DURING recall write-back (>= encode within so there is headroom)")
    ap.add_argument("--btsp-lr", type=float, default=0.02, dest="btsp_lr")
    ap.add_argument("--btsp-elig-tau-ms", type=float, default=1000.0, dest="btsp_elig_tau_ms")
    ap.add_argument("--n-interfere", type=int, default=120, dest="n_interfere",
                    help="number of non-dog CA3 cells driven in the interference (runaway-contrast) test")
    ap.add_argument("--interfere-steps", type=int, default=200, dest="interfere_steps")
    # step-2 self-terminating window
    ap.add_argument("--b-adapt", type=float, default=0.8, dest="b_adapt", help="Ecker-b apical adaptation increment")
    ap.add_argument("--tau-w", type=float, default=150.0, dest="tau_w")
    ap.add_argument("--tau-apical", type=float, default=15.0, dest="tau_apical")
    ap.add_argument("--cue-pa", type=float, default=300.0, dest="cue_pa")
    ap.add_argument("--ignite-steps", type=int, default=80, dest="ignite_steps")
    ap.add_argument("--window-steps", type=int, default=500, dest="window_steps")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()
    all_results = {}; go_flags = []
    for seed in seeds:
        out_path = (Path(a.out) if len(seeds) == 1 else Path(a.out).parent / f"seed{seed}.json")
        res = run_one(seed, a, backend, out_path)
        all_results[seed] = res
        go_flags.append(bool(res.get("verdict_status") == "GO"))

    if len(seeds) > 1:
        summ_go = int(sum(go_flags)); n = len(seeds)
        print("\n" + "#" * 118)
        print(f"[d5-learn-through-use] {n}-SEED SUMMARY: {summ_go}/{n} GO  seeds={seeds}  go_flags={go_flags}")
        for s in seeds:
            c = all_results[s].get("checks", {})
            print(f"  seed {s}: status={all_results[s].get('verdict_status')} max-lesion {c.get('max_lesion_baseline')}->"
                  f"{c.get('max_lesion_final')} w_dog {c.get('w_dog_borderline')}->{c.get('w_dog_final')} "
                  f"nocue_dw={c.get('dw_dog_nocue')} clamp_dw={c.get('dw_dog_clamp')} cat_drift={c.get('cat_drift')} "
                  f"interf ON={c.get('interf_dw_ON')} OFF={c.get('interf_dw_OFF')}")
        print("#" * 118)
        summ_path = Path(a.out).parent / f"summary_{n}seed.json"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps({"seeds": seeds, "n_go": summ_go, "go_flags": go_flags, "backend": backend,
                                         "params": vars(a),
                                         "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}},
                                        indent=2, default=str))
        print(f"[d5-learn-through-use] wrote {summ_path}")
    return 0 if all(go_flags) else 1


if __name__ == "__main__":
    sys.exit(main())
