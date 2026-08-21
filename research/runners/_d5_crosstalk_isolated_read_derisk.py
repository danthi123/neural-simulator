"""D5 CROSSTALK ISOLATED-READ de-risk — DECIDE board #73: does consolidating memory A move a NEIGHBOR memory B's
surfaced recall strength? (the long-open memory-separator crosstalk question, decidable NOW).

WHY THIS IS DECIDABLE NOW (verified, do NOT re-derive):
  * The prior te=40 crosstalk confirm (research/findings/2026-08-21-d5-pattern-separation-setpoint-...-te40-confirm,
    artifact research/findings/raw/_d5_separation/cupy_confirm_te40.json) rendered the crosstalk UNDECIDABLE — e.g.
    seed 43 ON: neighbor |dB|=1.79 mV *despite B's within-weights being BYTE-IDENTICAL* after consolidating A. A
    weight-untouched neighbor moving 1.79 mV can only be READ NOISE.
  * The stabilized-read finding (research/findings/2026-08-21-d5-stabilized-read-NEGATIVE.md, on main) DIAGNOSED that
    noise: the production `_apical_depth_read` uses only an INCOMPLETE reset (hard_silence + _reset_apical_latch) between
    reads, so residual bridge state carries over and the surfaced read becomes a stationary PERIOD-2 LIMIT CYCLE
    (a byte-identical neighbour still swings several mV of pure read noise). The CURE it validated: a COMPLETE reset — a
    snapshot_state/restore to the clean-rest baseline + inject the current weights — makes the read DETERMINISTIC
    (repeated-read std = 0 on 6/6). That deterministic read is a PURE FUNCTION OF THE STORED WEIGHTS.

THE EXPERIMENT (re-runs Layer B of _d5_pattern_separation_setpoint_derisk with the DETERMINISTIC snapshot-isolated read
instead of the period-2 live `_apical_depth_read`). Per seed x separation in {OFF=unmodified emergent assemblies /
ON=winner-fatigue set-point}: form A(slot0)+B(slot1) on the shared DAP readout at store_te, read neighbor B (isolated),
CONSOLIDATE A (re-form at consol_te = production te=40 => more within-A BTSP), read B again (isolated) -> |dB|_iso.
Classify each build by whether B's within-weights are byte-identical after consolidating A:
  * DISJOINT builds (byte-identical B-weights): a deterministic read that is a pure function of B's weights MUST give
    |dB|_iso == 0 EXACTLY -> the clean confirmation that consolidating A does NOT touch a non-overlapping neighbour, and
    the check that the read is genuinely WEIGHT-LOCAL (this is exactly where the live read gave 1.79 mV of read noise).
  * OVERLAPPING builds (shared cells => B's read-path weights DO change): |dB|_iso now reveals the TRUE weight-mediated
    crosstalk magnitude, no longer buried in read noise — quantified.
Also reports the repeated-read NOISE FLOOR of the isolated read (K reads, no consolidation between -> std ~0) alongside
the LIVE read's noise floor (the period-2 the isolation removes) -> proves decidability.

VERDICT (a MEASUREMENT, not a GO-hunt): crosstalk decidably CLOSED (isolated read deterministic; every byte-identical-B
build gives |dB|_iso == 0; the overlapping residual quantified as a REAL weight-mediated effect, not read noise) vs a
real residual remains (the read is NOT weight-local, or the isolated read is not deterministic).

BRAIN-BASED / SCOPE (NO sim/ edit): the read is the SAME spiking apical-dAP completion (cp_v_apical); the strengthening
is the substrate's OWN plateau-gated BTSP (_form_one_assembly). Host code is the encode selection, the winner-fatigue
set-point (host-applied hyperpolarizing current; the theta-crossing selection stays on-substrate), and the
snapshot/restore determinism guard (the biology's own return-to-rest between recalls). ADDITIVE, default-off; the binary
moat gate is unchanged.

  3-seed decisive: SIM_BACKEND=cupy python -m research.runners._d5_crosstalk_isolated_read_derisk --seeds 42 43 44
  6-seed:          SIM_BACKEND=cupy python -m research.runners._d5_crosstalk_isolated_read_derisk \
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

from sim.backend import get_backend, to_host  # noqa: E402
# Reuse Layer-B's VALIDATED machinery verbatim (formation, assemblies, overlap, the graded depth read) — only the READ
# is swapped to the deterministic snapshot-isolated read below. NO sim/ edit.
from research.runners._d5_pattern_separation_setpoint_derisk import (  # noqa: E402
    _assemblies, _apical_depth_read, _form_slot, _pairwise_overlaps, _max_shared, _whash, SEP_BIAS_DEFAULT)
from research.runners._gap5_dendritic_dap_readout_completion_derisk import (  # noqa: E402
    _build_dap_readout, _held_cue_perm, _reset_apical_latch)
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import make_readout  # noqa: E402
from research.runners._episodic_dap_dialogue_memory import GO_DEFAULTS  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

cp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_d5_crosstalk_iso" / "summary.json"

ZERO_TOL = 1e-6      # a byte-identical-B build's |dB|_iso must be <= this (a deterministic read is EXACTLY 0)
NOISE_TOL = 1e-4     # the isolated read is decidable if its repeated-read std <= this (~0)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The snapshot/restore determinism guard — PORTED from research/runners/_gap5_d5_latch_self_termination_derisk.py
# (that module is not on this topic branch; re-implemented here, byte-for-byte, per the task). It is the biology's
# own return-to-rest between recalls: copy every mutable cp_ ndarray + the sparse .data + current_time_step to a
# clean-rest snapshot, then restore it byte-identically before each read so the read is a pure function of the weights.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def snapshot_state(bridge):
    snap = {}
    for name in dir(bridge):
        if not name.startswith("cp_"):
            continue
        try:
            arr = getattr(bridge, name)
        except Exception:  # noqa: BLE001 -- some cp_ attrs are properties that can raise; skip them
            continue
        if arr is None or hasattr(arr, "tocoo"):
            continue
        if hasattr(arr, "dtype") and hasattr(arr, "shape") and hasattr(arr, "copy"):
            try:
                snap[name] = arr.copy()
            except Exception:  # noqa: BLE001
                pass
    if getattr(bridge, "cp_connections", None) is not None:
        snap["__conn_data__"] = bridge.cp_connections.data.copy()
    # ZERO the spike-history in the frozen state: hard_silence does NOT reset cp_prev_firing_states, so a prior read's
    # last-step spikes would otherwise be frozen in and, on restore, drive feedback inhibition suppressing ignition.
    for name in ("cp_prev_firing_states", "cp_firing_states"):
        if name in snap:
            snap[name][:] = False
    snap["__t__"] = int(bridge.runtime_state.current_time_step)
    return snap


def restore_state(bridge, snap):
    for name, arr in snap.items():
        if name.startswith("__"):
            continue
        cur = getattr(bridge, name, None)
        if cur is not None and hasattr(cur, "shape") and cur.shape == arr.shape:
            cur[:] = arr
        else:
            setattr(bridge, name, arr.copy())
    if "__conn_data__" in snap and getattr(bridge, "cp_connections", None) is not None:
        bridge.cp_connections.data[:] = snap["__conn_data__"]
    bridge.runtime_state.current_time_step = snap["__t__"]


def _iso_depth_read(bridge, R, snap, W_full, held_pos, cue_g, up_thresh, v_hold):
    """The DETERMINISTIC snapshot-isolated surfaced read: complete reset to the clean-rest snapshot + inject the current
    weights, then the SAME spiking apical-dAP recall + graded depth_hold as `_apical_depth_read` (which uses only the
    incomplete reset that produces the period-2 cycle). A pure function of `W_full` -> weight-attributable, std 0."""
    restore_state(bridge, snap)
    bridge.cp_connections.data[:] = cp.asarray(W_full)
    return _apical_depth_read(bridge, R, held_pos, cue_g, up_thresh, v_hold)


def _std(vals):
    return float(np.std(np.asarray(vals, dtype=np.float64))) if len(vals) else 0.0


def _crosstalk_one_iso(seed, *, separation, sep_bias, p, store_te, consol_te, n_patterns, up_thresh, v_hold,
                       k_noise, warm_reads, verbose):
    """Form A(slot0)+B(slot1) on the shared DAP readout, ISOLATED-read B & A, consolidate A (re-form at consol_te),
    ISOLATED-read again. Returns the deterministic neighbor shift |dB|_iso, A's own rise, the isolated + live
    repeated-read noise floors, the shared within-connection count, and B-weight byte-identity."""
    assemblies, r1_range = _assemblies(seed, n_patterns, separation=separation, sep_bias=sep_bias)
    n_ca3 = r1_range[2]
    sizes = [int(len(a)) for a in assemblies]
    overlaps = _pairwise_overlaps(assemblies)

    bridge = _build_dap_readout(seed, n_ca3=n_ca3, ca3_density=p["density"], ca3_fb_inhib=p["ca3_fb_inhib"],
                                k_thresh=p["kthresh"], plateau_strength=p["plateau_strength"], apical_R=p["apical_R"],
                                self_regen=p["self_regen"], v_hold=p["v_hold"], apical_kir_g=p["apical_kir_g"],
                                apical_gc=p["apical_gc"], apical_gc_read=p["apical_gc_read"], coincidence=True)
    R = make_readout(bridge, seed, assembly_frac=p["assembly_frac"], cue_frac=p["cue_frac"], drive_pA=p["drive_pA"],
                     warm_steps=p["warm_steps"], read_steps=p["read_steps"], silence_steps=p["silence_steps"],
                     assemblies_ext=assemblies)
    held_pos_by_asm, cue_by_asm, _perm = _held_cue_perm(R, seed)

    A_slot, B_slot = 0, 1
    mA = R.withinA_masks[A_slot]; mB = R.withinA_masks[B_slot]
    shared_conn = int(to_host(cp.sum(mA & mB)))   # within-connections in BOTH masks (both endpoints in A cap B)

    # STORE A then B at store_te (last write wins on any shared connection => B's value on the shared conns).
    m0, v0 = _form_slot(seed, n_ca3, assemblies, A_slot, store_te, p); R.C.data[m0] = v0
    m1, v1 = _form_slot(seed, n_ca3, assemblies, B_slot, store_te, p); R.C.data[m1] = v1
    Bw_before = _whash(R.C.data[mB])
    w_A_stored = float(to_host(cp.mean(R.C.data[mA])))

    # ── clean-rest snapshot: a warm read allocates cp_v_apical, then hard_silence + reset -> snapshot the baseline ──
    _apical_depth_read(bridge, R, held_pos_by_asm[B_slot], cue_by_asm[B_slot], up_thresh, v_hold)  # allocate cp_v_apical
    R.hard_silence(); _reset_apical_latch(bridge)
    snap = snapshot_state(bridge)
    W_stored = R.C.data.copy()

    # ── ISOLATED baseline reads (deterministic) ──
    B_before = _iso_depth_read(bridge, R, snap, W_stored, held_pos_by_asm[B_slot], cue_by_asm[B_slot], up_thresh, v_hold)
    A_before = _iso_depth_read(bridge, R, snap, W_stored, held_pos_by_asm[A_slot], cue_by_asm[A_slot], up_thresh, v_hold)

    # ── NOISE FLOOR: K repeated reads of the SAME neighbour B, NO consolidation between them (weights == W_stored) ──
    iso_noise = [_iso_depth_read(bridge, R, snap, W_stored, held_pos_by_asm[B_slot], cue_by_asm[B_slot],
                                 up_thresh, v_hold)["depth_hold"] for _ in range(k_noise)]
    iso_noise_std = _std(iso_noise)
    # LIVE contrast: warm the live bridge into its steady period-2 regime, then K repeated LIVE reads (weights unchanged
    # == W_stored, since the last iso read restored+injected it; live reads never touch the weights).
    for _ in range(warm_reads):
        _apical_depth_read(bridge, R, held_pos_by_asm[B_slot], cue_by_asm[B_slot], up_thresh, v_hold)
    live_noise = [_apical_depth_read(bridge, R, held_pos_by_asm[B_slot], cue_by_asm[B_slot],
                                     up_thresh, v_hold)["depth_hold"] for _ in range(k_noise)]
    live_noise_std = _std(live_noise)

    # ── CONSOLIDATE A: re-form A from baseline at consol_te (production te=40) -> stronger/reasserted within-A weights.
    #    On DISJOINT builds mA cap mB = {} => B's read-path weights are untouched (byte-identical). On OVERLAPPING
    #    builds the shared connections change (B's stored value -> A's consolidated value) => B's read changes. ──
    mc, vc = _form_slot(seed, n_ca3, assemblies, A_slot, consol_te, p); R.C.data[mc] = vc
    W_consol = R.C.data.copy()
    Bw_after = _whash(R.C.data[mB])
    w_A_consol = float(to_host(cp.mean(R.C.data[mA])))
    byte_id_B = bool(Bw_before == Bw_after)

    # ── ISOLATED post reads (deterministic) ──
    B_after = _iso_depth_read(bridge, R, snap, W_consol, held_pos_by_asm[B_slot], cue_by_asm[B_slot], up_thresh, v_hold)
    A_after = _iso_depth_read(bridge, R, snap, W_consol, held_pos_by_asm[A_slot], cue_by_asm[A_slot], up_thresh, v_hold)

    dB_iso = float(B_after["depth_hold"] - B_before["depth_hold"])
    dA_iso = float(A_after["depth_hold"] - A_before["depth_hold"])
    res = {
        "separation": bool(separation), "seed": seed, "assembly_sizes": sizes,
        "max_shared_cells": _max_shared(overlaps), "shared_within_conn": shared_conn,
        "B_within_weights_byte_identical": byte_id_B, "Bw_before": Bw_before, "Bw_after": Bw_after,
        "w_A_stored": round(w_A_stored, 4), "w_A_consol": round(w_A_consol, 4),
        "B_depth_before": round(B_before["depth_hold"], 6), "B_depth_after": round(B_after["depth_hold"], 6),
        "neighbor_shift_iso": round(dB_iso, 8), "neighbor_shift_iso_abs": round(abs(dB_iso), 8),
        "A_depth_before": round(A_before["depth_hold"], 6), "A_depth_after": round(A_after["depth_hold"], 6),
        "A_rise_iso": round(dA_iso, 6),
        "iso_noise_std": round(iso_noise_std, 8), "iso_noise_vals": [round(float(x), 5) for x in iso_noise],
        "live_noise_std": round(live_noise_std, 6), "live_noise_vals": [round(float(x), 5) for x in live_noise],
    }
    if verbose:
        tag = "ON " if separation else "OFF"
        print(f"[xtalk-iso {tag}] s{seed} sizes={sizes} shared_cells={res['max_shared_cells']} "
              f"shared_conn={shared_conn} byteid_B={byte_id_B} | B {res['B_depth_before']}->{res['B_depth_after']} "
              f"|dB|_iso={res['neighbor_shift_iso_abs']} | iso_noise_std={res['iso_noise_std']} "
              f"live_noise_std={res['live_noise_std']} | A {res['A_depth_before']}->{res['A_depth_after']} "
              f"(+{res['A_rise_iso']})", flush=True)
    del bridge, R
    return res


def run_seed(seed, *, sep_bias, p, store_te, consol_te, n_patterns, up_thresh, v_hold, k_noise, warm_reads, verbose):
    t0 = time.time()
    out = {"seed": seed}
    out["off"] = _crosstalk_one_iso(seed, separation=False, sep_bias=sep_bias, p=p, store_te=store_te,
                                    consol_te=consol_te, n_patterns=n_patterns, up_thresh=up_thresh, v_hold=v_hold,
                                    k_noise=k_noise, warm_reads=warm_reads, verbose=verbose)
    out["on"] = _crosstalk_one_iso(seed, separation=True, sep_bias=sep_bias, p=p, store_te=store_te,
                                   consol_te=consol_te, n_patterns=n_patterns, up_thresh=up_thresh, v_hold=v_hold,
                                   k_noise=k_noise, warm_reads=warm_reads, verbose=verbose)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _decide(results, seeds):
    """Earn the crosstalk decidability verdict. A build = one (seed, separation) crosstalk measurement."""
    builds = []
    for s in seeds:
        for k in ("off", "on"):
            b = results[s].get(k)
            if b:
                builds.append(b)
    byteid_builds = [b for b in builds if b["B_within_weights_byte_identical"]]
    overlap_builds = [b for b in builds if not b["B_within_weights_byte_identical"]]

    # decidability: the isolated read's repeated-read noise floor is ~0 on EVERY build.
    max_iso_noise = max((b["iso_noise_std"] for b in builds), default=0.0)
    max_live_noise = max((b["live_noise_std"] for b in builds), default=0.0)
    iso_deterministic = bool(max_iso_noise <= NOISE_TOL)
    # weight-local: every byte-identical-B build gives |dB|_iso == 0 EXACTLY.
    byteid_all_zero = bool(all(b["neighbor_shift_iso_abs"] <= ZERO_TOL for b in byteid_builds))
    max_byteid_shift = max((b["neighbor_shift_iso_abs"] for b in byteid_builds), default=0.0)
    # the overlapping residual: quantify the true weight-mediated crosstalk on builds where B genuinely changes.
    overlap_shifts = sorted((b["neighbor_shift_iso_abs"] for b in overlap_builds), reverse=True)
    max_overlap_shift = overlap_shifts[0] if overlap_shifts else 0.0

    v = Verdict("D5 memory-separator crosstalk (#73): is 'consolidating A moves neighbour B' DECIDABLE with the "
                "deterministic snapshot-isolated read, where the period-2 live read left it UNDECIDABLE?")
    v.disabled("host weight formula / plasticity on the read bridge",
               "the strengthening is the substrate's OWN plateau-gated BTSP (_form_one_assembly on a fresh formation "
               "bridge); the read bridge has plasticity FROZEN. Host code = encode selection + snapshot/restore guard.")
    v.disabled("live-bridge state carryover on the surfaced read",
               "the ISOLATED read completely resets to the clean-rest snapshot + injects the current weights before the "
               "read (the biology's return-to-rest between recalls) -> deterministic + weight-attributable")
    v.disabled("on-substrate intrinsic-plasticity conductance for the ON set-point",
               "the winner-fatigue sparsity bias is a host-applied hyperpolarizing current; the theta-crossing "
               "selection stays on-substrate. The spiking granule intrinsic-plasticity version is the tracked next step.")
    v.require("isolated read is DETERMINISTIC (repeated-read noise ~0 on every build)", iso_deterministic, expect=True,
              note=f"max iso_noise_std={max_iso_noise:.2e} over {len(builds)} builds (NOISE_TOL={NOISE_TOL})")
    v.require("byte-identical-B builds exist (a neighbour whose read-path weights are untouched)",
              len(byteid_builds) >= 1, expect=True,
              note=f"{len(byteid_builds)} byte-identical-B builds (disjoint / saturated-shared)")
    v.require("every byte-identical-B build gives |dB|_iso == 0 EXACTLY (read is WEIGHT-LOCAL)", byteid_all_zero,
              expect=True, note=f"max |dB|_iso on byte-identical-B builds = {max_byteid_shift:.2e} "
                                f"(the live read gave up to {max_live_noise:.3f} mV of read noise on these)")
    v.require("overlapping build(s) reproduced (B's read-path weights DO change) so the residual is quantifiable",
              len(overlap_builds) >= 1, expect=True,
              note=f"{len(overlap_builds)} overlapping builds; max |dB|_iso={max_overlap_shift:.4f} mV")
    v.control("isolated vs live repeated-read noise floor (the decidability upgrade)",
              treatment=max_live_noise, control=max_iso_noise, min_separation=0.0,
              note="the live read carries the period-2 noise the isolated read removes (iso std ~0)")

    # GO == crosstalk decidably CLOSED: read deterministic, disjoint gives exactly 0 (weight-local), overlapping
    # residual is a real weight-mediated number (not read noise). NO-GO would be a real read-side residual remaining.
    go = bool(iso_deterministic and len(byteid_builds) >= 1 and byteid_all_zero and len(overlap_builds) >= 1
              and max_live_noise > max_iso_noise)
    decided = v.decide(go=go)
    decided["_metrics"] = {
        "n_builds": len(builds), "n_byteid_builds": len(byteid_builds), "n_overlap_builds": len(overlap_builds),
        "max_iso_noise_std": round(max_iso_noise, 8), "max_live_noise_std": round(max_live_noise, 6),
        "max_byteid_shift_iso": round(max_byteid_shift, 8), "max_overlap_shift_iso": round(max_overlap_shift, 6),
        "overlap_shifts_iso": [round(x, 6) for x in overlap_shifts],
        "byteid_shifts_iso": sorted(round(b["neighbor_shift_iso_abs"], 8) for b in byteid_builds),
    }
    return decided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--sep-bias", type=float, default=SEP_BIAS_DEFAULT)
    ap.add_argument("--store-te", type=int, default=20, help="train_events for the initial store (non-saturated headroom)")
    ap.add_argument("--consol-te", type=int, default=40, help="train_events for the consolidation re-form (production te)")
    ap.add_argument("--n-patterns", type=int, default=3)
    ap.add_argument("--k-noise", type=int, default=8, dest="k_noise", help="repeated reads for the noise floor")
    ap.add_argument("--warm-reads", type=int, default=4, dest="warm_reads", help="live warm reads before its noise floor")
    ap.add_argument("--json", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    p = dict(GO_DEFAULTS)
    results = {}
    t0 = time.time()
    print("#" * 118)
    print(f"[d5-xtalk-iso] backend={backend} seeds={a.seeds} sep_bias={a.sep_bias} store_te={a.store_te} "
          f"consol_te={a.consol_te} k_noise={a.k_noise}", flush=True)
    for seed in a.seeds:
        try:
            results[seed] = run_seed(seed, sep_bias=a.sep_bias, p=p, store_te=a.store_te, consol_te=a.consol_te,
                                     n_patterns=a.n_patterns, up_thresh=p["up_thresh"], v_hold=p["v_hold"],
                                     k_noise=a.k_noise, warm_reads=a.warm_reads, verbose=True)
            print(f"[d5-xtalk-iso] seed {seed} done ({results[seed]['elapsed_s']}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            results[seed] = {"seed": seed, "error": repr(e)}
            traceback.print_exc()

    ok_seeds = [s for s in a.seeds if "off" in results.get(s, {}) and "on" in results.get(s, {})]
    verdict = _decide(results, ok_seeds) if ok_seeds else None

    payload = {"backend": backend, "seeds": a.seeds, "ok_seeds": ok_seeds,
               "params": {"sep_bias": a.sep_bias, "store_te": a.store_te, "consol_te": a.consol_te,
                          "n_patterns": a.n_patterns, "k_noise": a.k_noise, "warm_reads": a.warm_reads},
               "results": {str(s): results[s] for s in a.seeds},
               "elapsed_s": round(time.time() - t0, 1)}
    # LIFT the verdict + its preconditions to the TOP LEVEL so tools/gates/verdict_preconditions.py can enforce them.
    if verdict is not None:
        payload["verdict"] = verdict["status"]
        payload["go"] = verdict["go"]
        payload["preconditions"] = verdict["preconditions"]
        payload["disabled_processes"] = verdict["disabled_processes"]
        payload["verdict_full"] = verdict
        payload["metrics"] = verdict.get("_metrics")

    out_path = Path(a.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print("#" * 118)
    if verdict is not None:
        m = verdict["_metrics"]
        print(f"[d5-xtalk-iso] VERDICT: {verdict['status']}  (go={verdict['go']})")
        print(f"  builds={m['n_builds']} byteid_B={m['n_byteid_builds']} overlapping={m['n_overlap_builds']}")
        print(f"  iso noise floor (max std) = {m['max_iso_noise_std']:.2e}  |  live noise floor (max std) = "
              f"{m['max_live_noise_std']:.4f}")
        print(f"  byte-identical-B builds |dB|_iso = {m['byteid_shifts_iso']}  (must be 0 -> read is weight-local)")
        print(f"  OVERLAPPING builds |dB|_iso (true weight-mediated crosstalk) = {m['overlap_shifts_iso']}")
    print(f"[d5-xtalk-iso] wrote {out_path}  ({payload['elapsed_s']}s)")
    print("#" * 118)
    return 0 if (verdict is not None and verdict["status"] == "GO") else 1


if __name__ == "__main__":
    sys.exit(main())
