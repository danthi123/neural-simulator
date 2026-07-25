"""Consolidation dendritic OPERATING-POINT sweep (the design-gate's prescribed NEXT, 2026-07-25).

The confirmed boundary: the co-activation potentiation fix works (directional `ca1->slot` write), the two-compartment
bistable plateau ENGAGES, but at the operating points tried (Option-1 k-sweep {3..60}, Option-3 BTSP) the slots
OVER-FIRE non-selectively — a CLIFF with NO intermediate k that fires only the fact's own slot, because there is NO
per-slot c_drive SEPARATION (`2026-07-25-consolidation-dendritic-surpass-DESIGN...`). The design's NEXT (a): a
comprehensive operating-point sweep to find a NARROW selective plateau (self_regen down so it doesn't latch-all, lower
slot_drive so the write is only the strongly-co-active `ca1_i->slot_i`, a stronger WTA to force one-of-N), MEASURING
the per-slot c_drive DIRECTLY (does slot_i get more drive than slot_j under fact_i's tag?) rather than inferring from
the ignition cliff. NEXT (b): if NO operating point separates -> the deeper dendritic LINE/BUMP attractor is the named
mechanism.

Runs on the numpy CPU backend (verified) so the free mini-PC pool can sweep it untended. Each --config-index is an
ISOLATED subprocess (build+encode+replay+probe for BOTH the dendritic arm and its LINEAR control at one seed) so a
hang/crash costs one cell, not the window. NO sim/ edit (pure config reuse of nmda_compositional_consolidation).

  python -m research.runners._consol_dendritic_opsweep --list-configs        # -> N (total grid size)
  python -m research.runners._consol_dendritic_opsweep --config-index 7 --seed 42 --out research/findings/raw/consol_opsweep
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SIM_BACKEND", "numpy")   # the pool is CPU/numpy; GPU box can override to cupy
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay,
    _mean_gate_weight, CONSOLIDATED_FACTS, _try_tgate, _try_pgate)
from sim.backend import get_backend, to_host

# ---- the operating-point grid (the design's NEXT-(a) knobs) --------------------------------------------------------
GRID = dict(
    self_regen=[0.0, 0.05, 0.10, 0.15, 0.20],   # v-gated SUSTAIN latch; LOW so the plateau doesn't latch ALL slots
    k_thresh=[2.0, 3.0, 4.0, 5.0],              # coincidence threshold on the per-step weighted ca1->slot drive
    wta=[5.0, 10.0, 20.0, 40.0],                # comp_wta_weight; STRONGER to force one-of-N
    kir_g=[1.0, 3.0, 5.0],                      # apical KIR down-state depth (silent rest)
    slot_drive=[700.0, 1400.0],                 # co-activation slot drive; LOW so the write is only strongly-co-active
)
_KEYS = list(GRID.keys())
_CONFIGS = [dict(zip(_KEYS, vals)) for vals in itertools.product(*[GRID[k] for k in _KEYS])]

BASE = dict(ca1_concept_density=0.25, ca1_concept_weight=0.0, nmda_self_weight=12.0, nmda_self_density=0.15,
            nmda_recurrent_ratio=0.6, cross_pool_density=0.10, stdp_w_max=8.0, enable_global_nmda=False,
            enable_hebbian=True, skip_nmda_additions=True,
            comp_attractor_slots=len(CONSOLIDATED_FACTS), comp_attractor_n_per=120, comp_self_weight=12.0,
            comp_no_pool_slot=True)   # drop the concept->ALL-slots broadcast (the write-selectivity killer, confirmed)
N = len(CONSOLIDATED_FACTS)
REPLAY_CYCLES = 40   # potentiation is fast on numpy (8-cyc smoke already gave dw=0.0088 > the design's 100-cyc +0.0057);
                     # the scientific question is c_drive SEPARATION (operating-point-dependent), not cycle count, and
                     # the dendritic plateau OVER-FIRES -> more cycles = slower numpy for no separation gain.


def _slot_idx(bridge):
    rm = bridge.region_manager
    return {s: list(rm.indices(f"comp_attr_{s}")) for s in range(N)}


def cdrive_probe(bridge, tags):
    """PRIMARY measurement: under each fact's tag, read the mean coincidence-plateau drive (g_coincidence) + general
    exc drive (g_e) reaching EACH slot. SELECTIVE structure <=> c_drive[slot_i | fact_i] >> c_drive[slot_j | fact_i].
    Returns per-fact separation ratio = own-slot drive / mean(other-slot drive) (the r-iii c_drive diagnostic)."""
    cp, _ = get_backend()
    _try_tgate(bridge, "nmda_attractor", 1.0)
    _try_pgate(bridge, "ca1_to_comp_attr", 1.0)
    slots = _slot_idx(bridge)
    have_coinc = getattr(bridge, "cp_conductance_g_coincidence", None) is not None
    rows, ratios = [], []
    for i, tag in enumerate(tags):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(60):
            bridge._run_one_simulation_step()
        bridge.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        acc_c = {s: 0.0 for s in range(N)}
        acc_e = {s: 0.0 for s in range(N)}
        steps = 40
        for _ in range(steps):
            bridge._run_one_simulation_step()
            gc = to_host(bridge.cp_conductance_g_coincidence) if have_coinc else None
            ge = to_host(bridge.cp_conductance_g_e)
            for s in range(N):
                if gc is not None:
                    acc_c[s] += float(gc[slots[s]].mean())
                acc_e[s] += float(ge[slots[s]].mean())
        try:
            bridge.clear_tag_drive(tag)
        except Exception:
            pass
        cvec = [acc_c[s] / steps for s in range(N)]
        evec = [acc_e[s] / steps for s in range(N)]
        drive = cvec if (have_coinc and sum(cvec) > 1e-9) else evec
        own = drive[i]
        others = [drive[j] for j in range(N) if j != i]
        mo = (sum(others) / len(others)) if others else 0.0
        ratio = (own / mo) if mo > 1e-9 else (float("inf") if own > 1e-9 else 0.0)
        ratios.append(ratio if ratio != float("inf") else 999.0)
        rows.append(dict(fact=i, g_coinc=cvec, g_e=evec, own=own, mean_other=mo, ratio=round(min(ratio, 999.0), 3)))
    return dict(rows=rows, mean_ratio=round(float(np.mean(ratios)), 3),
                n_separated=int(sum(1 for r in ratios if r > 1.5)))   # >1.5x own-vs-other = a real separation


def slot_ignition(bridge, tags):
    """After consolidation: cue each fact's tag, read which slot ignites. SELECTIVE iff fact i -> slot i."""
    _try_tgate(bridge, "nmda_attractor", 1.0)
    _try_pgate(bridge, "ca1_to_comp_attr", 1.0)
    slots = _slot_idx(bridge)
    rows = []
    for i, tag in enumerate(tags):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(60):
            bridge._run_one_simulation_step()
        bridge.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        cnt = {s: 0 for s in range(N)}
        for _ in range(80):
            bridge._run_one_simulation_step()
            fs = to_host(bridge.cp_firing_states)
            for s in range(N):
                cnt[s] += int(fs[slots[s]].sum())
        # HOLD: drive off, does the winner stay latched? (bistable plateau signature)
        try:
            bridge.clear_tag_drive(tag)
        except Exception:
            pass
        bridge.cp_external_input_current[:] = 0.0
        hold = {s: 0 for s in range(N)}
        for _ in range(60):
            bridge._run_one_simulation_step()
            fs = to_host(bridge.cp_firing_states)
            for s in range(N):
                hold[s] += int(fs[slots[s]].sum())
        top = max(cnt, key=cnt.get)
        rows.append(dict(fact=i, top=top, cnt=cnt[top], all=cnt, hold_top=hold[top],
                         hold_total=int(sum(hold.values()))))
    sel = sum(1 for r in rows if r["top"] == r["fact"] and r["cnt"] > 0)
    ign = sum(1 for r in rows if r["cnt"] > 0)
    return dict(rows=rows, selective=sel, ignition=ign)


def run_arm(seed, op, dendritic):
    a = dict(BASE)
    a.update(comp_dendritic=bool(dendritic), comp_wta_weight=op["wta"],
             comp_k_thresh=op["k_thresh"], comp_self_regen=op["self_regen"], comp_kir_g=op["kir_g"])
    b = build_substrate(seed, SimpleNamespace(**a))
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    w0 = _mean_gate_weight(b, "ca1_to_comp_attr")
    coactivation_replay(b, CONSOLIDATED_FACTS, tags, REPLAY_CYCLES, seed,
                        coactivate=True, attractor_on=True, slot_drive_pA=op["slot_drive"])
    w1 = _mean_gate_weight(b, "ca1_to_comp_attr")
    cd = cdrive_probe(b, tags)
    ig = slot_ignition(b, tags)
    return dict(arm="dendritic" if dendritic else "linear", w_ca1slot_pre=round(w0, 5), w_ca1slot_post=round(w1, 5),
                dw=round(w1 - w0, 5), cdrive=cd, ignition=ig)


def main():
    global REPLAY_CYCLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-index", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep")
    ap.add_argument("--list-configs", action="store_true")
    ap.add_argument("--dendritic-only", action="store_true", help="skip the LINEAR control arm (half the runtime)")
    ap.add_argument("--cycles", type=int, default=None, help="replay cycles (default 100; lower for smoke)")
    args = ap.parse_args()
    if args.cycles is not None:
        REPLAY_CYCLES = int(args.cycles)
    if args.list_configs:
        print(len(_CONFIGS)); return
    ci = args.config_index
    if ci is None or not (0 <= ci < len(_CONFIGS)):
        print(f"ERR --config-index must be 0..{len(_CONFIGS)-1}"); sys.exit(2)
    op = _CONFIGS[ci]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    outp = Path(args.out) / f"op{ci:03d}_seed{args.seed}.json"
    t0 = time.time()
    rec = dict(config_index=ci, seed=args.seed, op=op, backend=get_backend()[1],
               replay_cycles=REPLAY_CYCLES, n_facts=N)

    def _verdict():
        d = rec.get("dendritic")
        if not d:
            return {}
        return dict(
            dend_selective=d["ignition"]["selective"], dend_ratio=d["cdrive"]["mean_ratio"],
            dend_separated=d["cdrive"]["n_separated"],
            lin_selective=(rec.get("linear") or {}).get("ignition", {}).get("selective"),
            # GO candidate iff dendritic separates AND is selective >= ceil(N/2) AND beats the linear control
            candidate=bool(d["cdrive"]["n_separated"] >= (N + 1) // 2
                           and d["ignition"]["selective"] >= (N + 1) // 2
                           and (rec.get("linear") is None
                                or d["ignition"]["selective"] > rec["linear"]["ignition"]["selective"])))

    def _flush():
        rec["VERDICT"] = _verdict()
        rec["elapsed_s"] = round(time.time() - t0, 1)
        outp.write_text(json.dumps(rec, indent=2))   # INCREMENTAL: preserve the dendritic arm if the linear arm times out

    try:
        rec["dendritic"] = run_arm(args.seed, op, dendritic=True)
        _flush()                                                      # <- dendritic result persisted before the slow linear arm
        if not args.dendritic_only:
            rec["linear"] = run_arm(args.seed, op, dendritic=False)   # coincidence OFF, SAME wires -> load-bearing check
            _flush()
    except Exception as e:
        import traceback
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
        _flush()
    v = rec.get("VERDICT", {})
    print(f"[op{ci:03d} seed{args.seed}] {op} -> dend_sel={v.get('dend_selective')} ratio={v.get('dend_ratio')} "
          f"sep={v.get('dend_separated')}/{N} lin_sel={v.get('lin_selective')} CANDIDATE={v.get('candidate')} "
          f"({rec['elapsed_s']}s){' ERR='+rec['error'] if 'error' in rec else ''}", flush=True)


if __name__ == "__main__":
    main()
