"""D3 integration — the WEAK-SUPERVISION-LEARNED transition EXECUTES ON SPIKES (closing the learning->spiking loop).
Two validated pieces so far were separate: RANK-1 (`_d3_weak_supervision_derisk`) LEARNED the transition delta from
END-STATE-ONLY supervision (no per-step teaching, rate); the one-loop (`_d3_spiking_onbridge_loop`) EXECUTED a
teacher-forced delta on the spiking FS-WTA re-discretization. THIS composes them: train delta by WEAK (end-state-only)
supervision, then roll it out with the SPIKING FS-WTA re-discretization (a real `SimulationBridge`, the winner pool
FIRES, the next state is decoded from spikes) -> the WEAKLY-LEARNED transition length-generalizes ON SPIKES. So the
whole story is on the substrate: delta LEARNED from weak supervision + delta EXECUTED on spikes, held-out-deeper.

ANTI-CHEATS: (a) the weakly-learned delta's SPIKING held-out-DEEPER state-track >> chance == the RANK-1 rate result;
(b) spiking-winner == host-argmax (faithful re-discretization of the weakly-learned scores); (c) a SHUFFLE-label-trained
delta (memorization floor) -> the spiking rollout COLLAPSES to chance (proves it executes the genuinely-learned transition,
not noise); (d) multi-seed. Reuse-by-import (RANK-1 `train_endstate` + rung-1 `build_fswta_score_bridge`/`fswta_drive`/
`spiking_rollout_eval`); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_weaklearned_spiking_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, build_group
from research.runners._d3_weak_supervision_derisk import train_endstate
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive, spiking_rollout_eval


def run_seed(group, seed, n_hid=160, epochs=80, n_eval=40, fs_inh=9.0, fs_settle=25):
    n_pool = 256 if group == "A5" else 64
    task = make_group_task(group, seed, n_pool=n_pool, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    K = task["K"]
    wk = train_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, supervise="state")              # RANK-1 weak-learn
    sh = train_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, supervise="state", shuffle_labels=True)  # mem-floor
    W, Wsh = wk["weights"], sh["weights"]
    sb = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)                                   # spiking FS-WTA
    spk = spiking_rollout_eval(task, W, "test_deeper", sb, K, seed=seed, settle=fs_settle, n_eval=n_eval, drive_fn=fswta_drive)
    sb2 = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)
    spk_sh = spiking_rollout_eval(task, Wsh, "test_deeper", sb2, K, seed=seed, settle=fs_settle, n_eval=n_eval, drive_fn=fswta_drive)
    return {"seed": seed, "K": K,
            "rate_deeper_prop": round(wk["deeper"]["prop"], 3), "rate_deeper_track": round(wk["deeper"]["state_track"], 3),
            "SPK_deeper_track": round(spk["spk_track"], 3), "SPK_host_agree": round(spk["spk_host_agree"], 3),
            "SHUFFLE_SPK_track": round(spk_sh["spk_track"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--fs-inh", type=float, default=9.0)
    ap.add_argument("--fs-settle", type=int, default=25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 weak-learned -> SPIKING] {a.group} | delta LEARNED from end-state-only supervision, EXECUTED on the spiking FS-WTA re-discretization", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(a.group, s, n_hid=a.n_hid, epochs=a.epochs, n_eval=a.n_eval, fs_inh=a.fs_inh, fs_settle=a.fs_settle)
        rows.append(r)
        print(f"  [seed {s}] weak-learn rate deeper (track={r['rate_deeper_track']}) || SPIKING deeper-track={r['SPK_deeper_track']} "
              f"(host-agree={r['SPK_host_agree']}) || SHUFFLE-learned SPIKING={r['SHUFFLE_SPK_track']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        sk, ag, shk = _m("SPK_deeper_track"), _m("SPK_host_agree"), _m("SHUFFLE_SPK_track")
        Kg = build_group(a.group)[1].shape[0]
        go = (sk > 0.90) and (ag > 0.95) and (sk - shk > 0.30)
        print(f"\n  AGGREGATE ({a.group}): weakly-learned-delta SPIKING deeper-track={sk:.3f} (host-agree {ag:.3f}) | SHUFFLE-learned SPIKING={shk:.3f} (chance={1.0/Kg:.3f})", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the transition delta LEARNED from END-STATE-only supervision (no per-step teaching) EXECUTES on the spiking FS-WTA re-discretization and length-generalizes to held-out-DEEPER ('+format(sk,'.2f')+', faithful == host argmax) >> the shuffle-learned floor ('+format(shk,'.2f')+') -> the whole story is on the substrate: delta LEARNED from weak supervision + delta EXECUTED on spikes' if go else 'the weakly-learned delta did not execute cleanly on spikes (read host-agree + the shuffle gap; tune epochs/fs-inh/fs-settle)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
