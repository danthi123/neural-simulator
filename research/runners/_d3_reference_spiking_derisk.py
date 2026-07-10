"""D3 -> LANGUAGE, ON SPIKES (the fully-spiking mission payoff): the discrete-attractor tracks the DISCOURSE REFERENT
("who holds the object") across a narrative with the RE-DISCRETIZATION running on the project's spiking substrate.

Composes the language reference-tracking task (`_d3_reference_tracking_derisk.make_reference_tracking_task`) with the
group-task spiking machinery: the transition delta is learned (teacher-forced, `discrete_attractor_rnn`; the honest
relational-learning residual for END-STATE-only supervision is documented separately), and the RE-DISCRETIZATION each
step is the spiking FS-WTA over K Izhikevich attractor pools on a real `SimulationBridge` (`build_fswta_score_bridge` /
`fswta_drive` / `spiking_rollout_eval`) -- the winner referent-pool FIRES, the next holder is decoded from spikes,
iterated. So the running discourse-referent state is maintained as a SPIKING attractor across the narrative.

ANTI-CHEATS: (a) SPIKING held-out-DEEPER holder-track >> chance == the rate result; (b) spiking-winner == host-argmax
(faithful re-discretization); (c) multi-seed. Reuse-by-import; numpy backend (small bridge); NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_reference_spiking_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_reference_tracking_derisk import make_reference_tracking_task
from research.runners._d3_group_composition_derisk import discrete_attractor_rnn
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive, spiking_rollout_eval


def run_seed(seed, K=6, n_pool=64, n_hid=192, epochs=60, n_eval=40, fs_inh=9.0, fs_settle=25, input_gain=1200.0):
    task = make_reference_tracking_task(seed, K=K, n_pool=n_pool, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    da = discrete_attractor_rnn(task, seed=seed, epochs=epochs, n_hid=n_hid)              # learn the reference delta
    W = da["weights"]
    sb = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)                       # spiking FS-WTA re-discretization
    spk = spiking_rollout_eval(task, W, "test_deeper", sb, K, seed=seed, input_gain=input_gain, settle=fs_settle, n_eval=n_eval, drive_fn=fswta_drive)
    return {"seed": seed, "K": K, "rate_deeper_track": round(da["state_deeper"], 3), "rate_step_delta": round(da["step_transition_acc"], 3),
            "SPK_deeper_track": round(spk["spk_track"], 3), "SPK_host_agree": round(spk["spk_host_agree"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--fs-inh", type=float, default=9.0)
    ap.add_argument("--fs-settle", type=int, default=25)
    ap.add_argument("--input-gain", type=float, default=1200.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 -> LANGUAGE ON SPIKES] K={a.K} | discourse-referent tracking with the re-discretization on the spiking FS-WTA substrate", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, K=a.K, n_hid=a.n_hid, epochs=a.epochs, n_eval=a.n_eval, fs_inh=a.fs_inh, fs_settle=a.fs_settle, input_gain=a.input_gain)
        rows.append(r)
        print(f"  [seed {s}] rate holder-track DEEPER={r['rate_deeper_track']} (step-delta={r['rate_step_delta']}) || "
              f"SPIKING holder-track DEEPER={r['SPK_deeper_track']} (host-agree={r['SPK_host_agree']})", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        sk, ag = _m("SPK_deeper_track"), _m("SPK_host_agree")
        go = (sk > 0.75) and (ag > 0.95)
        print(f"\n  AGGREGATE (K={a.K}, chance {1.0/a.K:.3f}): SPIKING holder-track DEEPER={sk:.3f} (host-agree {ag:.3f})", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the discrete-attractor tracks the DISCOURSE REFERENT (who-holds-it) across a held-out-DEEPER narrative with the re-discretization running ON SPIKES ('+format(sk,'.2f')+' holder-track, faithful == host argmax) -> D3 tracks who/what we are talking about across a conversation ON THE SPIKING SUBSTRATE = the fully-spiking mission-payoff language application' if go else 'the spiking reference tracking did not hold cleanly (tune fs-inh/fs-settle/epochs; read host-agree)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
