"""D3 spiking port, INTEGRATION rung: the WHOLE recurrent step in ONE spiking loop. Rungs 1 & 2 validated the two
halves SEPARATELY -- rung 1 (`_d3_spiking_attractor_derisk`) put the RE-DISCRETIZATION on the Izhikevich FS-WTA bridge
(S3 host-agree 1.0), rung 2 (`_d3_spiking_transition_derisk`) learned the TRANSITION through a spiking LIF hidden pool
(step-delta 1.0). BUT rung 1's rollout still computed the transition with the RATE tanh weights (host-computed). THIS
composes them: each step of the autoregressive rollout is (i) the SPIKING LIF transition forward (`[emb[state];input]`
-> W1 -> LIF hidden T steps -> W2 -> K scores) THEN (ii) the SPIKING FS-WTA re-discretization (drive K Izhikevich
attractor pools + shared inhibitory FS -> the winner FIRES) -> (iii) the spiking winner's `emb` FEEDS BACK as the next
state. So the entire discrete-attractor recurrent step -- transition AND re-discretization -- runs on spiking neurons,
in a loop, feeding back, to held-out-DEEPER depth.

If the full-spiking loop length-generalizes (deeper >> chance) AND the spiking winner is faithful (== the host argmax
over the LIF scores), the simulated recurrent sequence/language cortex step is realized END-TO-END on spikes.

ANTI-CHEATS: (a) full-spiking-loop held-out-DEEPER track >> chance == the rate result; (b) spiking-winner == host-argmax
over the LIF scores (the FS-WTA re-discretization is faithful to the spiking transition it reads); (c) the RE-DISCRETIZE
is LOAD-BEARING -- a NO-REDISCRETIZATION control (carry the SOFT LIF state `softmax(scores)@emb` forward instead of the
clean attractor winner) DRIFTS -> fails deeper (isolates the on-spikes re-discretization); (d) multi-seed. Reuse-by-import
(`spiking_transition` rung 2 + `build_fswta_score_bridge`/`fswta_drive` rung 1 + `make_group_task`); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_spiking_onbridge_loop_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, build_group
from research.runners._d3_spiking_transition_derisk import spiking_transition
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive


def lif_forward(feats, W):
    """The rung-2 SPIKING LIF transition forward, standalone from the returned weights: `[emb[state];input]` (feats) ->
    W1 -> LIF hidden pool (T hard-reset steps, rate = mean spikes) -> W2 -> K next-state scores. This is the SAME spiking
    nonlinearity the transition was trained through (surrogate grad), now driven in the loop."""
    W1, W2, b2 = W["W1"], W["W2"], W["b2"]
    T, leak, thr = W["T"], W["leak"], W["thr"]
    drive = np.asarray(feats, dtype=np.float32) @ W1.T                 # [B, n_hid] constant per-step drive
    B, n_hid = drive.shape
    v = np.zeros((B, n_hid), np.float32); acc = np.zeros((B, n_hid), np.float32); s = np.zeros((B, n_hid), np.float32)
    for _ in range(T):
        v = leak * v * (1.0 - s) + drive
        s = (v >= thr).astype(np.float32); acc += s
    rate = acc / T
    return rate @ W2.T + b2                                            # [B, K] spiking-transition scores


def _softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


def fullspiking_rollout(task, W, split, sb, K, seed=42, n_eval=40, input_gain=1200.0, settle=25, rediscretize=True):
    """Autoregressive rollout where EVERY step is spiking-transition + spiking-FS-WTA-re-discretization + feedback.
    rediscretize=True: the FS-WTA winner (a CLEAN attractor) feeds back as emb[winner] (the on-spikes re-discretization).
    rediscretize=False (CONTROL): carry the SOFT continuous state softmax(scores)@emb forward (no attractor -> drifts)."""
    emb = W["emb"]; ident = task["ident"]
    Xe, ye, Le, _, Se = task[split]
    rng = np.random.RandomState(seed + 7)
    idx = rng.choice(len(Le), min(n_eval, len(Le)), replace=False)
    ok = 0; agree = 0; steps = 0
    for n in idx:
        cur = ident                                                   # discrete state index
        cur_emb = emb[ident].copy()                                   # (control path) soft state embedding
        for t in range(int(Le[n])):
            state_emb = emb[cur] if rediscretize else cur_emb         # feedback: clean attractor vs soft carry
            feat = np.concatenate([state_emb, Xe[n, t]])[None, :]
            scores = lif_forward(feat, W)[0]                          # [K] SPIKING LIF transition scores
            nxt_host = int(np.argmax(scores))
            if rediscretize:
                sc = np.maximum(scores, 0.0); mx = sc.max()
                sc = sc / (mx + 1e-9) if mx > 0 else sc               # scale-normalize (argmax-preserving) for a
                #                                                       consistent FS-WTA drive across score magnitudes
                _, acc = fswta_drive(sb, K, sc, input_gain=input_gain, settle=settle)   # SPIKING FS-WTA re-discretization
                nxt_spk = int(np.argmax(acc)) if acc.max() > 0 else ident
                agree += int(nxt_spk == nxt_host); steps += 1
                cur = nxt_spk                                         # ROLL OUT on the SPIKING winner (feedback)
            else:
                p = _softmax(scores); cur_emb = p @ emb               # CONTROL: soft state carried (no re-discretize)
                cur = nxt_host
        ok += int(cur == Se[n, int(Le[n]) - 1])
    return {"track": ok / len(idx), "host_agree": agree / max(steps, 1)}


def run_seed(group_name, seed, n_pool=None, n_hid=192, epochs=25, fs_inh=9.0, fs_settle=25, n_eval=40,
             deep_lens=(8, 12, 16), lif_T=16, nperlen=None):
    is_big = group_name == "A5"
    n_pool = n_pool if n_pool is not None else (256 if is_big else 64)
    nperlen = nperlen if nperlen is not None else (8000 if is_big else 1500)
    # Train the transition on SHALLOW (1,2,3); roll the full-spiking loop out MUCH DEEPER (8/12/16 = up to ~5x training
    # depth) -- there the soft-carry control ACCUMULATES drift and fails while the clean re-discretization holds (the
    # discrete-attractor's whole point: arbitrary depth). The transition is per-step teacher-forced -> depth-agnostic.
    task = make_group_task(group_name, seed, n_pool=n_pool, n_per_len=nperlen, train_lens=(1, 2, 3), test_lens=deep_lens)
    K = task["K"]
    tr = spiking_transition(task, seed=seed, n_hid=n_hid, T=lif_T, epochs=epochs, spiking=True)  # rung-2 SPIKING transition
    W = tr["weights"]
    sb = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)                          # rung-1 FS-WTA bridge
    full = fullspiking_rollout(task, W, "test_deeper", sb, K, seed=seed, settle=fs_settle, n_eval=n_eval, rediscretize=True)
    ctrl = fullspiking_rollout(task, W, "test_deeper", sb, K, seed=seed, settle=fs_settle, n_eval=n_eval, rediscretize=False)
    return {"seed": seed, "group": group_name, "K": K,
            "spk_step_delta_same": round(tr["step_delta_same"], 3),
            "FULLSPK_deeper_track": round(full["track"], 3), "FULLSPK_host_agree": round(full["host_agree"], 3),
            "NO_REDISC_deeper_track": round(ctrl["track"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--fs-inh", type=float, default=9.0)
    ap.add_argument("--fs-settle", type=int, default=25)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--deep-lens", default="8,12,16", help="held-out DEEP rollout lengths (>> train 1,2,3)")
    ap.add_argument("--lif-T", type=int, default=16, help="LIF transition rate resolution (finer -> more separable at large K)")
    ap.add_argument("--n-per-len", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    deep_lens = tuple(int(x) for x in a.deep_lens.replace(",", " ").split())
    print(f"[D3 spiking ONE-LOOP] {a.group} | the WHOLE step in one spiking loop: LIF transition -> FS-WTA re-discretize -> feedback", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(a.group, s, n_hid=a.n_hid, epochs=a.epochs, fs_inh=a.fs_inh, fs_settle=a.fs_settle,
                     n_eval=a.n_eval, deep_lens=deep_lens, lif_T=a.lif_T, nperlen=a.n_per_len)
        rows.append(r)
        print(f"  [seed {s}] spk-step-delta={r['spk_step_delta_same']} || FULL-SPIKING loop DEEPER={r['FULLSPK_deeper_track']} "
              f"(host-agree={r['FULLSPK_host_agree']}) || NO-REDISCRETIZE control DEEPER={r['NO_REDISC_deeper_track']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        fd, fa, cd = _m("FULLSPK_deeper_track"), _m("FULLSPK_host_agree"), _m("NO_REDISC_deeper_track")
        Kg = build_group(a.group)[1].shape[0]
        go = (fd > 0.90) and (fa > 0.95) and (fd - cd > 0.15)
        print(f"\n  AGGREGATE ({a.group}): FULL-SPIKING loop deeper={fd:.3f} (host-agree {fa:.3f}) | NO-REDISCRETIZE control deeper={cd:.3f} (chance={1.0/Kg:.3f})", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the WHOLE discrete-attractor recurrent step runs in ONE spiking loop (spiking LIF transition -> spiking FS-WTA re-discretization -> emb feedback), length-generalizing to held-out-DEEPER ('+format(fd,'.2f')+' >> the no-rediscretize control '+format(cd,'.2f')+') and faithful (== host argmax) -> the simulated recurrent sequence/language cortex step is realized END-TO-END on the project spiking substrate' if go else 'the full-spiking loop did not hold cleanly (tune fs-inh/fs-settle/input-gain/epochs; read host-agree + the control gap)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
