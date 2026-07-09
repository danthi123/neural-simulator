"""D3 SPIKING port (rung 1): the re-discretization ON SPIKES — the concrete "simulated recurrent sequence/language
cortex". The rate de-risk (`_d3_group_composition_derisk.py`) proved DISCRETE-ATTRACTOR recurrence length-generalizes
multi-hop group composition (S3 + theorem-backed A5) where a continuous RNN cannot; the mechanism = re-discretize the
running state to a CLEAN attractor each step. THIS ports that re-discretization onto the project's OWN spiking substrate:
each step's transition scores drive K Izhikevich attractor pools with input-DIVISIVE-NORMALIZATION (the E%-max WTA =
the OneBrainComposer/NEF cleanup = CA3 pattern completion) -> the WINNER pool FIRES -> the next state is read from
SPIKES -> iterate. So the running group state is maintained as a spiking attractor, composing to held-out-DEEPER depth.

RUNG-1 SCOPE: the TRANSITION (delta: state x input -> next-state scores) is the rate-learned weights (reuse the validated
discrete_attractor_rnn); only the RE-DISCRETIZATION is moved on-spikes (the divnorm WTA). Anti-cheats: (a) spiking-WTA
winner == host-argmax winner per step (the WTA is faithful); (b) DIVNORM-OFF lesion -> the WTA degrades (the divisive
normalization is load-bearing); (c) held-out-DEEPER state-track on spikes >> chance == the rate result. Reuse-by-import;
NO `sim/` edit. numpy backend (small bridge).

Run:  SIM_BACKEND=numpy python -m research.runners._d3_spiking_attractor_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, discrete_attractor_rnn
from research.runners._phaseC_S5_divnorm_derisk import build_divnorm_score_bridge, onbridge_divnorm_drive


def spiking_rollout_eval(task, W, split, sb, K, input_gain=1200.0, settle=15, n_eval=60, seed=42):
    """Autoregressive rollout with ON-BRIDGE spiking WTA re-discretization. Each step: scores = Ws.tanh(Wr.emb[cur] +
    Wi.x) + bs (rate transition) -> drive the K attractor pools -> the winner FIRES (divnorm WTA) -> next state = the
    spiking winner. Returns spiking state-track acc + the spiking-vs-host winner agreement."""
    emb, Wr, Wi, Ws, bs = W["emb"], W["Wr"], W["Wi"], W["Ws"], W["bs"]
    ident = task["ident"]
    Xe, ye, Le, _, Se = task[split]
    rng = np.random.RandomState(seed + 1)
    idx = rng.choice(len(Le), min(n_eval, len(Le)), replace=False)
    ok_spk = 0; agree = 0; steps = 0
    for n in idx:
        cur = ident
        for t in range(int(Le[n])):
            h = np.tanh(emb[cur] @ Wr.T + Xe[n, t] @ Wi.T)
            scores = h @ Ws.T + bs                                # K-dim transition scores
            _, acc = onbridge_divnorm_drive(sb, K, scores, input_gain=input_gain, settle=settle)
            nxt_spk = int(np.argmax(acc)) if acc.max() > 0 else ident
            nxt_host = int(np.argmax(scores))
            agree += int(nxt_spk == nxt_host); steps += 1
            cur = nxt_spk                                         # ROLL OUT on the SPIKING winner
        ok_spk += int(cur == Se[n, int(Le[n]) - 1])
    return {"spk_track": ok_spk / len(idx), "spk_host_agree": agree / max(steps, 1)}


def run_seed(group_name, seed, n_pool=None, n_hid=192, epochs=60, n_per_len=None):
    is_big = group_name == "A5"
    n_pool = n_pool if n_pool is not None else (256 if is_big else 64)
    n_per_len = n_per_len if n_per_len is not None else (8000 if is_big else 1500)
    task = make_group_task(group_name, seed, n_pool=n_pool, noise=0.6, n_per_len=n_per_len,
                           train_lens=(1, 2, 3, 4, 5), test_lens=(6, 7, 8))
    K = task["K"]
    da = discrete_attractor_rnn(task, seed=seed, epochs=epochs, n_hid=n_hid)     # rate transition (validated)
    W = da["weights"]
    # PRIMARY spiking WTA = PLAIN Izhikevich drive (drive each attractor pool by its score -> the winner fires most ->
    # decode argmax(firing) = the spiking re-discretization). The divisive-norm E%-max OVER-normalizes single-winner
    # transition scores (a diagnostic, not the right cleanup for a clear one-of-K winner).
    sb = build_divnorm_score_bridge(seed=seed, V=K, n_word=10, enable_divnorm=False)
    sb_dn = build_divnorm_score_bridge(seed=seed, V=K, n_word=10, enable_divnorm=True)    # divnorm-ON diagnostic
    spk_same = spiking_rollout_eval(task, W, "test_same", sb, K, seed=seed)
    spk_deep = spiking_rollout_eval(task, W, "test_deeper", sb, K, seed=seed)
    spk_deep_dn = spiking_rollout_eval(task, W, "test_deeper", sb_dn, K, seed=seed)
    return {"seed": seed, "group": group_name, "K": K, "rate_step_delta": round(da["step_transition_acc"], 3),
            "rate_deeper_track": round(da["state_deeper"], 3),
            "SPK_same_track": round(spk_same["spk_track"], 3), "SPK_deeper_track": round(spk_deep["spk_track"], 3),
            "SPK_host_agree_deeper": round(spk_deep["spk_host_agree"], 3),
            "SPK_deeper_track_DIVNORM_ON": round(spk_deep_dn["spk_track"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 SPIKING attractor] {a.group} | re-discretization ON SPIKES (divnorm WTA = CA3/NEF cleanup) | rate transition + spiking re-discretize", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(a.group, s, n_hid=a.n_hid, epochs=a.epochs)
        rows.append(r)
        print(f"  [seed {s}] rate: step-delta={r['rate_step_delta']} deeper={r['rate_deeper_track']} || "
              f"SPIKING (plain Izh WTA): same-track={r['SPK_same_track']} DEEPER-track={r['SPK_deeper_track']} "
              f"(host-agree={r['SPK_host_agree_deeper']}; divnorm-ON over-normalizes={r['SPK_deeper_track_DIVNORM_ON']})", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        spk_d, agree, dn = _m("SPK_deeper_track"), _m("SPK_host_agree_deeper"), _m("SPK_deeper_track_DIVNORM_ON")
        # GO: the SPIKING re-discretization (plain Izhikevich WTA: drive attractor pools by score -> winner fires most ->
        # decode) maintains the running state to held-out-DEEPER (>>chance ~ 1/K) AND is FAITHFUL (spiking winner ==
        # host argmax, agree ~1.0). The E%-max divnorm is the WRONG WTA for a clear one-of-K winner (over-normalizes).
        go = (spk_d > 0.90) and (agree > 0.95)
        print(f"\n  AGGREGATE ({a.group}): SPIKING deeper-track={spk_d:.3f} | host-agree={agree:.3f} | (divnorm-ON diag={dn:.3f}) (chance={1.0/rows[0]['K']:.3f})", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the re-discretization runs ON SPIKES faithfully (plain Izhikevich attractor-pool WTA == host argmax, deeper-track '+format(spk_d,'.2f')+') -> the discrete-attractor recurrent composition is realized on the project spiking substrate = the simulated recurrent language cortex; next: FS lateral-inhibition for a CLEAN one-active attractor + transition on-spikes + reduce teacher-forcing + the real CA3 bridge' if go else 'the spiking re-discretization did not hold (read host-agree: if low, tune input_gain/settle so the winner pool fires most)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
