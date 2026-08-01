"""gap#4 arc B -- SCRATCH CPU-rate de-risk (report only, do NOT commit).

THE SINGLE-VARIABLE QUESTION: the CPU-rate learned-microcircuit GO (2026-07-24) learns W^PI and MEASURES its
apical-silent property, but the FEEDFORWARD weights are still driven by the RAW fixed-random feedback
(v_api = e_upper @ Y). The learned interneuron cancellation NEVER drives the FF weights -- "runner-supplied
not learned". This de-risk changes exactly ONE thing: at the TOP hidden layer, make the FF credit ride on the
LEARNED cancelled residual  v_apical = onehot@Y - softmax@W^PI  (W^PI learned from a NOISY init by the Eq.9
self-prediction rule), instead of the raw (onehot-softmax)@Y. At the self-predicting fixed point W^PI==Y the
two are byte-identical, so a CONVERGED W^PI reproduces fixed_fa; a noisy/learning W^PI drives the FF weights
with the LEARNED instructive signal. This is the swap the whole "learned instructive signal" crux asks for,
done at CPU rate first (the cheap read before the on-bridge e-prop port).

ARMS (held-out inheritance accuracy, the metric):
  reservoir            : hidden FROZEN (credit-independent floor).
  fixed_fa             : raw fixed-random FA credit drives FF                    = the control the learned signal must MATCH.
  micro_drives_ff      : LEARNED cancelled residual (W^PI plastic) drives FF     = THE ARM.
  frozen_wpi_drives_ff : cancelled residual with W^PI FROZEN-noisy drives FF     = isolates that LEARNING is load-bearing.
  transport_ceiling    : Y:=W^T (cheat upper bound; no-weight-transport guard MUST fail).

GO gate: micro_drives_ff ~ fixed_fa (within margin, >=5/6 seeds) AND >> reservoir ; apical-silent EARNED
(silent_ratio << frozen) ; frozen_wpi_drives_ff < micro_drives_ff (learning load-bearing) ; anti-cheats:
permuted->chance, no-weight-transport True for micro_drives_ff.

Reuse-by-import ONLY (no sim/ edit): subclasses MicroNet, overriding train_step's top-layer credit source.

  SIM_BACKEND=numpy nice -n 12 .venv/bin/python -u -m research.runners._gap4_learned_instructive_drives_ff_DERISK_SCRATCH \
      --seeds 42 43 44 100 101 102 --hidden 96 --deep-layers 2 --epochs 250 --lr 0.3
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sim.dendritic_mlp import DendriticMLP  # noqa: E402
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, _acc_on)
from research.runners._gnw_d1_spiking_bdsp_derisk import (  # noqa: E402
    _train, _no_weight_transport, _cos, _softmax, _MOMENTUM)
from research.runners._gap4_learned_microcircuit_selfpredict_derisk import (  # noqa: E402
    MicroNet, TransportCeilingNet)


class MicroDrivesFFNet(MicroNet):
    """MicroNet + the SINGLE-VARIABLE swap: at the top hidden layer, the FF credit rides on the LEARNED
    interneuron-cancelled residual (W^PI plastic) instead of the raw fixed feedback. drives_ff toggles it;
    wpi_frozen keeps W^PI at its noisy init (the 'fixed instructive signal' control)."""

    def __init__(self, *args, drives_ff=True, wpi_frozen=False, **kw):
        super().__init__(*args, **kw)
        self.drives_ff = bool(drives_ff)
        self.wpi_frozen = bool(wpi_frozen)

    def train_step(self, X, y, mode, lr):
        acts, lg = self._forward(X); y = np.asarray(y)
        nW = len(self.W); nhid = nW - 1
        top = nhid - 1
        delta_out = _softmax(lg).copy(); delta_out[np.arange(len(y)), y] -= 1.0
        if mode == "wrong_sign":
            delta_out = -delta_out
        upd = [None] * nW
        upd[-1] = np.zeros_like(self.W[-1]) if mode == "wpi_warmup" else -(acts[-1].T @ delta_out)
        e_hid = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out
        if mode == "shufE":
            e_hid = e_hid[self._shuf_rng.permutation(e_hid.shape[0])]
        e_upper = e_hid
        # the learned-instructive sources at the TOP layer
        src_pred = _softmax(lg)
        src_target = np.zeros_like(src_pred); src_target[np.arange(len(y)), y] = 1.0
        for k in range(nhid - 1, -1, -1):
            E = acts[k + 1]
            Yk = np.zeros_like(self.Y[k]) if mode == "apical_lesion" else self.Y[k]
            if k == top and mode == "bdsp" and self.drives_ff:
                # LEARNED instructive signal drives the FF weights (the swap). apical_lesion zeroes both terms.
                Wpi = np.zeros_like(self.W_PI[top]) if mode == "apical_lesion" else self.W_PI[top]
                v_api = src_target @ Yk - src_pred @ Wpi
            else:
                v_api = e_upper @ Yk                                    # raw fixed FA (fixed_fa arm + lower layers)
            soma_err = (E * (1.0 - E)) * v_api
            soma_err = self._homeo_scale(k, soma_err)
            freeze = (mode == "reservoir") or (mode == "wpi_warmup") or (mode == "freeze_deepest" and k == 0)
            upd[k] = np.zeros_like(self.W[k]) if freeze else (acts[k].T @ soma_err)
            e_upper = soma_err
        # learned Eq.9 self-prediction (drives W^PI -> Y); frozen control skips it. wpi_warmup = interneuron converges
        # to its self-predicting fixed point with the FF weights FROZEN (the fast free-phase before the plastic phase).
        if self.wpi_plastic and (not self.wpi_frozen) and mode in ("bdsp", "wpi_warmup"):
            self._wpi_selfpredict_update(src_pred, lr)
        if mode in ("bdsp", "wpi_warmup"):
            self._selfpred_cos.append(_cos(self.W_PI[top], self.Y[top]))
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [np.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]


def _new_net(arm, sizes, seed, a):
    if arm == "transport_ceiling":
        return TransportCeilingNet(sizes, seed=seed, feedback="fixed", wpi_plastic=False, wpi_init="noisy")
    if arm == "fixed_fa":
        return MicroDrivesFFNet(sizes, seed=seed, feedback="fixed", wpi_plastic=False, drives_ff=False)
    if arm == "micro_drives_ff":
        return MicroDrivesFFNet(sizes, seed=seed, feedback="fixed", wpi_plastic=True,
                                wpi_init="noisy", wpi_lr=a.wpi_lr, drives_ff=True, wpi_frozen=False)
    if arm == "frozen_wpi_drives_ff":
        return MicroDrivesFFNet(sizes, seed=seed, feedback="fixed", wpi_plastic=True,
                                wpi_init="noisy", wpi_lr=a.wpi_lr, drives_ff=True, wpi_frozen=True)
    if arm == "fixedpoint_wpi_drives_ff":
        # W^PI == Y from the start, frozen there: residual == (onehot-softmax)@Y == the fixed_fa credit EXACTLY.
        # Verifies the wiring + isolates that any collapse of micro_drives_ff is noisy-init co-adaptation, not formulation.
        return MicroDrivesFFNet(sizes, seed=seed, feedback="fixed", wpi_plastic=True,
                                wpi_init="fixedpoint", wpi_lr=a.wpi_lr, drives_ff=True, wpi_frozen=True)
    # reservoir uses the fixed_fa net but trained in reservoir mode
    return MicroDrivesFFNet(sizes, seed=seed, feedback="fixed", wpi_plastic=False, drives_ff=False)


def _train_arm(arm, sizes, task, idx, seed, a):
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    net = _new_net(arm, sizes, seed, a)
    _train(net, Xtr, ytr, "reservoir" if arm == "reservoir" else "bdsp", a.epochs, a.lr, a.batch, seed)
    return net, {"inherit_heldout": float(_acc_on(net, Xte, yte, idx["inh_idx"])),
                 "train": float(net.accuracy(Xtr, ytr))}


def run_seed(seed, a):
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_semantic_inheritance(
        seed, n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
        n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
    task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    n_in = Xtr.shape[1]; k = meta["k_classes"]
    sizes = [n_in] + [a.hidden] * int(a.deep_layers) + [k]
    inh_idx = idx["inh_idx"]
    chance = float(max(np.mean(yte[inh_idx] == c) for c in np.unique(yte[inh_idx]))) if len(inh_idx) else float("nan")

    # oracle (task validity)
    onet = DendriticMLP(sizes, seed=seed)
    r = np.random.default_rng(seed + 777)
    for _ in range(a.epochs):
        p = r.permutation(len(ytr))
        for i in range(0, len(ytr), a.batch):
            onet.train_step(Xtr[p[i:i + a.batch]], ytr[p[i:i + a.batch]], mode="oracle", lr=a.lr)
    oracle = float(_acc_on(onet, Xte, yte, inh_idx))

    arms, nets = {}, {}
    for arm in ["reservoir", "fixed_fa", "micro_drives_ff", "frozen_wpi_drives_ff",
                "fixedpoint_wpi_drives_ff", "transport_ceiling"]:
        nets[arm], arms[arm] = _train_arm(arm, sizes, task, idx, seed, a)

    # NAMED BIOLOGICAL FIX TEST: warm up the interneuron (fast free-phase) to its self-predicting fixed point with the
    # FF weights FROZEN, THEN let the learned residual drive FF. Rescue => the crux is the interneuron TIMESCALE.
    wnet = _new_net("micro_drives_ff", sizes, seed, a)
    _train(wnet, Xtr, ytr, "wpi_warmup", a.warmup_epochs, a.lr, a.batch, seed)
    cos_after_warmup = float(wnet._selfpred_cos[-1]) if wnet._selfpred_cos else float("nan")
    _train(wnet, Xtr, ytr, "bdsp", a.epochs, a.lr, a.batch, seed)
    nets["micro_warmup_drives_ff"] = wnet
    arms["micro_warmup_drives_ff"] = float(_acc_on(wnet, Xte, yte, inh_idx))

    # apical-silent EARNED (the moat property): plastic micro vs frozen-noisy W^PI
    apical = {"micro_drives_ff": nets["micro_drives_ff"].apical_silent_stats(Xte, yte),
              "frozen_wpi": nets["frozen_wpi_drives_ff"].apical_silent_stats(Xte, yte)}

    # anti-cheat: permuted labels -> chance (on the micro_drives_ff arm)
    prng = np.random.default_rng(seed * 31 + 5)
    yperm = ytr[prng.permutation(len(ytr))]
    pnet = _new_net("micro_drives_ff", sizes, seed, a)
    _train(pnet, Xtr, yperm, "bdsp", a.epochs, a.lr, a.batch, seed)
    permuted = float(_acc_on(pnet, Xte, yte, inh_idx))

    # anti-cheat: no-weight-transport (Y never == a forward W/W^T); W^PI reads only activities (transport-free by code)
    nwt = bool(_no_weight_transport(nets["micro_drives_ff"]))
    ceiling_nwt = bool(_no_weight_transport(nets["transport_ceiling"]))   # MUST be False (the guard fires on the cheat)

    # ATTRIBUTION (tools.lab): whose is the effect, not merely both arms measured. The LEARNING is load-bearing
    # (plastic micro vs frozen-noisy W^PI) and the CREDIT is real (vs the credit-independent reservoir floor).
    from tools.lab import attributable_to
    _av = lambda a2: (arms[a2]["inherit_heldout"] if isinstance(arms[a2], dict) else arms[a2])
    attributable_to("learned residual drives FF (micro vs frozen-noisy W^PI)", _av("micro_drives_ff"), _av("frozen_wpi_drives_ff"))
    attributable_to("micro credit above the reservoir floor", _av("micro_drives_ff"), _av("reservoir"))

    return {"seed": seed, "chance": chance, "oracle": oracle,
            "arms": {a2: (arms[a2]["inherit_heldout"] if isinstance(arms[a2], dict) else arms[a2]) for a2 in arms},
            "cos_after_warmup": cos_after_warmup,
            "apical": apical, "permuted_micro": permuted,
            "nwt_micro": nwt, "ceiling_nwt": ceiling_nwt,
            "selfpred_cos_final": float(nets["micro_drives_ff"]._selfpred_cos[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--deep-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--wpi-lr", type=float, default=0.2)
    ap.add_argument("--warmup-epochs", type=int, default=80)
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    t0 = time.time()
    results = []
    for s in a.seeds:
        r = run_seed(s, a)
        results.append(r)
        print(f"[seed {s}] chance={r['chance']:.3f} oracle={r['oracle']:.3f} | "
              f"reservoir={r['arms']['reservoir']:.3f} fixed_fa={r['arms']['fixed_fa']:.3f} "
              f"micro_drives_ff={r['arms']['micro_drives_ff']:.3f} "
              f"frozen_wpi={r['arms']['frozen_wpi_drives_ff']:.3f} "
              f"fixedpoint_wpi={r['arms']['fixedpoint_wpi_drives_ff']:.3f} "
              f"micro_warmup={r['arms']['micro_warmup_drives_ff']:.3f}(cos_wu={r['cos_after_warmup']:.2f}) "
              f"ceiling={r['arms']['transport_ceiling']:.3f} | "
              f"silent(micro)={r['apical']['micro_drives_ff']['silent_ratio']:.3f} "
              f"silent(frozen)={r['apical']['frozen_wpi']['silent_ratio']:.3f} "
              f"cos={r['selfpred_cos_final']:.3f} perm={r['permuted_micro']:.3f} "
              f"nwt={r['nwt_micro']} ceilNWT={r['ceiling_nwt']}", flush=True)

    def _mean(key):
        return float(np.mean([r["arms"][key] for r in results]))
    agg = {a2: round(_mean(a2), 4) for a2 in
           ["reservoir", "fixed_fa", "micro_drives_ff", "frozen_wpi_drives_ff",
            "fixedpoint_wpi_drives_ff", "micro_warmup_drives_ff", "transport_ceiling"]}
    ff = np.array([r["arms"]["fixed_fa"] for r in results])
    mc = np.array([r["arms"]["micro_drives_ff"] for r in results])
    fz = np.array([r["arms"]["frozen_wpi_drives_ff"] for r in results])
    print("\n==== AGGREGATE (mean over seeds) ====", flush=True)
    print(f"  {agg}", flush=True)
    print(f"  micro>=fixed-0.03 in {int(np.sum(mc >= ff - 0.03))}/{len(mc)} seeds | "
          f"micro>reservoir margin mean {float(np.mean(mc - np.array([r['arms']['reservoir'] for r in results]))):+.3f} | "
          f"micro>frozen_wpi in {int(np.sum(mc > fz + 0.02))}/{len(mc)} seeds", flush=True)
    print(f"  apical-silent EARNED: micro ratio mean "
          f"{float(np.mean([r['apical']['micro_drives_ff']['silent_ratio'] for r in results])):.3f} "
          f"vs frozen {float(np.mean([r['apical']['frozen_wpi']['silent_ratio'] for r in results])):.3f} | "
          f"selfpred_cos mean {float(np.mean([r['selfpred_cos_final'] for r in results])):.3f}", flush=True)
    print(f"  permuted(micro) mean {float(np.mean([r['permuted_micro'] for r in results])):.3f} "
          f"(chance {float(np.mean([r['chance'] for r in results])):.3f}) | "
          f"nwt_micro all True={all(r['nwt_micro'] for r in results)} | "
          f"ceiling_nwt all False={all(not r['ceiling_nwt'] for r in results)}", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"agg": agg, "results": results}, indent=2))
        print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
