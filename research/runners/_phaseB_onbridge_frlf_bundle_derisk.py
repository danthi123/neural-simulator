"""On-bridge binding build STEP 1 (cheap-first, NO protected sim/ edit) — does the fixed-role + learned-filler
BUNDLED bind survive REAL LIF spiking on the SimulationBridge at the parity dimension?

The dendritic-build gate (the 6-seed A/B + the D_h capacity sweep, 2026-06-17) established: a FIXED self-inverse
role (+-1) + LEARNED filler codes (+ a learned read-out) bundles multi-attribute facts to ~1.000 at D_h=256
(numpy) -- full parity with the fixed FHRR algebra, where a learned LINEAR-inverse and additive could not. The
owner greenlit the on-bridge binding build. Its FIRST step reuses the validated on-bridge ON/OFF LIF substrate
(`_phaseB_onbridge_bind_nonlinearity_derisk`: drive `bind_pos` with relu(h), `bind_neg` with relu(-h), read
per-neuron spike RATES = the spiking ON/OFF bound) -- but with the FIXED +-1 role + LEARNED filler (the
`FixedRoleLearnedFillerBinder` that reached parity) and the BUNDLED (3-way superposition) eval, at D_h=256.

THE QUESTION: does the real LIF spiking ON/OFF rate-code (threshold/refractory/finite-count dynamics, population
code N_PER per dim for SNR) preserve the FR+LF BUNDLED held-out recall vs the numpy reference? If yes, the binding
is already realizable on the substrate with the existing ON/OFF infrastructure + external-current drive -- the
+-1 product is a fixed per-dim sign/channel-swap (linear routing), so the genuine on-substrate question is whether
the spiking superposition survives, which the population rate-code should carry (the CYCLE-91 lift). A GO means
the protected sim/ edit (if any) is minimal wiring, not a new mechanism. If it degrades, localize (rate band /
population size / the coincidence-plateau cleanup of the bundled superposition) before committing the wiring.

GATE (3 seeds, escalate to 6 on GO): on-bridge (LIF-rate ON/OFF) FR+LF BUNDLED held-out ~ the numpy FR+LF
reference (>= 0.75x) AND >> memorization floor. Reuse-by-import (FixedRoleLearnedFillerBinder + the on-bridge LIF
driver + the systematicity protocol). GPU (the real bridge).
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_frlf_bundle_derisk [--dh 256] [--seeds 42,43,44]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax, score_memorization_floor)
from research.runners._phaseB_fixed_role_learned_filler_bundling_derisk import FixedRoleLearnedFillerBinder  # noqa: E402
import research.runners._phaseB_onbridge_bind_nonlinearity_derisk as onb  # noqa: E402

R, F, N_SPLITS = 4, 16, 2          # 2 splits (a GPU bridge per combo is the cost; matches the on-bridge harness)
N_FACT_STEPS = 24000               # bundle-aware training (== the A/B)
N_EVAL_FACTS = 40                  # bundled eval facts per split (== the A/B)
LR = 0.005


def build_frlf_binder(codes, seed, D_h):
    """Train a FixedRoleLearnedFillerBinder (fixed +-1 role, learned W_F/W_O) bundle-aware, per the A/B."""
    fillers = codes[:F]
    D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng_pm1 = np.random.default_rng(seed * 31 + 5)
    R_proj = rng_pm1.standard_normal((D_in, D_h)) / np.sqrt(D_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)          # [R, D_h] in {+-1} -- fixed self-inverse
    return role_pm1, fillers


def run_split(codes, seed, D_h, split, bridge, pos_idx, neg_idx):
    fillers = codes[:F]
    D_in = fillers.shape[1]
    role_pm1, _ = build_frlf_binder(codes, seed, D_h)
    rng = np.random.default_rng(seed * 53 + 9)
    tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
    if min(len(tr_by_role[r]) for r in range(3)) == 0:
        return None
    binder = FixedRoleLearnedFillerBinder(D_in=D_in, role_pm1=role_pm1, D_h=D_h, lr=LR, lam=1e-4, seed=seed)
    for _ in range(N_FACT_STEPS):
        fa = rng.choice(tr_by_role[0]); fv = rng.choice(tr_by_role[1]); fo = rng.choice(tr_by_role[2])
        binder.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], fillers, int(rng.integers(3)))

    def analog_bundle(fids):
        return sum(role_pm1[r] * (fillers[fids[r]] @ binder.W_F) for r in range(3))   # signed [D_h]

    # calibrate the LIF rate magnitude to the analog bundle the unbind was trained on (one demo bundle)
    demo = analog_bundle(rng.choice(F, 3, replace=False))
    np_mag = float(np.mean(np.abs(demo)) + 1e-9)
    lif_demo = onb.lif_onoff(bridge, pos_idx, neg_idx, demo, onb.DRIVE_SCALE)          # [2*D_h] on/off rates
    lif_mag = float(np.mean(lif_demo) + 1e-9)
    cal = np_mag / lif_mag

    train_set = set(split["train"])
    ob_tr_ok = ob_tr = ob_h_ok = ob_h = nn_h_ok = nn_h = 0
    for _ in range(N_EVAL_FACTS):
        fids = rng.choice(F, 3, replace=False)
        bundle_np = analog_bundle(fids)                                               # numpy reference bundle
        lif = onb.lif_onoff(bridge, pos_idx, neg_idx, bundle_np, onb.DRIVE_SCALE) * cal
        bundle_spk = (lif[:D_h] - lif[D_h:])                                          # reconstruct signed bundle
        for r in range(3):
            ob = int(native_argmax(binder.unbind(bundle_spk, r), fillers) == fids[r])  # on-bridge (spiking) unbind
            nn = int(native_argmax(binder.unbind(bundle_np, r), fillers) == fids[r])   # numpy reference unbind
            if (r, int(fids[r])) in train_set:
                ob_tr_ok += ob; ob_tr += 1
            else:
                ob_h_ok += ob; ob_h += 1
                nn_h_ok += nn; nn_h += 1
    memf = score_memorization_floor(split["train"], split["held_out"], fillers)["held_out_acc"]
    return {"ob_held": ob_h_ok / ob_h if ob_h else 0.0,
            "ob_train": ob_tr_ok / ob_tr if ob_tr else 0.0,
            "nn_held": nn_h_ok / nn_h if nn_h else 0.0, "mem_floor": memf}


def run_seed(codes, seed, D_h):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    bridge, pos_idx, neg_idx = onb.build_bind_bridge(D_h, seed)
    obh, obt, nnh, mf = [], [], [], []
    for split in splits:
        r = run_split(codes, seed, D_h, split, bridge, pos_idx, neg_idx)
        if r is None:
            continue
        obh.append(r["ob_held"]); obt.append(r["ob_train"]); nnh.append(r["nn_held"]); mf.append(r["mem_floor"])
    row = {"seed": seed, "ob_held": float(np.mean(obh)), "ob_train": float(np.mean(obt)),
           "nn_held": float(np.mean(nnh)), "mem_floor": float(np.mean(mf))}
    print(f"  [seed {seed} D_h={D_h}] ON-BRIDGE FR+LF bundled held-out {row['ob_held']:.3f} "
          f"(train {row['ob_train']:.3f}) | numpy reference {row['nn_held']:.3f} | mem-floor {row['mem_floor']:.3f}",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dh", type=int, default=256)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw", "_phaseB_onbridge_frlf_bundle.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[on-bridge FR+LF bundle de-risk] does the REAL LIF spiking ON/OFF preserve the fixed-role+learned-filler "
          f"BUNDLED bind at D_h={args.dh} (numpy parity ~1.000)? seeds={seeds}", flush=True)
    rows = [run_seed(codes, s, args.dh) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    ob, nn, mf = m("ob_held"), m("nn_held"), m("mem_floor")
    chance = 1.0 / F
    print(f"\n{'='*98}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds, D_h={args.dh}): ON-BRIDGE FR+LF bundled held-out {ob:.3f} | "
          f"numpy reference {nn:.3f} | mem-floor {mf:.3f} | chance {chance:.3f}", flush=True)
    go = ob >= mf + 0.25 and ob >= 0.75 * nn
    if go:
        print(f"  GO: the real LIF spiking ON/OFF preserves the FR+LF BUNDLED bind -- on-bridge {ob:.3f} = "
              f"{ob/max(nn,1e-9):.0%} of the numpy reference ({nn:.3f}), >> mem-floor {mf:.3f}. The fixed-role + "
              f"learned-filler bundling survives real spiking at the parity dimension. ==> the on-bridge binding is "
              f"realizable on the existing ON/OFF substrate; the protected wiring (production composer path) is the "
              f"next step, not a new mechanism.", flush=True)
    elif ob >= mf + 0.10:
        print(f"  PARTIAL: the spiking ON/OFF partly preserves the bundled bind ({ob:.3f} vs numpy {nn:.3f}) -- "
              f"localize the rate band / population size / a coincidence-plateau cleanup of the superposition.",
              flush=True)
    else:
        print(f"  NEGATIVE: the spiking ON/OFF does NOT preserve the bundled bind ({ob:.3f} vs floor {mf:.3f}) -- "
              f"the bundled superposition hits the rate-code wall at this D_h/population; needs the coincidence-"
              f"plateau nonlinearity (the protected edit) to clean the superposition.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*98}", flush=True)
    out = {"verdict": "GO" if go else ("PARTIAL" if ob >= mf + 0.10 else "NEGATIVE"),
           "D_h": args.dh, "seeds": seeds, "ob_held": ob, "nn_held": nn, "mem_floor": mf, "chance": chance,
           "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
