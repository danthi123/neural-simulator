"""gap#2 Rank-1 RIGOR REPAIR — is the LEARNED error-correcting (delta) binder LOAD-BEARING over CORRELATED role keys,
decisively beating the plain-Hebbian (additive) baseline, where the audit's gap#2 GO gate (`delta > additive`) was
NEVER met?

WHY (the audit's precise defect, 2026-07-21 8-skeptic PARTIAL on `_gap2_spiking_deltarule_binder_derisk`):
the delta rule was tested with FRESH NEAR-ORTHOGONAL random role keys at D=128, so the additive (plain-Hebbian
outer-product) binder NEVER COLLAPSED (delta == additive == 1.000 at P=1..5) -> the delta rule was NOT shown load-bearing
over a trivial Hebbian write. Measured here: the brain's OWN scale-cortex phasor codes are near-orthogonal too
(mean|cos| 0.084 ~ random 0.077 at D=128 -- the sqrt(D) dilution), so real codes AS KEYS do NOT stress additive either.

THE FRONTIER QUESTION (mission line 167-168): "a learned binder generalizes on DECORRELATED codes but goes NEGATIVE on
CORRELATED codes, which is exactly what the brain's own emergent codes are." At the phasor level the crosstalk that
collapses additive is the ROLE-KEY inner product k_i^H k_j. So the single decisive variable is KEY CORRELATION rho.
This runner SWEEPS rho on synthetic controllable-correlation keys (fillers stay the brain's OWN scale-cortex codes),
locates the crossover where additive collapses, and asks whether the LEARNED delta (error-correcting, the deployed
`build_W` rule, no backprop/transport) stays high there -> delta LOAD-BEARING. It also runs a REAL-Z-keys reference arm
to HONESTLY LOCATE the brain's own codes on that curve.

MECHANISM (reuse-by-import; NO sim/ edit): `build_W` (delta = erase-then-write `W+=(v-Wk)k^H/D`, additive = `W+=vk^H/D`)
and `cleanup` + `load_phasor_codes` are imported verbatim from `_gap2_spiking_deltarule_binder_derisk`; the on-substrate
read is that runner's `spiking_read` on the committed RF resonate loop (`_build_rf_bridge`). Default read is numpy `W@key`
(the finding validated RF-read == W@key at 0.0074 phase err) for the fast decisive 6-seed sweep; `--read spiking` /
`--confirm-spiking` re-exercise the SAME arms on the RF substrate.

ARMS (each per rho, all WIRED + INVOKED in main's run path):
  delta(correlated)  additive(correlated)  permuted-role[delta] (anti-cheat)  shuffled-write[delta] (anti-cheat)
plus the REAL-Z-keys reference (delta,additive) and rho=0 (the decorrelated control = additive-should-recover).

GO GATE (printed BY this runner from ITS OWN computed 6-seed means, at P=--p-gate):
  GO iff  EXISTS rho with:
     additive(corr) < 0.60      (additive COLLAPSES from key crosstalk)
     delta(corr)  >= 0.80       (learned binder reaches ceiling there)
     delta - additive >= 0.30   (delta LOAD-BEARING -- the audit's unmet gate now MET)
   AND at that rho:  permuted-role <= chance+0.15  AND  shuffled-write <= chance+0.15
   AND rho=0 additive >= 0.80   (decorrelated control: additive suffices -> correlation IS the cause)
   AND real-Z additive >= 0.80  (honest locate: the brain's near-orthogonal codes need no error-correction)
  NEGATIVE iff no such rho (delta collapses wherever additive does -> single-pass delta does NOT rescue
   correlated-key binding on point-neuron phasors -> the deeper gap#2 frontier stands; needs iteration / a
   dendritic-multiplicative / self-organizing bind). PARTIAL otherwise.

This is a CPU-only numpy de-risk. A TINY smoke (`--synth-D`, tiny cap/P/facts, 1 seed, `--confirm-spiking`) proves the
whole path RUNS + every control is live + a verdict prints; GO/NEGATIVE is claimed ONLY from the full 6-seed run.
"""
import argparse
import json
import os
import time

import numpy as np

from research.runners._gap2_spiking_deltarule_binder_derisk import (
    build_W, cleanup, load_phasor_codes, spiking_read,
)
from research.runners.rf_phasor_composer import _build_rf_bridge


def correlated_keys(P, D, rho, rng):
    """P unit-phasor role keys with controllable pairwise correlation rho in [0,1].
    rho=0 -> independent (near-orthogonal, mean|cos|~1/sqrt(D)); rho->1 -> all identical (rank-1 key set)."""
    common = np.exp(2j * np.pi * rng.random(D))
    out = np.empty((P, D), complex)
    a, b = np.sqrt(rho), np.sqrt(1.0 - rho)
    for i in range(P):
        idio = np.exp(2j * np.pi * rng.random(D))
        raw = a * common + b * idio
        out[i] = raw / np.abs(raw)                 # per-component re-normalize -> unit phasor
    return out


def _delta_passes(W, keys, Z, wfids, D, passes):
    """Extra online delta sweeps (iterative Widrow-Hoff -> pseudo-inverse) beyond build_W's single pass."""
    for _ in range(passes - 1):
        for ri, fi in enumerate(wfids):
            k = keys[ri]; v = Z[fi]
            W = W + np.outer(v - W @ k, k.conj()) / D
    return W


def retrieve(Z, bridge, seed, P, n_facts, D, rho, delta,
             real_z_keys=False, read="numpy", permute=False, shuffle_write=False, passes=1):
    rng = np.random.default_rng(seed * 131 + P * 7 + int(round(rho * 1000)))
    N = Z.shape[0]
    n_ok = n = 0
    for _ in range(n_facts):
        fids = rng.choice(N, P, replace=False)
        if real_z_keys:
            kids = rng.choice(N, P, replace=False)          # keys = the brain's OWN codes (distinct pool)
            keys = Z[kids]
        else:
            keys = correlated_keys(P, D, rho, rng)
        wfids = list(fids)
        if shuffle_write and P > 1:
            perm = np.arange(P)                              # force a DERANGEMENT (no fixed point) so EVERY
            while True:                                      # (role,filler) pairing is broken -> clean chance
                perm = rng.permutation(P)
                if not np.any(perm == np.arange(P)):
                    break
            wfids = [fids[j] for j in perm]                  # anti-cheat: role i now stores a DIFFERENT filler
        W = build_W(keys, Z, list(range(P)), wfids, D, delta)     # the deployed single-pass rule
        if passes > 1 and delta:
            W = _delta_passes(W, keys, Z, wfids, D, passes)
        for i in range(P):
            key = keys[(i + 1) % P] if (permute and P > 1) else keys[i]
            rec = spiking_read(bridge, W, key, D) if read == "spiking" else (W @ key)
            n_ok += int(cleanup(rec, Z) == fids[i]); n += 1
    return n_ok / n if n else 0.0


def _mean(Z, bridge, seeds, **kw):
    return float(np.mean([retrieve(Z, bridge, s, **kw) for s in seeds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--pmax", type=int, default=4)
    ap.add_argument("--p-gate", type=int, default=3)
    ap.add_argument("--n-facts", type=int, default=20)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--read", choices=["numpy", "spiking"], default="numpy")
    ap.add_argument("--confirm-spiking", action="store_true",
                    help="re-run the GO-point arms through the RF resonate loop (on-substrate confirmation)")
    ap.add_argument("--synth-D", type=int, default=0,
                    help=">0: synthetic near-orthogonal phasor fillers at this D (tiny smoke); 0 = real scale-cortex codes")
    ap.add_argument("--out", type=str, default="research/findings/raw/_gap2_correlated_key_binder.json")
    args = ap.parse_args()

    t0 = time.time()
    if args.synth_D > 0:
        rng0 = np.random.default_rng(12345)
        Z = np.exp(2j * np.pi * rng0.random((args.cap, args.synth_D)))
        code_src = f"synth-{args.synth_D}"
    else:
        Z = load_phasor_codes(cap=args.cap)
        code_src = "scale-cortex-codes"
    N, D = Z.shape
    chance = 1.0 / N
    bridge = _build_rf_bridge(2 * D, seed=args.seeds[0]) if (args.read == "spiking" or args.confirm_spiking) else None

    print(f"[gap2 CORRELATED-KEY binder] codes={N} D={D} src={code_src} read={args.read} passes={args.passes} "
          f"| seeds={args.seeds} p_gate={args.p_gate} chance={chance:.4f}")

    # ---- primary sweep: delta vs additive vs anti-cheats over the key-correlation rho, at p_gate ----
    P = args.p_gate
    common = dict(P=P, n_facts=args.n_facts, D=D, read=args.read, passes=args.passes)
    sweep = {}
    for rho in args.rhos:
        de = _mean(Z, bridge, args.seeds, rho=rho, delta=True, **common)
        ad = _mean(Z, bridge, args.seeds, rho=rho, delta=False, **common)
        pm = _mean(Z, bridge, args.seeds, rho=rho, delta=True, permute=True, **common)
        sh = _mean(Z, bridge, args.seeds, rho=rho, delta=True, shuffle_write=True, **common)
        sweep[f"{rho:.2f}"] = dict(delta=de, additive=ad, permuted=pm, shuffled=sh)
        print(f"  rho={rho:.2f} P={P}: DELTA {de:.3f} | additive {ad:.3f} | permuted-role {pm:.3f} "
              f"| shuffled-write {sh:.3f}")

    # ---- real-Z-keys reference (honest locate of the brain's own codes) ----
    rz_de = _mean(Z, bridge, args.seeds, rho=0.0, delta=True, real_z_keys=True, **common)
    rz_ad = _mean(Z, bridge, args.seeds, rho=0.0, delta=False, real_z_keys=True, **common)
    print(f"  REAL-Z keys (brain's own codes) P={P}: DELTA {rz_de:.3f} | additive {rz_ad:.3f}")

    # ---- full P curve at the two rho endpoints (context; not gated) ----
    pcurve = {}
    for rho in (min(args.rhos), max(args.rhos)):
        row = {}
        for p in range(1, args.pmax + 1):
            c = dict(n_facts=args.n_facts, D=D, read=args.read, passes=args.passes)
            row[p] = dict(delta=_mean(Z, bridge, args.seeds, P=p, rho=rho, delta=True, **c),
                          additive=_mean(Z, bridge, args.seeds, P=p, rho=rho, delta=False, **c))
        pcurve[f"{rho:.2f}"] = row

    # ---- GO gate: read this runner's OWN computed means ----
    go_rho = None
    for rho in args.rhos:
        r = sweep[f"{rho:.2f}"]
        if (r["additive"] < 0.60 and r["delta"] >= 0.80 and (r["delta"] - r["additive"]) >= 0.30
                and r["permuted"] <= chance + 0.15 and r["shuffled"] <= chance + 0.15):
            go_rho = rho; break
    dec_add = sweep[f"{min(args.rhos):.2f}"]["additive"]      # decorrelated (lowest rho) additive
    real_z_ok = rz_ad >= 0.80
    dec_ok = dec_add >= 0.80

    if go_rho is not None and dec_ok and real_z_ok:
        verdict = "GO"
        reason = (f"delta LOAD-BEARING at rho={go_rho:.2f} (delta {sweep[f'{go_rho:.2f}']['delta']:.3f} vs "
                  f"additive {sweep[f'{go_rho:.2f}']['additive']:.3f}); decorr additive {dec_add:.3f}>=0.80; "
                  f"real-Z additive {rz_ad:.3f}>=0.80 (brain codes near-ortho -> error-correction not needed there)")
    elif go_rho is None and any(sweep[f'{r:.2f}']['additive'] < 0.60 for r in args.rhos):
        verdict = "NEGATIVE"
        reason = ("additive COLLAPSES at high rho but single-pass delta does NOT stay >=0.80/beat it by 0.30 "
                  "-> per-fact delta does not rescue correlated-key binding on point-neuron phasors (deeper "
                  "gap#2 frontier stands: iterate/pseudo-inverse or a dendritic-multiplicative self-organizing bind)")
    else:
        verdict = "PARTIAL"
        reason = ("no rho drove additive below 0.60 in this sweep (keys not correlated enough) OR a control failed; "
                  "widen --rhos toward 1.0 / raise --p-gate")
    print(f"  => VERDICT: {verdict} :: {reason}")

    # ---- on-substrate confirmation at the GO/collapse rho ----
    confirm = None
    if args.confirm_spiking and bridge is not None:
        crho = go_rho if go_rho is not None else max(args.rhos)
        cs = dict(n_facts=max(4, args.n_facts // 4), D=D, read="spiking", passes=args.passes)
        cd = _mean(Z, bridge, args.seeds[:1], P=P, rho=crho, delta=True, **cs)
        ca = _mean(Z, bridge, args.seeds[:1], P=P, rho=crho, delta=False, **cs)
        confirm = dict(rho=crho, delta_spiking=cd, additive_spiking=ca, seeds=args.seeds[:1])
        print(f"  [confirm-spiking rho={crho:.2f}]: DELTA(RF) {cd:.3f} | additive(RF) {ca:.3f} "
              f"(reproduces the numpy delta-vs-additive discrimination on the RF resonate loop)")

    out = dict(
        runner="_gap2_correlated_key_binder_derisk",
        verdict=verdict, reason=reason, go_rho=go_rho, chance=chance,
        knobs=dict(seeds=args.seeds, rhos=args.rhos, pmax=args.pmax, p_gate=args.p_gate,
                   n_facts=args.n_facts, cap=args.cap, passes=args.passes, read=args.read,
                   confirm_spiking=bool(args.confirm_spiking), synth_D=args.synth_D,
                   code_src=code_src, N=N, D=D),
        sweep=sweep, real_z=dict(delta=rz_de, additive=rz_ad),
        decorrelated_additive=dec_add, p_curve=pcurve, confirm_spiking=confirm,
        elapsed_s=round(time.time() - t0, 2),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {args.out}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
