"""gap#5 EMERGENT-DG — R0 (risk-first): does a CA3 assembly EMERGE from an input via the trisynaptic loop
(language_input -> ec -> dg -> mossy -> ca3), instead of a pre-assigned random mask? Reuses the existing EC->DG->CA3
wiring (`_build`) + the drive/read pattern; NO sim/ edit, NO BTSP/recall yet (that is R1). Per the emergent-DG scoping
(`2026-07-18-gap5-emergent-DG-scoping.md`).

MECHANISM: drive a sparse input on language_input, run the feedforward loop (learning OFF), read the NATURAL CA3
assembly A_m = {ca3 cells firing in >= theta of the drive steps} (NOT top-k -- top-k would launder the sparsity claim).

R0 GO (6-seed): the DG-selected assembly is SPARSE (|A_m|/n_ca3 in [0.005, 0.08]), RIGHT-SIZED (|A_m| ~ 6-30), STABLE
(re-present same input -> Jaccard >= 0.6), and SEPARATED (distinct inputs -> assembly cos < 0.4). Anti-cheats: NO-INPUT
-> |A_noinput| <= 0.2*mean|A_m| (the moat); PERMUTE-INPUT -> a DIFFERENT assembly (cos < 0.4 vs original = input-driven,
not hand-assigned); DG/MOSSY-LESION (mossy_weight=0) -> no CA3 assembly (provenance). Run (GPU):
  SIM_BACKEND=cupy python -m research.runners._gap5_emergent_dg_selection_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._riii_ca3_coincidence_completion_derisk import _build  # noqa: E402
from research.runners.validate_trisynaptic_loop import build_drive_pattern  # noqa: E402

cp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_gap5_emergent_dg_selection.json"


def _build_bridge(seed, n_ca3, dg_ffi_weight, ca3_fb_inhib, mossy_weight, mossy_density, n_lang=384, n_dg=300,
                  ca3_ff_inhib=None, amplify=False, amp_ca3w=4.0):
    if amplify:
        # LAYER-2 AMPLIFICATION (2026-07-19): the layer-1 default (ca3w=1.5, coincidence/train off) gives 0 CA3 firing
        # (raw R0 boundary). The finding's 15-26-cell selection needs a MODERATE recurrent (ca3w~4) + the dendritic-
        # plateau coincidence read + two-compartment + plastic recurrent + the bistability keystone, so a synchronized
        # mossy seed AMPLIFIES into a sparse assembly. NO sim/ edit (all _build params).
        b = _build(seed, n_lang=n_lang, n_dg=n_dg, n_ca3=n_ca3, ca3_density=0.05, ca3w=float(amp_ca3w),
                   coincidence=True, two_comp=True, train=True, ca3_fb_inhib=ca3_fb_inhib, dg_ffi_weight=dg_ffi_weight,
                   mossy_weight=mossy_weight, mossy_density=mossy_density, enable_ou=False, ca3_ff_inhib=ca3_ff_inhib,
                   plateau_self_regen=0.15, apical_kir_g=3.0)
    else:
        b = _build(seed, n_lang=n_lang, n_dg=n_dg, n_ca3=n_ca3, ca3_density=0.05, ca3w=1.5, coincidence=False,
                   two_comp=False, train=False, ca3_fb_inhib=ca3_fb_inhib, dg_ffi_weight=dg_ffi_weight,
                   mossy_weight=mossy_weight, mossy_density=mossy_density, enable_ou=False, ca3_ff_inhib=ca3_ff_inhib)
    b.core_config.enable_hebbian_learning = False   # read pass: no plasticity (the feedforward always conducts)
    return b


def _select(bridge, lang, ca3_arr, dg_arr, pat_idx, drive_pA=200.0, n_events=6, reset_steps=10, drive_steps=40, theta=0.3,
            sync=False, g_on=3, g_off=3):
    """Drive language_input[pat_idx], run the loop, return (assembly set A over ca3-LOCAL indices, ca3 rate vec, dg_sparsity).
    sync=True gamma-pulses the drive (g_on on / g_off off) -> a SYNCHRONIZED DG volley (the finding's amplification prereq:
    coincident mossy fibers detonate CA3)."""
    lang_arr = np.asarray(lang, dtype=np.int64)
    drv = cp.asarray(lang_arr[pat_idx], dtype=cp.int64) if len(pat_idx) else None
    ca3_spk = cp.zeros(len(ca3_arr), dtype=cp.float32); dg_spk = cp.zeros(len(dg_arr), dtype=cp.float32); nrec = 0
    _period = g_on + g_off
    for ev in range(n_events):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
        for _t in range(drive_steps):
            bridge.cp_external_input_current[:] = 0.0
            _drive_now = (not sync) or ((_t % _period) < g_on)     # sync: gamma-pulsed volley
            if drv is not None and _drive_now:
                bridge.cp_external_input_current[drv] = float(drive_pA)
            bridge._run_one_simulation_step()
            if ev >= n_events - 3:                        # record the last 3 events (settled)
                ca3_spk += bridge.cp_firing_states[cp.asarray(ca3_arr)].astype(cp.float32)
                dg_spk += bridge.cp_firing_states[cp.asarray(dg_arr)].astype(cp.float32)
                nrec += 1
    bridge.cp_external_input_current[:] = 0.0
    ca3_rate = np.asarray(to_host(ca3_spk)) / max(1, nrec)
    dg_rate = np.asarray(to_host(dg_spk)) / max(1, nrec)
    A = set(int(i) for i in np.where(ca3_rate >= theta)[0])   # NATURAL assembly (>= theta firing), NOT top-k
    dg_sparsity = float(np.mean(dg_rate >= theta))
    return A, ca3_rate, dg_sparsity


def _jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 1e-9 and nv > 1e-9 else 0.0


def run(seed, n_ca3=400, n_inputs=4, dg_ffi_weight=6.0, ca3_fb_inhib=20.0, mossy_weight=8.0, mossy_density=0.10,
        ca3_ff_inhib=None, amplify=False, amp_ca3w=4.0, sync=False, sel_drive=200.0):
    import functools
    _sel = functools.partial(_select, sync=sync, drive_pA=sel_drive)
    b = _build_bridge(seed, n_ca3, dg_ffi_weight, ca3_fb_inhib, mossy_weight, mossy_density, ca3_ff_inhib=ca3_ff_inhib,
                      amplify=amplify, amp_ca3w=amp_ca3w)
    rm = b.region_manager
    lang = list(rm.indices("language_input")); ca3_arr = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    dg_arr = np.asarray(list(rm.indices("dg")), dtype=np.int64)
    pats = [build_drive_pattern(len(lang), 0.1, seed * 100 + m) for m in range(n_inputs)]
    sel = [_sel(b, lang, ca3_arr, dg_arr, p) for p in pats]
    A = [s[0] for s in sel]; rates = [s[1] for s in sel]; dg_sp = float(np.mean([s[2] for s in sel]))
    sizes = [len(a) for a in A]
    ca3_sparsity = float(np.mean([len(a) / len(ca3_arr) for a in A]))
    # stability: re-present each input from a fresh run -> Jaccard
    sel2 = [_sel(b, lang, ca3_arr, dg_arr, p) for p in pats]
    stability = float(np.mean([_jacc(A[m], sel2[m][0]) for m in range(n_inputs) if len(A[m]) or len(sel2[m][0])] or [0.0]))
    # separation: pairwise cos of the rate vectors + Jaccard of the assemblies
    pair_cos, pair_jac = [], []
    for i in range(n_inputs):
        for j in range(i + 1, n_inputs):
            pair_cos.append(_cos(rates[i], rates[j])); pair_jac.append(_jacc(A[i], A[j]))
    sep_cos = float(np.mean(pair_cos)) if pair_cos else 0.0; sep_jac = float(np.mean(pair_jac)) if pair_jac else 0.0
    # anti-cheats
    A_noin, _, _ = _sel(b, lang, ca3_arr, dg_arr, np.array([], dtype=np.int64))           # no-input -> the moat
    perm = np.random.default_rng(seed + 555).permutation(len(lang))[:len(pats[0])]           # a scrambled input
    A_perm, r_perm, _ = _sel(b, lang, ca3_arr, dg_arr, perm)
    perm_cos = _cos(rates[0], r_perm)                                                         # different from input 0 => input-driven
    bl = _build_bridge(seed, n_ca3, dg_ffi_weight, ca3_fb_inhib, 0.0, mossy_density, ca3_ff_inhib=ca3_ff_inhib,
                       amplify=amplify, amp_ca3w=amp_ca3w)  # mossy-LESION (dg->ca3 weight 0)
    rm2 = bl.region_manager
    A_les, _, _ = _sel(bl, list(rm2.indices("language_input")),
                          np.asarray(list(rm2.indices("ca3")), dtype=np.int64),
                          np.asarray(list(rm2.indices("dg")), dtype=np.int64), pats[0])
    mean_size = float(np.mean(sizes)) if sizes else 0.0
    return {"seed": seed, "dg_sparsity": dg_sp, "ca3_sizes": sizes, "mean_size": mean_size,
            "ca3_sparsity": ca3_sparsity, "stability": stability, "sep_cos": sep_cos, "sep_jac": sep_jac,
            "noinput_size": len(A_noin), "perm_cos": perm_cos, "lesion_size": len(A_les)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-ca3", type=int, default=400)
    ap.add_argument("--dg-ffi", type=float, default=6.0)
    ap.add_argument("--ca3-fb", type=float, default=20.0)
    ap.add_argument("--ca3-ff", dest="ca3_ff", type=float, default=None,
                    help="E%-max FEEDFORWARD ca3 inhibition (dg-afferent-driven basket) -> robust sparse selection across "
                         "inputs (emergent-DG fragility fix). None (default) = feedback-only (byte-identical).")
    ap.add_argument("--mossy-w", type=float, default=8.0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s, n_ca3=a.n_ca3, dg_ffi_weight=a.dg_ffi, ca3_fb_inhib=a.ca3_fb, mossy_weight=a.mossy_w,
                    ca3_ff_inhib=a.ca3_ff)
            per.append(r)
            print(f"  [seed {s}] dg_sp {r['dg_sparsity']:.3f} | ca3 size {r['mean_size']:.1f} sparsity {r['ca3_sparsity']:.3f} "
                  f"| stability {r['stability']:.2f} | sep_cos {r['sep_cos']:.2f} sep_jac {r['sep_jac']:.2f} || "
                  f"noinput {r['noinput_size']} perm_cos {r['perm_cos']:.2f} lesion {r['lesion_size']} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mean(k): return float(np.mean([p[k] for p in per]))
        msz, msp, mst, mco = mean("mean_size"), mean("ca3_sparsity"), mean("stability"), mean("sep_cos")
        mni, mpc, mle = mean("noinput_size"), mean("perm_cos"), mean("lesion_size")
        sparse = all(0.005 <= p["ca3_sparsity"] <= 0.08 and 6 <= p["mean_size"] <= 30 for p in per)
        stable = all(p["stability"] >= 0.6 for p in per)
        separated = all(p["sep_cos"] < 0.4 for p in per)
        moat = all(p["noinput_size"] <= max(1, 0.2 * p["mean_size"]) for p in per)
        input_driven = all(p["perm_cos"] < 0.5 for p in per)
        lesion_ok = all(p["lesion_size"] <= max(1, 0.2 * p["mean_size"]) for p in per)
        go = bool(sparse and stable and separated and moat and input_driven and lesion_ok)
        if go:
            verdict = (f"GO -- a CA3 assembly EMERGES from the input via the trisynaptic loop: SPARSE (size {msz:.1f}, "
                       f"sparsity {msp:.3f}), STABLE (re-present Jaccard {mst:.2f}), SEPARATED (distinct inputs -> "
                       f"assembly cos {mco:.2f} < 0.4). Input-driven (permute-input cos {mpc:.2f}), no-input moat "
                       f"({mni:.1f}), mossy-lesion collapses ({mle:.1f}). 6-seed. => the assembly is DG-SELECTED from "
                       f"experience, not a hand-assigned mask. NEXT: R1 = BTSP-store the emergent assembly + bistable complete.")
        else:
            miss = []
            if not sparse: miss.append(f"not sparse/right-sized (size {msz:.1f}, sparsity {msp:.3f})")
            if not stable: miss.append(f"not stable (Jaccard {mst:.2f})")
            if not separated: miss.append(f"not separated (cos {mco:.2f})")
            if not moat: miss.append(f"no-input leak ({mni:.1f})")
            if not input_driven: miss.append(f"not input-driven (perm_cos {mpc:.2f})")
            if not lesion_ok: miss.append(f"mossy-lesion didn't collapse ({mle:.1f})")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". Per THE LAW: sweep dg_ffi/ca3_fb/mossy for the sparsity+"
                       "separation working point, NOT a stop.")
    else:
        go = False; verdict = f"ERROR -- {err}"
    summary = {"probe": "gap5_emergent_dg_selection_R0", "GO": go, "verdict": verdict, "seeds": a.seeds,
               "config": {"n_ca3": a.n_ca3, "dg_ffi": a.dg_ffi, "ca3_fb": a.ca3_fb, "mossy_w": a.mossy_w},
               "elapsed_seconds": round(time.time()-t0, 1), "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[emergent-dg-R0] VERDICT: {verdict}\n[emergent-dg-R0] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
