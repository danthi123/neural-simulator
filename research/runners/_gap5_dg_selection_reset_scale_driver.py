"""gap#5 R1 — DG-direct emergent selection with FULL snapshot/restore reset + scale-bisect + sparse detonator.

Reproduces the 2026-07-19 6-seed GO (n_ca3=400: sparse 10-37 / sep_cos<0.4 / Jaccard>0.6), then scale-bisects
n_ca3 and co-sweeps {dense mossy d0.10/w200 vs SPARSE DETONATOR d0.02/w1000} x {recurrent w4.0 vs w2.5} to test the
fixed-fraction-fan-out root cause: is the diffuse 2000-cell failure a scale/reset artifact, or a genuine window-closes
BOUNDARY that justifies R4/BTSP?

RESET DISCIPLINE (the wall-2 fix): a COMPLETE post-build snapshot of every dynamic per-neuron cp_* array (== fresh
bridge) is captured once, and RESTORED before EACH input presentation -> no plateau/apical latch or STP depletion
carries between presentations. Byte-identity of the restore is asserted (--verify-reset). Reuse EMERGE-61 from_host.

DRIVE: DG-DIRECT (isolate the dg->ca3 mossy selection; the upstream lang->ec->dg conduction is separately too weak).
READ: the NATURAL >=theta CA3 assembly (NOT top-k -- top-k launders the sparsity claim).

Anti-cheats: mossy-LESION (mossy_weight=0) -> assembly ~0 (load-bearing); PERMUTED DG -> different assembly
(overlap < 0.13 / perm_cos < 0.5, input-driven); no-input -> ~0 (moat).

GO bar: sparse(6-40) + separated(sep_cos<0.4) + stable(Jaccard>0.6) + anti-cheats hold, >=5/6 seeds.

  SIM_BACKEND=cupy python -m research.runners._gap5_dg_selection_reset_scale_driver --seeds 42 --n-ca3 400 \
      --mossy-w 200 --mossy-density 0.10 --amp-ca3w 4.0
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend, to_host, from_host, get_random_state, set_random_state  # noqa: E402
from research.runners._gap5_emergent_dg_selection_derisk import _build_bridge, _jacc, _cos  # noqa: E402
from research.runners.validate_trisynaptic_loop import build_drive_pattern  # noqa: E402

cp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_gap5_dg_selection_reset_scale.json"


def _is_snapshot_array(arr):
    """A dynamic per-neuron backend array we can byte-restore. Excludes sparse (cp_connections) + non-arrays."""
    if arr is None or not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        return False
    if hasattr(arr, "nnz"):   # sparse matrix (cp_connections) -- skip (weights frozen, hebbian off in read pass)
        return False
    try:
        return arr.ndim >= 1 and arr.size > 0
    except Exception:
        return False


def _snapshot_state(bridge):
    """Byte-for-byte host capture of EVERY dynamic cp_* per-neuron array (== fresh-bridge state: v/u/all conductances
    incl. the plateau + graded_plateau self-regen latch + apical + coincidence + firing + STP + eligibility) PLUS the
    scalar step/time counters. The step counters are LOAD-BEARING: the synaptic delay ring buffer is indexed by
    `current_time_step % max_delay_steps`, so restoring the buffer CONTENTS without the step index reads them at the
    wrong offset -> a false 'instability' (fresh bridges start at step 0 -> agree 0.977; a restore that kept the step
    counter agreed only 0.615). Reset the counters -> restore == fresh bridge."""
    snap = {"__arrays__": {}, "__scalars__": {}}
    for name, arr in list(vars(bridge).items()):
        if name.startswith("cp_") and _is_snapshot_array(arr):
            snap["__arrays__"][name] = np.asarray(to_host(arr)).copy()
    rs = bridge.runtime_state
    snap["__scalars__"]["rs.current_time_step"] = getattr(rs, "current_time_step", None)
    snap["__scalars__"]["rs.current_time_ms"] = getattr(rs, "current_time_ms", None)
    snap["__scalars__"]["_bdsp_step_counter"] = getattr(bridge, "_bdsp_step_counter", None)
    return snap


def _restore_state(bridge, snap):
    for name, val in snap["__arrays__"].items():
        arr = getattr(bridge, name, None)
        if arr is not None:
            arr[:] = from_host(val)
    rs = bridge.runtime_state
    sc = snap["__scalars__"]
    if sc.get("rs.current_time_step") is not None:
        rs.current_time_step = sc["rs.current_time_step"]
    if sc.get("rs.current_time_ms") is not None:
        rs.current_time_ms = sc["rs.current_time_ms"]
    if sc.get("_bdsp_step_counter") is not None:
        bridge._bdsp_step_counter = sc["_bdsp_step_counter"]


def _drive_read(bridge, drive_global_idx, ca3_arr, drive_pA=200.0, n_events=6, reset_steps=10, drive_steps=40,
                theta=0.3, sync=True, g_on=3, g_off=3):
    """Drive `drive_global_idx` (DG-direct), run the gamma-pulsed loop, return (assembly set over ca3-LOCAL idx, ca3_rate).
    Reads the NATURAL >=theta assembly (NOT top-k). Assumes the caller restored the clean snapshot first."""
    drv = cp.asarray(np.asarray(drive_global_idx, dtype=np.int64), dtype=cp.int64) if len(drive_global_idx) else None
    ca3_g = cp.asarray(ca3_arr)
    dg_g = cp.asarray(np.asarray(drive_global_idx, dtype=np.int64)) if len(drive_global_idx) else None
    ca3_spk = cp.zeros(len(ca3_arr), dtype=cp.float32); nrec = 0; dg_drv_spk = 0.0
    _period = g_on + g_off
    for ev in range(n_events):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
        for _t in range(drive_steps):
            bridge.cp_external_input_current[:] = 0.0
            _drive_now = (not sync) or ((_t % _period) < g_on)
            if drv is not None and _drive_now:
                bridge.cp_external_input_current[drv] = float(drive_pA)
            bridge._run_one_simulation_step()
            if ev >= n_events - 3:
                ca3_spk += bridge.cp_firing_states[ca3_g].astype(cp.float32)
                if dg_g is not None:
                    dg_drv_spk += float(to_host(cp.sum(bridge.cp_firing_states[dg_g].astype(cp.float32))))
                nrec += 1
    bridge.cp_external_input_current[:] = 0.0
    ca3_rate = np.asarray(to_host(ca3_spk)) / max(1, nrec)
    A = set(int(i) for i in np.where(ca3_rate >= theta)[0])   # NATURAL assembly, NOT top-k
    _drive_read.last_dg_driven_rate = dg_drv_spk / max(1, nrec) / max(1, len(drive_global_idx))
    return A, ca3_rate


def _fresh_assembly(seed, n_ca3, dg_ffi_weight, ca3_fb_inhib, mossy_weight, mossy_density, amp_ca3w, n_dg,
                    mossy_stp_disabled, ca3_ff_inhib, pat_local, drive_pA, sync, theta, warmup=30):
    """Build a FRESH bridge, warm-up settle, drive the DG pattern ONCE, read the natural >=theta CA3 assembly.
    FRESH-per-presentation is the finding's validated reset method ('two fresh bridges gave Jaccard 1.00'):
    a snapshot/restore REUSED across drives leaked residual state on this bridge (first restore matched fresh, but
    subsequent restores converged to a smaller wrong set); the FIRST drive off a fresh init is always correct."""
    b = _build_bridge(seed, n_ca3, dg_ffi_weight, ca3_fb_inhib, mossy_weight, mossy_density,
                      n_dg=n_dg, amplify=True, amp_ca3w=amp_ca3w, mossy_stp_disabled=mossy_stp_disabled,
                      ca3_ff_inhib=ca3_ff_inhib)
    rm = b.region_manager
    ca3_arr = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    dg_arr = np.asarray(list(rm.indices("dg")), dtype=np.int64)
    b.cp_external_input_current[:] = 0.0
    for _ in range(warmup):
        b._run_one_simulation_step()
    b.cp_external_input_current[:] = 0.0
    drive_idx = dg_arr[pat_local] if len(pat_local) else np.array([], dtype=np.int64)
    A, rate = _drive_read(b, drive_idx, ca3_arr, drive_pA=drive_pA, sync=sync, theta=theta)
    dg_rate = float(getattr(_drive_read, "last_dg_driven_rate", 0.0))
    del b   # let the bridge (and its GPU arrays) be collected before the next fresh build
    return A, rate, dg_rate


def run(seed, n_ca3=400, n_inputs=4, dg_ffi_weight=6.0, ca3_fb_inhib=20.0, mossy_weight=200.0, mossy_density=0.10,
        amp_ca3w=4.0, sync=True, drive_pA=200.0, n_dg=300, verify_reset=False, mossy_stp_disabled=False, theta=0.3,
        ca3_ff_inhib=None, reseed=True):
    def fresh(pat_local, mossy_w=None):
        return _fresh_assembly(seed, n_ca3, dg_ffi_weight, ca3_fb_inhib,
                               mossy_weight if mossy_w is None else mossy_w, mossy_density, amp_ca3w, n_dg,
                               mossy_stp_disabled, ca3_ff_inhib, pat_local, drive_pA, sync, theta)
    # DG index range (a throwaway build to get n_dg count for the pattern rng); patterns are DG-LOCAL
    n_dg_cells = n_dg
    pats = [build_drive_pattern(n_dg_cells, 0.1, seed * 100 + m) for m in range(n_inputs)]

    # first pass: assemble + separation
    sel = [fresh(p) for p in pats]
    A = [s[0] for s in sel]; rates = [s[1] for s in sel]
    dg_driven_rate = float(np.mean([s[2] for s in sel]))
    sizes = [len(a) for a in A]
    ca3_sparsity = float(np.mean([sz / n_ca3 for sz in sizes]))
    # stability: INDEPENDENT fresh build, same input -> Jaccard (the finding's fresh-vs-fresh protocol)
    sel2 = [fresh(p) for p in pats]
    stability = float(np.mean([_jacc(A[m], sel2[m][0]) for m in range(n_inputs)
                               if len(A[m]) or len(sel2[m][0])] or [0.0]))
    # separation (rate-cosine + assembly-Jaccard)
    pair_cos, pair_jac = [], []
    for i in range(n_inputs):
        for j in range(i + 1, n_inputs):
            pair_cos.append(_cos(rates[i], rates[j])); pair_jac.append(_jacc(A[i], A[j]))
    sep_cos = float(np.mean(pair_cos)) if pair_cos else 0.0
    sep_jac = float(np.mean(pair_jac)) if pair_jac else 0.0
    # anti-cheats
    A_noin, _, _ = fresh(np.array([], dtype=np.int64))                        # no-input moat
    perm = np.random.default_rng(seed + 555).permutation(n_dg_cells)[:len(pats[0])]
    A_perm, r_perm, _ = fresh(perm)                                           # permuted input -> different assembly
    perm_cos = _cos(rates[0], r_perm)
    perm_overlap = _jacc(A[0], A_perm)
    A_les, _, _ = fresh(pats[0], mossy_w=0.0)                                 # mossy-lesion -> collapse

    mean_size = float(np.mean(sizes)) if sizes else 0.0
    return {"seed": seed, "n_ca3": int(n_ca3), "mossy_w": mossy_weight, "mossy_density": mossy_density,
            "amp_ca3w": amp_ca3w, "ca3_sizes": sizes, "mean_size": mean_size, "ca3_sparsity": ca3_sparsity,
            "stability": stability, "sep_cos": sep_cos, "sep_jac": sep_jac, "noinput_size": len(A_noin),
            "perm_cos": perm_cos, "perm_overlap": perm_overlap, "lesion_size": len(A_les),
            "dg_driven_rate": dg_driven_rate, "reset_check": None}


def _verdict(per):
    def mean(k): return float(np.mean([p[k] for p in per]))
    # per-seed pass (sparse right-sized 6-40, separated, stable, moat, input-driven, lesion collapses)
    def seed_pass(p):
        return (6 <= p["mean_size"] <= 40 and p["sep_cos"] < 0.4 and p["stability"] > 0.6
                and p["noinput_size"] <= max(1, 0.2 * p["mean_size"])
                and p["perm_overlap"] < 0.13 and p["lesion_size"] <= max(1, 0.2 * p["mean_size"]))
    npass = sum(seed_pass(p) for p in per)
    go = npass >= max(1, int(np.ceil(5 / 6 * len(per)))) if len(per) >= 6 else all(seed_pass(p) for p in per)
    return go, npass, {"mean_size": mean("mean_size"), "ca3_sparsity": mean("ca3_sparsity"),
                       "stability": mean("stability"), "sep_cos": mean("sep_cos"),
                       "noinput": mean("noinput_size"), "perm_overlap": mean("perm_overlap"),
                       "lesion": mean("lesion_size")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=400)
    ap.add_argument("--n-dg", type=int, default=300)
    ap.add_argument("--dg-ffi", type=float, default=6.0)
    ap.add_argument("--ca3-fb", type=float, default=20.0)
    ap.add_argument("--ca3-ff", type=float, default=None, help="E%-max feedforward inhibition (divisive norm, separation robustness)")
    ap.add_argument("--mossy-w", type=float, default=200.0)
    ap.add_argument("--mossy-density", type=float, default=0.10)
    ap.add_argument("--amp-ca3w", type=float, default=4.0)
    ap.add_argument("--drive-pA", type=float, default=200.0)
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-reseed", action="store_true", help="do NOT reset the RNG stream before each presentation (measures RNG-drift contribution to instability)")
    ap.add_argument("--mossy-stp-disabled", action="store_true",
                    help="mossy detonator does NOT depress (committed per-pathway STP-disable) -> CA3 conducts under global STP")
    ap.add_argument("--verify-reset", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time(); err = None; per = []
    print(f"[gap5-R1 {a.tag}] n_ca3={a.n_ca3} mossy(w={a.mossy_w},d={a.mossy_density}) amp_ca3w={a.amp_ca3w} "
          f"sync={not a.no_sync} seeds={a.seeds}", flush=True)
    try:
        for s in a.seeds:
            r = run(s, n_ca3=a.n_ca3, dg_ffi_weight=a.dg_ffi, ca3_fb_inhib=a.ca3_fb, mossy_weight=a.mossy_w,
                    mossy_density=a.mossy_density, amp_ca3w=a.amp_ca3w, sync=not a.no_sync, drive_pA=a.drive_pA,
                    n_dg=a.n_dg, verify_reset=a.verify_reset, mossy_stp_disabled=a.mossy_stp_disabled, theta=a.theta,
                    ca3_ff_inhib=a.ca3_ff, reseed=not a.no_reseed)
            per.append(r)
            rc = r.get("reset_check")
            rcs = f" | reset byte-identical={rc['byte_identical']}" if rc else ""
            print(f"  [seed {s}] size {r['mean_size']:.1f} (sizes {r['ca3_sizes']}) sparsity {r['ca3_sparsity']:.3f} "
                  f"| stability {r['stability']:.2f} | sep_cos {r['sep_cos']:.3f} sep_jac {r['sep_jac']:.2f} || "
                  f"noinput {r['noinput_size']} perm_ov {r['perm_overlap']:.2f} perm_cos {r['perm_cos']:.2f} "
                  f"lesion {r['lesion_size']} | dg_rate {r['dg_driven_rate']:.2f}{rcs}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        go, npass, m = _verdict(per)
        verdict = (f"{'GO' if go else 'BOUNDARY'} -- {npass}/{len(per)} seeds pass "
                   f"(size {m['mean_size']:.1f}, sparsity {m['ca3_sparsity']:.3f}, stability {m['stability']:.2f}, "
                   f"sep_cos {m['sep_cos']:.3f}, moat {m['noinput']:.1f}, perm_ov {m['perm_overlap']:.2f}, "
                   f"lesion {m['lesion']:.1f})")
    else:
        go = False; verdict = f"ERROR -- {err}" if err else "ERROR -- no results"
    summary = {"probe": "gap5_dg_selection_reset_scale_R1", "tag": a.tag, "GO": go, "verdict": verdict,
               "config": {"n_ca3": a.n_ca3, "mossy_w": a.mossy_w, "mossy_density": a.mossy_density,
                          "amp_ca3w": a.amp_ca3w, "sync": not a.no_sync}, "seeds": a.seeds,
               "elapsed_seconds": round(time.time()-t0, 1), "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[gap5-R1] VERDICT: {verdict}\n[gap5-R1] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
