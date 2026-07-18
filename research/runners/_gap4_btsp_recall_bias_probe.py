"""GAP#4<->GAP#5 unification (FUNCTIONAL link): are the BTSP-STORED recurrent weights functionally meaningful -- does a
PARTIAL cue of a BTSP-stored assembly drive the HELD-OUT assembly partners more than non-assembly cells (a recall BIAS,
the precursor to full completion)? Reuses the storing setup (`_gap4_btsp_stores_recurrent_assembly_derisk._build`) +
the two committed edits; NO new sim/ edit.

PROTOCOL: (1) STORE the assembly via BTSP (co-fire the assembly + a brief plateau pulse -> within-assembly recurrent
weights potentiate one-shot). (2) RECALL: drive only HALF the assembly (the partial cue), BTSP/BDSP learning OFF, no
plateau; run recall steps; measure the mean firing of the HELD-OUT assembly half vs the non-assembly cells. A recall
BIAS = held-out partners fire more than non-assembly (the stored recurrent weights carry the cue to the partners).

GO (6-seed): heldout_rate > 2*nonassembly_rate (recall bias -- the stored weights are functional) AND a NO-STORE control
(skip step 1) collapses the bias (heldout ~ nonassembly -> the bias is from the STORED weights, not the topology). This
is a BIAS, NOT full attractor completion (that needs the gap#5 trilemma config -- the next rung). Run:
  python -m research.runners._gap4_btsp_recall_bias_probe --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._gap4_btsp_stores_recurrent_assembly_derisk import _build  # noqa: E402

xp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_gap4_btsp_recall_bias.json"


def _run(seed, store, n_ca3=40, assembly_n=12, store_steps=200, pulse_steps=15, pulse_pA=120.0,
         recall_steps=60, recall_drive=900.0):
    sb = _build(enable_btsp=True, seed=seed, n_ca3=n_ca3)
    rm = sb.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")))
    rng = np.random.default_rng(seed)
    assembly = np.sort(rng.choice(ca3_idx, size=assembly_n, replace=False))
    n = sb.cp_membrane_potential_v.size
    non_assembly = np.array([i for i in ca3_idx if i not in set(assembly.tolist())])
    cue = assembly[: assembly_n // 2]                  # partial cue = first half
    heldout = assembly[assembly_n // 2:]               # held-out partners = second half
    sb.cp_bdsp_apical_drive = xp.zeros(n, dtype=xp.float32)

    # --- (1) STORE (optional: the no-store control skips it) ---
    if store:
        drive = np.zeros(n, dtype=np.float32); drive[assembly] = 900.0
        for step in range(store_steps):
            sb.cp_external_input_current[:] = xp.asarray(drive)
            cur = np.zeros(n, dtype=np.float32)
            if 20 <= step < 20 + pulse_steps:
                cur[assembly] = pulse_pA
            sb.cp_bdsp_apical_drive = xp.asarray(cur)
            sb._run_one_simulation_step()

    # --- (2) RECALL: partial cue only, learning OFF, no plateau; measure firing of held-out vs non-assembly ---
    sb.core_config.enable_btsp = False; sb.core_config.bdsp_learning_rate = 0.0
    sb.cp_bdsp_apical_drive = xp.zeros(n, dtype=xp.float32)
    # settle reset (clear the encoding's membrane state so recall is cue-driven, not residual)
    sb.cp_membrane_potential_v[:] = xp.float32(-65.0)
    heldout_spikes = np.zeros(len(heldout)); non_spikes = np.zeros(len(non_assembly))
    cue_drive = np.zeros(n, dtype=np.float32); cue_drive[cue] = recall_drive
    for step in range(recall_steps):
        sb.cp_external_input_current[:] = xp.asarray(cue_drive)
        sb._run_one_simulation_step()
        fired = np.asarray(to_host(sb.cp_firing_states)).astype(float)
        heldout_spikes += fired[heldout]; non_spikes += fired[non_assembly]
    return {"heldout_rate": float(heldout_spikes.mean() / recall_steps),
            "nonassembly_rate": float(non_spikes.mean() / recall_steps)}


def run(seed):
    stored = _run(seed, store=True)
    nostore = _run(seed, store=False)
    return {"seed": seed, "heldout_rate": stored["heldout_rate"], "nonassembly_rate": stored["nonassembly_rate"],
            "nostore_heldout_rate": nostore["heldout_rate"], "nostore_nonassembly_rate": nostore["nonassembly_rate"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s); per.append(r)
            print(f"  [seed {s}] STORED heldout {r['heldout_rate']:.3f} vs non {r['nonassembly_rate']:.3f} | "
                  f"NO-STORE heldout {r['nostore_heldout_rate']:.3f} vs non {r['nostore_nonassembly_rate']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mean(k): return float(np.mean([p[k] for p in per]))
        ho, na = mean("heldout_rate"), mean("nonassembly_rate")
        nho, nna = mean("nostore_heldout_rate"), mean("nostore_nonassembly_rate")
        bias = all(p["heldout_rate"] > 2.0 * max(p["nonassembly_rate"], 1e-6) for p in per)
        nostore_collapses = all(p["nostore_heldout_rate"] <= 1.5 * max(p["nostore_nonassembly_rate"], 1e-6) + 0.02
                                for p in per)
        go = bool(bias and nostore_collapses)
        if go:
            verdict = (f"GO -- the BTSP-STORED recurrent weights are FUNCTIONAL: a partial cue drives the HELD-OUT "
                       f"assembly partners ({ho:.3f}) far more than non-assembly cells ({na:.3f}, {ho/max(na,1e-6):.1f}x) "
                       f"= a recall BIAS through the stored recurrent weights. The NO-STORE control collapses the bias "
                       f"(heldout {nho:.3f} ~ non {nna:.3f}) -> the bias is from the STORED weights, not the topology. "
                       f"6-seed. => the BTSP-stored assembly carries a partial cue to its partners (the functional bridge "
                       f"to completion). NOT full attractor completion (the gap#5 trilemma config is the next rung).")
        else:
            miss = []
            if not bias: miss.append(f"no recall bias (heldout {ho:.3f} vs non {na:.3f})")
            if not nostore_collapses: miss.append(f"no-store didn't collapse (heldout {nho:.3f} vs non {nna:.3f})")
            verdict = "BOUNDARY -- " + "; ".join(miss) + ". Per THE LAW: tune the store/recall drive or assembly, NOT a stop."
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "gap4_btsp_recall_bias", "GO": go, "verdict": verdict,
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "A recall BIAS (partial cue -> stored partners fire more), NOT full attractor completion "
                              "(needs the gap#5 trilemma config: bistable dendrites + selective inhib + structural sep). "
                              "Reuses the storing setup + the two committed edits; NO new sim/ edit."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-btsp-recall] VERDICT: {verdict}", flush=True)
    print(f"[gap4-btsp-recall] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
