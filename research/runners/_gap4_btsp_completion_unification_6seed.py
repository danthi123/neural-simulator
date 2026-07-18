"""gap#4<->gap#5 UNIFICATION — 6-seed: does BTSP plateau-gated ONE-SHOT encoding produce a CA3 assembly that the
bistable CA3 COMPLETES from a partial cue (the gap#5 completion), unifying the two gaps on the shared dendritic
bistability keystone? Runs the validated gap#5 completion config with encode_btsp=True instead of the rate-window
Hebbian encode. NO new sim/ edit (reuses the encode_btsp path + the two committed edits).

GO (per gap#5's own gate, per seed): held_cue >= 0.20 AND >= 3*held_nocue AND >= 3*held_perm AND held_nocue <= 0.10.
Anti-cheats (all must hold): NO-ENCODE (no plateau + no co-fire) -> held_cue <= 0.10 (the completion is on the STORED
assembly, not a drive/leak artifact); the built-in PERMUTED cue (a random non-assembly set) -> held_perm low (specificity);
NO-CUE -> held_nocue low (bistable rest, not self-sustaining). Reports w_within (the BTSP-stored recurrent scale).

Run (GPU): SIM_BACKEND=cupy python -m research.runners._gap4_btsp_completion_unification_6seed --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_btsp_completion_unification_6seed.json"

GO_CFG = dict(n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000.0, no_sync=True,
              recall_k_thresh=110.0, recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False,
              enable_ou=False, selective_inhib=True, structural_sep=1, plateau_self_regen=0.15, apical_kir_g=3.0,
              apical_gc=1.0, apical_gc_read=5.0)
# BTSP-encode config tuned for the plateau-gated UNIFORM within-assembly distribution: init recurrent LOW (0.5) so BTSP
# builds it; btsp_w_max=300 (moderate scale — over-strong uniform weights OVER-drive the recall, a non-monotonic effect);
# recall_k_thresh LOWERED to 60 (BTSP's uniform coincident drive is spread evenly, so it wants a lower dendritic
# threshold than Hebbian's 110; structural_sep=1 keeps permuted-cue specificity structurally, so lowering it is safe).
BTSP_CFG = {**GO_CFG, "encode_btsp": True, "encode_ca3w": 0.5, "encode_plateau_pA": 250.0, "btsp_lr": 0.02,
            "hebb_max": 300.0, "train_events": 30, "recall_k_thresh": 40.0}


def one(seed):
    r = run(seed, **BTSP_CFG)
    ne = run(seed, **{**BTSP_CFG, "encode_plateau_pA": 0.0, "encode_drive": 0.0})   # no-encode anti-cheat
    return {"seed": seed, "w_within": r["w_within"], "held_cue": r["held_cue"], "held_nocue": r["held_nocue"],
            "held_perm": r["held_perm"], "rest_firing": r["rest_firing"], "noencode_cue": ne["held_cue"],
            "seed_go": bool(r["held_cue"] >= 0.20 and r["held_cue"] >= 3.0*(r["held_nocue"]+1e-6)
                            and r["held_cue"] >= 3.0*(r["held_perm"]+1e-6) and r["held_nocue"] <= 0.10
                            and ne["held_cue"] <= 0.10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = one(s); per.append(r)
            print(f"  [seed {s}] w_within {r['w_within']:.2f} | cue {r['held_cue']:.3f} nocue {r['held_nocue']:.3f} "
                  f"perm {r['held_perm']:.3f} | no-encode cue {r['noencode_cue']:.3f} -> {'GO' if r['seed_go'] else 'no'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        n_go = sum(1 for p in per if p["seed_go"])
        mc = float(np.mean([p["held_cue"] for p in per])); mn = float(np.mean([p["held_nocue"] for p in per]))
        mp = float(np.mean([p["held_perm"] for p in per])); mne = float(np.mean([p["noencode_cue"] for p in per]))
        go = n_go >= 5   # 5/6 (matches the gap#5 magnitude-seed-variability standard)
        verdict = (f"{'GO' if go else 'BOUNDARY'} {n_go}/6 -- BTSP plateau-gated ONE-SHOT encoding {'PRODUCES' if go else 'did not reliably produce'} "
                   f"a CA3 assembly the bistable CA3 completes: cue {mc:.3f} vs nocue {mn:.3f} / perm {mp:.3f}; no-encode "
                   f"anti-cheat cue {mne:.3f}. {'=> the gap#4 credit rule (BTSP) STORES and the gap#5 bistable CA3 COMPLETES -- the two gaps unified on the shared dendritic-bistability keystone, on the spiking substrate.' if go else 'Per THE LAW: tune the encode plateau/ca3w/btsp_lr/train_events, NOT a stop.'}")
    else:
        go = False; verdict = f"ERROR -- {err}"; n_go = 0
    summary = {"probe": "gap4_btsp_completion_unification", "GO": go, "n_go": n_go, "verdict": verdict,
               "seeds": a.seeds, "elapsed_seconds": round(time.time()-t0, 1), "per_seed": per,
               "config": {"BTSP": BTSP_CFG}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[gap4-btsp-unify] VERDICT: {verdict}\n[gap4-btsp-unify] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
