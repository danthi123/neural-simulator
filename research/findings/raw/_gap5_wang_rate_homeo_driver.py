"""Gap #5 — does a PER-NEURON RATE HOMEOSTATIC (Turrigiano intrinsic excitability) robustify the Wang-NMDA bistable
completion working point across seeds? Baseline (no rate_homeo) reproduces the 1/6 fragility; rate_homeo should
equalize the low state (suppress the self-sustaining seeds' over-firers) -> >=5/6 genuine bistable+specific completion
with the mandatory no-cue + permuted anti-cheats. GPU.
"""
import argparse, os, sys, time
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--n-ca3", type=int, default=2000)
    ap.add_argument("--assembly-frac", type=float, default=0.008)
    ap.add_argument("--ca3-density", type=float, default=0.5)     # Guzman-Jonas biological ~0.02-0.05 gives specificity
    ap.add_argument("--dendritic", action="store_true", help="dendritic dAP readout (nmda_recurrent=False), the mode that completes")
    ap.add_argument("--k-thresh", type=float, default=18.0)
    ap.add_argument("--recall-k-thresh", type=float, default=None)   # decouple encode (low) vs recall (high) dAP threshold
    ap.add_argument("--fb-inhib", type=float, default=20.0)
    ap.add_argument("--selective-inhib", action="store_true", help="assembly-selective inhibition (spare own engram, Kim-Kim 2025)")
    ap.add_argument("--lam-dep-wi", type=float, default=0.5)
    ap.add_argument("--encode-drive", type=float, default=3000.0)   # formation lever: continuous strong drive
    ap.add_argument("--hebb-lr", type=float, default=2.0)           # formation lever
    ap.add_argument("--no-sync", action="store_true", default=True) # continuous (no gamma off-gap decay)
    ap.add_argument("--nmda-tau", type=float, default=100.0)
    ap.add_argument("--nmda-ratio", type=float, default=1.0)   # scale the recurrent NMDA conductance (drive at fixed weak weights)
    ap.add_argument("--recall-steps", type=int, default=150)
    ap.add_argument("--recall-drive", type=float, default=800.0)
    ap.add_argument("--homeo-target", type=float, default=800.0)
    ap.add_argument("--no-weightsum-homeo", action="store_true")
    ap.add_argument("--rate-homeo", action="store_true")
    ap.add_argument("--rh-target", type=float, default=0.02)
    ap.add_argument("--rh-adapt", type=float, default=15.0)
    ap.add_argument("--rh-steps", type=int, default=400)
    ap.add_argument("--rh-cap", type=float, default=800.0)
    ap.add_argument("--no-ou", action="store_true", help="disable OU noise (isolate deterministic bistability)")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    t0 = time.time(); ngo = 0
    tag = f"rate_homeo={a.rate_homeo}(t={a.rh_target},adapt={a.rh_adapt},cap={a.rh_cap})"
    print(f"[gap5 wang rate-homeo] n_ca3={a.n_ca3} frac={a.assembly_frac} nmda_tau={a.nmda_tau} "
          f"recall_steps={a.recall_steps} recall_drive={a.recall_drive} weightsum_T={a.homeo_target} {tag}", flush=True)
    for s in seeds:
        r = run(s, n_ca3=a.n_ca3, assembly_frac=a.assembly_frac, ca3_density=a.ca3_density, bistable=True,
                nmda_recurrent=(not a.dendritic), k_thresh=a.k_thresh, recall_k_thresh=a.recall_k_thresh,
                ca3_fb_inhib=a.fb_inhib, lam_dep_wi=a.lam_dep_wi, selective_inhib=a.selective_inhib,
                encode_drive=a.encode_drive, hebb_lr=a.hebb_lr, no_sync=a.no_sync,
                nmda_tau=a.nmda_tau, nmda_ratio=a.nmda_ratio, recall_steps=a.recall_steps, recall_drive=a.recall_drive,
                homeostatic=((not a.no_weightsum_homeo) and not a.dendritic), homeo_target=a.homeo_target,
                rate_homeo=a.rate_homeo, rate_homeo_target=a.rh_target, rate_homeo_adapt=a.rh_adapt,
                rate_homeo_steps=a.rh_steps, rate_homeo_cap=a.rh_cap, enable_ou=(not a.no_ou))
        ngo += int(r["go"])
        print(f"  PERSEED [seed {s}] w_within={r.get('w_within',0):.1f} cue={r['held_cue']:.3f} nocue={r['held_nocue']:.3f} "
              f"perm={r['held_perm']:.3f} rest={r['rest_firing']:.3f} bias={r['mean_bias']:.0f} "
              f"-> {'GO' if r['go'] else 'no'} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  RESULT: {ngo}/{len(seeds)} GO  {tag}", flush=True)


if __name__ == "__main__":
    main()
