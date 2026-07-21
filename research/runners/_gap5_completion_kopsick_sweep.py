"""gap#5 — CA3 COMPLETION Kopsick recipe sweep: surpass the functionally-silent completion boundary.

The 2026-07-08/17/21 diagnosis: the within-assembly recurrent weights ARE potentiated (w_ratio 88-212) but the
completion is FUNCTIONALLY SILENT (h_comp=0) — the absolute recurrent drive from a partial cue is ~1000x too weak to
fire the held-out members. Simple encode/train tuning does NOT fix it (probe confirmed).

The pinned NEXT MECHANISM (Kopsick-Ascoli 2024 + the project's own generalization-arc amplifiers): raise the absolute
recurrent drive via biologically-grounded amplifiers, one lever at a time from the h_comp=0 baseline, to ISOLATE what
lifts completion:
  A NMDA-recurrent   — slow NMDA temporally integrates the sparse recurrent drive past threshold (the exact mechanism
                        that resolved the rate-code wall in the generalization arc). #1 lever.
  B bistable+plateau — dendritic-plateau bistable attractor (Bittner/Kopsick plateau potential amplifier).
  C less-inhib+dense — lower CA3 feedback inhibition + denser recurrent connectivity (more, less-suppressed drive).
  D easier-threshold — lower k_thresh + stronger recall (partial) cue.
  E Kopsick-combined — strong synchronous encoding + denser assemblies + moderate fb-inhib + NMDA-recurrent.

GATE (this seed-42 lever-finding pass): h_comp >= 0.3 on ANY config => the lever that surpasses the boundary. Then
6-seed + anti-cheats on the winning config. VERIFY-FIRST: also prints w_ratio (weights potentiated) + n_comp so a
'completion' that is really the cue itself (n_comp<=cue) is caught. GPU.
"""
import argparse
import json

from research.runners._riii_ca3_synchronous_assembly_derisk import run

# Each config = a single hypothesis-driven lever set OVER the h_comp=0 baseline (n_ca3=1000, n_mem=2, defaults).
CONFIGS = {
    "baseline":       dict(),                                                        # the confirmed h_comp=0 point
    "A_nmda":         dict(nmda_recurrent=True, nmda_ratio=1.5, nmda_tau=120.0),
    "B_bistable":     dict(bistable=True, plateau_strength=220.0),
    "C_lessinhib":    dict(ca3_fb_inhib=5.0, ca3_density=0.8),
    "D_easythresh":   dict(k_thresh=10.0, recall_drive=420.0),
    "E_kopsick":      dict(encode_drive=1400.0, train_events=300, sync_on=3,
                           assembly_frac=0.02, ca3_fb_inhib=8.0,
                           nmda_recurrent=True, nmda_ratio=1.5, nmda_tau=120.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ca3", type=int, default=1000)
    ap.add_argument("--n-mem", type=int, default=2)
    ap.add_argument("--only", type=str, default=None, help="run ONE config by name")
    ap.add_argument("--out", type=str, default="research/findings/raw/_gap5_completion_kopsick_sweep.json")
    args = ap.parse_args()

    names = [args.only] if args.only else list(CONFIGS.keys())
    rows = []
    best = None
    for name in names:
        kw = CONFIGS[name]
        try:
            r = run(args.seed, n_ca3=args.n_ca3, n_mem=args.n_mem, **kw)
        except Exception as e:
            print(f"  [{name:14s}] ERROR {type(e).__name__}: {e}", flush=True)
            rows.append({"config": name, "error": f"{type(e).__name__}: {e}"})
            continue
        h = float(r.get("h_comp", 0.0)); ncomp = int(r.get("n_comp", 0)); wr = float(r.get("w_ratio", 0.0))
        go = float(r.get("go", 0.0))
        rows.append({"config": name, "h_comp": h, "n_comp": ncomp, "w_ratio": wr, "go": go})
        flag = "  <-- SURPASSED (h_comp>=0.3)" if h >= 0.3 else ""
        print(f"  [{name:14s}] h_comp={h:.3f}  n_comp={ncomp}  w_ratio={wr:.1f}  go={go}{flag}", flush=True)
        if best is None or h > best[1]:
            best = (name, h)

    print("=" * 84, flush=True)
    if best:
        surpassed = best[1] >= 0.3
        print(f"[gap5 Kopsick sweep] seed={args.seed} n_ca3={args.n_ca3} n_mem={args.n_mem} | "
              f"best={best[0]} h_comp={best[1]:.3f} | "
              f"{'LEVER FOUND -> 6-seed next' if surpassed else 'still-silent -> next amplifier'}", flush=True)
    json.dump({"rows": rows, "best": best}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
