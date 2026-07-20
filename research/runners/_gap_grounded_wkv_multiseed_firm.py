"""Multi-seed firming of the grounded-fluent WKV renderer (De-risk 2 -> 6-seed GO standard).

The single-seed De-risk 2 GO (format fine-tune: focused-grounded 0.83, RA-faithful 1.00) is firmed to the CLAUDE.md
6-seed bar. For each seed [42,43,44 (dev) | 100,101,102 (blind)]: fine-tune the SAME base checkpoint with that TRAINING
seed (varies frame sampling + marker init + torch RNG), then eval focused-grounded + RA-faithful on ALL 22 curriculum
SVO facts (HELD OUT of training). Reports dev and blind SEPARATELY (per the project standard). NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

PY = sys.executable
DEV = [42, 43, 44]
BLIND = [100, 101, 102]


def run(seed, steps, base, tmpdir):
    ck = f"{tmpdir}/wkv_grounded_ft_seed{seed}.npz"
    ev = f"{tmpdir}/wkv_grounded_ft_seed{seed}_eval.json"
    env = dict(os.environ, SIM_BACKEND="cupy")
    ft = subprocess.run([PY, "-m", "research.runners._gap_grounded_wkv_finetune",
                         "--ckpt", base, "--out", ck, "--steps", str(steps),
                         "--n-tiny", "40000", "--lr", "3e-4", "--grounded-frac", "0.65",
                         "--seed", str(seed), "--eval-every", "3000"],
                        env=env, capture_output=True, text=True)
    if not os.path.exists(ck):
        return {"seed": seed, "error": ft.stderr[-500:] or ft.stdout[-500:]}
    env2 = dict(os.environ, SIM_BACKEND="numpy")
    subprocess.run([PY, "-m", "research.runners._gap_grounded_wkv_ceiling_probe",
                    "--ckpt", ck, "--max-new", "8", "--all-facts", "--out", ev, "--show", "0"],
                   env=env2, capture_output=True, text=True)
    ppl = json.load(open(ck.replace(".npz", "_meta.json"))) if os.path.exists(ck.replace(".npz", "_meta.json")) else {}
    r = json.load(open(ev))
    n = r["n"]
    return {"seed": seed, "n": n,
            "focused_grounded": round(r["cont"]["verified"] / n, 3),
            "confab": round(r["cont"]["confab"] / n, 3),
            "ra_faithful": round(r["ra_faithful"]["follows"] / max(1, r["ra_faithful"]["n"]), 3),
            "ra_bias": r["ra_faithful"]["bias"],
            "ppl_before": round(ppl.get("ppl_before", 0), 2), "ppl_after": round(ppl.get("ppl_after", 0), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bridges/wkv_ckpt/wkv_ssmU_v4000_d256_big_seed42.npz")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tmpdir", default="bridges/wkv_ckpt/_ms_firm")
    ap.add_argument("--out", default="research/findings/raw/_gap_grounded_wkv_multiseed_firm.json")
    args = ap.parse_args()
    os.makedirs(args.tmpdir, exist_ok=True)

    results = []
    for grp, seeds in [("DEV", DEV), ("BLIND", BLIND)]:
        for s in seeds:
            r = run(s, args.steps, args.base, args.tmpdir)
            r["group"] = grp
            results.append(r)
            if "error" in r:
                print(f"[{grp} seed {s}] ERROR: {r['error'][:200]}", flush=True)
            else:
                print(f"[{grp} seed {s}] focused-grounded={r['focused_grounded']:.2f} RA-faithful={r['ra_faithful']:.2f} "
                      f"confab={r['confab']:.2f} ppl {r['ppl_before']}->{r['ppl_after']} (n={r['n']})", flush=True)

    def agg(grp):
        rs = [r for r in results if r.get("group") == grp and "error" not in r]
        if not rs:
            return {}
        return {"focused_grounded": round(sum(r["focused_grounded"] for r in rs) / len(rs), 3),
                "ra_faithful": round(sum(r["ra_faithful"] for r in rs) / len(rs), 3),
                "confab": round(sum(r["confab"] for r in rs) / len(rs), 3),
                "min_focused": round(min(r["focused_grounded"] for r in rs), 3),
                "min_ra": round(min(r["ra_faithful"] for r in rs), 3), "n_seeds": len(rs)}
    dev, blind = agg("DEV"), agg("BLIND")
    print(f"\n=== MULTI-SEED FIRMING (n_facts=22, held out of training) ===")
    print(f"DEV  (42/43/44):   focused-grounded {dev.get('focused_grounded')} (min {dev.get('min_focused')})  "
          f"RA-faithful {dev.get('ra_faithful')} (min {dev.get('min_ra')})  confab {dev.get('confab')}")
    print(f"BLIND(100/101/102): focused-grounded {blind.get('focused_grounded')} (min {blind.get('min_focused')})  "
          f"RA-faithful {blind.get('ra_faithful')} (min {blind.get('min_ra')})  confab {blind.get('confab')}")
    json.dump({"results": results, "dev": dev, "blind": blind}, open(args.out, "w"), indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
