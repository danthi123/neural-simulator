"""Aggregate the N9 nav A/B: neural critic (delta=r-V via spiking SNc + GABA_B) vs the host-V
scaffold baseline (the cheat being replaced). nav_sum = sum of phase_stats.final_quarter_mean_distance
(lower = better). Verdict per BRAIN-BASED-ONLY: neural <= ~host (within ~15%) => the biologization
succeeds (no regression); neural >> host => honest negative (substrate can't yet match the shortcut).

    python research/findings/raw/g11_bg/_n9ab_aggregate.py
"""
import json, glob, os


def nav_sum(path):
    d = json.load(open(path))
    return sum(p["final_quarter_mean_distance"] for p in d.get("phase_stats", []))


def collect(tag):
    out = {}
    for f in glob.glob(f"research/findings/raw/g11_bg/_n9ab_{tag}_seed*.json"):
        seed = int(os.path.basename(f).split("seed")[1].split(".")[0])
        try:
            out[seed] = nav_sum(f)
        except Exception as e:
            out[seed] = None
            print(f"  [WARN] {os.path.basename(f)}: {e}")
    return out


def main():
    on = collect("ON")    # neural delta = r - V
    off = collect("OFF")  # host-V scaffold (the cheat)
    seeds = sorted(set(on) & set(off))
    print("=" * 64)
    print("N9 nav A/B  (nav_sum = sum final_quarter_mean_distance; LOWER = better)")
    print(f"{'seed':>6} | {'ON neural d=r-V':>16} | {'OFF host-V':>12} | {'neural/host':>11}")
    print("-" * 64)
    rows = []
    for s in seeds:
        a, b = on.get(s), off.get(s)
        if a is None or b is None:
            print(f"{s:>6} | {'(missing)':>16} | {'(missing)':>12} |")
            continue
        ratio = a / b if b > 1e-9 else float("inf")
        rows.append((s, a, b, ratio))
        print(f"{s:>6} | {a:16.3f} | {b:12.3f} | {ratio:11.3f}")
    if rows:
        ma = sum(r[1] for r in rows) / len(rows)
        mb = sum(r[2] for r in rows) / len(rows)
        mr = ma / mb if mb > 1e-9 else float("inf")
        print("-" * 64)
        print(f"{'mean':>6} | {ma:16.3f} | {mb:12.3f} | {mr:11.3f}")
        print("=" * 64)
        if mr <= 1.15:
            print(f"VERDICT: neural delta=r-V is within 15% of (or beats) the host-V cheat "
                  f"(ratio {mr:.2f}) => BIOLOGIZATION SUCCEEDS (no nav regression). "
                  f"Next: 6-seed A/B + anti-cheats (place-shuffle / sensor-ablation / GABA_B-lesion).")
        else:
            print(f"VERDICT: neural delta=r-V is {(mr-1)*100:.0f}% WORSE than the host-V cheat "
                  f"(ratio {mr:.2f}) => HONEST NEGATIVE at this operating point (the spiking substrate "
                  f"does not yet match the host shortcut). A valid BRAIN-BASED-ONLY deliverable; "
                  f"diagnose (critic-rate / GABA_B operating point / value-train) before the 6-seed.")
        print("NOTE: directional (2-seed, non-deterministic nav); a clean read needs full-run "
              "determinism (extend cfg.deterministic_transpose_matvec to the coincidence/GABA_B "
              "matvecs) + 6 seeds. Anti-cheats NOT yet run.")
    else:
        print("(no complete A/B pairs yet)")


if __name__ == "__main__":
    main()
