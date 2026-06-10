"""Aggregate the FULLY-BIOLOGIZED N9 nav A/B: the fully-spiking loop (reward_us US->SNc + FS->critic
inhibition + N5 perceived reward) vs the host-V baseline (OFF) and the GIRK-cap-masked version
(ONcap25). nav_sum = sum phase_stats.final_quarter_mean_distance (LOWER=better). Also checks the
(A) US-lesion anti-cheat (reward_us drive 0 -> US silent -> nav should REGRESS = the US is load-bearing).

    python research/findings/raw/g11_bg/_n9fullspike_aggregate.py
"""
import json, glob, os

D = "research/findings/raw/g11_bg"


def navsum(f):
    d = json.load(open(f))
    return sum(p["final_quarter_mean_distance"] for p in d.get("phase_stats", []))


def collect(tag):
    out = {}
    for f in glob.glob(f"{D}/_n9ab_{tag}_seed*.json"):
        try:
            s = int(os.path.basename(f).split("seed")[1].split(".")[0])
            out[s] = navsum(f)
        except Exception as e:
            print(f"  [WARN] {os.path.basename(f)}: {e}")
    return out


def main():
    full = collect("FULLSPIKE")      # reward_us US->SNc + FS->critic + N5 (fully spiking, GIRK cap off)
    cap = collect("ONcap25")         # GIRK-cap-masked neural critic
    off = collect("OFF")             # host-V scaffold baseline (the cheat)
    seeds = sorted(set(full) | set(cap) | set(off))
    print("=" * 90)
    print("N9 FULLY-SPIKING nav A/B  (nav_sum = sum final_quarter_mean_distance; LOWER = better)")
    print(f"{'seed':>5} | {'FULLSPIKE (A+B spiking)':>22} | {'ONcap25 (GIRK mask)':>20} | {'OFF host-V cheat':>17} | {'full/host':>9}")
    print("-" * 90)
    rows = []
    for s in seeds:
        a, c, b = full.get(s), cap.get(s), off.get(s)
        fmt = lambda x: ("%.3f" % x) if x is not None else "--"
        r = (a / b) if (a is not None and b) else None
        rows.append((s, a, c, b, r))
        print(f"{s:>5} | {fmt(a):>22} | {fmt(c):>20} | {fmt(b):>17} | {(('%.3f'%r) if r else '--'):>9}")
    paired = [r for r in rows if r[1] is not None and r[3]]
    if paired:
        ma = sum(r[1] for r in paired) / len(paired)
        mb = sum(r[3] for r in paired) / len(paired)
        print("-" * 90)
        print(f"{'mean':>5} | {ma:22.3f} | {'':>20} | {mb:17.3f} | {ma/mb:9.3f}")
        print("=" * 90)
        rr = ma / mb
        if rr <= 1.05:
            print(f"VERDICT: the FULLY-SPIKING N9 (reward_us + FS->critic, no host reward, no GIRK mask) "
                  f"MATCHES/BEATS the host cheat (mean {rr:.2f}) => the BRAIN-BASED-ONLY completion succeeds.")
        elif rr <= 1.25:
            print(f"VERDICT: the FULLY-SPIKING N9 is COMPETITIVE with the host cheat (mean {rr:.2f}, within 25%) "
                  f"-- a strong biologization; residual = value-train draw-variance / multi-seed.")
        else:
            print(f"VERDICT: the FULLY-SPIKING N9 is {(rr-1)*100:.0f}% worse than host (mean {rr:.2f}) "
                  f"=> honest negative (substrate vs host shortcut); diagnose.")
    # (A) US-lesion anti-cheat
    les = glob.glob(f"{D}/_n9ab_USLESION_seed*.json")
    if les:
        print("-" * 90)
        for f in sorted(les):
            s = int(os.path.basename(f).split("seed")[1].split(".")[0])
            ls = navsum(f); fs = full.get(s)
            print(f"(A) US-LESION seed {s}: nav_sum={ls:.3f}  vs FULLSPIKE={fs if fs else '--'}")
            if fs is not None:
                if ls > fs * 1.3:
                    print(f"    -> US-lesion REGRESSES nav ({fs:.2f}->{ls:.2f}) = the reward_us US chain is LOAD-BEARING (anti-cheat PASS).")
                else:
                    print(f"    -> US-lesion does NOT regress nav ({fs:.2f}->{ls:.2f}) = the agent navigates WITHOUT the US "
                          f"(via place-goal-readout?) -> the US isn't load-bearing here (honest; the reward drives LEARNING not the readout).")


if __name__ == "__main__":
    main()
