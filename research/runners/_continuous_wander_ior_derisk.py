"""DE-RISK the fix for the degenerate wander (2026-08-20-continuous-wander-content-degenerate, 6/6 'cat'):
INHIBITION-OF-RETURN via adaptation on the just-ignited basin.

The self-init wander picks the DOMINANT basin under a FIXED curiosity recurrent-gain (`gains_on`, biased toward the
most-novel concept) -> the same basin ('cat') wins every wander. Inhibition-of-return (how real spontaneous cognition
avoids fixation; SFA / short-term depression; local precedent 2026-08-14-gnw-rung2b-sfa-workspace-eviction) FATIGUES
the just-visited basin so the next wander moves elsewhere. Here the fatigue is applied at the neuromod-drive level:
after a wander selects concept i, multiply basin i's gain by IOR_STRENGTH (<1), then let all basins RECOVER toward
their base gain each step. This is the cheapest faithful test of the LEVER (does suppressing the winner yield varied
wandering?) before the deeper per-neuron SFA-current integration.

TWO ARMS (same organ, same seed, N successive wanders each):
  BASELINE : fixed gains_on (reproduces the 6/6 'cat' negative)
  IOR      : gains adapted between wanders (expect VARIETY: n_distinct > 1)

Run: SIM_BACKEND=cupy BRAIN_SELF_INITIATE_STORE=1 .venv/bin/python -m research.runners._continuous_wander_ior_derisk
Writes research/findings/raw/_continuous_live_cupy/wander_ior.json ; exit 0 iff IOR yields more variety than baseline.
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("BRAIN_SELF_INITIATE_STORE", "1")
N = int(os.environ.get("WANDER_N", "6"))
IOR_STRENGTH = float(os.environ.get("IOR_STRENGTH", "0.15"))   # multiply the just-won basin's gain by this
IOR_RECOVERY = float(os.environ.get("IOR_RECOVERY", "0.5"))    # fraction of the deficit recovered each step
_SUF = os.environ.get("IOR_OUT_SUFFIX", "")   # distinct artifact per sweep config (e.g. "_r0.2")
OUT = os.path.join("research", "findings", "raw", "_continuous_live_cupy", "wander_ior%s.json" % _SUF)


def _run_arm(ior: bool):
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan
    org = SelfInitiationOrgan(seed=42)
    org._ensure_mouth()
    base = list(org.gains_on)              # the fixed curiosity gains (the baseline drive)
    adapt = np.ones(len(base))             # per-basin adaptation multiplier (1.0 = rested)
    seq, times = [], []
    for _ in range(N):
        # apply the current adaptation to the drive BEFORE the wander
        org.gains_on = [float(base[j] * adapt[j]) for j in range(len(base))]
        t0 = time.time()
        out = org.speak(lesion=False)
        times.append(round(time.time() - t0, 1))
        c = out.get("concept")
        seq.append(c)
        if ior and c in org.agents:
            i = list(org.agents).index(c)
            adapt[i] *= IOR_STRENGTH                       # fatigue the just-won basin
            adapt = 1.0 - (1.0 - adapt) * (1.0 - IOR_RECOVERY)  # all basins recover toward 1.0
    valid = [c for c in seq if c]
    return {"sequence": seq, "per_wander_s": times,
            "n_distinct": len(set(valid)), "distinct": sorted(set(valid))}


def main() -> int:
    from sim.backend import get_backend
    _, backend = get_backend()
    print("=== BASELINE (fixed gains) ===", flush=True)
    baseline = _run_arm(ior=False)
    print(baseline["sequence"], "-> n_distinct", baseline["n_distinct"], flush=True)
    print("=== IOR (adapt just-won basin) ===", flush=True)
    ior = _run_arm(ior=True)
    print(ior["sequence"], "-> n_distinct", ior["n_distinct"], flush=True)

    result = {
        "runner": "research/runners/_continuous_wander_ior_derisk.py",
        "seed": 42, "backend": backend, "n_wanders": N,
        "ior_strength": IOR_STRENGTH, "ior_recovery": IOR_RECOVERY,
        "baseline": baseline, "ior": ior,
        "baseline_distinct": baseline["n_distinct"], "ior_distinct": ior["n_distinct"],
        "VERDICT": "GO" if ior["n_distinct"] > baseline["n_distinct"] and ior["n_distinct"] > 1 else "NO-GO",
        "interpretation": ("inhibition-of-return breaks the degenerate wander: adapting the just-won basin's drive "
                           "yields varied successive wanders (trains-of-thought now moves between concepts)"
                           if ior["n_distinct"] > baseline["n_distinct"] and ior["n_distinct"] > 1 else
                           "IOR at the neuromod-drive level did not yield variety -> the winner is set deeper than "
                           "the gain (basin-coverage or noise-amplitude cause); next probe the per-basin ignition margin"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("baseline_distinct", "ior_distinct", "VERDICT")}, indent=2), flush=True)
    return 0 if result["VERDICT"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
