"""STORE-COVERAGE residual (2026-08-20): the between-turn wander with inhibition-of-return reaches only 3 of the 4
stored concepts under ANY recovery/strength — the 4th never wins even when the top 3 are IOR-suppressed
(wander_ior_*.json). Diagnosis (2026-08-20-per-neuron-SFA-wrong-locus): the wander winner is set by the tonic
STEERING gain (curiosity), so the 4th concept never surfaces because its steering gain is too LOW to win even against
the IOR-suppressed top 3. This de-risks the named secondary lever: a curiosity-gain FLOOR.

MECHANISM: before applying the IOR adaptation each wander, clamp every basin's base gain UP to a floor
(gains = max(base, GAIN_FLOOR)); then the IOR fatigue applies on top. With a high-enough floor, when the top basins
are IOR-fatigued the previously-dead 4th basin has enough residual steering drive to win -> full 4/4 coverage. This is
a STEERING-drive change (the correct fatigue/drive locus per the diagnosis), not an intrinsic-excitability hack.

Two arms, same organ/seed, N successive wanders each, WITH inhibition-of-return in both:
  IOR-ONLY  : the current live behaviour (base gains + IOR)                       -> expect 3/4
  IOR+FLOOR : same + a gain floor clamp                                            -> target 4/4

Run: SIM_BACKEND=cupy BRAIN_SELF_INITIATE_STORE=1 WANDER_N=10 GAIN_FLOOR=1.4 \
       .venv/bin/python -m research.runners._continuous_wander_gainfloor_coverage_derisk
Writes research/findings/raw/_continuous_live_cupy/wander_gainfloor{IOR_OUT_SUFFIX}.json ; exit 0 iff FLOOR reaches
more distinct concepts than IOR-only (ideally full n_mem).
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("BRAIN_SELF_INITIATE_STORE", "1")
N = int(os.environ.get("WANDER_N", "10"))
IOR_STRENGTH = float(os.environ.get("IOR_STRENGTH", "0.15"))
IOR_RECOVERY = float(os.environ.get("IOR_RECOVERY", "0.3"))
GAIN_FLOOR = float(os.environ.get("GAIN_FLOOR", "1.4"))
_SUF = os.environ.get("IOR_OUT_SUFFIX", "")
OUT = os.path.join("research", "findings", "raw", "_continuous_live_cupy", "wander_gainfloor%s.json" % _SUF)


def _run_arm(floor):
    """One arm: N successive wanders with IOR; if floor>0, clamp base gains up to the floor before IOR fatigue."""
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan
    org = SelfInitiationOrgan(seed=42)
    org._ensure_mouth()
    base = np.asarray(org.gains_on, dtype=np.float64)
    if floor > 0:
        base = np.maximum(base, floor)          # the gain FLOOR (steering-drive change)
    adapt = np.ones(len(base))
    seq, times = [], []
    for _ in range(N):
        org.gains_on = [float(base[j] * adapt[j]) for j in range(len(base))]
        t0 = time.time()
        out = org.speak(lesion=False)
        times.append(round(time.time() - t0, 1))
        c = out.get("concept")
        seq.append(c)
        if c in org.agents:
            i = list(org.agents).index(c)
            adapt[i] *= IOR_STRENGTH
            adapt = 1.0 - (1.0 - adapt) * (1.0 - IOR_RECOVERY)
    valid = [c for c in seq if c]
    return {"sequence": seq, "per_wander_s": times, "n_distinct": len(set(valid)), "distinct": sorted(set(valid))}, org.n_mem


def main() -> int:
    from sim.backend import get_backend
    _, backend = get_backend()
    print("=== IOR-ONLY (no floor) ===", flush=True)
    ior_only, n_mem = _run_arm(0.0)
    print(ior_only["sequence"], "-> n_distinct", ior_only["n_distinct"], "of", n_mem, flush=True)
    print("=== IOR + GAIN FLOOR %.2f ===" % GAIN_FLOOR, flush=True)
    ior_floor, _ = _run_arm(GAIN_FLOOR)
    print(ior_floor["sequence"], "-> n_distinct", ior_floor["n_distinct"], "of", n_mem, flush=True)

    result = {
        "runner": "research/runners/_continuous_wander_gainfloor_coverage_derisk.py",
        "seed": 42, "backend": backend, "n_wanders": N, "n_mem": n_mem,
        "ior_strength": IOR_STRENGTH, "ior_recovery": IOR_RECOVERY, "gain_floor": GAIN_FLOOR,
        "ior_only": ior_only, "ior_floor": ior_floor,
        "ior_only_distinct": ior_only["n_distinct"], "ior_floor_distinct": ior_floor["n_distinct"],
        "full_coverage": bool(ior_floor["n_distinct"] >= n_mem),
        "VERDICT": ("GO" if ior_floor["n_distinct"] > ior_only["n_distinct"] else
                    ("UNDEFINED" if ior_floor["sequence"] == ior_only["sequence"] else "NO-GO")),
        "interpretation": (
            "a curiosity-gain FLOOR raises the previously-dead tail concept's steering drive enough that, when the "
            "top basins are IOR-fatigued, it wins -> wider wander coverage"
            if ior_floor["n_distinct"] > ior_only["n_distinct"] else
            "the gain floor did not widen coverage vs IOR-only -- the tail concept's basin ENCODING (not its steering "
            "gain) is the limit -> next lever is a stronger tail-basin encode, not a gain change"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("ior_only_distinct", "ior_floor_distinct", "full_coverage", "VERDICT")},
                     indent=2), flush=True)
    return 0 if result["VERDICT"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
