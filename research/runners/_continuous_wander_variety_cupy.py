"""Is the between-turn WANDER genuinely varied, or a fixed output masquerading as wandering?

The continuous engine claims a THOUGHT wanders between turns (trains-of-thought). If successive idle wanders on
one session always surface the SAME concept, that property is degenerate (a fixed read, not wandering) — the exact
hollow-faculty trap. This drives ONE self-initiation organ through N successive wanders (as an idle session would
across N idle ticks) and reports the concept SEQUENCE + how many DISTINCT concepts appear. Genuine trains-of-thought
=> variety (>1 distinct); degenerate => all identical.

On cupy the self-init wander is the stochastic multibasin CA3 wander (the varied path per the v1 finding); on numpy
it is the light curiosity-top path (expected stable). Run on cupy to test the production claim.

Run: SIM_BACKEND=cupy BRAIN_SELF_INITIATE_STORE=1 .venv/bin/python -m research.runners._continuous_wander_variety_cupy
Writes research/findings/raw/_continuous_live_cupy/wander_variety.json.
"""
import os, sys, json, time

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("BRAIN_SELF_INITIATE_STORE", "1")
N = int(os.environ.get("WANDER_N", "6"))
OUT = os.path.join("research", "findings", "raw", "_continuous_live_cupy", "wander_variety.json")


def main() -> int:
    from sim.backend import get_backend
    _, backend = get_backend()
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan

    org = SelfInitiationOrgan(seed=42)            # ONE session's organ, as the live server holds per cache_key
    concepts, times = [], []
    for i in range(N):
        t0 = time.time()
        try:
            out = org.speak(lesion=False)
            c = out.get("concept")
        except Exception as e:
            c = "ERR:%s" % type(e).__name__
        concepts.append(c)
        times.append(round(time.time() - t0, 1))
        print("wander %d/%d -> %r (%.1fs)" % (i + 1, N, c, times[-1]), flush=True)

    valid = [c for c in concepts if c and not str(c).startswith("ERR:")]
    distinct = sorted(set(valid))
    result = {
        "runner": "research/runners/_continuous_wander_variety_cupy.py",
        "seed": 42, "backend": backend, "n_wanders": N,
        "concept_sequence": concepts, "per_wander_s": times,
        "n_distinct": len(distinct), "distinct_concepts": distinct,
        "VERDICT": "VARIED" if len(distinct) > 1 else ("DEGENERATE-single-concept" if valid else "NO-WANDER"),
        "interpretation": ("trains-of-thought is genuine: successive idle wanders surface different concepts"
                           if len(distinct) > 1 else
                           "successive idle wanders surface ONE concept -> the wander is a fixed read, not wandering; "
                           "residual: widen the CA3 wander's stochasticity / concept basin coverage"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("backend", "n_distinct", "distinct_concepts", "VERDICT")}, indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
