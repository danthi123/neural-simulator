"""EMERGE-94 (probe) -- RUNG A.3 dt-reconciliation: does the on-bridge SPIKING reservoir parse at dt=1.0?

RUNG A.2 folded the composer + producer onto one bridge at dt=1.0. To fold the SPIKING reservoir (OnBridgeLSM, tuned at
dt=0.5) onto the SAME bridge, it must parse at the shared dt. This single-variable probe builds the reservoir
comprehender at dt=0.5 (baseline) and dt=1.0 (the shared-bridge dt) and measures transitive parse_acc on held-out
content. GO (dt=1.0 parses) -> RUNG A.3 proceeds at dt=1.0; BOUNDARY (dt=1.0 collapses) -> the reservoir needs
re-tuning at dt=1.0 or a per-phase dt switch (named next steps).

Run:  SIM_BACKEND=numpy python -u -m research.runners._emerge94_reservoir_dt_probe --seeds 42 43 44
"""
import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _content_pools, _gen, _TRAIN_KINDS,
)
from research.runners._emerge82_onbridge_lsm_derisk import OnBridgeLSM, _N_POOL, _N_TRAIN_PER  # noqa: E402
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import (  # noqa: E402
    ReservoirComprehender, _build_test_facts,
)


def _parse_acc_at_dt(seed, dt):
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    enc = Encoder(discovered)
    comp = ReservoirComprehender(seed, discovered, res=OnBridgeLSM(enc.dim, seed=seed, n=_N_POOL, dt=dt), enc=enc)
    comp.fit(_gen(_TRAIN_KINDS, _N_TRAIN_PER, np.random.default_rng(seed * 101 + 5), subj, verb, obj))
    test, _seen, _t = _build_test_facts(seed, subj, verb, obj)
    hit = sum(int(comp.comprehend(toks).get("agent") == s
                  and comp.comprehend(toks).get("action") == v3s
                  and comp.comprehend(toks).get("patient") == o)
              for toks, s, v3s, o in test)
    return hit / len(test), float(getattr(comp.res, "_last_mean_spikes", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        p05, spk05 = _parse_acc_at_dt(s, 0.5)
        p10, spk10 = _parse_acc_at_dt(s, 1.0)
        rows.append({"seed": s, "parse_dt05": p05, "spikes_dt05": round(spk05, 3),
                     "parse_dt10": p10, "spikes_dt10": round(spk10, 3)})
        print(f"[seed {s}] parse dt=0.5 {p05:.3f} (spk {spk05:.2f}) | parse dt=1.0 {p10:.3f} (spk {spk10:.2f})",
              flush=True)

    m05 = float(np.mean([r["parse_dt05"] for r in rows]))
    m10 = float(np.mean([r["parse_dt10"] for r in rows]))
    go = m10 >= 0.90
    verdict = "GO (dt=1.0 parses -> RUNG A.3 at dt=1.0)" if go else "BOUNDARY (dt=1.0 collapses -> re-tune/per-phase dt)"
    print(f"\n[emerge94] parse dt=0.5 {m05:.3f} | dt=1.0 {m10:.3f} -> {verdict}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "parse_dt05": m05, "parse_dt10": m10, "go_dt10": go}, fh, indent=2)


if __name__ == "__main__":
    main()
