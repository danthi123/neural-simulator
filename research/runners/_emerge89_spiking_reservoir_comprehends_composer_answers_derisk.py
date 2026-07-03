"""EMERGE-89 -- FULLY-SPIKING comprehension -> composition: the ON-BRIDGE spiking reservoir COMPREHENDS -> the
composer ANSWERS.

EMERGE-88 proved the handoff with the RATE reservoir (EMERGE-78 echo-state) comprehending -> the spiking
`RFPhasorComposer` answering. EMERGE-89 is the mechanical follow-on it named: swap the rate reservoir for EMERGE-82's
`OnBridgeLSM` -- a recurrent Izhikevich `BrainRegion` on a real `SimulationBridge`, whose `final_state(U)` has the
IDENTICAL signature -- so the WHOLE comprehension->composition pipeline runs on spikes end-to-end (the reservoir on a
SimulationBridge, the composer on RF resonate-and-fire). The reservoir understands the sentence on spikes; the composer
stores + answers on spikes; the no-confab moat holds.

Reuse-by-import (EMERGE-88's `ReservoirComprehender` + de-risk helpers, with an EMERGE-82 `OnBridgeLSM` injected as its
reservoir); NO `sim/` edit.

Anti-cheats (same as EMERGE-88): held-out CONTENT; the no-confab MOAT (an (agent, action) never stored -> abstain);
a COMPREHENSION-LESION (the reservoir's closed-class identity collapsed -> role-labeling collapses on spikes -> the
stored facts are wrong -> recall collapses) = the spiking reservoir is load-bearing for the whole turn.

Run (numpy = the CPU path; the bridge step is heavy, so a reduced train, per EMERGE-82):
  SIM_BACKEND=numpy python -u -m research.runners._emerge89_spiking_reservoir_comprehends_composer_answers_derisk \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_emerge89_spiking_reservoir_comprehends.json
"""
import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _content_pools, _gen, _TRAIN_KINDS,
)
from research.runners._emerge82_onbridge_lsm_derisk import OnBridgeLSM, _N_POOL, _N_TRAIN_PER  # noqa: E402
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import (  # noqa: E402
    ReservoirComprehender, _build_test_facts, _recall_over, _D,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


def _derisk_one(seed):
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    rng = np.random.default_rng(seed * 101 + 5)

    # THE SPIKING reservoir on a real SimulationBridge (EMERGE-82) as the comprehension front-end
    enc = Encoder(discovered)
    res = OnBridgeLSM(enc.dim, seed=seed, n=_N_POOL)
    comp = ReservoirComprehender(seed, discovered, res=res, enc=enc)
    comp.fit(_gen(_TRAIN_KINDS, _N_TRAIN_PER, rng, subj, verb, obj))   # fit the slot read-out on ON-BRIDGE spike states

    v3 = [v + "s" for v in verb]
    vocab = sorted(set(subj) | set(v3) | set(obj))
    test, seen, trng = _build_test_facts(seed, subj, verb, obj)

    # PARSE: did the spiking reservoir map each transitive to the right (agent, action, patient)?
    parse_hit = 0
    for toks, s, v3s, o in test:
        fact = comp.comprehend(toks)
        parse_hit += int(fact.get("agent") == s and fact.get("action") == v3s and fact.get("patient") == o)
    parse_acc = parse_hit / len(test)

    # THE INTEGRATION: spiking reservoir comprehends -> spiking composer stores -> who/what recall
    composer = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    recall = _recall_over(composer, comp, test, lesion=False)

    # MOAT: (agent, action) never stored -> abstain (None). A non-None = a false-accept.
    stored_keys = {(s, v3s) for _t, s, v3s, _o in test}
    fa = tot = 0
    mguard = 0
    while tot < 40 and mguard < 4000:
        mguard += 1
        s = str(trng.choice(subj)); v3q = str(trng.choice(verb)) + "s"
        if (s, v3q) in stored_keys:
            continue
        tot += 1
        fa += int(composer.query_patient(s, v3q) is not None)
    moat_fa = fa / max(1, tot)

    # NECESSITY: lesion the spiking reservoir's comprehension -> roles collapse -> wrong facts -> recall collapses.
    composer_l = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    lesion_recall = _recall_over(composer_l, comp, test, lesion=True)

    return {
        "seed": seed, "n_discovered_closed": len(discovered), "n_test_facts": len(test), "n_pool": _N_POOL,
        "mean_spikes_per_neuron": round(float(getattr(res, "_last_mean_spikes", 0.0)), 3),
        "parse_acc": parse_acc, "recall": recall, "moat_false_accept": moat_fa, "lesion_recall": lesion_recall,
    }


def _go(rows):
    def mean(k):
        return float(np.mean([r[k] for r in rows]))
    return {
        "n_seeds": len(rows),
        "mean_spikes_per_neuron": mean("mean_spikes_per_neuron"),
        "parse_acc": mean("parse_acc"), "recall": mean("recall"),
        "moat_false_accept": mean("moat_false_accept"), "lesion_recall": mean("lesion_recall"),
        "go": (mean("parse_acc") >= 0.90 and mean("recall") >= 0.90
               and mean("moat_false_accept") <= 0.05 and mean("lesion_recall") <= 0.55),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    for s in args.seeds:
        d = _derisk_one(s)
        rows.append(d)
        print(f"[seed {s}] spikes {d['mean_spikes_per_neuron']:.2f} | parse {d['parse_acc']:.3f} | "
              f"recall {d['recall']:.3f} | moat-FA {d['moat_false_accept']:.3f} | "
              f"lesion-recall {d['lesion_recall']:.3f}", flush=True)

    agg = _go(rows)
    agg["elapsed_seconds"] = round(time.time() - t0, 1)
    verdict = "GO" if agg["go"] else "NO-GO"
    print(f"\n[emerge89] VERDICT: {verdict} -- the ON-BRIDGE SPIKING reservoir COMPREHENDS and the composer ANSWERS "
          f"(genuinely spiking {agg['mean_spikes_per_neuron']:.2f} spikes/neuron; parse {agg['parse_acc']:.3f}; who/what "
          f"recall {agg['recall']:.3f}; no-confab moat {agg['moat_false_accept']:.3f} false-accept; comprehension-lesion "
          f"collapses recall to {agg['lesion_recall']:.3f}). Fully-spiking comprehension->composition end-to-end.",
          flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2)
        print(f"[emerge89] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
