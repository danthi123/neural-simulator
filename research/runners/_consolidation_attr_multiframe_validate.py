"""Consolidation validation (richer-syntax #1 + #2 on the PRODUCTION agent, the LEARNED 320 codes) — does folding
attributed single-attribute entities + auto-selected multi-frame comprehension into the production
BrainConversationalAgent (composer_kind="onebrain") HOLD on the 320 stream-learned cortex codes, with the no-confab
moat intact and the flat-SVO path un-regressed?

Pre-registered by `2026-06-19-conversational-scaling-next-lever-scoping.md` (#1 = consolidation) and scoped by
`2026-06-19-resonator-on-learned-codes-derisk.md` (ship single-attribute + multi-frame, NOT the F=3 two-attribute
path, which degrades to ~29% on the correlated learned codes).

WHAT IT RUNS, all on ONE production agent built on the learned 320 codes (the same grounding map the
consolidated-320 demo uses, verbatim):
  - FLAT SVO who/what/yes-no + the moat  (the validated production matrix; the NON-REGRESSION baseline).
  - ATTRIBUTED single-attribute facts ("dog eat big apple") -> what_does returns "big apple"  (the 2-factor path).
  - MULTI-FRAME comprehension: the SAME fact heard in VSO ("eat lion cake") / OSV ("garden rabbit jump") word order
    -> the agent auto-selects the frame (verb-position -> neural FrameSelector) and comprehends it -> who/what work.
  - The no-confab MOAT, asserted THROUGHOUT (an unstored attributed/multi-frame/flat cue -> None; 0 false-accepts).

The richer paths are NEURAL (the attributed parse is the spiking AttributedBridgeParser; the multi-frame parse is the
spiking FrameSelector + MultiFrameParser; the bind/unbind is the resonate-and-fire composer). The host steps stay the
environment (token string) + body (emit the words) + the known-verb lexical lookup (the morphology front end).

PRE-REGISTERED GATE (FROZEN; the 320 codes give 3 neural + 3 host seed-runs = 6; >=5/6):
  GO       = on the consolidated agent, for every seed-run:
               flat-SVO recall == 1.0 AND flat-SVO == the flags-OFF baseline (NON-REGRESSION) AND
               one-attribute attributed recall >= 0.90 AND
               multi-frame comprehension correct (who/what on the auto-selected-frame facts) AND
               the moat holds (0 false-accepts on the unstored attributed/multi-frame/flat cues),
             on >=5/6 seed-runs.
  PARTIAL  = flat un-regressed + moat intact, but attributed OR multi-frame is seed-fragile (< the bar) -> localize.
  MOAT_BREACH (HARD STOP) = any false-accept on the consolidated agent -> the no-confab guarantee failed.

GPU (SIM_BACKEND=cupy) for the 320-scale matrix (the on-bridge parsers train on the substrate). numpy = a tiny smoke.
Run:  SIM_BACKEND=cupy python -u -m research.runners._consolidation_attr_multiframe_validate --readout both
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8  # noqa: E402
from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories  # noqa: E402
# the grounding map (the step-3 pattern, verbatim from the production demo)
from research.runners.consolidated_320_conversation_demo import _projection, grounded_phases  # noqa: E402

D = 128

# --- the conversation, all words real 320-taxonomy tokens -------------------
# FLAT SVO facts (the production non-regression baseline -- the same shape the consolidated-320 demo uses).
FLAT_FACTS = [
    ("dog", "eat", "apple"),
    ("cat", "play", "ball"),
    ("bird", "sleep", "tree"),
    ("mouse", "walk", "house"),
]
# unstored (agent, action) cues that MUST abstain (real words, never stored together).
FLAT_ABSENT = [("dog", "sing"), ("cat", "run"), ("bird", "eat"), ("mouse", "jump")]

# ATTRIBUTED single-attribute facts: 'S V adj N' -> what_does(S,V) == 'adj N'. All real taxonomy words.
ATTR_FACTS = [
    ("fish", "eat", "red", "cake"),     # fish eat red cake -> 'red cake'
    ("frog", "look", "big", "box"),     # frog look big box -> 'big box'
    ("bear", "play", "blue", "kite"),   # bear play blue kite -> 'blue kite'
    ("duck", "walk", "green", "road"),  # duck walk green road -> 'green road'
]
ATTR_ABSENT = [("fish", "sleep"), ("frog", "run")]    # unstored attributed cues -> abstain

# MULTI-FRAME facts: stored in a NON-NATIVE frame the agent must auto-select. (agent, action, patient) is the TRUTH;
# the sentence presents them in VSO ("action agent patient") or OSV ("patient agent action") order.
MF_FACTS = [
    ("lion", "eat", "milk", "VSO"),     # VSO sentence: 'eat lion milk'   -> agent=lion action=eat patient=milk
    ("rabbit", "jump", "garden", "OSV"),  # OSV sentence: 'garden rabbit jump' -> agent=rabbit action=jump patient=garden
]
MF_ABSENT = [("lion", "sleep"), ("rabbit", "walk")]   # unstored multi-frame cues -> abstain

# the known-verb lexicon (the morphology front end the FrameSelector uses to find the verb position).
VERBS = {"run", "jump", "walk", "play", "look", "eat", "sleep", "sing"}

FRAME_ORDER = {  # frame -> the position of (agent, action, patient) in the surface sentence
    "SVO": ("agent", "action", "patient"),
    "VSO": ("action", "agent", "patient"),
    "OSV": ("patient", "agent", "action"),
}


def _mf_sentence(agent, action, patient, frame):
    role_word = {"agent": agent, "action": action, "patient": patient}
    return " ".join(role_word[r] for r in FRAME_ORDER[frame])


def _build_agent(seed, grounded, concepts, attributed, multiframe):
    return BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded,
                                    composer_kind="onebrain", enable_neural_render=False,
                                    enable_attributed=attributed, enable_multiframe=multiframe)


def run_seed(seed, codes, vocab, readout):
    proj = _projection(D, codes.shape[1], seed)
    grounded = {vocab[i]: grounded_phases(codes[i], proj) for i in range(len(vocab))}
    concepts = {vocab[i]: codes[i] for i in range(len(vocab))}

    # ---- (A) the CONSOLIDATED agent (flags ON) ----
    agent = _build_agent(seed, grounded, concepts, attributed=True, multiframe=True)
    for a, v, o in FLAT_FACTS:
        agent.hear(f"{a} {v} {o}", polarity="AFFIRM")
    for a, v, adj, n in ATTR_FACTS:
        agent.hear_attributed(f"{a} {v} {adj} {n}", polarity="AFFIRM")
    for a, v, o, frame in MF_FACTS:
        agent.hear_multiframe(_mf_sentence(a, v, o, frame), VERBS, polarity="AFFIRM")

    # flat-SVO recall (who AND what) -- the production matrix
    flat_ok = flat_tot = 0
    for a, v, o in FLAT_FACTS:
        flat_tot += 2
        flat_ok += int(agent.what_does(a, v) == o) + int(agent.who_does(v, o) == a)
    flat_recall = flat_ok / flat_tot

    # attributed single-attribute recall: what_does(S,V) == 'adj N'
    attr_ok = 0
    for a, v, adj, n in ATTR_FACTS:
        attr_ok += int(agent.what_does(a, v) == f"{adj} {n}")
    attr_recall = attr_ok / len(ATTR_FACTS)

    # multi-frame comprehension: who/what on the auto-selected-frame facts
    mf_ok = mf_tot = 0
    for a, v, o, _frame in MF_FACTS:
        mf_tot += 2
        mf_ok += int(agent.what_does(a, v) == o) + int(agent.who_does(v, o) == a)
    mf_recall = mf_ok / mf_tot

    # the no-confab moat THROUGHOUT (flat + attributed + multi-frame unstored cues -> None)
    false_accept, breaches = 0, []
    for a, v in FLAT_ABSENT + ATTR_ABSENT + MF_ABSENT:
        ans = agent.what_does(a, v)
        if ans is not None:
            false_accept += 1
            breaches.append(f"what_does({a},{v}) -> {ans!r} (should abstain)")

    # ---- (B) the NON-REGRESSION baseline (flags OFF = the byte-identical default onebrain path) ----
    base = _build_agent(seed, grounded, concepts, attributed=False, multiframe=False)
    for a, v, o in FLAT_FACTS:
        base.hear(f"{a} {v} {o}", polarity="AFFIRM")
    base_ok = base_tot = 0
    for a, v, o in FLAT_FACTS:
        base_tot += 2
        base_ok += int(base.what_does(a, v) == o) + int(base.who_does(v, o) == a)
    base_recall = base_ok / base_tot
    flat_unregressed = flat_recall >= base_recall and flat_recall == 1.0

    go = (flat_recall == 1.0 and flat_unregressed and attr_recall >= 0.90 and mf_recall == 1.0
          and false_accept == 0)
    return {
        "seed": seed, "readout": readout,
        "flat_recall": flat_recall, "base_recall": base_recall, "flat_unregressed": bool(flat_unregressed),
        "attr_recall": attr_recall, "mf_recall": mf_recall,
        "false_accept": false_accept, "moat_breach": false_accept > 0, "breaches": breaches, "go": bool(go),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", choices=["neural", "host", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_consolidation_attr_multiframe.json")
    a = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")

    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    readouts = ["neural", "host"] if a.readout == "both" else [a.readout]

    print(f"[consolidation attr+multiframe] richer-syntax #1+#2 on the production agent, the LEARNED 320 codes -- "
          f"does it HOLD with the moat intact + flat un-regressed?\n", flush=True)
    results, hard_stop = [], False
    for readout in readouts:
        suffix = "neural_seed" if readout == "neural" else "seed"
        for seed in a.seeds:
            cpath = os.path.join(_REPO, "research", "findings", "raw",
                                 f"_phaseB_stream_codes_320_{suffix}{seed}.npy")
            if not os.path.exists(cpath):
                print(f"  [{readout} seed {seed}] SKIP -- no codes at {cpath}", flush=True)
                continue
            codes = np.load(cpath)
            codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
            r = run_seed(seed, codes, vocab, readout)
            results.append(r)
            tag = "GO" if r["go"] else ("MOAT_BREACH" if r["moat_breach"] else "PARTIAL")
            print(f"  [{readout} seed {seed}] flat {r['flat_recall']:.2f} (base {r['base_recall']:.2f}, "
                  f"un-regressed {r['flat_unregressed']}) | attr {r['attr_recall']:.2f} | "
                  f"multi-frame {r['mf_recall']:.2f} | false-accept {r['false_accept']}  ==> {tag}", flush=True)
            for b in r["breaches"]:
                print(f"      !! {b}", flush=True)
            hard_stop = hard_stop or r["moat_breach"]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2, default=str)

    n_go = sum(r["go"] for r in results)
    n = len(results)
    bar = int(np.ceil(5 / 6 * n)) if n else 0
    print(f"\n{'='*100}", flush=True)
    if hard_stop:
        print("  MOAT_BREACH (HARD STOP): the consolidated agent accepted an unstored query on the learned codes -- "
              "the no-confab guarantee failed; do NOT ship.", flush=True)
    elif n and n_go >= bar:
        def mean(k):
            return float(np.mean([r[k] for r in results]))
        print(f"  GO ({n_go}/{n} seed-runs, bar {bar}): the production agent now does FLAT SVO (recall "
              f"{mean('flat_recall'):.2f}, un-regressed) + single-ATTRIBUTE 'big apple' (recall {mean('attr_recall'):.2f}) "
              f"+ auto-selected MULTI-FRAME comprehension (recall {mean('mf_recall'):.2f}) on the 320 stream-learned "
              "codes, with 0 false-accepts (the moat held). The F=3 two-attribute path stays the documented boundary "
              "(~29% on the correlated learned codes).", flush=True)
    elif n:
        print(f"  PARTIAL ({n_go}/{n} seed-runs GO, bar {bar}): the moat held (no breach) + flat un-regressed, but "
              "attributed or multi-frame is seed-fragile -- localize which capability + why.", flush=True)
    else:
        print("  NO CODES -- run the 320 stream cortex first to produce the cached codes.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
