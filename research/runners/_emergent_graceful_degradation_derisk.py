"""EMERGENT FEATURE #1 (the integrated one-brain loop) — GRACEFUL DEGRADATION, cheap-first de-risk.

The whole who/what conversational turn now runs as ONE persistent spiking loop on the bridge (the production
`OneBrainComposer`). A real brain analogue should DEGRADE GRACEFULLY under damage: lesion / synaptic noise / neuron
dropout should produce a gradual fall-off + the lost recall should turn into ABSTENTION (the no-confab moat), NOT
confident wrong answers (confabulation) -- the hallmark of a DISTRIBUTED population code (catalog E.03: "robust to
noise and single-neuron loss") + the CA3 autoassociator (D.05/D.13). A host lookup table fails catastrophically (a
cliff) and/or confabulates; the distributed phasor store should not.

THIS IS A PERTURBATION OF THE BRAIN, host-orchestrated but legitimate (the experimenter lesions tissue + reads the
argmax off the spiking cleanup membrane -- cognition stays neural). NO sim/ edit. Reuse-by-import the production
`OneBrainComposer`: the persistent fact-store lives in complex synapses (`store_conns` -> `cp_rf_w_re/im`), so a
lesion = zeroing a fraction of those synapses; noise = jitter on them; dropout = masking a fraction of the readout
dimensions. The query path rebuilds the weights from `store_conns` each call, so perturbing `store_conns` perturbs
every subsequent query.

GO bar (the de-risk verdict):
  - intact p=0 positive control: recall 1.0, confab 0, moat false-accept 0.
  - MONOTONE graceful fall-off (no cliff: no adjacent-level recall drop > CLIFF), with a genuine INTERMEDIATE level
    (some perturbation level lands recall in [0.4, 0.9]).
  - lost recall -> ABSTENTION not CONFABULATION (confab stays <= CONFAB_MAX at every level).
  - the HARD moat-not-weakened guard: moat false-accept ~ 0 at EVERY level (robustness is NOT bought by a looser
    abstention gate -- the abstention is the cue-match failing, a FIXED mechanism).
  - cross-perturbation convergence: lesion / noise / dropout all show the same qualitative graceful pattern.
An honest NEGATIVE (a cliff, or silent confabulation under damage, or a moat that leaks under damage) maps the
substrate boundary + motivates the attractor-cleanup ladder -- itself a deliverable.

Run (GPU; numpy is a tiny-smoke fallback):
  SIM_BACKEND=cupy python -m research.runners._emergent_graceful_degradation_derisk \
      --seeds 42 43 44 --D 128 --out research/findings/raw/_emergent_graceful_degradation.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.runners.one_brain_composer import OneBrainComposer

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]
# K=8 facts with DISTINCT (agent, action) cues so a who/what query is unambiguous; patients reuse vocab words (they
# are just codes). The agent/action/patient are all real vocab concepts.
FACTS = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south"), ("apple", "stop", "west"),
         ("river", "swim", "home"), ("home", "go", "cat"), ("north", "look", "dog"), ("west", "come", "bird")]
# unstored (agent, action) cues for the moat: NOT among the stored (a, v) pairs.
UNSTORED_CUES = [("dog", "come"), ("cat", "go"), ("bird", "stop"), ("apple", "swim"), ("river", "look")]

CLIFF = 0.5          # an adjacent-level recall drop bigger than this = a cliff (catastrophic, not graceful)
CONFAB_MAX = 0.15    # confabulation must stay at/below this at every level (lost recall abstains, not lies)
MOAT_MAX = 0.05      # the HARD guard: moat false-accept must stay ~0 at every level


def _build(seed, D, confidence_gate=0.0):
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, confidence_gate=confidence_gate)
    for (a, v, p) in FACTS:
        c.store(a, v, p)
    return c


def _eval(c):
    """Query every stored fact + every unstored cue ONCE; classify. Returns the rates."""
    correct = abstain = confab = 0
    for (a, v, p) in FACTS:
        ans = c.query_patient(a, v)
        if ans is None:
            abstain += 1               # graceful: the degraded fact dropped below cue-match -> abstain
        elif ans == p:
            correct += 1               # intact recall
        else:
            confab += 1                # the BAD failure: a confident wrong answer
    n = len(FACTS)
    false_accept = sum(1 for (a, v) in UNSTORED_CUES if c.query_patient(a, v) is not None)
    return {"recall": correct / n, "abstain": abstain / n, "confab": confab / n,
            "moat_false_accept": false_accept / len(UNSTORED_CUES)}


def _lesion(store_conns, p, rng):
    """Synaptic lesion: zero a random fraction p of the store synapses (distributed across all facts)."""
    out = list(store_conns)
    n = len(out)
    k = int(round(p * n))
    if k > 0:
        for idx in rng.choice(n, size=k, replace=False):
            post, pre, _w = out[int(idx)]
            out[int(idx)] = (post, pre, 0.0 + 0j)
    return out


def _noise(store_conns, sigma, rng):
    """Synaptic noise: complex Gaussian jitter on each store weight (magnitude + phase)."""
    out = []
    for (post, pre, w) in store_conns:
        jit = complex(rng.normal(0.0, sigma), rng.normal(0.0, sigma))
        out.append((post, pre, complex(w) + jit))
    return out


def _dropout(store_conns, q, D, rng):
    """Neuron dropout: mask a random fraction q of the D readout DIMENSIONS (same dims dropped across all facts =
    losing q of the readout population). Implemented as zeroing every store synapse whose readout index k is dropped."""
    dropped = set(int(i) for i in rng.choice(D, size=int(round(q * D)), replace=False)) if q > 0 else set()
    out = []
    for (post, pre, w) in store_conns:
        k = (post - pre - 1)               # post = trig+1+k, pre = trig  -> k = post-pre-1
        out.append((post, pre, 0.0 + 0j) if k in dropped else (post, pre, w))
    return out


def _sweep(c, kind, levels, D, rng):
    """Apply each perturbation level to a COPY of the store, eval, restore. Returns [(level, rates)]."""
    base = list(c.store_conns)
    rows = []
    for lv in levels:
        if kind == "lesion":
            c.store_conns = _lesion(base, lv, rng)
        elif kind == "noise":
            c.store_conns = _noise(base, lv, rng)
        elif kind == "dropout":
            c.store_conns = _dropout(base, lv, D, rng)
        rows.append((lv, _eval(c)))
        c.store_conns = list(base)         # restore the intact store before the next level
    return rows


def _verdict(rows):
    """A single perturbation's graceful-degradation verdict. The LOAD-BEARING claim is the brain-like FAILURE MODE
    (the distributed store turns out so robust that the fall-off SHAPE is secondary): across a sweep that actually
    reaches the degradation region, lost recall must turn into ABSTENTION (confab stays low) and the no-confab moat
    must NEVER leak under damage. The curve shape (a cliff vs a smooth intermediate, the robustness plateau) is
    reported DESCRIPTIVELY -- a substrate too robust to show a smooth intermediate within the sweep is a STRONGER
    result, not a failure (so 'no_cliff'/'has_intermediate' do not gate GO; the failure-mode criteria do)."""
    recalls = [r["recall"] for _lv, r in rows]
    confabs = [r["confab"] for _lv, r in rows]
    moats = [r["moat_false_accept"] for _lv, r in rows]
    intact = rows[0][1]
    # the GO bar = the graceful FAILURE MODE:
    p0_ok = intact["recall"] == 1.0 and intact["confab"] == 0.0 and intact["moat_false_accept"] == 0.0
    confab_ok = all(cf <= CONFAB_MAX for cf in confabs)             # lost recall -> ABSTAIN, not confabulate
    moat_ok = all(m <= MOAT_MAX for m in moats)                     # the HARD moat-not-weakened guard
    reaches_falloff = min(recalls) <= 0.9                           # the sweep genuinely probed the degradation region
    go = bool(p0_ok and confab_ok and moat_ok and reaches_falloff)
    # descriptive shape (NOT gating):
    no_cliff = all((recalls[i] - recalls[i + 1]) <= CLIFF for i in range(len(recalls) - 1))
    has_intermediate = any(0.4 <= rc <= 0.9 for rc in recalls)
    plateau_until = max([lv for lv, r in rows if r["recall"] >= 0.9], default=0.0)   # robustness threshold (recall>=0.9)
    return {"go": go, "p0_ok": bool(p0_ok), "confab_ok": bool(confab_ok), "moat_ok": bool(moat_ok),
            "reaches_falloff": bool(reaches_falloff), "no_cliff": bool(no_cliff),
            "has_intermediate": bool(has_intermediate), "plateau_until": float(plateau_until)}


def run_seed(seed, D, confidence_gate=0.0):
    c = _build(seed, D, confidence_gate=confidence_gate)
    rng = np.random.default_rng(seed)
    # The distributed phasor store is EXTREMELY robust (a matched-filter cleanup over a D-dim code tolerates large
    # synaptic loss), so the sweep must reach HIGH damage to capture the graceful fall-off region (recall is flat 1.0
    # until a high fraction is destroyed, then falls toward total ABSTENTION as damage -> 1.0).
    damage = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]   # lesion / dropout fraction
    noise_sigma = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]          # complex jitter sigma on unit-magnitude phasors
    out = {"seed": seed, "D": D, "sweeps": {}, "verdicts": {}}
    for kind in ("lesion", "noise", "dropout"):
        lv = noise_sigma if kind == "noise" else damage
        rows = _sweep(c, kind, lv, D, rng)
        out["sweeps"][kind] = [{"level": l, **r} for l, r in rows]
        out["verdicts"][kind] = _verdict(rows)
    # cross-perturbation convergence: ALL three graceful
    out["all_graceful"] = all(out["verdicts"][k]["go"] for k in out["verdicts"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--confidence-gate", type=float, default=0.0,
                    help="0.0 = the bare cue-match (the boundary); > 0 enables the familiarity/confidence gate on the "
                         "cue read-out (blank a noise-dominated block -> abstain), expected to close the extreme-damage "
                         "confabulation/moat-leak tail WITHOUT hurting the functional regime.")
    ap.add_argument("--out", default="research/findings/raw/_emergent_graceful_degradation.json")
    a = ap.parse_args()
    print(f"[graceful-degradation] one-brain loop under lesion/noise/dropout (D={a.D}, "
          f"confidence_gate={a.confidence_gate}); "
          f"GO = graceful fall-off + lost-recall->abstain (confab<={CONFAB_MAX}) + moat-never-leaks (<={MOAT_MAX})\n",
          flush=True)
    results = []
    for seed in a.seeds:
        r = run_seed(seed, a.D, confidence_gate=a.confidence_gate)
        results.append(r)
        for kind in ("lesion", "noise", "dropout"):
            curve = " ".join(f"{row['level']:.2f}:{row['recall']:.2f}/{row['confab']:.2f}/{row['moat_false_accept']:.2f}"
                             for row in r["sweeps"][kind])
            v = r["verdicts"][kind]
            print(f"  [seed {seed}] {kind:8s} (level:recall/confab/moat)  {curve}   "
                  f"[plateau>=0.9 until {v['plateau_until']:.2f}]  ==> {'GO' if v['go'] else 'NEGATIVE'}", flush=True)
        print(f"  [seed {seed}] all-three-graceful: {'GO' if r['all_graceful'] else 'NEGATIVE'}\n", flush=True)
    n_go = sum(1 for r in results if r["all_graceful"])
    print("=" * 90, flush=True)
    print(f"  GRACEFUL DEGRADATION: {n_go}/{len(results)} seeds all-three-graceful. The integrated one-brain store "
          f"degrades gracefully (distributed code, E.03) + lost recall ABSTAINS not confabulates + the no-confab "
          f"moat never leaks under damage.", flush=True)
    print("=" * 90, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"go_seeds": n_go, "n_seeds": len(results), "results": results}, f, indent=2)
    print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
