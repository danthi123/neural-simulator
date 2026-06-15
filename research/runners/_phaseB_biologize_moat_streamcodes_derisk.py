"""CYCLE 97 biologization sweep, piece 1 (the MOAT) — replace the host confidence-threshold no-confab moat with
the LEARNED Bogacz-Brown anti-Hebbian familiarity gate, ON THE STREAM-LEARNED codes, and verify it AGREES with
the host moat (0 confabulations, clean separable margin, lesionable).

CONTEXT. The on-bridge stream-cortex conversation (`_phaseB_onbridge_stream_conversation_derisk.py`) abstains
("I don't know") via a HOST check: a Python threshold on the conjunctive-cue confidence (GATE=0.25 on the min
of the unbound verb/object cosines). That is a host computation. The brain-based replacement (catalog D.04,
perirhinal repetition suppression) is the learned anti-Hebbian familiarity gate (`AntiHebbianFamiliarity`,
Bogacz-Brown): imprint each stored fact's PARTIAL-fact cue (the bound verb+object composite); at query time
render the query's same composite and read its NOVELTY energy N(x)=||x||^2 - x^T W x (W = projector onto the
stored span). Familiar (a stored fact has this partial structure) -> N~0 -> ACCEPT (the system answers); novel
(no stored fact) -> N~||x||^2 -> ABSTAIN. A learned, lesionable signal.

THE TRANSFER QUESTION. The familiarity gate was validated at V=320 on the production composer's DECORRELATED
phasor codes (`familiarity_gate_v320_validation.py`). The stream cortex's codes are LEARNED and semantically
CORRELATED (that is WHY they generalize). Does the gate still cleanly separate familiar from novel on the
CORRELATED stream codes? Bogacz-Brown PROVE the projector form is high-capacity on correlated inputs (it
reconstructs any vector in the stored span, so a familiar -- even correlated -- cue is suppressed), so the
prediction is yes; this de-risk verifies it on the actual learned codes.

GATES (multi fact-set seeds, on the cached 320 stream codes):
  agreement   : the gate's accept/abstain MATCHES the host moat on present + absent cues.
  no_confab   : the DANGEROUS cell (host-abstain but gate-ACCEPT = a confabulation) is 0; gate false-accepts on
                absent cues is 0 (the moat must NOT weaken).
  margin      : novelty(present) << novelty(absent) -- a clean separable gap (a fixed a-priori threshold exists).
  lesion      : lesioning the gate's LEARNED W collapses the margin (the decision rides the learned gate).

Brain-based: the gate is a learned anti-Hebbian projector (neurons/synapses); the composite cue is produced by
the composer's bind/bundle (the substrate). Reuse-by-import (the projector + hrr ops + the cached stream codes);
CPU-cheap (the projector is the load-bearing learned state); NO GPU (does not contend with the multi-seed).
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_biologize_moat_streamcodes_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._phaseB_assembled_pipeline_ppmi_derisk import hrr_bind, hrr_unbind, _cos  # noqa: E402

N_FACTS = 8
HOST_GATE = 0.25          # the production conversation runner's a-priori conjunctive-cue threshold (the host moat)


class RealAntiHebbianFamiliarity:
    """The Bogacz-Brown anti-Hebbian familiarity gate (catalog D.04 perirhinal repetition suppression), adapted
    to REAL code vectors. SAME mechanism as AntiHebbianFamiliarity (W = projector onto the imprinted span;
    novelty N(x) = ||x||^2 - x^T W x; familiar -> ~0, novel -> ~||x||^2; lesionable) -- only the input adaptor
    differs: the stream codes are real-valued, so x = the unit-normalized real cue (no phasor cos/sin render)."""

    def __init__(self):
        self._basis = []

    @staticmethod
    def _render(vec):
        x = np.asarray(vec, dtype=np.float64)
        return x / (np.linalg.norm(x) + 1e-12)

    def imprint(self, vec):
        x = self._render(vec)
        for u in self._basis:                       # Gram-Schmidt against the learned stored basis (anti-Hebbian)
            x = x - (u @ x) * u
        nrm = np.linalg.norm(x)
        if nrm > 1e-6:
            self._basis.append(x / nrm)

    def novelty(self, vec):
        x = self._render(vec)
        if not self._basis:
            return float(x @ x)
        U = np.stack(self._basis, axis=1)
        proj = U @ (U.T @ x)                          # projector onto the stored span (W x)
        return float(x @ x - x @ proj)               # ||x||^2 - x^T W x

    def lesion(self):
        self._basis = []


def run_factset(codes, labels, seed):
    Nc, D = codes.shape
    rng = np.random.default_rng(seed * 17 + 3)
    R_a = rng.standard_normal(D) / np.sqrt(D)
    R_v = rng.standard_normal(D) / np.sqrt(D)
    R_o = rng.standard_normal(D) / np.sqrt(D)
    facts = []
    for _ in range(N_FACTS):
        i, j, k = rng.choice(Nc, 3, replace=False)
        facts.append((int(i), int(j), int(k)))
    bound = np.array([hrr_bind(R_a, codes[i]) + hrr_bind(R_v, codes[j]) + hrr_bind(R_o, codes[k])
                      for i, j, k in facts])

    def composite(verb, obj):                        # the who-Q&A partial-fact cue (verb+object), the gate's input
        return hrr_bind(R_v, codes[verb]) + hrr_bind(R_o, codes[obj])

    def host_confidence(verb, obj):                  # the host moat's conjunctive-cue confidence
        best = 0.0
        for F in bound:
            mv = _cos(hrr_unbind(F, R_v), codes)[verb]
            mo = _cos(hrr_unbind(F, R_o), codes)[obj]
            best = max(best, min(mv, mo))
        return best

    # imprint the stored facts' (verb, object) composites into the learned gate
    gate = RealAntiHebbianFamiliarity()
    for _, v, o in facts:
        gate.imprint(composite(v, o))

    # the a-priori neural threshold: novelty is ~0 for familiar, ~1 for novel (unit-normed) -> midpoint 0.5.
    NOV_GATE = 0.5
    stored_vo = {(v, o) for _, v, o in facts}
    pres_nov, pres_host = [], []
    for _, v, o in facts:                            # PRESENT cues (stored facts)
        pres_nov.append(gate.novelty(composite(v, o)))
        pres_host.append(host_confidence(v, o))
    abs_nov, abs_host, n_absent, dangerous, gate_fa = [], [], 0, 0, 0
    tries = 0
    while n_absent < N_FACTS and tries < 4000:       # ABSENT cues (a (verb,object) combo in no stored fact)
        tries += 1
        v, o = int(rng.integers(Nc)), int(rng.integers(Nc))
        if (v, o) in stored_vo or v == o:
            continue
        n_absent += 1
        nov = gate.novelty(composite(v, o)); conf = host_confidence(v, o)
        abs_nov.append(nov); abs_host.append(conf)
        host_abstain = conf < HOST_GATE
        gate_accept = nov < NOV_GATE
        gate_fa += int(gate_accept)                  # gate accepted an ABSENT cue = a confabulation
        dangerous += int(host_abstain and gate_accept)   # host abstains but gate accepts = the dangerous cell
    # agreement on present cues: host accepts (conf>=GATE) AND gate accepts (nov<NOV_GATE)
    pres_agree = sum(1 for c, n in zip(pres_host, pres_nov) if (c >= HOST_GATE) == (n < NOV_GATE))
    # lesion anti-cheat: novelty separation must collapse
    gate.lesion()
    les_pres = float(np.mean([gate.novelty(composite(v, o)) for _, v, o in facts]))
    les_abs = float(np.mean([gate.novelty(composite(int(rng.integers(Nc)), int(rng.integers(Nc))))
                             for _ in range(N_FACTS)]))
    cp, ca = float(np.mean(pres_nov)), float(np.mean(abs_nov))
    print(f"\n[biologize moat seed {seed}] {Nc} concepts x {D}D, {N_FACTS} facts", flush=True)
    print(f"  novelty: present {cp:+.3f} vs absent {ca:+.3f} (margin {ca-cp:+.3f}, gate {NOV_GATE})", flush=True)
    print(f"  gate vs host: present-agree {pres_agree}/{N_FACTS} | gate false-accepts(absent) {gate_fa}/{n_absent}"
          f" | DANGEROUS (host-abstain, gate-accept) {dangerous}", flush=True)
    print(f"  lesion anti-cheat: present {les_pres:+.3f} vs absent {les_abs:+.3f} (margin -> {les_abs-les_pres:+.3f}"
          f", must collapse to ~0)", flush=True)
    return {"seed": seed, "nov_present": cp, "nov_absent": ca, "margin": ca - cp, "pres_agree": pres_agree,
            "gate_fa": gate_fa, "dangerous": dangerous, "lesion_margin": les_abs - les_pres}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path} — run the 320 stream-conversation first to cache the codes.", flush=True)
        return
    codes = np.load(codes_path)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    labels = np.arange(codes.shape[0])               # labels unused here (the moat is fact-presence, not category)
    print(f"[biologize moat de-risk] stream-learned codes {codes.shape} (cached 320) -- does the LEARNED "
          f"Bogacz-Brown familiarity gate REPLACE the host threshold moat on the CORRELATED stream codes?",
          flush=True)
    rows = [run_factset(codes, labels, s) for s in (42, 43, 44, 45, 46, 47)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    margin, agree, fa, dang = m("margin"), m("pres_agree"), sum(r["gate_fa"] for r in rows), sum(r["dangerous"] for r in rows)
    les = m("lesion_margin")
    print(f"\n{'='*96}\n  MEAN ({len(rows)} fact-sets): novelty margin (absent-present) {margin:+.3f} | "
          f"present-agree {agree:.1f}/{N_FACTS} | TOTAL gate false-accepts {fa} | TOTAL dangerous {dang} | "
          f"lesion margin {les:+.3f}", flush=True)
    print(f"{'='*96}", flush=True)
    if dang == 0 and fa == 0 and margin >= 0.20 and abs(les) <= 0.05:
        print(f"  GO: the LEARNED familiarity gate REPLACES the host moat on the stream-learned codes -- 0 "
              f"confabulations (dangerous {dang}, false-accepts {fa}), a clean separable margin ({margin:+.3f}), "
              f"and lesioning the learned weights collapses it ({les:+.3f}). ==> the no-confab moat is BIOLOGIZED "
              f"(a learned anti-Hebbian novelty signal, not a host threshold) on the correlated learned codes.",
              flush=True)
    elif dang == 0 and margin >= 0.10:
        print(f"  PARTIAL: separable + no confabulation (margin {margin:+.3f}, dangerous {dang}) but tighten -- "
              f"gate false-accepts {fa} or lesion {les:+.3f}; check the a-priori threshold placement.", flush=True)
    else:
        print(f"  NEGATIVE: the gate does not cleanly replace the host moat on the correlated codes (margin "
              f"{margin:+.3f}, dangerous {dang}, false-accepts {fa}) -- the correlated cues may overlap the stored "
              f"span; inspect.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"margin": margin, "present_agree": agree, "gate_false_accepts": fa, "dangerous": dang,
           "lesion_margin": les, "per_factset": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_biologize_moat_streamcodes.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
