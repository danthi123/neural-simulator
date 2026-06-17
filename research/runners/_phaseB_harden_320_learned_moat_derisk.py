"""HARDEN the 320-concept stream cortex, piece 1 (the no-confab MOAT) — replace the fixed HOST confidence
threshold (GATE=0.25) with the LEARNED Bogacz-Brown familiarity gate in the FULL who-Q&A conversation, and
verify it closes the seed-43 false-accept that the host gate left, on EACH seed's OWN stream-learned codes.

WHY (the owner-approved "harden the 320 cortex"). The 320-concept on-bridge stream cortex is 3-seed validated
(`_phaseB_onbridge_stream_conv_320_*` logs): who-Q&A recall 1.00 every seed; the no-confab moat abstains 1.00 on
seeds 42 + 44 (0 false-accepts) but 0.88 on seed 43 (1 false-accept) at the FIXED host gate 0.25 — the present/
absent confidences ARE separable there (+0.464 vs +0.064), so it is a GATE-PLACEMENT artifact of the fixed host
threshold, NOT a binding failure. The brain-based fix is the LEARNED anti-Hebbian familiarity gate (catalog D.04,
perirhinal repetition suppression; Bogacz-Brown), already validated CLEANER than the host moat on seed-42 codes
(`_phaseB_biologize_moat_streamcodes_derisk.py`, 0 false-accepts). THE OPEN QUESTION this closes: that de-risk only
ran on SEED-42's codes with varied fact-sets — it never tested the gate on SEED-43's actual (lower-fidelity) codes,
which is exactly where the host gate failed. This runner runs the learned gate on EACH seed's OWN cached codes in
the full who-Q&A pipeline.

GATE (the 3 cached seeds 42/43/44, extendable):
  recall    : who-Q&A recall on PRESENT facts >= 0.70 every seed (binding intact — accept decision is the learned
              gate, not the host threshold).
  no_confab : the learned gate's false-accepts on ABSENT (verb,object) cues == 0 on EVERY seed (the moat must NOT
              weaken; seed-43 must go 1 -> 0).
  margin    : novelty(absent) >> novelty(present) — a clean a-priori-separable gap every seed.
  lesion    : lesioning the gate's LEARNED projector collapses the margin (the decision rides the learned weights).
ANTI-CHEAT: the novelty threshold NOV_GATE=0.5 is a-priori (novelty ~0 familiar, ~1 novel for unit-normed cues —
the midpoint), NOT tuned on the test; the gate imprints only the STORED facts (train); absent cues are genuinely
absent; codes are the bridge's own stream-learned codes (loaded per-seed, not re-derived).

Reuse-by-import (RealAntiHebbianFamiliarity + hrr ops + the cached per-seed stream codes). CPU/numpy, NO GPU
(does not contend with any GPU run). NO sim/ edit.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_harden_320_learned_moat_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._phaseB_assembled_pipeline_ppmi_derisk import hrr_bind, hrr_unbind, _cos  # noqa: E402
from research.runners._phaseB_biologize_moat_streamcodes_derisk import RealAntiHebbianFamiliarity  # noqa: E402

N_FACTS = 8
HOST_GATE = 0.25     # the production conversation runner's fixed host threshold (for the side-by-side comparison)
NOV_GATE = 0.5       # the learned gate's a-priori novelty threshold (unit-norm midpoint; NOT tuned on the test)


def run_seed(codes, seed):
    """The EXACT CYCLE-90 / conversation-runner fact construction + who-Q&A recall, but the ACCEPT/ABSTAIN decision
    is the LEARNED familiarity gate instead of the fixed host threshold. Returns recall + the learned-gate moat +
    the host-gate moat (side by side) + the lesion control."""
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    Nc, D = codes.shape
    rng = np.random.default_rng(seed * 17 + 3)          # SAME stream as the conversation runner -> same facts
    R_a = rng.standard_normal(D) / np.sqrt(D)
    R_v = rng.standard_normal(D) / np.sqrt(D)
    R_o = rng.standard_normal(D) / np.sqrt(D)
    facts = []
    for _ in range(N_FACTS):
        i, j, k = rng.choice(Nc, 3, replace=False)
        facts.append((int(i), int(j), int(k)))
    bound = np.array([hrr_bind(R_a, codes[i]) + hrr_bind(R_v, codes[j]) + hrr_bind(R_o, codes[k])
                      for i, j, k in facts])

    def composite(verb, obj):                            # the partial-fact cue the learned gate reads
        return hrr_bind(R_v, codes[verb]) + hrr_bind(R_o, codes[obj])

    def cue_match(verb, obj):                            # the host conjunctive-cue confidence + the best fact index
        scores = []
        for F in bound:
            mv = _cos(hrr_unbind(F, R_v), codes)[verb]
            mo = _cos(hrr_unbind(F, R_o), codes)[obj]
            scores.append(min(mv, mo))
        scores = np.array(scores)
        return int(np.argmax(scores)), float(scores.max())

    # learn the gate on the STORED facts' (verb,object) composites (train only)
    gate = RealAntiHebbianFamiliarity()
    for _, v, o in facts:
        gate.imprint(composite(v, o))

    # --- PRESENT facts: recall the agent IF accepted (learned gate), measure recall + present novelty/host-conf ---
    recall_learned, recall_host, pres_nov, pres_host = 0, 0, [], []
    for (i, j, k), F in zip(facts, bound):
        bf, conf = cue_match(j, k)
        nov = gate.novelty(composite(j, k))
        pres_nov.append(nov); pres_host.append(conf)
        # learned-gate-accepted recall (the production decision under the new moat)
        if nov < NOV_GATE:
            pred = int(np.argmax(_cos(hrr_unbind(bound[bf], R_a), codes)))
            recall_learned += int(pred == i)
        # host-gate-accepted recall (for the side-by-side)
        if conf >= HOST_GATE:
            pred = int(np.argmax(_cos(hrr_unbind(bound[bf], R_a), codes)))
            recall_host += int(pred == i)

    # --- ABSENT (verb,object) cues: count false-accepts under BOTH gates ---
    stored_vo = {(v, o) for _, v, o in facts}
    abs_nov, abs_host, n_absent, fa_learned, fa_host, tries = [], [], 0, 0, 0, 0
    while n_absent < N_FACTS and tries < 4000:
        tries += 1
        v, o = int(rng.integers(Nc)), int(rng.integers(Nc))
        if (v, o) in stored_vo or v == o:
            continue
        n_absent += 1
        nov = gate.novelty(composite(v, o)); _, conf = cue_match(v, o)
        abs_nov.append(nov); abs_host.append(conf)
        fa_learned += int(nov < NOV_GATE)               # learned gate accepted an ABSENT cue = confabulation
        fa_host += int(conf >= HOST_GATE)               # host gate's false-accept (the seed-43 failure)

    # --- lesion anti-cheat: wipe the learned projector -> the novelty separation must collapse ---
    gate.lesion()
    les_pres = float(np.mean([gate.novelty(composite(j, k)) for _, j, k in facts]))
    les_abs = float(np.mean([gate.novelty(composite(int(rng.integers(Nc)), int(rng.integers(Nc))))
                             for _ in range(N_FACTS)]))

    cp_n, ca_n = float(np.mean(pres_nov)), float(np.mean(abs_nov))
    out = {"seed": seed, "n_concepts": int(Nc),
           "recall_learned": recall_learned / N_FACTS, "recall_host": recall_host / N_FACTS,
           "fa_learned": int(fa_learned), "fa_host": int(fa_host), "n_absent": int(n_absent),
           "nov_present": cp_n, "nov_absent": ca_n, "nov_margin": ca_n - cp_n,
           "lesion_margin": les_abs - les_pres}
    print(f"\n[harden moat seed {seed}] {Nc} concepts x {D}D, {N_FACTS} facts", flush=True)
    print(f"  recall: learned-gate {out['recall_learned']:.2f} (host-gate {out['recall_host']:.2f}) | "
          f"false-accepts: LEARNED {fa_learned}/{n_absent}  vs  HOST {fa_host}/{n_absent}", flush=True)
    print(f"  novelty: present {cp_n:+.3f} vs absent {ca_n:+.3f} (margin {out['nov_margin']:+.3f}, gate {NOV_GATE}) "
          f"| lesion margin -> {out['lesion_margin']:+.3f} (must collapse ~0)", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--codes-template", default="research/findings/raw/_phaseB_stream_codes_320_seedSEED.npy")
    p.add_argument("--out", default="research/findings/raw/_phaseB_harden_320_learned_moat.json")
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[harden 320 moat] does the LEARNED familiarity gate close the seed-43 false-accept the FIXED host gate "
          f"left, on EACH seed's OWN cached 320 stream codes? seeds={seeds}", flush=True)

    rows = []
    for s in seeds:
        cpath = os.path.join(_REPO, a.codes_template.replace("SEED", str(s)))
        if not os.path.exists(cpath):
            print(f"  [missing] {cpath} — skip seed {s}", flush=True)
            continue
        rows.append(run_seed(np.load(cpath), s))

    if not rows:
        print("  [no codes found] run the 320 stream-conversation first to cache the per-seed codes.", flush=True)
        raise SystemExit(2)

    recall = float(np.mean([r["recall_learned"] for r in rows]))
    fa_learned = sum(r["fa_learned"] for r in rows)
    fa_host = sum(r["fa_host"] for r in rows)
    margin = float(np.mean([r["nov_margin"] for r in rows]))
    les = float(np.mean([r["lesion_margin"] for r in rows]))
    all_recall_ok = all(r["recall_learned"] >= 0.70 for r in rows)
    all_moat_ok = all(r["fa_learned"] == 0 for r in rows)
    all_margin_ok = all(r["nov_margin"] >= 0.20 for r in rows)
    lesion_ok = abs(les) <= 0.05

    go = bool(all_recall_ok and all_moat_ok and all_margin_ok and lesion_ok)
    verdict = "GO" if go else "NEGATIVE"
    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({len(rows)} seeds): recall {recall:.2f} | LEARNED-gate false-accepts {fa_learned} "
          f"(HOST-gate {fa_host}) | novelty margin {margin:+.3f} | lesion {les:+.3f}", flush=True)
    print(f"  per-seed false-accepts (learned vs host): "
          + " ".join(f"s{r['seed']}:{r['fa_learned']}v{r['fa_host']}" for r in rows), flush=True)
    print(f"  ==> {verdict}\n{'='*100}", flush=True)
    if go:
        print(f"  GO: the LEARNED familiarity gate HARDENS the 320 moat — recall {recall:.2f} intact, and "
              f"false-accepts go to {fa_learned} on EVERY seed (host left {fa_host}, incl. seed-43's 1). A clean "
              f"a-priori-separable novelty margin ({margin:+.3f}); lesioning the learned projector collapses it "
              f"({les:+.3f}). ==> the seed-43 gate-placement gap is closed by the BRAIN-BASED moat (not a tuned "
              f"threshold), all on the bridge's own stream-learned codes. NEXT: piece 2 (on-brain read-out "
              f"normalization) + fold into the production 320 conversation.", flush=True)
    else:
        why = []
        if not all_recall_ok: why.append(f"recall<0.70 on a seed (binding)")
        if not all_moat_ok: why.append(f"learned-gate false-accepts remain ({fa_learned}) — the gate did NOT close seed-43")
        if not all_margin_ok: why.append("novelty margin <0.20 on a seed")
        if not lesion_ok: why.append(f"lesion did not collapse ({les:+.3f})")
        print(f"  NEGATIVE: {'; '.join(why)}. Honest — the learned gate is not a clean drop-in on these codes; "
              f"route to the +stream lever (more windows widens the gap) or a per-seed-train-calibrated threshold "
              f"(NOT tuned on test). The host moat is NOT weakened meanwhile.", flush=True)

    os.makedirs(os.path.dirname(os.path.join(_REPO, a.out)), exist_ok=True)
    with open(os.path.join(_REPO, a.out), "w") as fh:
        json.dump({"verdict": verdict, "recall": recall, "fa_learned": fa_learned, "fa_host": fa_host,
                   "nov_margin": margin, "lesion_margin": les, "per_seed": rows}, fh, indent=2, default=str)
    print(f"  [saved] {a.out}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    raise SystemExit(0 if go else 1)


if __name__ == "__main__":
    main()
