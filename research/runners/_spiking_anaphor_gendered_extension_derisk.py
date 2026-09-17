"""EXTEND the spiking CA3 anaphor-detection organ to he/she/him/her -- the EMPIRICAL verification of the
additive, opt-in `extra_anaphors` capability landed on `spiking_anaphor_detection_organ.SpikingAnaphorDetectorOrgan`
(mechanism de-risk this extends: `_spiking_anaphor_detection_derisk.py`, finding
`2026-09-09-spiking-anaphor-detection-CA3-pattern-completion-6seed-GO.md`; production wire-in for the original
5-token set default-ON since 2026-09-16, `2026-09-16-wiring-flips-novelty-anaphor-qroute-DEFAULT-ON-integrated-
no-regression-GO.md`). See the organ module's own "CAPABILITY EXTENSION" docstring section for the mechanism
argument (SAME `SpikingLoopContextBuffer`, SAME Hebbian-outer-product attractor install, SAME `decide_pronoun`
threshold read -- the only change is a longer concept list passed at construction); this runner is the proof.

THE CLAIM UNDER TEST. Two things, both required, neither implied by the other:
  (1) The 4 NEW tokens (he/she/him/her) recruit CA3 attractor assemblies via the identical mechanism and inherit
      the SAME properties the original de-risk proved for it/that/they/them/this: clean-cue recall, noisy/
      partial-cue pattern-completion SURPASSING exact-match, specificity against content words, and lesion
      (attractor_weight=0) collapse.
  (2) Recruiting them does NOT perturb the original 5 -- a baseline organ (extra_anaphors=None) and an EXTENDED
      organ (extra_anaphors=EXTRA_ANAPHORS), same seed, must produce IDENTICAL detection decisions AND
      IDENTICAL peak firing rates on every original anaphor, not merely "the same verdict".

GATE (pre-registered, 6 project-standard seeds [42,43,44,100,101,102], numpy-CPU, NO sim/ edit -- the ONLY
non-test code change is the organ's own additive `extra_anaphors` kwarg, already landed):
  G0 BYTE-IDENTICAL DIFFERENTIAL: baseline vs extended organ, same seed, agree EXACTLY (same is_anaphor bool AND
     bit-identical peak firing rate) on all 5 original anaphors and on N_CONTENT unknown-string probes.
  G1 clean-cue recall (the 4 NEW tokens): mean accuracy over their own clean-cue probes             >= 0.90
  G2 noisy/partial-cue completion (the 4 NEW tokens), KEEP_FRAC=20% (80% corrupted, the de-risk's own
     calibrated operating point, reused verbatim via `organ.probe_corrupted_cue`):                   >= 0.85
  G3 specificity on the EXTENDED (9-concept) substrate: false-positive rate over N_CONTENT genuine
     unused-pool cues (drawn directly, mirroring the de-risk's own G3 methodology exactly)            <= 0.15
  G3b (auxiliary, exact-all, not a threshold): a batch of realistic content-word STRINGS, run through the
     organ's actual public `is_anaphor()` surface (the production call), are ALL rejected.
  G4 lesion collapse (extended organ, attractor_weight=0): combined clean+noisy accuracy on he/she/him/her
     <= 0.30 AND attributable_to(intact, lesion) >= 0.5
Per-seed GO = G0 and G1 and G2 and G3 and G3b and G4. Board GO = >=5/6 seeds (the same project convention the
de-risk this extends used).

Run:
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._spiking_anaphor_gendered_extension_derisk \\
      --seeds 42 43 44 100 101 102 --n-noisy-repeats 4 --n-content 10 \\
      --out research/findings/raw/_spiking_anaphor_gendered_extension/decisive_6seed.json

Smoke (tiny, 1 seed, matches the commit-time check for this build):
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._spiking_anaphor_gendered_extension_derisk \\
      --seeds 42 --n-noisy-repeats 1 --n-content 3 \\
      --out research/findings/raw/_spiking_anaphor_gendered_extension/smoke_1seed.json
"""
from __future__ import annotations

import os
import sys
import json
import argparse

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to, void_if  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from research.runners._spiking_anaphor_detection_derisk import ANAPHORS, KEEP_FRAC, PATTERN_SIZE  # noqa: E402
from research.runners.spiking_anaphor_detection_organ import (  # noqa: E402
    EXTRA_ANAPHORS, SpikingAnaphorDetectorOrgan)

# Realistic content-word strings for G3b -- the production `is_anaphor()` surface, exercised with tokens a live
# conversation would actually present (distinct from G3's direct unused-pool neuron draw, which tests the
# SUBSTRATE's specificity property rather than the host string-lookup shortcut in front of it).
CONTENT_WORDS = ["dog", "garden", "quickly", "yesterday", "office", "blue", "think", "under", "apple", "river"]


def _byte_identical_differential(seed, n_unknown_probes):
    """G0: a baseline organ (5 tokens) and an EXTENDED organ (9 tokens), same seed, must agree EXACTLY -- same
    boolean AND identical peak firing rate, not just the same verdict -- on every original anaphor and on
    `n_unknown_probes` unknown-string tokens. Empirical confirmation of the organ module's own structural
    argument ("WHY THE ORIGINAL 5 STAY BYTE-IDENTICAL"), not a substitute for reading it."""
    base = SpikingAnaphorDetectorOrgan(seed=seed, lesion=False)
    ext = SpikingAnaphorDetectorOrgan(seed=seed, lesion=False, extra_anaphors=EXTRA_ANAPHORS)
    mismatches = []
    for w in ANAPHORS:
        db, de = base.detect(w), ext.detect(w)
        if db["is_anaphor"] != de["is_anaphor"] or db["peak"] != de["peak"] or db["winner"] != de["winner"]:
            mismatches.append({"word": w, "baseline": db, "extended": de})
    for i in range(n_unknown_probes):
        w = "unkword%d" % i
        db, de = base.detect(w), ext.detect(w)
        if db["is_anaphor"] != de["is_anaphor"] or db["peak"] != de["peak"]:
            mismatches.append({"word": w, "baseline": db, "extended": de})
    return mismatches


def _extended_probe_set(seed, n_noisy_repeats, n_content, attractor_lesion, rng):
    """One full probe battery on the EXTENDED (9-concept) organ: clean+noisy recall for the 4 NEW tokens
    (he/she/him/her) via the organ's own public surface (`detect`, `probe_corrupted_cue`), plus a genuine
    substrate-level specificity check (G3) drawing directly from the organ's own unused-neuron pool --
    mirroring `_spiking_anaphor_detection_derisk.run_probe_set` for the extended word list."""
    org = SpikingAnaphorDetectorOrgan(seed=seed, lesion=attractor_lesion, extra_anaphors=EXTRA_ANAPHORS)
    org._ensure()   # populate _full_patterns / _unused_global so G3 can draw a real unused-pool cue below
                    # (detect()/probe_corrupted_cue() would do this lazily; explicit here since G3 also needs
                    # direct access to _unused_global, exactly what the de-risk's own run_probe_set reads off
                    # its scratch buffer).

    clean_correct, noisy_correct = [], []
    for c in EXTRA_ANAPHORS:
        d = org.detect(c)
        clean_correct.append(bool(d["is_anaphor"] and d["winner"] == c))
        for _r in range(n_noisy_repeats):
            res = org.probe_corrupted_cue(c, keep_frac=KEEP_FRAC, rng=rng)
            noisy_correct.append(bool(res["recovered"]))

    false_positives = []   # G3: genuine unused-pool cues driven directly on the substrate (de-risk's own G3)
    for _w in range(n_content):
        probe = rng.choice(org._unused_global, size=PATTERN_SIZE, replace=False)
        winner_c, _peak_c = org._probe(probe)
        false_positives.append(bool(winner_c is not None))

    public_api_rejections = []   # G3b: realistic content-word STRINGS through the actual production surface
    for i in range(n_content):
        w = CONTENT_WORDS[i % len(CONTENT_WORDS)]
        public_api_rejections.append(bool(org.is_anaphor(w)))   # True here would be a FALSE ALARM

    return {
        "clean_accuracy": float(np.mean(clean_correct)),
        "noisy_accuracy": float(np.mean(noisy_correct)),
        "false_positive_rate": float(np.mean(false_positives)),
        "public_api_false_alarms": int(np.sum(public_api_rejections)),
        "n_clean": len(clean_correct), "n_noisy": len(noisy_correct), "n_content": len(false_positives),
        "n_public_api_probes": len(public_api_rejections),
    }


def run_seed(seed, n_noisy_repeats=4, n_content=10):
    mismatches = _byte_identical_differential(seed, n_unknown_probes=min(n_content, 5))
    g0 = len(mismatches) == 0
    if not g0:
        void_if(True, "seed %d: byte-identical differential broke on %d probe(s): %r"
                % (seed, len(mismatches), mismatches[:2]))

    rng = np.random.default_rng(seed * 104729 + 3)    # a stream distinct from the base de-risk's seed*7919+{1,2}
    intact = _extended_probe_set(seed, n_noisy_repeats, n_content, attractor_lesion=False, rng=rng)
    rng2 = np.random.default_rng(seed * 104729 + 4)
    lesion = _extended_probe_set(seed, n_noisy_repeats, n_content, attractor_lesion=True, rng=rng2)

    g1 = bool(intact["clean_accuracy"] >= 0.90)
    g2 = bool(intact["noisy_accuracy"] >= 0.85)
    g3 = bool(intact["false_positive_rate"] <= 0.15)
    g3b = bool(intact["public_api_false_alarms"] == 0)
    lesion_combined_acc = float(np.mean([lesion["clean_accuracy"], lesion["noisy_accuracy"]]))
    intact_combined_acc = float(np.mean([intact["clean_accuracy"], intact["noisy_accuracy"]]))
    attrib = attributable_to("gendered-anaphor pattern-completion (attractor weight)", intact_combined_acc,
                              lesion_combined_acc, warn_below=0.5)
    g4_collapse = bool(lesion_combined_acc <= 0.30)
    g4 = bool(g4_collapse and attrib is not None and attrib >= 0.5)

    go = bool(g0 and g1 and g2 and g3 and g3b and g4)
    return {
        "seed": seed,
        "n_byte_identical_mismatches": len(mismatches),
        "G0_byte_identical": g0,
        "clean_accuracy": round(intact["clean_accuracy"], 4),
        "noisy_accuracy": round(intact["noisy_accuracy"], 4),
        "false_positive_rate": round(intact["false_positive_rate"], 4),
        "public_api_false_alarms": intact["public_api_false_alarms"],
        "lesion_clean_accuracy": round(lesion["clean_accuracy"], 4),
        "lesion_noisy_accuracy": round(lesion["noisy_accuracy"], 4),
        "lesion_combined_accuracy": round(lesion_combined_acc, 4),
        "attribution_to_attractor": None if attrib is None else round(float(attrib), 4),
        "G1_clean_recall": g1, "G2_noisy_surpass": g2, "G3_specificity": g3, "G3b_public_api": g3b,
        "G4_lesion_collapse": g4, "GO": go,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-noisy-repeats", type=int, default=4)
    ap.add_argument("--n-content", type=int, default=10)
    ap.add_argument("--out", default="research/findings/raw/_spiking_anaphor_gendered_extension/derisk.json")
    a = ap.parse_args()

    rows = [run_seed(s, n_noisy_repeats=a.n_noisy_repeats, n_content=a.n_content) for s in a.seeds]
    for r in rows:
        print(f"[anaphor-ext s{r['seed']}] byte_id_mismatches={r['n_byte_identical_mismatches']} "
              f"clean={r['clean_accuracy']:.3f} noisy={r['noisy_accuracy']:.3f} "
              f"fp_rate={r['false_positive_rate']:.3f} public_api_false_alarms={r['public_api_false_alarms']} "
              f"lesion_acc={r['lesion_combined_accuracy']:.3f} attrib={r['attribution_to_attractor']} || "
              f"{'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    n = len(rows)
    v = Verdict("spiking-anaphor gendered-extension board (>=5/6 of the 6 project-standard seeds)")
    v.require("all 6 project-standard seeds present", n == 6, expect=True, note="seeds run: %d" % n)
    v.floor("seed-GO count", measured=ngo, floor=4.5, note="board bar is >=5/6 -> floor=4.5 excludes 4/6")
    decided = v.decide(go=(n == 6 and ngo >= 5))
    verdict = decided["status"]
    print(f"[anaphor-ext] {ngo}/{n} seed-GO (board bar >=5/6) || verdict={verdict}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {"rows": rows, "n_go": ngo, "n_seeds": n, "verdict": verdict}
    payload.update(decided)
    json.dump(payload, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
