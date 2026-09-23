"""PHASE 2A next-method attempt (2026-09-22, affect->tone sub-arc): swap the DECODE ROUTE the NO-GO's own 6-seed
directional independent-lexicon ruler measured (`webapp/wkv_mouth_generator.py::_apply_affect_bias`, a HOST
additive, single-candidate-concentrated, margin-to-top1-saturating logit bias) for the ALREADY-BUILT,
default-OFF, brain-based alternative (`BRAIN_WKV_MOUTH_AFFECT_NEURAL=1` -> `_affect_pool_gains` +
`FewSpikeWordRead.set_mood`, a real neuromodulator concentration on the genuine spiking Izhikevich population
read), and re-run the IDENTICAL 6-seed directional GO-gate the NO-GO used, completely unchanged -- a
single-variable A/B on the coupling MECHANISM, holding the harness/prompts/lexicon/scorer/preregistered-delta/
anti-cheats fixed (reuse by import, not by copy, of
`research.runners._lbf_affect_tone_open_output_derisk`).

WHY THIS IS THE NEXT METHOD, NOT A RE-TUNE OF THE FAILED ONE (HARD RULE 2 -- bank the failing method, take a
NEW one). PHASE 1's diagnosis (`research.runners._lbf_affect_tone_decode_ceiling_diagnose`,
research/findings/raw/_affect_tone_decode_ceiling/diagnose_verdict.json) measured the RAW, UNBIASED linattn
mouth's own full-vocab softmax on the NO-GO's own 10 free-talk tone prompts (all 6 seeds, greedy decode,
production repetition controls, no affect bias/fact boost) and found: negative-valence independent-lexicon
vocabulary carries probability mass within ~1.5x of positive-valence vocabulary (mean ratio 1.46, range 1.05-2.5
across seeds -- NOT near-zero), and a negative-lexicon word reaches the top-64 candidate window on 5.3-11.3% of
generation steps (positive: 11-23%) -- present, just rarer, not structurally absent. Under the preregistered
DECODE-CEILING/DATA-LIMIT rule (ratio>=50x AND top64-frac<10% => DATA-LIMIT), every one of the 6 seeds reads
DECODE-CEILING. So the NO-GO's asymmetry is a property of the DECODE MECHANISM, not the training corpus: the
host bias only ever assists the SINGLE mood-congruent candidate closest to the current top-1 margin (a
deterministic nearest-margin comparison); on this wiki-descriptive corpus that nearest candidate is reliably
positive (nice/good/great sit closer to a confident encyclopedic top-1 than dreadful/miserable do) but the host
mechanism never gives any OTHER, deeper negative candidate a chance -- exactly the "we substituted a constant
for a competitive process" pattern CLAUDE.md's wall-reframe names. The neural mechanism already restores that
competition (EVERY mood-congruent top-64 candidate gets its own excitability nudge; the actual winner is a
genuine stochastic Izhikevich population race, not a deterministic margin check) -- it was built 2026-09-04 but
never re-verified against the directional/independent-lexicon ruler this NO-GO introduced. This runner is that
re-verification, not a re-tuned version of the banked host mechanism.

BRAIN-BASED-ONLY BOUNDARY (respected, not merely asserted): the congruence gate (`_affect_bias_ids`, the
Warriner-normed appraisal map ALREADY reused project-wide for the live user-message mood read -- not a fresh
host sentiment formula invented for this task) is UNCHANGED between the two mechanisms; an A/B on "which
substrate carries the effect" must not also silently vary "what counts as mood-congruent" (mirrors
`_affect_pool_gains`'s own docstring on this point). What changes is the DESTINATION only: a neuromodulator
concentration handed to `sim/bridge.py`'s own unmodified per-step current computation, consumed by a real
spiking competition, INSTEAD OF host arithmetic added directly to the output logits before any neuron sees
them. No host lexicon selects or injects a word into the reply here -- it tags candidate POOLS with an
excitability value; the substrate's own noisy accumulation decides which pool actually fires.

NO sim/ edit. NO webapp/ edit -- `BRAIN_WKV_MOUTH_AFFECT_NEURAL` already ships, default-OFF, in
`webapp/wkv_mouth_generator.py` (2026-09-04 scaffold-retirement; see `wkv_mouth_affect_neural_enabled`'s own
docstring there). This module ONLY sets that one env var into the harness's own subprocess environment (by
rebinding `_BASE._ENV_BASE`, the SAME dict `_BASE._spawn` already reads to build each worker subprocess's
environment) and then calls `_BASE.run_controller`/`_BASE.score_and_gate` UNCHANGED against a FRESH output
directory -- every anti-cheat (preregistered delta from the lesion arm's own noise band, the attribution
control, content-identity + moat, fluency, determinism, lexicon-disjointness) is the SAME code path the NO-GO
ran, not a reimplementation.

Run (CPU-forced; numpy backend; wrap with tools/memcap.sh for the RAM ceiling):
  CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy bash tools/memcap.sh 8 -- \\
      .venv/bin/python -m research.runners._lbf_affect_tone_neural_coupling_derisk --controller --parallel 2
  CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy .venv/bin/python -m \\
      research.runners._lbf_affect_tone_neural_coupling_derisk --score-only
  CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy .venv/bin/python -m \\
      research.runners._lbf_affect_tone_neural_coupling_derisk --selftest
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

import research.runners._lbf_affect_tone_open_output_derisk as _BASE  # noqa: E402

# THE ONE VARIABLE THIS MODULE CHANGES (see module docstring): every other key in `_BASE._ENV_BASE` (recurrence,
# tokenizer, scope, LTM defaults, fact-sentence suppression) is inherited UNCHANGED from the NO-GO's own harness.
# Rebinding (not mutating in place) so any other importer holding a reference to the ORIGINAL dict is unaffected.
_BASE._ENV_BASE = dict(_BASE._ENV_BASE)
_BASE._ENV_BASE["BRAIN_WKV_MOUTH_AFFECT_NEURAL"] = "1"

DEFAULT_OUT_DIR = "research/findings/raw/_affect_tone_neural_coupling"
ARTIFACT_NAME = "affect_tone_neural_coupling_verdict.json"


def selftest():
    """The base harness's own selftest (pure logic: lexicon/scorer/gate, no brain build) must still pass
    UNCHANGED through this wrapper -- proves the rebind above did not disturb the shared scorer/gate code this
    module reuses by import. Plus one check that the env rebind actually took effect."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        ok = ok and cond

    check("neural-coupling env var wired into the harness's own _ENV_BASE",
          _BASE._ENV_BASE.get("BRAIN_WKV_MOUTH_AFFECT_NEURAL") == "1")
    check("every OTHER _ENV_BASE key still matches the NO-GO harness's own defaults",
          all(_BASE._ENV_BASE[k] == v for k, v in {
              "CUDA_VISIBLE_DEVICES": "", "SIM_BACKEND": "numpy", "BRAIN_OPEN_ENDED": "1",
              "BRAIN_WKV_MOUTH_RECURRENCE": "linattn", "BRAIN_WKV_MOUTH_SCOPE": "broad",
          }.items()))
    check("shared scorer/gate selftest (imported, not reimplemented) still passes", _BASE.selftest())
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--controller", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--score-only", action="store_true", help="re-score collected arms without re-running")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", type=str, default="pos")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--memcap-gb", type=int, default=8)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    elif args.worker:
        # a single fresh-subprocess arm, IDENTICAL to the base harness's own worker (the env rebind above is
        # what makes this process's `webapp.wkv_mouth_generator.wkv_mouth_affect_neural_enabled()` read True)
        _BASE._LEX, _ = _BASE.load_indep_lexicon()
        out = args.out or _BASE._worker_out(args.out_dir, args.seed, args.arm)
        _BASE.run_worker(args.seed, args.arm, out)
    elif args.score_only:
        artifact = _BASE.score_and_gate(args.out_dir, write_artifact=False)
        artifact["runner"] = "_lbf_affect_tone_neural_coupling_derisk (reuses _lbf_affect_tone_open_output_derisk's scorer/gate)"
        artifact["coupling_mechanism"] = "neural (BRAIN_WKV_MOUTH_AFFECT_NEURAL=1) -- see module docstring"
        ap_path = os.path.join(args.out_dir, ARTIFACT_NAME)
        os.makedirs(args.out_dir, exist_ok=True)
        json.dump(artifact, open(ap_path, "w"), indent=2, default=str)
        print("[controller] wrote %s" % ap_path)
    else:
        artifact = _BASE.run_controller(args.out_dir, parallel=args.parallel, memcap_gb=args.memcap_gb)
        artifact["runner"] = "_lbf_affect_tone_neural_coupling_derisk (reuses _lbf_affect_tone_open_output_derisk's scorer/gate)"
        artifact["coupling_mechanism"] = "neural (BRAIN_WKV_MOUTH_AFFECT_NEURAL=1) -- see module docstring"
        ap_path = os.path.join(args.out_dir, ARTIFACT_NAME)
        json.dump(artifact, open(ap_path, "w"), indent=2, default=str)
        print("[controller] re-tagged + wrote %s" % ap_path)
        print("[controller] GO=%s" % artifact.get("GO"))
