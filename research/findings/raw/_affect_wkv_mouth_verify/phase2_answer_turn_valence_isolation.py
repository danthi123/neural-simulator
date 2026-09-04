"""Phase 2 -- ISOLATED `webapp.open_ended_chat.answer_turn` valence/arousal sweep, per recurrence family.

Mirrors the ORIGINAL 2026-09-03 linattn live-verification's own `phase4_5_sentence_facts_and_valence_isolation.py`
`valence_isolation` probe EXACTLY (same call shape, same topic, same seed) -- except this time the expectation
flips: that finding measured `raw_identical_across_valence: true` (the hollow gap); this run is expected to show
`false` now that `answer_turn` passes `valence`/`arousal` through to `_WKV.generate()`. Uses an UNKNOWN topic
("kanton genf" is not present in this worktree's LTM store unless the real bundle is loaded) with fact-routing
flags forced OFF, so the reply is genuinely written by `_free_gen`/`_free_gen_linattn`'s own free generation
(never the fact-clause short-circuit, which is intentionally affect-neutral by design -- see the module
docstring) -- an honest worst case for the mechanism (no grounding fact to lean on, matching Phase 1's own
adversarial framing).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["ssm", "linattn"], default="ssm")
args = ap.parse_args()

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "1"
# force the mouth's OWN free-gen to answer (isolate the coupling under test from the fact-clause/fact-boost
# levers, which are independent, orthogonal, and -- for the fact-clause path -- intentionally affect-neutral).
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE"] = "0"
os.environ["BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK"] = "0"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND"] = "0"
if args.family == "linattn":
    os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
    os.environ["BRAIN_WKV_MOUTH_CKPT"] = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"
    os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
    os.environ["BRAIN_WKV_MOUTH_SCOPE"] = "broad"
else:
    os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "ssm"
    os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "word"
    os.environ["BRAIN_WKV_MOUTH_SCOPE"] = "broad"   # broad so an out-of-TinyStories-vocab msg still routes here

from webapp import open_ended_chat as OE  # noqa: E402  (deliberately late: env-fixed constants read at import)

MSG = "Tell me about kanton genf"
SEED = 42

r_neu = OE.answer_turn(MSG, None, 0.0, 0.3, ltm_bundle=None, brain_bundle=None, seed=SEED)
r_lo = OE.answer_turn(MSG, None, -0.9, 0.9, ltm_bundle=None, brain_bundle=None, seed=SEED)
r_hi = OE.answer_turn(MSG, None, 0.9, 0.9, ltm_bundle=None, brain_bundle=None, seed=SEED)

print(f"--- {args.family} ---")
print("wkv_used (neu/lo/hi):", r_neu["wkv_mouth_used"], r_lo["wkv_mouth_used"], r_hi["wkv_mouth_used"])
print("generator (neu/lo/hi):", r_neu["generator"], r_lo["generator"], r_hi["generator"])
print("raw neutral: ", r_neu["raw"])
print("raw valence=-0.9:", r_lo["raw"])
print("raw valence=+0.9:", r_hi["raw"])
print("raw_identical_lo_hi:", r_lo["raw"] == r_hi["raw"])
print("raw_identical_neu_hi:", r_neu["raw"] == r_hi["raw"])
print("state carries different valence:", r_lo["state"], "vs", r_hi["state"])

out = {
    "family": args.family, "msg": MSG, "seed": SEED,
    "wkv_used": {"neutral": r_neu["wkv_mouth_used"], "lo": r_lo["wkv_mouth_used"], "hi": r_hi["wkv_mouth_used"]},
    "generator": {"neutral": r_neu["generator"], "lo": r_lo["generator"], "hi": r_hi["generator"]},
    "raw": {"neutral": r_neu["raw"], "lo": r_lo["raw"], "hi": r_hi["raw"]},
    "raw_identical_lo_hi": r_lo["raw"] == r_hi["raw"],
    "raw_identical_neu_hi": r_neu["raw"] == r_hi["raw"],
    "state": {"neutral": r_neu["state"], "lo": r_lo["state"], "hi": r_hi["state"]},
}
out_path = f"research/findings/raw/_affect_wkv_mouth_verify_phase2_answer_turn_{args.family}.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", out_path)
