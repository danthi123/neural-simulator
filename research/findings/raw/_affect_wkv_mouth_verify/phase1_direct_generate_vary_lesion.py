"""Phase 1 -- ISOLATED direct `webapp.wkv_mouth_generator.generate()` vary/lesion, ONE recurrence family per
process invocation (module-level constants like `_CKPT_TEMPLATE` are fixed at import time, so env vars must be
set BEFORE the first `import webapp.wkv_mouth_generator` -- the same discipline the original linattn live-
verification's own phase scripts used). Select the family with `--family ssm|linattn` (default ssm).

Cheapest possible probe of the NEW affect-bias mechanism (no full brain build, no LTM store): calls `generate()`
directly at a fixed prompt/seed, sweeping ONLY `valence`/`arousal`. Also exercises the `BRAIN_WKV_MOUTH_AFFECT=0`
kill switch (must reproduce the `valence=0.0` neutral output byte-for-byte) and reports `_affect_bias_ids`
coverage counts for the checkpoint's own vocabulary (so a silent-zero-coverage false fix would show up here,
not just be assumed away).
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
if args.family == "linattn":
    os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
    os.environ["BRAIN_WKV_MOUTH_CKPT"] = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"
    os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
    os.environ["BRAIN_WKV_MOUTH_SCOPE"] = "broad"
else:
    os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "ssm"
    os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "word"
    os.environ["BRAIN_WKV_MOUTH_SCOPE"] = "vocab"

import webapp.wkv_mouth_generator as WKV  # noqa: E402  (deliberately late: env-fixed constants read at import)

PROMPT = "Tell me about it."
SEED = 42
MAX_NEW = 40
# the SAME production repetition guard `answer_turn`'s WKV-mouth branch always passes
# (webapp/open_ended_chat.py: `repetition_penalty=1.3, no_repeat_ngram_size=3`) -- calling generate() without it
# (the library defaults, 1.0/0 = off) is a KNOWN pre-existing residual for short/vague prompts (named by the
# 2026-09-03 linattn live-verification finding's Property (i)), unrelated to the affect coupling under test here.
RP = dict(repetition_penalty=1.3, no_repeat_ngram_size=3)

aff_ids = WKV._affect_bias_ids(SEED)
n_pos = sum(1 for v in aff_ids.values() if v > 0)
n_neg = sum(1 for v in aff_ids.values() if v < 0)

text_neutral, _ = WKV.generate(PROMPT, seed=SEED, max_new_tokens=MAX_NEW, valence=0.0, arousal=0.0, **RP)
text_lo, _ = WKV.generate(PROMPT, seed=SEED, max_new_tokens=MAX_NEW, valence=-0.9, arousal=0.9, **RP)
text_hi, _ = WKV.generate(PROMPT, seed=SEED, max_new_tokens=MAX_NEW, valence=0.9, arousal=0.9, **RP)
text_lowaro, _ = WKV.generate(PROMPT, seed=SEED, max_new_tokens=MAX_NEW, valence=0.9, arousal=0.05, **RP)
text_hiaro, _ = WKV.generate(PROMPT, seed=SEED, max_new_tokens=MAX_NEW, valence=0.9, arousal=0.95, **RP)
os.environ["BRAIN_WKV_MOUTH_AFFECT"] = "0"
text_killed, _ = WKV.generate(PROMPT, seed=SEED, max_new_tokens=MAX_NEW, valence=0.9, arousal=0.9, **RP)
del os.environ["BRAIN_WKV_MOUTH_AFFECT"]

row = {
    "label": args.family,
    "recurrence_mode": WKV.recurrence_mode(),
    "tokenizer_mode": WKV.tokenizer_mode(),
    "affect_ids_count": len(aff_ids),
    "affect_ids_pos": n_pos,
    "affect_ids_neg": n_neg,
    "text_neutral_v0": text_neutral,
    "text_valence_lo_m0.9": text_lo,
    "text_valence_hi_p0.9": text_hi,
    "text_arousal_lo_0.05": text_lowaro,
    "text_arousal_hi_0.95": text_hiaro,
    "text_killswitch_at_hi_valence": text_killed,
    "lo_vs_hi_differ": text_lo != text_hi,
    "neutral_vs_hi_differ": text_neutral != text_hi,
    "lowaro_vs_hiaro_differ": text_lowaro != text_hiaro,
    "killswitch_matches_neutral": text_killed == text_neutral,
}
print(f"--- {args.family} ---")
for k in ("affect_ids_count", "affect_ids_pos", "affect_ids_neg", "lo_vs_hi_differ", "neutral_vs_hi_differ",
          "lowaro_vs_hiaro_differ", "killswitch_matches_neutral"):
    print(f"  {k}: {row[k]}")
print("  neutral:          ", text_neutral)
print("  valence=-0.9:      ", text_lo)
print("  valence=+0.9:      ", text_hi)
print("  arousal=0.05@v+0.9:", text_lowaro)
print("  arousal=0.95@v+0.9:", text_hiaro)
print("  killswitch@hi-val: ", text_killed)

out_path = f"research/findings/raw/_affect_wkv_mouth_verify_phase1_direct_{args.family}.json"
with open(out_path, "w") as fh:
    json.dump({"prompt": PROMPT, "seed": SEED, "max_new_tokens": MAX_NEW, "row": row}, fh, indent=1)
print("wrote", out_path)
