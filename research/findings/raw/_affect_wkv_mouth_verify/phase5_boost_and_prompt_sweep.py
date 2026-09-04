"""Phase 5 (2026-09-04) -- the `affect_boost` CALIBRATION sweep for the sharpness-aware/saturating/habituating
`_apply_affect_bias` coupling (`webapp/wkv_mouth_generator.py`), re-measured at the magnitude that actually
reaches the mouth LIVE (see that function's own mechanism comment for the full diagnosis): the 2026-09-03
calibration swept `valence=+-0.9` against a fixed absolute bias; the live pipeline was then measured (phase4)
to present `valence~0.16` (organ differential ~0.04 through `_valence_from_differential`'s `clip(4*diff,-1,1)`)
-- ~5.6x smaller. This sweep calibrates directly against the REALISTIC magnitude (`valence=0.16, arousal=0.65`,
representative of the phase4 'thrilled/overjoyed/wonderful' priming scenario) on BOTH families, BOTH mood
directions, TWO prompts (the phase4 known-topic prompt + the original 2026-09-03 calibration prompt), and
cross-checks the pre-existing `valence=+-0.9` extreme sweep for safety (does NOT need to reproduce the OLD
formula's behavior there -- just must not collapse into worse word-salad than the old `affect_boost>=8`
failure it replaces).

Direct calls to `webapp.wkv_mouth_generator.generate()` (no full brain build, no LTM store, no `webapp.server`)
-- the cheapest correct probe of THIS function's own behavior, matching `phase1_direct_generate_vary_lesion.py`'s
own methodology (one recurrence family per process, `--family ssm|linattn`, module-level constants fixed at
import time). Run twice (once per family) and merged by the CLI-less orchestrator block at the bottom into ONE
combined artifact.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["ssm", "linattn"], default=None)
ap.add_argument("--boosts", default="8,10,15")
args = ap.parse_args()

BOOSTS = [float(b) for b in args.boosts.split(",")]
PROMPTS = {"known_topic": "Tell me about frank_lincoln_wright", "generic": "Tell me about it."}
RP = dict(repetition_penalty=1.3, no_repeat_ngram_size=3)
SEED = 42
MAX_NEW = 40


def salad_frac(text):
    toks = (text or "").split()
    if not toks:
        return 0.0
    return Counter(toks).most_common(1)[0][1] / len(toks)


def _run_family(family):
    """Runs INSIDE this process -- only called when --family was passed (module-level env is already fixed)."""
    import webapp.wkv_mouth_generator as WKV

    out = {"family": family, "recurrence_mode": WKV.recurrence_mode(), "boosts": {}}
    for boost in BOOSTS:
        rows = {}
        for pname, prompt in PROMPTS.items():
            neutral, _ = WKV.generate(prompt, seed=SEED, max_new_tokens=MAX_NEW, valence=0.0, arousal=0.0,
                                       affect_boost=boost, **RP)
            real_pos, _ = WKV.generate(prompt, seed=SEED, max_new_tokens=MAX_NEW, valence=0.16, arousal=0.65,
                                        affect_boost=boost, **RP)
            real_neg, _ = WKV.generate(prompt, seed=SEED, max_new_tokens=MAX_NEW, valence=-0.16, arousal=0.65,
                                        affect_boost=boost, **RP)
            ext_lo, _ = WKV.generate(prompt, seed=SEED, max_new_tokens=MAX_NEW, valence=-0.9, arousal=0.9,
                                      affect_boost=boost, **RP)
            ext_hi, _ = WKV.generate(prompt, seed=SEED, max_new_tokens=MAX_NEW, valence=0.9, arousal=0.9,
                                      affect_boost=boost, **RP)
            rows[pname] = {
                "prompt": prompt,
                "neutral": neutral, "realistic_pos": real_pos, "realistic_neg": real_neg,
                "extreme_lo": ext_lo, "extreme_hi": ext_hi,
                "realistic_pos_differs_neutral": real_pos != neutral,
                "realistic_neg_differs_neutral": real_neg != neutral,
                "realistic_pos_vs_neg_differ": real_pos != real_neg,
                "salad_frac": {"neutral": salad_frac(neutral), "realistic_pos": salad_frac(real_pos),
                               "realistic_neg": salad_frac(real_neg), "extreme_lo": salad_frac(ext_lo),
                               "extreme_hi": salad_frac(ext_hi)},
            }
        max_salad = max(v for row in rows.values() for v in row["salad_frac"].values())
        both_realistic_move = all(rows[p]["realistic_pos_differs_neutral"] and rows[p]["realistic_neg_differs_neutral"]
                                   for p in PROMPTS)
        out["boosts"][str(boost)] = {"rows": rows, "max_salad_frac": max_salad,
                                      "both_realistic_directions_move_both_prompts": both_realistic_move,
                                      "not_worse_than_old_collapse": max_salad < 0.5}  # old >=8 failure was ~0.5+
    return out


if args.family is not None:
    print(json.dumps(_run_family(args.family)))
    sys.exit(0)

# ---- orchestrator: no --family given -> subprocess each family (module-level env must be fixed pre-import,
# so both families cannot run in ONE interpreter) and merge into one combined artifact ----
T0 = time.time()
os.environ.setdefault("SIM_BACKEND", "numpy")
combined = {"seed": SEED, "backend": os.environ.get("SIM_BACKEND"), "boosts_tested": BOOSTS,
            "prompts": PROMPTS, "families": {}}
for family in ("ssm", "linattn"):
    env = dict(os.environ)
    if family == "linattn":
        env["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
        env["BRAIN_WKV_MOUTH_CKPT"] = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"
        env["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
        env["BRAIN_WKV_MOUTH_SCOPE"] = "broad"
    else:
        env["BRAIN_WKV_MOUTH_RECURRENCE"] = "ssm"
        env["BRAIN_WKV_MOUTH_TOKENIZER"] = "word"
        env["BRAIN_WKV_MOUTH_SCOPE"] = "vocab"
    env.setdefault("SIM_BACKEND", "numpy")
    r = subprocess.run([sys.executable, __file__, "--family", family, "--boosts", args.boosts],
                        env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        combined["families"][family] = {"error": r.stderr[-4000:]}
        print(f"[{family}] FAILED: {r.stderr[-2000:]}", file=sys.stderr)
        continue
    combined["families"][family] = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"[{family}] done in {time.time()-T0:.1f}s")

# choose the recommended boost: the smallest tested value for which BOTH families move in both realistic
# directions on both prompts AND stay safely below the old collapse threshold, on EVERY tested boost.
recommended = None
for boost in sorted(BOOSTS):
    ok = True
    for family in ("ssm", "linattn"):
        fam = combined["families"].get(family, {})
        b = fam.get("boosts", {}).get(str(boost))
        if not b or not b["both_realistic_directions_move_both_prompts"] or not b["not_worse_than_old_collapse"]:
            ok = False
    if ok:
        recommended = boost
        break
combined["recommended_affect_boost"] = recommended
combined["wall_seconds"] = round(time.time() - T0, 1)

out_path = "research/findings/raw/_affect_wkv_mouth_verify_phase5_boost_and_prompt_sweep.json"
with open(out_path, "w") as fh:
    json.dump(combined, fh, indent=1)
print("wrote", out_path, "recommended_affect_boost =", recommended)
