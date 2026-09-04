"""NEURAL-vs-HOST affect coupling verification for the WKV/SSM mouth (2026-09-04, scaffold-retirement task).

CONTEXT. `webapp/wkv_mouth_generator.py`'s existing mood coupling (`_apply_affect_bias`) is HOST decode-time
arithmetic: it adds a signed, saturating bias directly to the full-vocab logits `lg` BEFORE `FewSpikeWordRead.
read(p)` ever runs -- the genuine spiking population-coded WTA read only ever sees an already mood-cooked
probability vector, never the mood signal itself. This rung adds a genuinely SUBSTRATE-mediated alternative
(`BRAIN_WKV_MOUTH_AFFECT_NEURAL=1`, default OFF): the SAME Warriner-gated congruence/habituation policy instead
drives a per-candidate-pool NEUROMODULATOR CONCENTRATION (`FewSpikeWordRead.set_mood`, `sim/neuromodulators.py`'s
existing, UNMODIFIED `excitability_drive` target applied by `sim/bridge.py`'s own per-step current computation --
the SAME general subsystem already used project-wide for dopamine/ACh/dynorphin/per-action-DA/NE-vigilance-gain,
see `research.runners._ne_lc_gain_vigilance_realbridge_derisk` for the closest precedent), so mood colors the
ACTUAL spiking competition among candidate pools rather than a pre-softmax probability the population never sees.

THIS RUNNER answers three questions, each as its own phase (fresh-SUBPROCESS-per-arm throughout -- matching the
project's own "phase6 clean isolation" discipline for this exact coupling: `research/findings/raw/
_affect_wkv_mouth_verify_phase6_clean_isolation.json`, previously verified via a full brain-chat pipeline; THIS
script verifies at the `webapp.wkv_mouth_generator.generate()` boundary the coupling itself lives at, with
`valence`/`arousal` passed directly -- exactly the values `answer_turn` already assembles from the real spiking
AffectProductionOrgan's `read_differential`/`valence_from_affect`, see that module. A separate subprocess per arm
is NOT optional: `_get_readout`/`_affect_bias_ids` cache per-seed state and `_RNG` keeps a CONTINUING private RNG
timeline across calls in-process, so two `generate()` calls in the SAME process would confound "different mood"
with "different consumed RNG history" -- exactly the class of bug the project's `_RngIsolation`/`_isolated`
precedent exists to avoid one level up):

  (1) CALIBRATE (--phase calibrate): sweep `affect_boost` (which multiplies `_AFFECT_NEURAL_PA_AT_REFERENCE_
      BOOST` -- see `webapp/wkv_mouth_generator.py`'s own comment above that constant) at the SAME realistic
      valence/arousal operating point the host mechanism's own `affect_boost=10.0` was calibrated against
      (valence=0.16, arousal=0.65 -- `generate()`'s own docstring), to find a `affect_boost` value where the
      neural coupling measurably moves output without word-salad collapse.
  (2) LOADBEARING (--phase loadbearing): fresh-session A (positive mood) vs B (negative mood, SAME prompt/seed)
      vs C (repeat of A) -- affect_loadbearing := text(A) != text(B); determinism := text(A) == text(C). Run for
      BOTH mechanisms (host, neural) on the SAME prompts/seeds so the two mechanisms are compared like-for-like.
  (3) COMPARE (--phase compare): host vs neural side-by-side on the same (prompt, valence, arousal) grid --
      self-NLL (fluency) and an affect-word FRACTION of the generated continuation (Warriner-strong-word share,
      the same salad-risk proxy the host mechanism's own calibration comment uses under the name "salad_frac").

Run (CPU-forced -- the bank is a few hundred Izhikevich neurons, GPU is not needed and may be busy):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_verify --phase calibrate \\
      --json research/findings/raw/_wkv_mouth_affect_neural_calibration_sweep.json
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_verify --phase loadbearing \\
      --json research/findings/raw/_wkv_mouth_affect_neural_loadbearing.json
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_verify --phase compare \\
      --json research/findings/raw/_wkv_mouth_affect_neural_vs_host_compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Content words the host mechanism's own congruence gate would recognize -- reused BY IMPORT (not re-derived) so
# the "how affect-saturated is the output" proxy metric below uses the IDENTICAL salience gate as the coupling
# itself (the same Warriner-strong-word test `_affect_bias_ids`/`appraise_text` already apply).
sys.path.insert(0, str(_REPO_ROOT))
from research.runners.affect_production_organ import WARRINER, STOP, _STRONG_MARGIN  # noqa: E402
# `lever` -- makes every "did arm X differ from arm Y" check below an INSTRUMENTED assertion (prints
# before/after, returns whether it moved) instead of a bare `!=`/`==` a reader has to trust was actually
# checked. `required=False` throughout: a NON-divergence is itself data this runner explicitly goes looking
# for (the calibrate phase's whole point is finding the boost region where nothing moves yet), never a reason
# to abort the sweep.
from tools.lab import lever  # noqa: E402


def _affect_word_fraction(text: str) -> float:
    """Fraction of `text`'s words that are a STRONGLY affect-bearing Warriner word (excluding STOP) -- the same
    salad-risk proxy the host mechanism's own calibration comment calls 'salad_frac'. 0.0 for empty text."""
    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words:
        return 0.0
    hits = 0
    for w in words:
        if w in STOP or w not in WARRINER:
            continue
        v9, _a9 = WARRINER[w]
        if abs(v9 - 5.0) >= _STRONG_MARGIN:
            hits += 1
    return hits / len(words)


def _run_arm(seed: int, prompt: str, valence: float, arousal: float, affect_boost: float, neural: bool,
             max_new_tokens: int = 48, recurrence: str = "ssm", timeout_s: int = 180) -> dict:
    """ONE `generate()` call in a FRESH subprocess -- complete isolation from every other arm (no shared
    `_CKPT_CACHE`/`_affect_bias_ids` cache, no continuing `_RNG` private timeline). Returns {"text", "secs",
    "wall_s", "affect_word_frac"}; raises RuntimeError with the captured stdout/stderr tail on any failure (never
    silently returns a placeholder)."""
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env["BRAIN_WKV_MOUTH_AFFECT"] = "1"
    env["BRAIN_WKV_MOUTH_AFFECT_NEURAL"] = "1" if neural else "0"
    env["BRAIN_WKV_MOUTH_RECURRENCE"] = recurrence
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "from webapp.wkv_mouth_generator import generate\n"
        f"text, secs = generate({prompt!r}, seed={int(seed)}, max_new_tokens={int(max_new_tokens)}, "
        f"valence={float(valence)!r}, arousal={float(arousal)!r}, affect_boost={float(affect_boost)!r})\n"
        "print('RESULT_JSON:' + json.dumps({'text': text, 'secs': secs}))\n"
    )
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=timeout_s, cwd=str(_REPO_ROOT))
    wall = time.time() - t0
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")), None)
    if line is None:
        raise RuntimeError(
            f"arm failed (seed={seed} valence={valence} arousal={arousal} neural={neural}): "
            f"rc={proc.returncode}\nSTDOUT-tail={proc.stdout[-2000:]}\nSTDERR-tail={proc.stderr[-4000:]}"
        )
    d = json.loads(line[len("RESULT_JSON:"):])
    d["wall_s"] = round(wall, 2)
    d["affect_word_frac"] = _affect_word_fraction(d["text"])
    return d


PROMPTS = ["the little girl was", "tom and his dog were", "one day the boy"]
POS_V, POS_A = 0.16, 0.65     # the realistic-magnitude operating point generate()'s own affect_boost=10.0 default
NEG_V, NEG_A = -0.16, 0.65    # was calibrated against (webapp/open_ended_chat.py::valence_from_affect measured
                              # ~0.16 live for a real 'thrilled/overjoyed/wonderful' priming turn).


def phase_calibrate(seed: int, out: dict) -> None:
    prompt = PROMPTS[0]
    rows = []
    for boost in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        pos = _run_arm(seed, prompt, POS_V, POS_A, boost, neural=True)
        neg = _run_arm(seed, prompt, NEG_V, NEG_A, boost, neural=True)
        neu = _run_arm(seed, prompt, 0.0, 0.0, boost, neural=True)   # lesion-equivalent reference (no mood)
        pos_ne_neg = lever(f"calibrate pos-vs-neg @boost={boost}", pos["text"], neg["text"], required=False)
        pos_ne_neutral = lever(f"calibrate pos-vs-neutral @boost={boost}", pos["text"], neu["text"],
                               required=False)
        neg_ne_neutral = lever(f"calibrate neg-vs-neutral @boost={boost}", neg["text"], neu["text"],
                               required=False)
        rows.append({
            "affect_boost": boost,
            "pos_text": pos["text"], "neg_text": neg["text"], "neutral_text": neu["text"],
            "pos_ne_neg": pos_ne_neg,
            "pos_ne_neutral": pos_ne_neutral,
            "neg_ne_neutral": neg_ne_neutral,
            "pos_affect_frac": pos["affect_word_frac"], "neg_affect_frac": neg["affect_word_frac"],
            "neutral_affect_frac": neu["affect_word_frac"],
        })
        print(f"[calibrate boost={boost}] pos!=neg={rows[-1]['pos_ne_neg']} "
              f"pos!=neutral={rows[-1]['pos_ne_neutral']} affect_frac(pos/neg/neu)="
              f"{pos['affect_word_frac']:.3f}/{neg['affect_word_frac']:.3f}/{neu['affect_word_frac']:.3f}",
              flush=True)
    out["calibrate"] = {"prompt": prompt, "seed": seed, "rows": rows}


def phase_loadbearing(seed: int, affect_boost: float, out: dict) -> None:
    result = {}
    for neural in (False, True):
        mech = "neural" if neural else "host"
        per_prompt = []
        for prompt in PROMPTS:
            a = _run_arm(seed, prompt, POS_V, POS_A, affect_boost, neural=neural)      # A: positive mood
            b = _run_arm(seed, prompt, NEG_V, NEG_A, affect_boost, neural=neural)      # B: negative mood
            c = _run_arm(seed, prompt, POS_V, POS_A, affect_boost, neural=neural)      # C: repeat of A
            lesion = _run_arm(seed, prompt, 0.0, 0.0, affect_boost, neural=neural)     # lesion (valence==0.0)
            tag = f"{mech} seed={seed} prompt={prompt!r}"
            moved_ab = lever(f"mood A(pos) vs B(neg) [{tag}]", a["text"], b["text"], required=False)
            moved_ac = lever(f"determinism A vs C(repeat) [{tag}]", a["text"], c["text"], required=False)
            moved_la = lever(f"lesion vs A(pos) [{tag}]", lesion["text"], a["text"], required=False)
            moved_lb = lever(f"lesion vs B(neg) [{tag}]", lesion["text"], b["text"], required=False)
            row = {
                "prompt": prompt,
                "A_pos": a["text"], "B_neg": b["text"], "C_pos_repeat": c["text"], "L_lesion": lesion["text"],
                "affect_loadbearing_A_ne_B": moved_ab,
                "determinism_A_eq_C": not moved_ac,
                "lesion_ne_A": moved_la,     # lesion should usually differ from a moody arm
                "lesion_ne_B": moved_lb,
            }
            per_prompt.append(row)
            print(f"[loadbearing {mech} seed={seed} prompt={prompt!r}] "
                  f"A!=B={row['affect_loadbearing_A_ne_B']} A==C={row['determinism_A_eq_C']}", flush=True)
        result[mech] = per_prompt
    n_lb = sum(1 for r in result["neural"] if r["affect_loadbearing_A_ne_B"])
    n_det = sum(1 for r in result["neural"] if r["determinism_A_eq_C"])
    result["neural_summary"] = {
        "affect_loadbearing_frac": n_lb / len(result["neural"]),
        "determinism_frac": n_det / len(result["neural"]),
        "GO": (n_lb >= 1) and (n_det == len(result["neural"])),
    }
    out["loadbearing"] = {"seed": seed, "affect_boost": affect_boost, **result}


def phase_compare(seed: int, affect_boost: float, out: dict) -> None:
    rows = []
    for prompt in PROMPTS:
        for label, v, a in (("positive", POS_V, POS_A), ("negative", NEG_V, NEG_A), ("neutral", 0.0, 0.0)):
            host = _run_arm(seed, prompt, v, a, affect_boost, neural=False)
            neural = _run_arm(seed, prompt, v, a, affect_boost, neural=True)
            rows.append({
                "prompt": prompt, "mood": label, "valence": v, "arousal": a,
                "host_text": host["text"], "neural_text": neural["text"],
                "host_affect_frac": host["affect_word_frac"], "neural_affect_frac": neural["affect_word_frac"],
                "host_secs": host["secs"], "neural_secs": neural["secs"],
            })
            print(f"[compare seed={seed} prompt={prompt!r} mood={label}] "
                  f"host_frac={host['affect_word_frac']:.3f} neural_frac={neural['affect_word_frac']:.3f}",
                  flush=True)
    out["compare"] = {"seed": seed, "affect_boost": affect_boost, "rows": rows}


def phase_summary(seeds: list, in_pattern: str, out: dict) -> None:
    """Aggregates the per-seed `loadbearing` artifacts (produced by `--phase loadbearing` on each of `seeds`,
    default filename pattern `in_pattern.format(seed=s)`) into the headline fractions cited in the finding --
    kept as its own runner phase (rather than a one-off analysis script) so the aggregate itself carries the
    SAME automatic run provenance (argv/git-SHA/env) every other artifact in this project does."""
    per_mech: dict = {}
    for mech in ("host", "neural"):
        n = lb = det = lesion_pos = lesion_neg = 0
        for s in seeds:
            d = json.loads(Path(in_pattern.format(seed=s)).read_text())
            for row in d["loadbearing"][mech]:
                n += 1
                lb += int(row["affect_loadbearing_A_ne_B"])
                det += int(row["determinism_A_eq_C"])
                lesion_pos += int(row["lesion_ne_A"])
                lesion_neg += int(row["lesion_ne_B"])
        per_mech[mech] = {
            "n_prompt_seed_combos": n,
            "affect_loadbearing_A_ne_B": f"{lb}/{n}",
            "determinism_A_eq_C": f"{det}/{n}",
            "lesion_diverges_positive_direction": f"{lesion_pos}/{n}",
            "lesion_diverges_negative_direction": f"{lesion_neg}/{n}",
        }
        print(f"[summary {mech}] loadbearing={lb}/{n} determinism={det}/{n} "
              f"lesion_pos={lesion_pos}/{n} lesion_neg={lesion_neg}/{n}", flush=True)
    out["summary"] = {"seeds": seeds, "in_pattern": in_pattern, "per_mechanism": per_mech}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["calibrate", "loadbearing", "compare", "summary", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--in-pattern", type=str,
                    default="research/findings/raw/_wkv_mouth_affect_neural_loadbearing_s{seed}.json")
    ap.add_argument("--affect-boost", type=float, default=10.0)
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_mouth_affect_neural_verify.json")
    args = ap.parse_args()

    t0 = time.time()
    out: dict = {"seed": args.seed, "affect_boost": args.affect_boost}
    phases = [args.phase] if args.phase != "all" else ["calibrate", "loadbearing", "compare"]
    for ph in phases:
        if ph == "calibrate":
            phase_calibrate(args.seed, out)
        elif ph == "loadbearing":
            phase_loadbearing(args.seed, args.affect_boost, out)
        elif ph == "compare":
            phase_compare(args.seed, args.affect_boost, out)
        elif ph == "summary":
            seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
            phase_summary(seeds, args.in_pattern, out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[done] {out['elapsed_s']}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
