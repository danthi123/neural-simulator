"""PROMOTION validation harness for the neural affect coupling (BRAIN_WKV_MOUTH_AFFECT_NEURAL) -- build-ahead
prep, 2026-09-04, NOT YET RUN AT THE 6-SEED SCALE.

CONTEXT. `research/findings/2026-09-04-affect-coupling-neural-not-host-PARTIAL.md` (merged 87631edf) landed the
neuromodulatory alternative to the host logit-bias mood coupling: `FewSpikeWordRead.set_mood`
(`research/runners/_wkv_fewspike_read_derisk.py`) drives a real `sim.neuromodulators` `excitability_drive`
concentration on the genuine Izhikevich word-read population, gated `BRAIN_WKV_MOUTH_AFFECT_NEURAL` (default
OFF). That finding's own 6-seed (42/43/44/100/101/102) x 3-prompt load-bearing battery measured mood-vs-mood
divergence (A!=B) and determinism at a clean 18/18, matching the pre-existing host mechanism -- but the
STRICTER per-direction check (does EACH mood arm individually diverge from the neutral/lesion baseline) showed
the neural mechanism undershooting on the NEGATIVE side: only 11/18 combinations, vs 18/18 positive and 18/18
for the host mechanism's own negative direction. Collapsed to PER-SEED (a seed counts as passing a direction
only if it holds on ALL 3 of its prompts -- see "Why per-seed, not per-combination" below), the ALREADY-
COMMITTED per-seed artifacts (`research/findings/raw/_wkv_mouth_affect_neural_loadbearing_s{42,43,44,100,101,
102}.json`, re-read by this module's own docstring build, not re-derived) show:

    positive direction (lesion_ne_A), all-3-prompts-pass: 6/6 seeds  (42,43,44,100,101,102 all pass)
    negative direction (lesion_ne_B), all-3-prompts-pass: 1/6 seeds  (only 43; 42/44/100/102 miss 1 of 3
                                                                       prompts each, 101 misses all 3)

The finding named this an honest residual and proposed two untried next steps: (a) an asymmetric, stronger
negative-direction pA scale, or (b) a corpus/checkpoint "positivity prior" explanation (TinyStories skews
cheerful, so the unconditioned/lesion continuation may already sit close to a mild positive nudge, leaving less
room for a negative nudge to move it) -- NEITHER attempted in that finding, to avoid conflating "does the
mechanism work" with "is it perfectly tuned."

THIS MODULE builds the validation to decide DEFAULT-ON promotion, testing both named next steps plus the
promotion-specific safety checks the finding did not need (it was reporting a PARTIAL, not proposing a flip):

  1. DIRECTIONAL, per-seed GO gate (`--phase directional` + `--phase summary`): re-runs the SAME 4-arm
     (A pos / B neg / C repeat / L lesion) x 3-prompt battery, but scored per-seed (>=5/6 seeds must pass EACH
     direction, not just an aggregate 18/18-style combo count) -- see the threshold rationale below. Takes an
     experimental `--neg-pa-scale` (default 1.0 = the exact pre-existing behavior): when the arm is NEGATIVE
     mood AND `neg_pa_scale != 1.0`, `_run_arm_scaled` MONKEYPATCHES `webapp.wkv_mouth_generator._affect_pool_
     gains` inside a fresh, throwaway subprocess to multiply the returned pA dict by `neg_pa_scale` -- testing
     next-step (a) with ZERO edits to `webapp/wkv_mouth_generator.py` (the scaling is a research-runner-side
     experiment; promoting a chosen scale into the shipped `_AFFECT_NEURAL_PA_AT_REFERENCE_BOOST` calibration
     block would be a separate, reviewed edit once a value is shown to work at the full 6-seed scale).
  2. CHARACTERIZE (`--phase characterize`): tests next-step (b) two ways -- (i) the OUTPUT-level Warriner-sign
     skew of the lesion arm's own continuation (does the checkpoint's unconditioned output already lean
     positive on the Warriner scale), and (ii) the mechanistic SOURCE-level question `_run_arm_instrumented`
     answers by transparently instrumenting (not manipulating) `_affect_pool_gains`: at MATCHED
     |valence|=0.16/|arousal|=0.65, does the negative arm see a lower congruent-candidate HIT RATE and/or a
     lower mean delivered pA per hit than the positive arm? (i) speaks to what the checkpoint tends to say
     regardless of mood; (ii) speaks to whether the candidate pool surfaced at each step is itself
     mood-asymmetric before any pA scaling is even applied -- distinct questions, both left open by the
     original finding.
  3. NO-REGRESSION (`check_no_regression`, folded into `--phase summary`): whatever `neg_pa_scale` is used must
     not cost determinism, positive-direction divergence, mood-loadbearing (A!=B), or push the affect-word
     ("salad") fraction toward the collapse zone the original calibration sweep measured (0.4-0.65 at
     affect_boost>=80) -- compared against the BASELINE_* constants below, sourced from the two already-
     committed artifacts named there.
  4. MOAT-HOLDS (`--phase moat`): re-verifies the finding's own scope claim -- `render_fact_sentence`'s
     closed-class fact-clause path (reached via `generate(..., sentence_facts=...)`) is architecturally
     unreachable from `_affect_pool_gains` (confirmed by reading `generate()`'s `_run()` closure: the
     `sentence_facts` branch returns before `_free_gen`/`_free_gen_linattn` -- the only callers of
     `_affect_pool_gains` -- are ever invoked), re-checked EMPIRICALLY here (flag on/off x 3 mood points) rather
     than assumed from the code reading alone.
  5. BYTE-IDENTICAL-OFF (`--phase byte_identical`): DELEGATES to the existing, already-committed
     `research.runners._wkv_mouth_affect_neural_byte_identical_check` (re-run, not re-implemented) -- this
     rung's own new call sites (the neg_pa_scale monkeypatch, the moat's sentence_facts calls) exercise the
     SAME underlying `generate()`/`_affect_pool_gains`, so re-confirming the flag-off and lesion-equivalent
     proofs now (rather than trusting 2026-09-04's run still holds) is the right paranoia level for a
     promotion decision.

WHY PER-SEED, NOT PER-COMBINATION, AND WHY ALL-3-PROMPTS (not a majority). A majority-of-3 rule would call the
CURRENT, unfixed baseline "5/6 seeds passing" already (see the numbers above: a 2-of-3 rule reads 42/43/44/100/
102 as passing negative, only 101 failing = 5/6, with `neg_pa_scale` never touched) -- exactly the lenient-
metric trap `docs/TERMS.md`'s "selective"/"works" entries exist to name. Requiring all 3 prompts makes the
per-seed gate genuinely test whether a fix helps, not merely re-describe the existing 11/18 in seed-shaped units.

STATUS OF THIS MODULE: a HARNESS, not a verdict. Per `docs/TERMS.md` ("GO" may be used only when the gate's own
verdict is positive), no phase below has been run past a 1-seed/1-prompt smoke -- the full 6-seed run is
reserved for when compute frees (fan-out commands below). `--phase go` will print NOT-YET until it has real
6-seed `directional`/`summary` + `moat` + `byte_identical` artifacts to read.

Run (CPU-forced throughout -- the bank is a few hundred Izhikevich neurons or a tiny SpikingClauseProducer;
GPU is not needed and may be busy):
  # smoke (cheap, ~1 seed x 1 prompt, confirms every new code path parses/imports/produces a step):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_promote_validate \\
      --phase smoke --json research/findings/raw/_wkv_mouth_affect_neural_promote_smoke.json

  # full validation, ONE PROCESS PER SEED (mechanical-parallelism: fan out, do not loop seeds serially) --
  # for each seed s in 42 43 44 100 101 102, at the candidate fix scale (repeat once per --neg-pa-scale
  # candidate, e.g. 1.0 for the reproduce-baseline arm and 2.0/3.0 for the fix attempt):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_promote_validate \\
      --phase directional --seed <s> --neg-pa-scale <scale> \\
      --json research/findings/raw/_wkv_mouth_affect_neural_promote_directional_s<s>.json
  # then, once all 6 seeds at the chosen scale have landed:
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_promote_validate \\
      --phase summary --json research/findings/raw/_wkv_mouth_affect_neural_promote_summary.json
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_promote_validate \\
      --phase characterize --seed 42 \\
      --json research/findings/raw/_wkv_mouth_affect_neural_promote_characterize_s42.json
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_promote_validate \\
      --phase moat --json research/findings/raw/_wkv_mouth_affect_neural_promote_moat.json
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_promote_validate \\
      --phase byte_identical --seed <s> \\
      --json research/findings/raw/_wkv_mouth_affect_neural_promote_byte_identical_s<s>.json   # per seed
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_promote_validate \\
      --phase go --json research/findings/raw/_wkv_mouth_affect_neural_promote_go.json
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
sys.path.insert(0, str(_REPO_ROOT))

# Reuse BY IMPORT, never re-derived: the exact arm-runner, salad-fraction proxy, prompt/mood-point config, and
# lever-instrumentation this module's own measurements must stay comparable to.
from research.runners._wkv_mouth_affect_neural_verify import (  # noqa: E402
    _run_arm, _affect_word_fraction, PROMPTS, POS_V, POS_A, NEG_V, NEG_A,
)
from research.runners.affect_production_organ import WARRINER, STOP, _STRONG_MARGIN  # noqa: E402
from tools.lab import lever  # noqa: E402


# ── Baseline reference values (measured 2026-09-04, cited not re-derived; neg_pa_scale==1.0 by construction --
# that run predates this knob existing). `check_no_regression` below compares against these. ──
# research/findings/raw/_wkv_mouth_affect_neural_loadbearing_aggregate.json (neural row):
BASELINE_NEURAL_LOADBEARING_A_NE_B = "18/18"
BASELINE_NEURAL_DETERMINISM = "18/18"
BASELINE_NEURAL_LESION_POS = "18/18"
BASELINE_NEURAL_LESION_NEG = "11/18"          # the undershoot this rung targets
# research/findings/raw/_wkv_mouth_affect_neural_calibration_sweep.json, affect_boost==10.0 row:
BASELINE_AFFECT_WORD_FRAC_AT_BOOST10 = 0.077  # positive-mood arm, the calibration sweep's own reported value
SALAD_COLLAPSE_FRAC = 0.4                     # the sweep's own >=affect_boost=80 collapse floor (0.4-0.65)

# ── This rung's OWN config (not inherited) ──
SEEDS = [42, 43, 44, 100, 101, 102]
MIN_SEEDS_PASS = 5
TOTAL_SEEDS = 6
SALAD_FRAC_CEILING = 0.20   # well under the sweep's 0.4 collapse floor, above the ~0.06-0.10 neutral band
NEG_PA_SCALE_CANDIDATES = [1.0, 1.5, 2.0, 3.0]   # 1.0 = reproduce-baseline arm; others = the fix attempt


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Arm runners
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _run_arm_scaled(seed: int, prompt: str, valence: float, arousal: float, affect_boost: float, neural: bool,
                     neg_pa_scale: float = 1.0, max_new_tokens: int = 48, recurrence: str = "ssm",
                     timeout_s: int = 180) -> dict:
    """Same contract as `_wkv_mouth_affect_neural_verify._run_arm`, plus one experimental knob: `neg_pa_scale`
    multiplies the pA `_affect_pool_gains` (webapp/wkv_mouth_generator.py) returns, but ONLY on the
    negative-valence branch (`valence < 0.0`) -- testing the finding's own named next step ("an asymmetric,
    stronger negative-direction pA scale") without editing production code. When `neural=False` or
    `neg_pa_scale==1.0` this is an EXACT delegate to `_run_arm` (no monkeypatch injected at all) -- so the
    reproduce-baseline arm (`--neg-pa-scale 1.0`) is not merely "should behave the same as before", it is
    LITERALLY the same call.

    The scaling is injected via a MONKEYPATCH of the module-level `_affect_pool_gains` name inside a FRESH,
    throwaway subprocess -- scoped to that one subprocess only, so production and every other runner import of
    `webapp.wkv_mouth_generator` are completely unaffected. This is a research-runner-side experiment; promoting
    a chosen scale into the shipped calibration constant is a separate, reviewed edit once a value is shown to
    help at the full 6-seed scale."""
    if not neural or neg_pa_scale == 1.0:
        return _run_arm(seed, prompt, valence, arousal, affect_boost, neural,
                         max_new_tokens=max_new_tokens, recurrence=recurrence, timeout_s=timeout_s)
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env["BRAIN_WKV_MOUTH_AFFECT"] = "1"
    env["BRAIN_WKV_MOUTH_AFFECT_NEURAL"] = "1"
    env["BRAIN_WKV_MOUTH_RECURRENCE"] = recurrence
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "import webapp.wkv_mouth_generator as _M\n"
        "_orig_gains = _M._affect_pool_gains\n"
        f"_SCALE = {float(neg_pa_scale)!r}\n"
        "def _scaled_gains(cand, affect_ids, valence, arousal, affect_boost, recent_ids=None):\n"
        "    g = _orig_gains(cand, affect_ids, valence, arousal, affect_boost, recent_ids=recent_ids)\n"
        "    if valence < 0.0 and g:\n"
        "        g = {k: v * _SCALE for k, v in g.items()}\n"
        "    return g\n"
        "_M._affect_pool_gains = _scaled_gains\n"
        f"text, secs = _M.generate({prompt!r}, seed={int(seed)}, max_new_tokens={int(max_new_tokens)}, "
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
            f"scaled arm failed (seed={seed} valence={valence} arousal={arousal} neg_pa_scale={neg_pa_scale}): "
            f"rc={proc.returncode}\nSTDOUT-tail={proc.stdout[-2000:]}\nSTDERR-tail={proc.stderr[-4000:]}"
        )
    d = json.loads(line[len("RESULT_JSON:"):])
    d["wall_s"] = round(wall, 2)
    d["affect_word_frac"] = _affect_word_fraction(d["text"])
    return d


def _run_arm_instrumented(seed: int, prompt: str, valence: float, arousal: float, affect_boost: float,
                          max_new_tokens: int = 48, timeout_s: int = 180) -> dict:
    """Neural arm that CAPTURES the raw `_affect_pool_gains` output at every generation step via the SAME
    monkeypatch technique `_run_arm_scaled` uses -- here for MEASUREMENT, not manipulation (the wrapper returns
    the ORIGINAL gains dict unchanged). Tests the mechanistic half of the "positivity prior" hypothesis: does
    the negative-mood arm see a lower congruent-candidate HIT RATE and/or lower mean delivered pA per hit than
    the positive arm at MATCHED |valence|/|arousal| -- i.e. is the candidate pool itself mood-asymmetric before
    any pA scaling is applied, as distinct from `_signed_warriner_stats` below (which reads the OUTPUT text,
    not the modulatory current that produced it). Returns {"text", "secs", "totals": {"sum_pA", "n_congruent",
    "n_steps"}, "wall_s", "affect_word_frac"}."""
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env["BRAIN_WKV_MOUTH_AFFECT"] = "1"
    env["BRAIN_WKV_MOUTH_AFFECT_NEURAL"] = "1"
    env["BRAIN_WKV_MOUTH_RECURRENCE"] = "ssm"
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "import webapp.wkv_mouth_generator as _M\n"
        "_orig_gains = _M._affect_pool_gains\n"
        "_totals = {'sum_pA': 0.0, 'n_congruent': 0, 'n_steps': 0}\n"
        "def _measured_gains(cand, affect_ids, valence, arousal, affect_boost, recent_ids=None):\n"
        "    g = _orig_gains(cand, affect_ids, valence, arousal, affect_boost, recent_ids=recent_ids)\n"
        "    _totals['n_steps'] += 1\n"
        "    _totals['sum_pA'] += sum(g.values())\n"
        "    _totals['n_congruent'] += len(g)\n"
        "    return g\n"
        "_M._affect_pool_gains = _measured_gains\n"
        f"text, secs = _M.generate({prompt!r}, seed={int(seed)}, max_new_tokens={int(max_new_tokens)}, "
        f"valence={float(valence)!r}, arousal={float(arousal)!r}, affect_boost={float(affect_boost)!r})\n"
        "print('RESULT_JSON:' + json.dumps({'text': text, 'secs': secs, 'totals': _totals}))\n"
    )
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=timeout_s, cwd=str(_REPO_ROOT))
    wall = time.time() - t0
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")), None)
    if line is None:
        raise RuntimeError(f"instrumented arm failed (seed={seed} valence={valence} arousal={arousal}): "
                           f"rc={proc.returncode}\nSTDOUT-tail={proc.stdout[-2000:]}\nSTDERR-tail={proc.stderr[-4000:]}")
    d = json.loads(line[len("RESULT_JSON:"):])
    d["wall_s"] = round(wall, 2)
    d["affect_word_frac"] = _affect_word_fraction(d["text"])
    return d


_MOAT_SENTENCE_FACTS = [("bounce_around_the_ground", "country", "united_kingom")]
# ^ verbatim worked example from webapp/wkv_mouth_generator.py::render_fact_sentence's own docstring. "country"
# IS in RELATION_LEXICON (research/runners/_wkv_fact_to_sentence_lexicon_lever.py: "is located in"/"the"), so
# this triple is GUARANTEED covered (renders a real sentence), not a guess that might silently fall through to
# free-gen and defeat the point of the check.


def _run_fact_sentence_arm(seed: int, valence: float, arousal: float, affect_boost: float, neural: bool,
                            prompt: str = "tell me about", max_new_tokens: int = 24,
                            timeout_s: int = 120) -> dict:
    """One `generate(..., sentence_facts=_MOAT_SENTENCE_FACTS)` call in a fresh subprocess -- the closed-class
    fact-clause path, architecturally upstream of `_free_gen`/`_affect_pool_gains` (see `generate()`'s `_run()`
    closure: the `sentence_facts` branch returns BEFORE `_get_readout`/`FewSpikeWordRead` are even
    constructed), so `max_new_tokens` is irrelevant to this branch -- kept small regardless for a fast smoke."""
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env["BRAIN_WKV_MOUTH_AFFECT"] = "1"
    env["BRAIN_WKV_MOUTH_AFFECT_NEURAL"] = "1" if neural else "0"
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "from webapp.wkv_mouth_generator import generate\n"
        f"text, secs = generate({prompt!r}, seed={int(seed)}, max_new_tokens={int(max_new_tokens)}, "
        f"sentence_facts={_MOAT_SENTENCE_FACTS!r}, valence={float(valence)!r}, arousal={float(arousal)!r}, "
        f"affect_boost={float(affect_boost)!r})\n"
        "print('RESULT_JSON:' + json.dumps({'text': text, 'secs': secs}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=timeout_s, cwd=str(_REPO_ROOT))
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")), None)
    if line is None:
        raise RuntimeError(f"fact-sentence moat arm failed (seed={seed} neural={neural}): "
                           f"rc={proc.returncode}\nSTDOUT-tail={proc.stdout[-2000:]}\nSTDERR-tail={proc.stderr[-4000:]}")
    return json.loads(line[len("RESULT_JSON:"):])


def _signed_warriner_stats(text: str) -> dict:
    """Per-word SIGNED Warriner valence (val9-5.0, so >0=pleasant/<0=unpleasant) over STRONG affect words in
    `text` (the SAME gate `_affect_word_fraction` uses) -- the raw ingredient `_affect_word_fraction` throws
    away (it only counts hits, not their sign). Returns {"n", "mean_signed", "frac"} -- `mean_signed` is None
    when `n==0` (UNDEFINED, not 0.0 -- `tools.lab.undefined_if_empty`'s own reasoning: a text with zero
    affect-bearing words says nothing about polarity, and reporting 0.0 would silently claim 'neutral' for 'no
    data')."""
    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words:
        return {"n": 0, "mean_signed": None, "frac": 0.0}
    signed = []
    for w in words:
        if w in STOP or w not in WARRINER:
            continue
        v9, _a9 = WARRINER[w]
        if abs(v9 - 5.0) >= _STRONG_MARGIN:
            signed.append(v9 - 5.0)
    if not signed:
        return {"n": 0, "mean_signed": None, "frac": 0.0}
    return {"n": len(signed), "mean_signed": sum(signed) / len(signed), "frac": len(signed) / len(words)}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Phases
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def phase_directional(seed: int, affect_boost: float, neg_pa_scale: float, out: dict) -> None:
    """The 4-arm (A pos / B neg / C repeat / L lesion) x 3-prompt battery, for BOTH mechanisms (host always at
    neg_pa_scale==1.0 -- the scale only ever applies to the neural mechanism's negative branch), scored the
    SAME way `_wkv_mouth_affect_neural_verify.phase_loadbearing` does per-row, PLUS each row's own
    `affect_word_frac` for the two mood arms (feeds `check_no_regression`'s salad-ceiling check without a
    second pass over the same text)."""
    result = {}
    for neural in (False, True):
        mech = "neural" if neural else "host"
        per_prompt = []
        for prompt in PROMPTS:
            run = (lambda v, a: _run_arm_scaled(seed, prompt, v, a, affect_boost, neural,
                                                neg_pa_scale=neg_pa_scale))
            a = run(POS_V, POS_A)          # A: positive mood
            b = run(NEG_V, NEG_A)          # B: negative mood
            c = run(POS_V, POS_A)          # C: repeat of A
            lesion = run(0.0, 0.0)         # lesion (valence==0.0 -- neg_pa_scale never applies, guarded upstream)
            tag = f"{mech} seed={seed} prompt={prompt!r} neg_pa_scale={neg_pa_scale}"
            moved_ab = lever(f"mood A(pos) vs B(neg) [{tag}]", a["text"], b["text"], required=False)
            moved_ac = lever(f"determinism A vs C(repeat) [{tag}]", a["text"], c["text"], required=False)
            moved_la = lever(f"lesion vs A(pos) [{tag}]", lesion["text"], a["text"], required=False)
            moved_lb = lever(f"lesion vs B(neg) [{tag}]", lesion["text"], b["text"], required=False)
            row = {
                "prompt": prompt,
                "A_pos": a["text"], "B_neg": b["text"], "C_pos_repeat": c["text"], "L_lesion": lesion["text"],
                "affect_loadbearing_A_ne_B": moved_ab,
                "determinism_A_eq_C": not moved_ac,
                "lesion_ne_A": moved_la,
                "lesion_ne_B": moved_lb,
                "A_pos_affect_frac": a["affect_word_frac"], "B_neg_affect_frac": b["affect_word_frac"],
            }
            per_prompt.append(row)
            print(f"[directional {mech} seed={seed} scale={neg_pa_scale} prompt={prompt!r}] "
                  f"A!=B={row['affect_loadbearing_A_ne_B']} A==C={row['determinism_A_eq_C']} "
                  f"L!=A={row['lesion_ne_A']} L!=B={row['lesion_ne_B']}", flush=True)
        result[mech] = per_prompt
    out["directional"] = {"seed": seed, "affect_boost": affect_boost, "neg_pa_scale": neg_pa_scale, **result}


def phase_characterize(seed: int, affect_boost: float, out: dict) -> None:
    """Tests the finding's own named, NOT-independently-verified "positivity prior" hypothesis two ways: (i)
    OUTPUT-level -- does the lesion arm's own continuation already skew positive on the Warriner scale; (ii)
    SOURCE-level -- at matched |valence|/|arousal|, does the negative arm see a lower congruent-candidate hit
    rate and/or lower mean pA per hit than the positive arm (via `_run_arm_instrumented`'s transparent capture
    of `_affect_pool_gains`'s own output). A null result on both would REFUTE the hypothesis and point back at
    the pA calibration formula itself as the more likely explanation."""
    rows = []
    for prompt in PROMPTS:
        lesion = _run_arm(seed, prompt, 0.0, 0.0, affect_boost, neural=True)
        pos = _run_arm_instrumented(seed, prompt, POS_V, POS_A, affect_boost)
        neg = _run_arm_instrumented(seed, prompt, NEG_V, NEG_A, affect_boost)

        def _derived(d: dict) -> dict:
            n_steps = d["totals"]["n_steps"]
            n_cong = d["totals"]["n_congruent"]
            return {
                "sum_pA": d["totals"]["sum_pA"], "n_congruent": n_cong, "n_steps": n_steps,
                "congruent_hit_rate": (n_cong / n_steps) if n_steps else None,
                "mean_pA_per_congruent_hit": (d["totals"]["sum_pA"] / n_cong) if n_cong else None,
                "warriner": _signed_warriner_stats(d["text"]),
            }

        row = {
            "prompt": prompt,
            "lesion_warriner": _signed_warriner_stats(lesion["text"]),
            "positive": _derived(pos), "negative": _derived(neg),
        }
        rows.append(row)
        print(f"[characterize seed={seed} prompt={prompt!r}] "
              f"lesion_mean_signed={row['lesion_warriner']['mean_signed']} "
              f"pos(hit_rate={row['positive']['congruent_hit_rate']}, "
              f"mean_pA={row['positive']['mean_pA_per_congruent_hit']}) "
              f"neg(hit_rate={row['negative']['congruent_hit_rate']}, "
              f"mean_pA={row['negative']['mean_pA_per_congruent_hit']})", flush=True)

    def _avg(key1: str, key2: str):
        vals = [r[key1][key2] for r in rows if r[key1].get(key2) is not None]
        return (sum(vals) / len(vals)) if vals else None

    lesion_means = [r["lesion_warriner"]["mean_signed"] for r in rows
                    if r["lesion_warriner"]["mean_signed"] is not None]
    out["characterize"] = {
        "seed": seed, "affect_boost": affect_boost, "rows": rows,
        "lesion_mean_signed_valence_avg": (sum(lesion_means) / len(lesion_means)) if lesion_means else None,
        "positivity_prior_supported_by_output_skew": bool(lesion_means)
                                                       and (sum(lesion_means) / len(lesion_means) > 0.0),
        "avg_congruent_hit_rate_positive": _avg("positive", "congruent_hit_rate"),
        "avg_congruent_hit_rate_negative": _avg("negative", "congruent_hit_rate"),
        "avg_mean_pA_per_hit_positive": _avg("positive", "mean_pA_per_congruent_hit"),
        "avg_mean_pA_per_hit_negative": _avg("negative", "mean_pA_per_congruent_hit"),
    }


def phase_moat(seed: int, affect_boost: float, out: dict) -> None:
    """The finding's own "Moat / scope" claim, RE-VERIFIED empirically for this rung: `render_fact_sentence`'s
    closed-class path must be COMPLETELY unaffected by BRAIN_WKV_MOUTH_AFFECT_NEURAL or by any mood value."""
    arms = []
    for neural in (False, True):
        for valence, arousal in ((0.0, 0.0), (POS_V, POS_A), (NEG_V, NEG_A)):
            d = _run_fact_sentence_arm(seed, valence, arousal, affect_boost, neural)
            arms.append({"neural": neural, "valence": valence, "arousal": arousal, "text": d["text"]})
    baseline = arms[0]["text"]
    identical = all(a["text"] == baseline for a in arms)
    covered = bool(baseline.strip())   # a covered relation renders non-empty; empty would mean it fell through
    out["moat"] = {
        "seed": seed, "affect_boost": affect_boost, "arms": arms,
        "moat_identical_regardless_of_flag_or_mood": identical, "fact_sentence_covered": covered,
    }
    print(f"[moat seed={seed}] identical={identical} covered={covered} text={baseline!r}", flush=True)


def phase_byte_identical(seed: int, affect_boost: float, out: dict, before_ref: str = "HEAD") -> None:
    """Delegates to the EXISTING, already-committed `_wkv_mouth_affect_neural_byte_identical_check` (re-run,
    not re-implemented)."""
    tmp = Path(f"research/findings/raw/_wkv_mouth_affect_neural_promote_byte_identical_s{seed}.json")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "research.runners._wkv_mouth_affect_neural_byte_identical_check",
           "--before-ref", before_ref, "--seed", str(seed), "--affect-boost", str(affect_boost),
           "--json", str(tmp)]
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, cwd=str(_REPO_ROOT))
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"byte-identical checker failed: rc={proc.returncode}\n"
                           f"STDOUT-tail={proc.stdout[-2000:]}\nSTDERR-tail={proc.stderr[-2000:]}")
    d = json.loads(tmp.read_text())
    out["byte_identical"] = {
        "seed": seed, "delegated_artifact": str(tmp),
        "off_by_default_all_byte_identical": d["off_by_default_all_byte_identical"],
        "lesion_equivalent_all_byte_identical": d["lesion_equivalent_all_byte_identical"],
    }
    print(f"[byte_identical seed={seed}] off_by_default={d['off_by_default_all_byte_identical']} "
          f"lesion_equivalent={d['lesion_equivalent_all_byte_identical']}", flush=True)


def _seed_passes_direction(rows: list, key: str) -> bool:
    return all(r[key] for r in rows)


def check_no_regression(directional_by_seed: dict) -> dict:
    """Safety-net check, INDEPENDENT of whether the fix improved the negative direction: whatever neg_pa_scale
    was used must not have made anything ELSE worse, vs the BASELINE_* constants (measured 2026-09-04,
    neg_pa_scale==1.0 by construction)."""
    all_rows_neural = [r for rows in directional_by_seed.values() for r in rows["neural"]]
    all_rows_host = [r for rows in directional_by_seed.values() for r in rows["host"]]
    n = len(all_rows_neural)
    lb = sum(1 for r in all_rows_neural if r["affect_loadbearing_A_ne_B"])
    det = sum(1 for r in all_rows_neural if r["determinism_A_eq_C"])
    lesion_pos = sum(1 for r in all_rows_neural if r["lesion_ne_A"])
    host_lb = sum(1 for r in all_rows_host if r["affect_loadbearing_A_ne_B"])
    host_det = sum(1 for r in all_rows_host if r["determinism_A_eq_C"])
    max_frac = max((r.get("B_neg_affect_frac", 0.0) for r in all_rows_neural), default=0.0)
    checks = {
        "determinism_still_100pct": det == n,
        "loadbearing_A_ne_B_still_100pct": lb == n,
        "positive_direction_still_100pct": lesion_pos == n,
        "host_mechanism_unaffected_loadbearing": (not all_rows_host) or (host_lb == len(all_rows_host)),
        "host_mechanism_unaffected_determinism": (not all_rows_host) or (host_det == len(all_rows_host)),
        "salad_frac_under_ceiling": max_frac <= SALAD_FRAC_CEILING,
    }
    checks["no_regression"] = all(checks.values())
    checks["_n_combos"] = n
    checks["_max_B_neg_affect_word_frac"] = max_frac
    return checks


def phase_summary(seeds: list, directional_pattern: str, out: dict) -> None:
    """Aggregates the per-seed `directional` artifacts into the per-seed pass/fail the GO gate reads, plus
    `check_no_regression`. Kept as its own runner phase (not a one-off analysis script) so the aggregate itself
    carries the SAME automatic run provenance (argv/git-SHA/env) every other artifact in this project does."""
    directional_by_seed = {}
    neg_pa_scale = None
    for s in seeds:
        d = json.loads(Path(directional_pattern.format(seed=s)).read_text())
        directional_by_seed[s] = d["directional"]
        neg_pa_scale = d["directional"]["neg_pa_scale"]

    pos_pass = {s: _seed_passes_direction(directional_by_seed[s]["neural"], "lesion_ne_A") for s in seeds}
    neg_pass = {s: _seed_passes_direction(directional_by_seed[s]["neural"], "lesion_ne_B") for s in seeds}
    n_pos_pass = sum(pos_pass.values())
    n_neg_pass = sum(neg_pass.values())

    no_reg = check_no_regression(directional_by_seed)
    directional_go = (n_pos_pass >= MIN_SEEDS_PASS) and (n_neg_pass >= MIN_SEEDS_PASS) and no_reg["no_regression"]

    out["summary"] = {
        "seeds": seeds, "neg_pa_scale": neg_pa_scale,
        "positive_direction_seeds_passing": f"{n_pos_pass}/{len(seeds)}",
        "negative_direction_seeds_passing": f"{n_neg_pass}/{len(seeds)}",
        "per_seed_positive_pass": pos_pass, "per_seed_negative_pass": neg_pass,
        "no_regression": no_reg,
        "min_seeds_required": MIN_SEEDS_PASS, "total_seeds": TOTAL_SEEDS,
        "directional_go_component": directional_go,
    }
    print(f"[summary] neg_pa_scale={neg_pa_scale} positive={n_pos_pass}/{len(seeds)} "
          f"negative={n_neg_pass}/{len(seeds)} no_regression={no_reg['no_regression']} "
          f"directional_go_component={directional_go}", flush=True)


def phase_go(summary_path: str, moat_path: str, byte_identical_paths: list, out: dict) -> None:
    """Overall PROMOTE-READY verdict = directional GO component AND moat holds AND every seed's byte-identical
    check holds. Reads already-produced artifacts; does not run anything itself. Per `docs/TERMS.md`, this
    prints NOT-YET (never "GO") until real 6-seed artifacts back every input -- a smoke-only run has no such
    artifacts and this phase is not exercised by `--phase smoke`."""
    summary = json.loads(Path(summary_path).read_text())["summary"]
    moat = json.loads(Path(moat_path).read_text())["moat"]
    byte_ident_all = []
    for p in byte_identical_paths:
        d = json.loads(Path(p).read_text())["byte_identical"]
        byte_ident_all.append(bool(d["off_by_default_all_byte_identical"]) and
                              bool(d["lesion_equivalent_all_byte_identical"]))
    go = bool(summary["directional_go_component"]) and bool(moat["moat_identical_regardless_of_flag_or_mood"]) \
        and bool(moat["fact_sentence_covered"]) and bool(byte_ident_all) and all(byte_ident_all)
    out["go"] = {
        "GO": go, "summary": summary,
        "moat_holds": moat["moat_identical_regardless_of_flag_or_mood"],
        "byte_identical_holds": bool(byte_ident_all) and all(byte_ident_all),
    }
    print(f"[GO-GATE] overall={'GO' if go else 'NOT-YET'}", flush=True)


def phase_smoke(out: dict) -> None:
    """Confirms the harness IMPORTS/PARSES/RUNS A STEP -- 1 seed, 1 prompt, the smallest arm set that still
    exercises every new code path this rung adds (the `neg_pa_scale` monkeypatch, the instrumented-capture
    monkeypatch, the moat's `sentence_facts` call, the signed-Warriner diagnostic). NOT a claim about the GO
    gate -- see the module docstring for why the full 6-seed run is reserved for when compute frees."""
    seed, prompt = 42, PROMPTS[0]
    boost = 10.0
    lesion = _run_arm(seed, prompt, 0.0, 0.0, boost, neural=True, max_new_tokens=12)
    neg_unscaled = _run_arm(seed, prompt, NEG_V, NEG_A, boost, neural=True, max_new_tokens=12)
    neg_scaled = _run_arm_scaled(seed, prompt, NEG_V, NEG_A, boost, neural=True, neg_pa_scale=2.0,
                                 max_new_tokens=12)
    pos_instrumented = _run_arm_instrumented(seed, prompt, POS_V, POS_A, boost, max_new_tokens=12)
    moat_off = _run_fact_sentence_arm(seed, 0.0, 0.0, boost, neural=False, max_new_tokens=12)
    moat_on = _run_fact_sentence_arm(seed, POS_V, POS_A, boost, neural=True, max_new_tokens=12)

    # exploratory only (`required=False`): 1 seed/1 prompt/12 tokens is too small a sample to REQUIRE movement,
    # this just reports whether the scale=2.0 monkeypatch had a visible effect at this one tiny sample point.
    scale_moved = lever("smoke: neg_pa_scale=2.0 vs 1.0 (unscaled)", neg_unscaled["text"], neg_scaled["text"],
                        required=False)
    warriner = _signed_warriner_stats(lesion["text"])
    all_nonempty = all(bool(t.strip()) for t in
                       (lesion["text"], neg_unscaled["text"], neg_scaled["text"], pos_instrumented["text"],
                        moat_off["text"], moat_on["text"]))
    out["smoke"] = {
        "seed": seed, "prompt": prompt,
        "lesion_text": lesion["text"], "neg_unscaled_text": neg_unscaled["text"],
        "neg_scaled_text": neg_scaled["text"], "neg_scale_used": 2.0, "neg_scale_moved_output": scale_moved,
        "pos_instrumented_text": pos_instrumented["text"], "pos_instrumented_totals": pos_instrumented["totals"],
        "moat_off_text": moat_off["text"], "moat_on_text": moat_on["text"],
        "moat_identical": moat_off["text"] == moat_on["text"],
        "lesion_warriner_stats": warriner,
        "all_arms_nonempty": all_nonempty,
        "harness_ready": all_nonempty,   # the ONLY claim this phase makes: every new code path ran to a result
    }
    print(f"[smoke] all_arms_nonempty={all_nonempty} moat_identical={out['smoke']['moat_identical']} "
          f"neg_scale_moved_output={scale_moved} harness_ready={out['smoke']['harness_ready']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "directional", "characterize", "moat", "byte_identical",
                                        "summary", "go"], default="smoke")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--affect-boost", type=float, default=10.0)
    ap.add_argument("--neg-pa-scale", type=float, default=1.0)
    ap.add_argument("--before-ref", type=str, default="HEAD")
    ap.add_argument("--directional-pattern", type=str,
                    default="research/findings/raw/_wkv_mouth_affect_neural_promote_directional_s{seed}.json")
    ap.add_argument("--summary-json", type=str,
                    default="research/findings/raw/_wkv_mouth_affect_neural_promote_summary.json")
    ap.add_argument("--moat-json", type=str,
                    default="research/findings/raw/_wkv_mouth_affect_neural_promote_moat.json")
    ap.add_argument("--byte-identical-pattern", type=str,
                    default="research/findings/raw/_wkv_mouth_affect_neural_promote_byte_identical_s{seed}.json")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wkv_mouth_affect_neural_promote_validate.json")
    args = ap.parse_args()

    t0 = time.time()
    out: dict = {"phase": args.phase, "seed": args.seed, "affect_boost": args.affect_boost,
                "neg_pa_scale": args.neg_pa_scale}
    if args.phase == "smoke":
        phase_smoke(out)
    elif args.phase == "directional":
        phase_directional(args.seed, args.affect_boost, args.neg_pa_scale, out)
    elif args.phase == "characterize":
        phase_characterize(args.seed, args.affect_boost, out)
    elif args.phase == "moat":
        phase_moat(args.seed, args.affect_boost, out)
    elif args.phase == "byte_identical":
        phase_byte_identical(args.seed, args.affect_boost, out, before_ref=args.before_ref)
    elif args.phase == "summary":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        phase_summary(seeds, args.directional_pattern, out)
    elif args.phase == "go":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        byte_paths = [args.byte_identical_pattern.format(seed=s) for s in seeds]
        phase_go(args.summary_json, args.moat_json, byte_paths, out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[done] phase={args.phase} {out['elapsed_s']}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
