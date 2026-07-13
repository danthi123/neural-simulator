# The SSM-extract's multi-timescale RECURRENCE does NOT robustly help deep-context LANGUAGE — 6-seed NEGATIVE (blind-seed-clean); the memory-horizon GO was pure RETENTION, and retention ≠ prediction. The plain random ESN (nonlinear mixing) is the robustly-best fixed reservoir at every depth

**Date:** 2026-07-13
**Runner:** `research/runners/_ssm_context_depth_derisk.py` (numpy-CPU; REUSE-BY-IMPORT of the VALIDATED context-depth machinery — the proper softmax delta-rule read-out `train_readout`, the per-context-depth reservoir-minus-bigram MARGIN, the bag-of-prefix CONFOUND control; NO `sim/` edit, NO BPTT).
**Status:** ❌ HONEST 6-seed NEGATIVE — a first-class deliverable that CHARACTERIZES exactly where the SSM/HiPPO multi-timescale mechanism helps (pure retention) and where it does not (language prediction), and REDIRECTS the lever.

## The two prior GOs this reconciles (a-1 done — credit + build on our own record)
1. **The memory-horizon GO** (`2026-07-13-SSM-fixed-structured-multitimescale-reservoir-SURPASSES-fading-memory-ceiling-6seed-GO.md`): a fixed multi-timescale diagonal holds a cue UNBOUNDED (decode 1.000 at gap 150) where a random ESN fades. **This finding shows that GO was PURE RETENTION — decode a HELD cue — and does NOT transfer to language.**
2. **The multi-timescale-READ GO** (`2026-07-12-multitimescale-reservoir-read-...6seed-GO.md`): reading the SAME reservoir at TWO timescales (slow+fast concat) beats single-read, 6-seed — a READ-OUT feature-diversity lever, honestly scoped as a WITHIN-Ueda-ceiling gain. **That is a DIFFERENT lever (the READ) from the RECURRENCE structure tested here; the two are consistent — the read helps, the recurrence structure does not.**

## Result 1 — PER-SENTENCE (≤16-token) deep-context, 6-seed (42/43/44 dev + 100/101/102 blind), TinyStories V=200, raw feature
Reservoir-minus-bigram CE margin at DEEP context (6+), mean over the two deep buckets:
| arm | mean deep margin (6 seeds) | beats plain ESN at deep? |
|---|---|---|
| **plain random ESN** (`tanh(W@x)`, nonlinear mixing) | **+0.193** | — (the reference) |
| hetero-leaky-ESN (mixing + heterogeneous time constants) | +0.184 | **2/6 seeds only** (both DEV; all 3 BLIND reversed) |
| multitimescale diagonal (the SSM extract) | +0.155 | worse on average |
| bag-of-prefix (memoryless control) | −0.021 | — |
- **hetero beats random at deep on only 2/6 seeds — both DEV seeds (42, 43); ALL THREE BLIND seeds (100/101/102) REVERSED it.** The dev-seed positive was a FALSE POSITIVE — exactly what the 3-dev/3-blind rule guards against.
- The plain random ESN has the BEST mean deep margin. Every reservoir beats the bag at deep (the reservoir's recurrent dynamics carry deep signal beyond a memoryless bag — the already-documented result), but the multi-timescale STRUCTURE adds nothing robust over plain mixing.

## Result 2 — LONG-CONTEXT (concat=5, ~45-token discourse sequences), depth 16–999 where long time constants SHOULD help most
`--concat 5` groups 5 CONSECUTIVE (same-story) sentences into one sequence so context depth extends to ~45 tokens — the regime where long τ (up to 400) finally have TIME to integrate (the memory GO used 150 steps; ≤16 is too short). **6-seed (42/43/44/100/101/102), deep = depth 16+, n-weighted margin over the bigram:**
| arm | mean deep margin (6 seeds) | beats plain ESN at deep? |
|---|---|---|
| **plain random ESN** | **+0.199** (robustly + every seed, +0.16→+0.24) | — |
| multitimescale diagonal | **−0.055** | **0/6** |
| hetero-leaky-ESN | **−0.035** | **0/6** |
| bag-of-prefix | −0.61 (hugely negative — a diffuse ~45-token count is uninformative) | — |
- At GENUINE long context, the plain ESN robustly BEATS the bigram (+0.199 mean, positive every seed), while **every multi-timescale reservoir is near-zero-or-NEGATIVE — 0/6 beats the plain ESN.** The result is DECISIVE, 6-seed, and in the OPPOSITE direction to the "long timescales should help long context" hypothesis. Raw: `raw/_ssm_ctxdepth_concat5_6seed.json`.

## The mechanism (why retention ≠ prediction — the load-bearing insight)
- The multi-timescale diagonal is **LINEAR**: its state is a per-unit time-weighted SUM of past inputs. Over ~45 tokens the slow units integrate a large diffuse sum → **blurred, low-discriminability** (its long-context behaviour degenerates toward the bag, which is hugely negative here). It can PERFECTLY PRESERVE a single held cue (the memory task — pure retention, disjoint distractors, no prediction) but cannot COMPUTE the nonlinear higher-order features (n-gram conjunctions, "token A AND token B present") that language next-token PREDICTION rewards.
- The plain random ESN's **bounded nonlinear tanh mixing** computes those conjunctions AND keeps a discriminative (non-blurred) state → it is the robustly-best fixed reservoir at EVERY depth, short and long.
- ⇒ **the memory-horizon GO was a DECEPTIVE positive for language:** retention (hold a cue) is not prediction (compute over the context). Language is prediction; the multi-timescale recurrence's retention strength does not buy a prediction advantage — the nonlinear mixing does.

## ⇒ Honest verdict + the redirect (boundary = a mapped mechanism, the lever is elsewhere)
- The SSM/HiPPO multi-timescale RECURRENCE (a fixed structured recurrence) is NOT a robust lever for deep-context LANGUAGE at tractable scale — neither the pure diagonal nor the mixing+timescales leaky-ESN robustly beats a plain random ESN. The memory-horizon GO stands as a RETENTION result only.
- What DOES help language (from our record, unchanged): the plain reservoir's nonlinear mixing (beats bigram + bag at deep, robustly) + the multi-timescale READ feature-diversity lever (prior 6-seed GO) — both WITHIN the Ueda-bounded fixed-reservoir ceiling.
- The path PAST that ceiling remains **learned recurrence / deep credit** (the parked dendritic frontier), NOT fixed-reservoir structure tweaks. This negative RE-CONFIRMS that conclusion from a fresh angle (a principled structured recurrence, tested rigorously, does not beat random mixing on language).
- **A negative worth recording:** the fresh-mechanism-class gate (spiking SSMs) surfaced a genuinely-new class; its emergence-compatible extract cleanly surpasses the fading-memory ceiling on RETENTION but does NOT transfer to language PREDICTION — a precise, blind-seed-clean characterization of the mechanism's reach, reached by the cheap-first + a-1 + 6-seed + reuse-the-validated-machinery discipline.

## Files
`_ssm_context_depth_derisk.py` (`--feature raw --concat N`); `research/findings/raw/_ssm_ctxdepth_{raw,concat5}_6seed.json`. Supersedes the language-relevance question left open by `2026-07-13-SSM-language-escalation-toy-scale-NULL-DISCRIMINATOR-*` (that toy runner was a null discriminator; THIS runner, with the validated softmax read-out + by-depth margin + bag control, is a VALID discriminator and gives the decisive answer). NO `sim/` edit.
