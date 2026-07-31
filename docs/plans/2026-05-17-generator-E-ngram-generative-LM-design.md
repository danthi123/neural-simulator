---
type: plan
status: live
date: 2026-05-17
---

# Generator-E — Self-Contained N-gram Generative LM through the SAME Hardened Gate — Design (ACTIVE)

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). Continuous autonomous arc
> (user 2026-05-17: work the arc a week, no stopping/asking, no
> config-cranking a terminated mechanism, self-contained at RUNTIME,
> local 3090, public training corpus authorized). Supersedes the
> "non-spiking reservoir" pre-stage in the Generator-D successor
> section, REDIRECTED by the falsify-cheaply ESN probe (below).

## Why this is the genuinely-different, evidence-indicated next step

Converged evidence across the whole arc:
- **9 honest negatives**: char-BPTT (Inc-1/2/3), controllers/predictors
  over an order-blind pool (G1/G1.5/P), order-intrinsic readback,
  subword spiking LM hard-target (Generator-S: ~230x worse than
  random), and the SAME spiking LM with the strongest signal (dense
  soft-target distillation, Generator-D: best 1.57x worse than random,
  0/3 — distillation closed ~99.3% of the gap but still FAIL).
- **Falsify-cheaply ESN probe (recorded `bmbl0qcvv`)**: a NON-spiking
  echo-state reservoir LM at comparable cheap scale gets held-out ppl
  **471 vs uniform-random 513** — only ~8% better than random, NOT
  meaningfully competent. So "spiking vs non-spiking" is NOT the
  dividing line: a cheap *non-spiking learned* sequence model is also
  near-random at feasible local scale.
- **The ONLY thing competent on this corpus is the non-neural
  count-based n-gram** (the Generator-D teacher: held-out ppl ~14-15
  vs random 513, grounded `ba1jyepwf`). It has **never been put
  through the full hardened gate as the runtime generative model** —
  it was only Generator-D's distillation *target* (the gate judged
  the spiking student).

The decision-relevant question the project's actual goal demands
("conversational capability, self-contained, local, no cheats") is
therefore NOT another learned-substrate variant (that would
config-crank the broader pattern). It is: **is there ANY
self-contained, in-constraints, no-cheat generative LM that clears
the SAME rigorous pre-registered hardened gate at feasible local
scale?** The n-gram is the only competent candidate in hand and is
untested as a generator. Gating it is genuinely different (a
non-neural statistical generative LM, never gated this way),
in-constraints (self-contained at runtime: back-off count tables +
BPE merge JSON only; zero external deps; corpus is training-time
only, authorized), and catalog-grounded (classical back-off n-gram
LM; Jelinek-Mercer / Kneser-Ney lineage).

## The honest anti-cheat crux (why this is a real test, not a reframe-to-win)

An n-gram LM is the classic regurgitation risk — which is **exactly
what the hardened gate's load-bearing bars catch**:
- `verbatim_copy_fraction(gen, train, n=8) <= 0.20`: an n-gram that
  "generates" by replaying long training spans FAILS here. (A trigram
  sampler conditions on only 2 prior tokens, so 8-token spans are
  stochastic recombinations — but TinyStories is formulaic, so this
  bar is genuinely load-bearing and may well fail; that is the point.)
- BPE-invariant **word-shuffle control**: an n-gram trained on
  order-destroyed text loses trigram structure -> the real n-gram
  must beat that control's held-out ppl by >= 20%.
- **absolute-competence floor** (the Generator-S-lesson bar): held-out
  ppl < uniform-random.
- `distinct_ngram_ratio >= 0.5`: degenerate/looping generation FAILS.
- **>= 3 seeds** (sampling RNG seed varied; the n-gram counts are
  deterministic, so the seed varies the GENERATION sampling and the
  word-shuffle control — a real generative result must hold across
  sampling seeds, not be a lucky sample).

So Generator-E is a sharp, honest, pre-registered adjudication: a
competent self-contained statistical LM either clears the SAME bars
that 9 neural attempts failed (a legitimate POSITIVE under the
project's own anti-cheat standard, strictly within its honest
ceiling) OR it fails the anti-regurgitation/structure bars (an honest
negative precisely characterizing even the strongest self-contained
option's ceiling). No bar is added, removed, or tuned; the HARDENED
`subword_lm_gate_core` is reused byte-UNMODIFIED.

## Honest ceiling (no overclaiming — stated up front, never spun)

A back-off n-gram LM at ppl ~15 on TinyStories produces **locally
coherent** simple text. This is **n-gram-class fluency, explicitly
NOT LLM-class**, NOT globally coherent, NOT reasoning, NOT
conversational in the SOTA-LLM sense. If it clears the gate it is
reported as exactly that: "a self-contained, local, no-cheat
n-gram-class generative LM clears the same rigorous anti-cheat gate
that 9 neural attempts failed" — never as "an LLM", never as more
than the bars certify. The validated grounded continual memory +
no-confabulation abstention remains the separate, primary deliverable;
Generator-E, if positive, is a distinct, honestly-scoped generative
capability on top of it, not a replacement for the honesty about
LLM-class generation being terminally negative on this substrate.

## Architecture (net-new is tiny; everything load-bearing reused UNMODIFIED)

Reuse UNMODIFIED (DRY): `sim.ngram_teacher.NgramTeacher` (the
grounded competent model — `.train`, `.soft_dist`; it IS the
generative model now, not a teacher), `research.runners.corpus_fetch`,
`sim.bpe_tokenizer`, the HARDENED `subword_lm_gate_core`
(0.20/1.5/0.5/0.20 + abs-competence floor 1.0, >=3 seeds —
byte-UNMODIFIED; NO new bar), the `subword_lm_gate.py` orchestration
shape (`_word_shuffle`, per-seed kill-safe `.resume.json`, ASCII
verdict block, honest-propagation-is-controller's-job).

Net-new (small, pure-testable):
1. `sim/ngram_generate.py` — pure: `ngram_sample_next(teacher, ctx,
   rng, temperature)` (sample from `teacher.soft_dist(ctx)`) and
   `ngram_generate(teacher, prompt_ids, n_tokens, rng, temperature)`
   (autoregressive: slide the trigram context). Pure numpy/stdlib,
   CPU-unit-testable (valid index, seed-reproducible, deterministic
   at temp 0, back-off-safe).
2. `sim/ngram_ppl.py` — pure: `ngram_heldout_nll(teacher, ids)` ->
   per-token nll list via `-log teacher.soft_dist(ctx)[true]` (the
   exact grounded-probe formula). CPU-unit-testable.
3. `research/runners/generator_e_gate.py` — thin runner mirroring the
   `subword_lm_gate.py` orchestration: per seed, fetch+split (cached
   TinyStories), BPE, train the n-gram on the train split (the REAL
   model) AND on the word-shuffled train split (the control),
   compute held-out ppl (real, control, train) via `ngram_heldout_nll`
   + `perplexity`, generate from held-out prompts via `ngram_generate`,
   compute `distinct_ngram_ratio` + `verbatim_copy_fraction`, and
   `gs_verdict(..., uniform_ppl=V)` (HARDENED gate_core; fail-closed
   without it) + `gs_aggregate_multiseed`. >=3 seeds enforced
   (<3 -> exit 2). Kill-safe (trivial — n-gram train+gen is
   CPU-seconds). Honest-propagation is the CONTROLLER's post-run job.

## Data flow

cached TinyStories -> split -> BPE -> n-gram on train (real) +
n-gram on word-shuffled train (control) -> held-out ppl (real /
control / train) -> generate from held-out prompts -> distinct +
verbatim-copy -> HARDENED `gs_verdict(uniform_ppl=V)` ->
`gs_aggregate_multiseed` -> JSON. No GPU needed; CPU-seconds; trivially
kill-safe.

## Pre-registered gate (the SAME hardened gate_core; bars FROZEN, never tuned)

PASS iff multi-seed >=3, every seed: held-out ppl beats uniform-random
(abs-competence floor) AND beats the word-shuffle control by >=20% AND
held-out <= 1.5x train AND distinct-trigram >= 0.5 AND verbatim-copy
<= 0.20. MANDATORY post-run anti-cheat smell-test (scrutinize a
nominal PASS HARDER than a FAIL: confirm the held-out ppl is genuinely
competent AND the verbatim-copy bar is genuinely cleared — an n-gram's
chief failure mode is regurgitation; verify from recorded JSON, no
re-run, no bar-tuning). PASS => the project has a self-contained,
local, no-cheat n-gram-class generative LM that clears the SAME
rigorous gate 9 neural attempts failed (honest, ceiling-bounded
POSITIVE) -> Generator-F: integrate this generator with the validated
grounded-memory + no-confabulation arch (grounded n-gram-class
conversation), and/or scale the n-gram (higher order + Kneser-Ney).
FAIL => honest negative precisely characterizing even the strongest
self-contained option's ceiling (likely the verbatim-copy or
word-shuffle-control bar) -> propagate + the converged conclusion
that NO self-contained no-cheat generator clears this bar at feasible
local scale is itself the terminal, decision-relevant finding; the
validated grounded-memory asset stands. Either outcome is
decision-relevant and terminal-or-progressing; NOT config-cranked.

## LOAD-BEARING no-harm

Generator-E is PURELY ADDITIVE new files. `subword_lm_gate_core.py`
(frozen bars), `song_g1_core.py`, `sim/bridge.py`, `g20_*`,
`sim/ngram_teacher.py` (reused UNMODIFIED — it is the model), and all
validated runners are byte-UNTOUCHED across the whole Generator-E
range; the full existing suite stays green. NO new bar; the hardened
gate_core decides.

## Scientific basis (catalog)

Classical back-off n-gram language modelling (Jelinek-Mercer
interpolation; Kneser-Ney smoothing — Chen & Goodman 1998); Shannon's
n-gram language model; the project's own grounded NgramTeacher (probe
ba1jyepwf, held-out ppl ~14-15). The hardened gate's anti-cheat
discipline (held-out generalization, permuted/shuffled control,
verbatim-copy bound, absolute-competence floor, multi-seed) is the
adjudicator.

## Out of scope (YAGNI)

No external dependency at RUNTIME ever. No new bar; HARDENED gate_core
byte-UNMODIFIED. No config-cranking any terminated neural mechanism.
No higher-order/Kneser-Ney scaling in THIS slice (the cheap trigram
that is already grounded competent is the decisive test; scaling is a
PASS-branch increment). The pre-registered hardened gate decides;
FAIL is terminal-decision-relevant, PASS proceeds to Generator-F.
