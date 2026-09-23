---
type: finding
status: partial
claim_check: measured
date: 2026-09-22
lane: language (own-voice mouth / affect grounding) — roadmap §8 near-term-speech
seeds: [42, 43, 44, 100, 101, 102]
mechanism: PHASE 1 (complete) measures the RAW, unbiased `research.runners._wkv_fewspike_read_derisk.LinAttnReadout`
  forward pass (no affect coupling of either kind) to diagnose the NO-GO's negative-asymmetry. PHASE 2A (built,
  launched, execution incomplete — see status) swaps the NO-GO's host `_apply_affect_bias` decode-time logit bias
  for the already-shipped, default-OFF, brain-based `BRAIN_WKV_MOUTH_AFFECT_NEURAL` coupling
  (`_affect_pool_gains` + `FewSpikeWordRead.set_mood`, a real `sim.neuromodulators.NeuromodulatorManager`
  concentration consumed by the genuine spiking Izhikevich population read), re-running the NO-GO's own 6-seed
  directional independent-lexicon gate UNCHANGED via a thin wrapper runner (env-var swap only, no `webapp/` edit,
  no `sim/` edit).
verdict: PHASE 1 = DECODE-CEILING (measured, all 6 seeds, complete). PHASE 2A = UNDEFINED — NOT a NO-GO, NOT a GO;
  the 36-arm gate did not finish executing in this session (see "Honest status: Phase 2A execution incomplete").
  Reporting a Phase 2A verdict here would fabricate a result the run never produced.
artifact: research/findings/raw/_affect_tone_decode_ceiling/diagnose_verdict.json (Phase 1, complete, cited below);
  research/findings/raw/_affect_tone_neural_coupling/ (Phase 2A, in-progress — no verdict artifact yet)
runner: research/runners/_lbf_affect_tone_decode_ceiling_diagnose.py (Phase 1, complete);
  research/runners/_lbf_affect_tone_neural_coupling_derisk.py (Phase 2A, built + selftested + launched)
supersedes-scope-of: none — this rung follows, does not retract,
  research/findings/2026-09-22-affect-tone-open-output-directional-6seed-positive-asymmetric-NOGO.md
---

# Affect→tone next method: PHASE 1 decode-ceiling diagnosis (complete) + PHASE 2A neural-coupling re-verification (built, launched, execution incomplete)

## Context: the banked method and the question this rung had to answer FIRST

`research/findings/2026-09-22-affect-tone-open-output-directional-6seed-positive-asymmetric-NOGO.md` measured
that the live affect→tone coupling over the linattn WKV mouth's freely-generated reply
(`webapp/wkv_mouth_generator.py::_apply_affect_bias` — a HOST additive, saturating, margin-to-top1-concentrated
logit bias) is real and directional for POSITIVE mood on all 6 seeds, but never produces a correct-sign NEGATIVE
tonal shift on the independent, WARRINER-disjoint lexicon ruler, on any seed. That finding characterized the
likely cause in prose ("the ceiling is the DECODE ROUTE... positive evaluative words sit near the margin while
negative words do not") without measuring it directly, and named two banked next methods: (1) the already-built
brain-based `BRAIN_WKV_MOUTH_AFFECT_NEURAL` neuromodulator coupling, and (2) a decode mechanism that shifts the
DISTRIBUTION rather than concentrating an additive bias on one candidate.

HARD RULE 2 (bank the failing method, take a new one, never tune the old one to flip) makes answering one
question mandatory before building anything: is the negative-asymmetry a DECODE-CEILING (negative-valence
vocabulary carries real-but-sub-margin probability mass, so a different decode mechanism could plausibly surface
it) or a TRAINING-DATA LIMIT (the small from-scratch linattn mouth's wiki-descriptive corpus never taught it
negative-affect vocabulary here, so no decode trick — of any shape — could move a probability that is not
there)? `tools/before_you_build.sh` was run against this exact defect before any lever (recorded to
`research/queue/.corpus_checks.jsonl`, gitignored/local); it surfaced the NO-GO itself plus the two prior
affect-coupling findings already read into this rung, and no prior measurement of this specific decode-ceiling
question.

## PHASE 1 — decode-ceiling vs data-limit diagnosis (COMPLETE, 6/6 seeds)

**Runner:** `research/runners/_lbf_affect_tone_decode_ceiling_diagnose.py`. **Artifact:**
`research/findings/raw/_affect_tone_decode_ceiling/diagnose_verdict.json`.

**Method.** For each of the 6 seeds and the SAME 10 free-talk TONE prompts the NO-GO's own harness used
(`_lbf_affect_tone_open_output_derisk.TONE_PROMPTS`, imported not retyped, so this diagnosis characterizes the
identical operating point), a GREEDY (argmax, deterministic, no RNG draw) 110-token continuation is decoded
through the real `LinAttnReadout` forward pass with the same production repetition controls
(`repetition_penalty=1.3`, `no_repeat_ngram_size=3`, matching the live `webapp/open_ended_chat.py` call) but
WITHOUT any affect bias or fact boost — the natural next-token distribution the affect coupling would have to
act on. At every step the FULL-VOCAB natural softmax `pfull=softmax(lg)` (no top-K cut, no sampling) is read and
the probability mass on every in-vocab token matching a positive/negative word in the NO-GO's own independent,
WARRINER-disjoint lexicon (`_lbf_affect_tone_open_output_derisk.load_indep_lexicon`) is summed, together with
whether the closest such token (by raw logit) falls inside the production `topk=64` candidate window and its
margin-to-top1. This measures the model, not the mouth's decode pipeline — no brain build, no webapp server, no
GPU, `numpy`/`CUDA_VISIBLE_DEVICES=''`. Full run: 6 seeds x 10 prompts x 110 steps = 6600 measured steps, 10.2s
wall time total.

**Preregistered decision rule** (fixed before running, in the runner's own module docstring, not tuned after
seeing results): DATA-LIMIT iff `mean(pos_mass)/max(mean(neg_mass),1e-12) >= 50` AND
`frac_steps_neg_in_top64 < 0.10`; DECODE-CEILING otherwise. 50x and 10% are round, conservative numbers chosen to
separate "structurally near-absent" from "present but sub-margin," not fit to this run's own numbers — confirmed
by the runner's own selftest, which checks the rule fires DATA-LIMIT on a synthetic 100x-asymmetric/rare-topk
case and DECODE-CEILING on a synthetic comparable-mass/frequent-topk case (`_lbf_affect_tone_decode_ceiling_diagnose.py --selftest`, PASS).

**Result — DECODE-CEILING on all 6 seeds.**

<!--derived-->
Aggregate over 6600 steps: mean_pos_mass=0.003218, mean_neg_mass=0.002209, ratio=1.457 — nowhere near the 50x
DATA-LIMIT threshold. frac_steps_pos_in_top64=0.155, frac_steps_neg_in_top64=0.079 — negative-valence words reach
the real production top-64 candidate window on ~8% of steps (positive: ~16%), rarer but far from structurally
absent. mean_pos_margin_to_top1=6.139, mean_neg_margin_to_top1=6.641 — comparable orders of magnitude (not
"positive sits at the margin, negative nowhere close").

| seed | mean_pos_mass | mean_neg_mass | frac_pos_top64 | frac_neg_top64 |
|------|---------------|---------------|-----------------|-----------------|
| 42   | 0.0028461 | 0.0019182 | 0.1536 | 0.0727 |
| 43   | 0.0050124 | 0.0020023 | 0.1627 | 0.0527 |
| 44   | 0.0025440 | 0.0021589 | 0.1300 | 0.1127 |
| 100  | 0.0040383 | 0.0025876 | 0.2273 | 0.0791 |
| 101  | 0.0025849 | 0.0024892 | 0.1482 | 0.0682 |
| 102  | 0.0022840 | 0.0021005 | 0.1100 | 0.0891 |

n_pos_vocab_ids=35, n_neg_vocab_ids=28 on EVERY seed (one shared `bridges/wkv_ckpt/wkv_bpe8k.json` vocabulary
across all 6 per-seed checkpoints). Per-seed pos/neg ratios range 1.05x-2.5x — every single seed reads
DECODE-CEILING individually, not just in aggregate.

**Reading.** Negative-valence vocabulary is not structurally absent from the linattn mouth's own learned
distribution on these prompts — it carries real, comparable-order-of-magnitude probability mass and regularly
(not never) reaches the candidate window the real decode loop samples from. The NO-GO's asymmetry is therefore a
property of the DECODE MECHANISM — a single-candidate, margin-to-top1-concentrated additive bias that only ever
assists the ONE congruent word closest to the current top-1, which on this wiki-descriptive corpus is reliably
positive — not a property of the training corpus. Per HARD RULE 2, this licenses PHASE 2A: build/re-verify a
DIFFERENT decode mechanism, not tune the banked one.

## PHASE 2A — re-verify the already-built neural coupling against the NO-GO's own ruler

**Runner:** `research/runners/_lbf_affect_tone_neural_coupling_derisk.py` (NEW file, ~140 lines). **Design:** a
thin wrapper that imports `research.runners._lbf_affect_tone_open_output_derisk` BY REFERENCE (not by copy) and
changes exactly one thing — it rebinds that module's own `_ENV_BASE` dict to add
`BRAIN_WKV_MOUTH_AFFECT_NEURAL=1` before calling the base module's own `run_controller`/`score_and_gate`/
`run_worker`/`load_indep_lexicon` UNCHANGED. Every anti-cheat the NO-GO's gate already has — the preregistered
delta from the lesion arm's own tone-noise band, the attribution control (mood-decoupled valence injection), the
content-identity + moat check, the fluency ceiling, the determinism check (lesion==lesion_rep), the
lexicon-disjointness enforcement — is the SAME code path, not a reimplementation, because this wrapper never
duplicates that logic. This makes the comparison a genuine single-variable A/B: DECODE ROUTE (host additive bias
vs neural neuromodulator concentration) is the only thing that changes; harness, prompts, lexicon, scorer, delta
rule, and every other anti-cheat are held fixed by construction.

**Why the neural coupling is the right next method, not a re-tune of the banked one.** The neural coupling
(`_affect_pool_gains` + `FewSpikeWordRead.set_mood`, shipped 2026-09-04, `research/findings/2026-09-04-affect-coupling-neural-not-host-PARTIAL.md`)
uses the SAME Warriner-gated congruence map as the host mechanism (an A/B on "which substrate carries the
effect" must not also silently vary "what counts as mood-congruent") but changes the DESTINATION: instead of a
single concentrated additive bias on the closest-margin candidate, EVERY mood-congruent candidate already inside
the top-64 window gets its own neuromodulator-driven excitability nudge, and the actual winner is decided by a
genuine stochastic Izhikevich population competition (OU noise + accumulated firing over `read_window` steps),
not a deterministic nearest-margin comparison. This is exactly the CLAUDE.md wall-reframe applied concretely:
the host mechanism substituted a deterministic single-candidate comparison for what biology runs as a
competitive population process; the neural coupling already restores that competition — it was simply never
re-verified against the directional, independent-lexicon ruler this NO-GO introduced (its own 2026-09-04
load-bearing check used a looser "any divergence from lesion" criterion on the easier SSM/TinyStories checkpoint
family, and even there showed a milder version of the SAME asymmetry: negative-direction divergence 11/18 vs
positive 18/18 — an honest prior signal that this next method is plausible, not guaranteed).

**BRAIN-BASED-ONLY BOUNDARY respected, not merely asserted.** No host sentiment lexicon selects or injects a
word into the reply under this coupling — `_affect_bias_ids`'s Warriner lookup only tags candidate POOLS with an
excitability value (the same "labelled-line pool assignment is a legitimate host input" category
`drive_from_weights` already relies on project-wide); which pool actually fires is decided by the substrate's own
noisy spiking competition, never by host arithmetic on the output.

**Selftest:** `_lbf_affect_tone_neural_coupling_derisk.py --selftest` PASSES — verifies the env rebind took
effect, every OTHER `_ENV_BASE` key is untouched from the NO-GO's own defaults, and the shared scorer/gate
selftest (imported, not reimplemented) still passes unchanged.

## Honest status: PHASE 2A execution incomplete (infrastructure constraint, not a capability finding)

The 36-arm gate (6 seeds x {pos, neg, lesion, lesion_rep, ctrl_pos, ctrl_neg}) was launched three times in this
session and did NOT complete. This is reported here in full rather than silently retried past session end,
because the cause is diagnostic and load-bearing for anyone resuming this run:

1. **First launch** (`--parallel 2`, memcap 8G each): both worker processes grew to ~7.6GB RSS. Combined with
   THREE unrelated, concurrent `onebrain_regression_battery` worker processes from another session on this
   shared machine (each ~7.7GB, ~23GB combined, apparently not memcap-wrapped), system-wide available memory
   fell to 439MB and then 669MB twice, with swap approaching full (45Gi/46Gi used at the worst point) — a
   genuine near-OOM crisis on the shared box, not a crash of this runner. Each time, this session killed its own
   worker(s) to relieve pressure (verified: available memory recovered to double digits of GB immediately after
   each kill) rather than let a global OOM-kill pick an arbitrary victim.
2. **Root cause of the SLOWNESS** (distinct from the OOM risk): the per-arm wall time is dominated by
   `webapp/server.py::brain_reply`'s unconditional `_get_warm_qwen_renderer()` warm-up (a `SpikingQwenFaculty`
   load of a 290-shard weight checkpoint), which is UNRELATED to the WKV-mouth/affect-coupling mechanism under
   test but is on the same production call path the NO-GO's own harness already exercises (`S.brain_chat` →
   `brain_reply`). Under the same contention above, this one-time (per-process) load was directly observed
   taking >6m45s to reach only 205/290 shards (tqdm's own `6.75s/it` at that point, vs an initial fast burst of
   `23.66it/s` — a >150x slowdown attributable to disk/page-cache contention, not CPU). The banked NO-GO's own
   254.5s-per-arm timing (`research/findings/raw/_affect_tone_open_output/arm_s42_pos.json`, `wall_seconds`)
   almost certainly benefited from a warm page cache from prior activity in that session; this session's cold,
   repeated restarts never got that benefit and re-paid the full cost each time, compounded by the shared
   machine's concurrent load.
3. **This is an infrastructure/scheduling constraint of the shared machine at this moment, not a mechanism
   defect.** Nothing observed suggests the neural coupling itself is slow or broken — the process was
   consistently reported "R"/active, consuming real CPU, and the ONE thing directly measured as abnormally slow
   (Qwen-renderer weight loading) sits entirely outside the coupling being tested.

**What is real and complete regardless:** the code is built, selftested, and its wiring was traced end-to-end
(the env var correctly reaches `webapp.wkv_mouth_generator.wkv_mouth_affect_neural_enabled()` inside each fresh
worker subprocess, verified by static trace through `_spawn`'s `env=` construction and `run_worker`'s own
`os.environ` update loop, which never touches the neural-coupling key). PHASE 1's diagnosis, which was the
higher-leverage, cheap, decisive step per HARD RULE 2, is complete and robust.

**Resume path** (unattended, whenever this shared machine has headroom — check `free -h` shows several GB
available AND no concurrent `onebrain_regression_battery`/similar heavy worker before launching):

```bash
CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy bash tools/memcap.sh 8 -- \
    .venv/bin/python -m research.runners._lbf_affect_tone_neural_coupling_derisk --controller --parallel 1 --memcap-gb 8
CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy .venv/bin/python -m \
    research.runners._lbf_affect_tone_neural_coupling_derisk --score-only
```

`run_controller`'s own `resume=True` default means a re-launch picks up from whatever arms already have valid
output — no wasted recompute. `--parallel 1` (not 2) is the lesson this session's own near-OOM incidents
support: even ONE arm can reach ~8GB RSS, and this shared machine's OTHER concurrent load is not observable or
controllable in advance, so serial execution is the safer default until a lower-contention window is confirmed.

## What is NOT claimed here

No GO, no NO-GO, no "surpass," no "closure" for the neural-coupling method — none of those words apply, because
the gate never finished. `docs/TERMS.md`'s condition for "GO" ("the gate's own verdict is positive") and the
symmetric condition for NO-GO both require a verdict the gate actually reached; asserting either from an
incomplete run would be the exact overclaim class `tools/gates/verdict_preconditions.py` exists to block. The
next action on this branch is: resume the launch above when the shared machine has headroom, then
`score_and_gate` (`_lbf_affect_tone_neural_coupling_derisk.py`) will write the verdict artifact — named
`affect_tone_neural_coupling_verdict` (json) inside the `_affect_tone_neural_coupling` raw-findings output
directory named above — and a follow-up finding can report the real GO/NO-GO with per-seed numbers, exactly as
the banked NO-GO did for the host mechanism.

## Reproduce

```bash
# Phase 1 (fast, ~10s, safe to re-run any time)
CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy .venv/bin/python -m research.runners._lbf_affect_tone_decode_ceiling_diagnose \
    --out research/findings/raw/_affect_tone_decode_ceiling/diagnose_verdict.json
CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy .venv/bin/python -m research.runners._lbf_affect_tone_decode_ceiling_diagnose --selftest

# Phase 2A (multi-hour on a contended shared machine; see "Resume path" above for the safe invocation)
CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy .venv/bin/python -m research.runners._lbf_affect_tone_neural_coupling_derisk --selftest
```

Worktree note (honest, for reproducibility): this session's git worktree lacked
`bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{43,44,100,101,102}.npz` and `data/corpus/tinystories.txt`
(both untracked/gitignored — `data/` is excluded via `.git/info/exclude`, and the non-seed42 linattn checkpoints
are untracked large binaries). They were copied byte-for-byte (sha256-verified for the checkpoints) from the
primary checkout into this worktree before running; nothing tracked in git changed. A fresh worktree checkout
will need the same copy step before either phase's 6-seed run can execute.
