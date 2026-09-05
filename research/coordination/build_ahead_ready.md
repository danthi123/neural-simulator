# BUILD-AHEAD READY — de-risks staged to queue the moment compute frees

Prepared 2026-09-04d by build-ahead workflow `wbs2wsm1y` (6 agents, all smoke-tested, runners **on main**). Owner
ask: "stage upcoming work so compute never idles." Each entry is READY to run — fire it when its lane frees. This
file is the durable store (the workflow's own output is transient scratch); `tools/backlog.py` surfaces the anchor,
this is the concrete queue commands behind it. Fire order below is by readiness, not priority.

## READY NOW (no upstream dependency — fire when the lane frees)

**1. affect→neural default-on validation** — lane A·Affect (pool). Validates promoting `BRAIN_WKV_MOUTH_AFFECT_NEURAL`
to default-on + tests a `neg_pa_scale` fix for the negative-mood undershoot (finding `87631edf` PARTIAL). GO =
≥5/6 seeds BOTH directions (strict) + moat + byte-identical-off. Runner `_wkv_mouth_affect_neural_promote_validate.py`
(runner-side only, no sim/webapp edit). Fan-out = 12 directional + 6 characterize + 1 moat + 6 byte-identical, then
aggregate `--phase summary` + `--phase go`. Full command block: see the workflow output / the runner's docstring.

**2. vision-crossing anti-cheats (held-out-position + scramble-null)** — lane D·Perception (pool). Tests whether the
flat-capacity crossing is real or an ELM-overfit (finding `2026-09-03-vision-configural-binding-crossing...`, open Q1).
```
bash tools/pool_queue.sh add 'SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk --ridge 0.5 --conj-bind none --n-s2 1152 --heldout-position --scramble-null --seeds 42 43 44 100 101 102 --out research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_heldoutpos_scramblenull_6seed.json' --checked 'open Q1: does the width-matched ELM crossing survive a genuine held-out-position block split + a scramble-null on the learned readout'
```

## READY AFTER A SMALL FIX / NODE-SYNC

**3. one-brain Stage-2 — Touchpoint-A fact-clause retire** — lane E·Language / one-brain (pool). Retires the Surface-A
open-prose recall Qwen dependency (follow-on to Stage-1). Runner `_touchpoint_a_fact_clause_derisk.py`, GO = 3
structural invariants (content-preserved + scope-untouched + flag-off-inert). **2026-09-04 UPDATE: the prose fix (b)
below landed (`webapp/wkv_mouth_generator.py::render_fact_sentence` now punctuates; `slug_to_np` no longer doubles a
leading determiner; verified content-preserving via a git-stashed baseline comparison), but the full `--n-known 4`
battery still returns `structural_checks_passed=False` for TWO SEPARATE, PRE-EXISTING reasons unrelated to the
prose — an RNG-isolation gap in `RichAnswerComposer._render_one_verified`'s Touchpoint-A call site (fails
`scope_untouched`) and a fact-count/gather divergence from Touchpoint-A rescuing facts the old renderer couldn't
verify-render (fails `content_preserved`); both confirmed pre-existing (byte-identical on a clean baseline) and
logged in `research/FAILURE_LOG.md` (2026-09-04, two rows) with root cause + candidate fixes. NOT YET promotable —
those two need fixing first.** ⚠️ Two gates before the full n=4 run: (a) sync the new runner to pool nodes first
(`pool_queue.sh` SSH-validates `--help` on pool40/41/42); (b) ~~a real fluency regression the smoke exposed —
`render_fact_sentence` clauses carry no trailing punctuation, so `render_paragraph`'s join runs sentences together,
plus a `slug_to_np` determiner-doubling ("the The Republic of Turkey")~~ FIXED 2026-09-04, see above.
```
bash tools/pool_queue.sh add 'CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._touchpoint_a_fact_clause_derisk --n-known 4 --out research/findings/raw/_touchpoint_a_fact_clause_full.json' --checked 'one-brain Stage-2 Touchpoint-A retire n=4 battery (sync runner to pool nodes first; prose fixed 2026-09-04, two OTHER pre-existing structural failures still open, see research/FAILURE_LOG.md)'
```

## CONDITIONAL / DEFERRED-FOCUS (queue only when the trigger fires)

**4. num/den Tier-2 on-bridge** — lane E·Language / fully-spiking (GPU). BLOCKED on Tier-1's `--linattn-div`
full-stack eval-checkpoint-load path (shared dep, branch `research/linattn-shunt-gain-tier1-redo`) + 5 missing linattn
seed checkpoints. Runner `_linattn_shunt_gain_tier2_onbridge_derisk.py` written to plug into that slot once it lands.
Queue only after Tier-1 lands the eval path + the 6 checkpoints exist.
```
bash tools/gpu_queue.sh add 'SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._linattn_shunt_gain_tier2_onbridge_derisk --seeds 42,43,44,100,101,102 --n-settle-steps 40 --n-trials 64 --json research/findings/raw/_linattn_shunt_gain_tier2_onbridge.json'
```

**5. next-fluency `learnkey` (learned-key content-addressable memory)** — lane E·Language (GPU). Owner-sanctioned
DEFERRED wall #3 fallback: queue ONLY if the linattn production-scale fluency sweep plateaus. GO = 6/6 seeds
margin_vs_trigram ≥ the linattn baseline (+0.0505 mean). Runner is `--recurrence learnkey` in `_emerge_wkv_lm_derisk.py`.
```
bash tools/gpu_queue.sh add "SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk --recurrence learnkey --learnkey-slots 64 --n-layers 2 --uniform-decay --d-model 192 --batch 128 --tokenizer bpe --corpus data/corpus/simplewiki.txt --contiguous --max-len 40 --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 --seeds 42 43 44 100 101 102 --tok-cache --save-ssm bridges/wkv_ckpt/wkv_learnkey_depth2_contiguous --json research/findings/raw/_emerge_wkv_lm_learnkey_depth2_contiguous_6seed.json"
```

**6. gap#5 forward-band homeostatic scaling (learn-through-use #107)** — lane Learning (GPU, cheap CPU scan first).
DEFERRED-focus (continuous learning is arc item 5). Runner `_gap5_forward_band_homeostatic_scaling_ltu_derisk.py`.
Run the cheap single-seed `--scan-mult` first to pick `--fwd-scale-mult`, THEN the 6-seed. GO = ≥5/6 (directional +
rescale-target + weak-cue-gain + use-dependent anti-cheat).
```
SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk --scan-mult --seeds 42 --scan-mults 1.0 1.25 1.5 2.0 3.0
bash tools/gpu_queue.sh add 'SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk --fwd-scale-mult 1.5 --seeds 42 43 44 100 101 102 --out research/findings/raw/gap5_ecker_adex/forward_band_homeostatic_scaling_ltu_6seed.json'
```

## Side-discoveries the build-ahead surfaced (separate follow-ups, not in the queue above)
- **Pre-existing `sim/bridge.py` bug:** a brain-region config with zero `region_pathways` raises `UnboundLocalError`
  in the connectivity fallback. Worked around in the Tier-2 runner (declares a zero-weight pathway); a real fix is a
  separate task.
- **Board #104's 6-seed wander verify ALREADY RAN 2026-08-28** (1/6 pass, blend-balance collapse, cause
  uncharacterized) and sat undocumented 9 days — written up by the gap5 agent; the master roadmap wall-ledger sync
  for #104/#107 is still outstanding.
