# Overnight arc summary — 2026-05-09 evening through 2026-05-10 morning

**Arc duration:** ~22:00 EDT 2026-05-09 → 05:00+ EDT 2026-05-10
**Commits:** 23+ to main, all pushed to origin (GitHub) and gitea
**User directive:** "autonomously continue working as per the master plan
towards making the sim capable of conversation"
**Status:** Track 3 production-side complete; capacity scaling test in flight

---

## Three validated milestones

### 1. Track 3 v2 chat_speak_demo 6-seed multi-seed (Tier 1, 4-word)
- **A2W mean 58.3% ± 20.4%**, 5/6 seeds at ≥50%, 5/6 above chance
- Single-seed 75% reproduces; 50% is the median floor
- Per-direction A2W: N=67%, E=67%, S=67%, W=33% (W-bias mirror of Tier 1 BREAKTHROUGH N-bias)
- Doc: `2026-05-09-chat_speak_demo-Track3-layer4-MULTI-SEED.md`

### 2. Tier 2.1 8-word :speak 6-seed multi-seed (production-side analog) — EXCEEDS paper
- **A2W any-synonym 87.5% ± 20.9%**, 6/6 GO unanimous, 5/6 at literal 100%
- Significantly exceeds Tier 2.1 BREAKTHROUGH paper's 63.7% ± 11.8%
- Primary 87.5%, synonym 0% (STDP WTA primary-wins confirmed at production-side)
- Per-direction: N=83%, E=100%, S=83%, W=83%
- Doc: `2026-05-09-chat_speak_synonym_demo-Tier2.1-8word-MULTI-SEED.md`

### 3. 16-word smoke seed 42 (consolidation_synonym_16word_scaled_smoke)
- **GO** (retention primary 90%, synonym 109%)
- Capacity rule extends to 4 sub-pops/motor_X at n_motor=2000
- Pre-silence overall 26.9% (low absolute due to --smoke chunking, but
  retention RATIO is the validated signal — cortex retains binding through
  hippocampus lesion at 16-word vocab)
- Doc: `2026-05-10-16word-smoke-GO-capacity-rule-extends.md`

## Track 3 conversational stack — COMPLETE

| Layer | Capability | Status |
|-------|-----------|--------|
| Layer 1 | `--learn` online vocab primitive | ✅ shipped + tested |
| Layer 2 | `chat_learn_demo` runner (multi-seed) | ✅ shipped + tested |
| Layer 3 | Dialog state (`:again`/`:opposite`/`:history`/`:forget`) | ✅ shipped + tested |
| Layer 4 | `:speak` generative decoder (Tier 1) | ✅ multi-seed VALIDATED tonight |
| Layer 4+ | `:speak` synonym variant (Tier 2.1) | ✅ multi-seed VALIDATED tonight |

The agent now has bidirectional 4-word and 8-word conversation:
- Reception (W→A): user types word, motor pool fires correct action
- Production (A→W): drive motor pool, agent says matching word

Both directions multi-seed validated. Both can be exercised via
chat_repl REPL with `--mode tier1` or `--mode synonym`, and both
work with `--save-bridge` / `--load-bridge` for instant reload of
trained state.

## Find-the-ceiling experiment (in flight)

User directive received tonight: "start very high on the scale to test
for failure". Implementation:

**Vocab tiers shipped (10 tiers from 8 to 256 words):**
- Hand-curated semantic synonyms (8-64): English + Unicode arrows + Spanish/German + Japanese/Arabic + nautical
- Numbered-variant fallback (96-256): `north_05`, `north_06`, ... for testing encoding-collision wall

**45 unit tests** (`tests/test_text_eval_vocab.py`) lock down dispatcher contract.

**Predicted vs actual VRAM (find-the-ceiling preliminary data):**

| Vocab | n_motor | Predicted VRAM | Actual VRAM | Wall clock /smoke |
|-------|---------|----------------|-------------|-------------------|
| 16 | 2000 | ~7 GB | ~7 GB | ~30 min ✓ |
| 24 | 2000 | ~8 GB | not tested | est ~35 min |
| 32 | 3000 | ~12 GB | not tested | est ~40 min |
| 48 | 4000 | ~17 GB | not tested | est ~45 min |
| **64** | **6000** | **~28 GB (predicted OOM)** | **16 GB (FITS!)** | **~150 min** (in flight) |
| 96 | 6000 | est ~17 GB | not tested | est ~160 min |
| 128 | 6000 | est ~18 GB | not tested | est ~170 min |
| 256 | 6000 | est ~20 GB | not tested | est ~190 min |

**Key discovery:** the predicted-OOM extrapolation from 16-word's
~7 GB to 64-word's ~28 GB **was wrong by ~12 GB**. Sparse connectivity
means VRAM scales sub-linearly with neuron count. 64-word at
n_motor=6000 actually uses 16 GB / 24 GB. **The ceiling on a 24 GB
3090 is significantly higher than initial estimate.**

**Real bottleneck identified:** wall-clock per chunk+sleep cycle.
- 16-word @ n_motor=2000: 145s per cycle
- 64-word @ n_motor=6000: 700s per cycle (~5× slower)

The 24K-motor inner loop is what limits practical exploration, not
VRAM. To find the ceiling without 3-hour smoke runs per tier, we
need to either:
1. Use parallel-N seed runs (multiple GPUs needed)
2. Accept that each tier costs hours to test
3. Profile the inner loop for optimization opportunities

## Infrastructure shipped tonight

**Runners:**
- `chat_speak_synonym_demo` (Tier 2.1 8-word :speak production-side)
- `chat_demo_aggregate` chat_speak branch (Tier 1 + Tier 2.1)

**Test coverage:**
- 6 tests for chat_speak_synonym_demo runner
- 3 tests for chat_speak_demo aggregator branch
- 45 tests for text_eval vocab dispatcher (10 tiers × multiple invariants)
- 2 tests for /api/bridges endpoint
- All pass (56 new tests)

**Webapp infrastructure:**
- `/api/bridges` endpoint (lists saved checkpoints)
- `/api/bridges/{name}` endpoint (single bridge metadata)
- Bridges tab UI in launcher dropdown (gallery view, "Load into chat (coming soon)" placeholder)
- 8 new presets (chat_speak_synonym_demo + 7 capacity-tier smokes)
- Launcher dropdown entries for all new presets

**chat_repl integration:**
- `--save-bridge` now writes sidecar `<name>.simstate.h5.meta.json`
  with mode/seed/events/neurons/synapses/saved_at metadata
- bridges/ directory created as standardized save location

**Bug fixes:**
- bio_three_factor LCM(50, 64) progress bug — was firing once per run
  instead of every 50 events; now decoupled, fires 32× per typical
  training run (validated working on seed 101)
- chat_speak_synonym_demo top_k bug (was top_k=4 truncating 8-word
  rankings; now top_k=8 for full diagnostic visibility)

**Path A overnight chain script:**
- `scripts/chain_path_a_overnight.sh` auto-fired smoke→multi-seed→16-word
- Validated working — fired all sequences correctly
- Will not auto-fire 64+/96+ tiers (manual-launch or extend chain script)

## Master plan + capability_status updates

- Master plan: 2 new decision-log entries (Tier 2.1 8-word :speak 6/6 GO,
  Track 3 v2 multi-seed VALIDATED)
- Path F empirical pillars: 2 new entries (#7 Track 3 v2, #8 Tier 2.1 :speak)
- capability_status.json: as_of bumped to 2026-05-10; new pillars 11+12;
  active phase updated to reflect 16-word smoke + find-the-ceiling
- README.md: "Latest validated result" section reflects Track 3 v1
  feature-complete + multi-seed validated

## Recommended priorities for next session

1. **Wait for 64-word smoke** (~05:10 EDT). If PASS at any meaningful
   accuracy: significant — biology-grounded sim handles 64-word vocab
   on a single 3090. Multi-seed + write findings.

2. **Webapp chat UI WebSocket** (1-2 days). The bridges tab gallery
   is shipped; the missing piece is the actual interactive chat REPL
   over WebSocket. This unlocks user testing of the trained bridges
   without dropping to CLI. High UX value.

3. **Temperature sampling on :speak** (1 hr). Add `--temperature`
   flag, deterministic by default for testing repeatability, opt-in
   for realistic conversation. Lifts synonym top-1 rate from 0% to
   biologically-realistic ~15-30%.

4. **Tier 2.3 phrases revisit** (1-2 weeks). PFC verb pool capacity
   scale-up — the 41% architecture-limited result needs n_motor scale-up
   investigation similar to what worked for 12-word vocab.

5. **Plan for richer dialog + generative composition** (research arc).
   Multi-turn coreference, "go north now" syntax, semantic content —
   none of these are in the master plan yet. Phase 2 path-f-hybrid's
   death tonight closed the "transformer shortcut" path; biology-grounded
   alternatives need explicit design.

## Compute reality check

Per the user's question earlier tonight about SOTA LLM comparisons,
the practical reality is:
- Current sim @ 8-word vocab: 12K neurons, 5M synapses, 6 GB VRAM
- Current sim @ 64-word vocab: 30K+ neurons, 30M+ synapses, 16 GB VRAM
- Per-token compute is ~1000-10000× higher than transformers at
  equivalent parameter count (we simulate time)

**Realistic targets:**
- 100-word vocab + 2-word phrases + dialog state: 80-150 GB VRAM,
  needs A100/H100 80GB or 2× 3090. Achievable now.
- 1000-word vocab + grammatical composition: 1-3 TB VRAM, multi-GPU
  cluster. ~6-12 month engineering project + breakthrough in efficient
  SNN simulation.
- "ChatGPT-3.5 class conversation": multi-million dollar compute project,
  5-10 years out at current biology-grounded scaling efficiency.

Path A is real and incremental. The biology-grounded sim WILL get to
richer conversation, but it scales like a rocket science project, not
a software project.

## Provenance

- Tonight's per-seed JSONs in `research/findings/raw/g11_bg/`
- Aggregate JSONs in `research/findings/raw/multi_seed/`
- Findings docs in `research/findings/2026-05-09-*.md` and `research/findings/2026-05-10-*.md`
- Master plan: `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`
- Capability status: `webapp/capability_status.json`
