---
type: finding
status: partial
lane: E-language-open-ended-honesty
date: 2026-08-28
mechanism: open-ended-generation-time-consensus-veto-live-mouth-tokenid-vs-text-divergence
seeds: [42]
seed-waiver: This session's OWN instrument is a bounded, single-topic, single-seed (42) CPU sanity re-run of the
  already-built, unmodified harness (device=cpu, T=2, small token budgets) -- it exists ONLY to confirm the
  harness executes end-to-end against a REAL Qwen (not a fake tokenizer/model) and stays within the ~4GB RSS
  budget, BEFORE spending GPU queue time. It is explicitly NOT the decisive claim and is not treated as one (the
  reduced T=2/CPU/float32/1-sentence config is not representative of the production T=16/cuda/float16 config the
  decisive comparison needs). The DECISIVE 6-seed (42/43/44/100/101/102) live-mouth token_id-vs-text divergence
  comparison BOTH 2026-08-27 and 2026-08-28 parent findings named as their own NEXT is QUEUED
  (`research/queue/gpu.queue`) and NOT run this session -- this finding does not claim its result.
instrument: research/runners/_open_ended_gen_time_consensus_veto_derisk.py's `run_battery`/`main` (unmodified;
  already built by prior commits eeaa6fb7e/72b9f6fe6/990de0d4b, see "Verify-first" below) for the decisive
  6-seed GPU run; a new, throwaway sanity script (not committed -- lived at
  /tmp/claude-*/scratchpad/sanity_real_mouth.py) that imports and calls the SAME unmodified
  `build_consensus_chat` / `generate_with_generation_time_veto` functions directly, scoped to 1 topic / 1 seed /
  small budgets, for this session's own CPU pre-flight check.
runner: research/runners/_open_ended_gen_time_consensus_veto_derisk.py
external: NO NEW external search logged this session -- reuse of the already-logged, in-window, lane-tagged
  source (the `.external_searches.jsonl` log in `research/queue/`, entry timestamped 2026-08-28T13:47:13Z: "SentenceKV:
  Efficient LLM Inference via Sentence-Level Semantic KV Caching" arXiv:2504.00970 <!--derived-->, lane
  `e-language-open-ended-honesty`). This finding proposes NO new mechanism -- it RUNS an already-designed,
  already-wired, already-staged measurement the 2026-08-28 token-id-continuation finding named as its own NEXT,
  so no fresh mechanism-lever grounding is owed.
artifacts:
  - research/findings/raw/_open_ended_gen_time_live_mouth_tokenid_vs_text_6seed.json
---

# Generation-time honesty: the decisive live-mouth token_id-vs-text divergence-rate comparison, staged and queued for the first time (STAGED, not landed)

Artifact: `research/findings/raw/_open_ended_gen_time_live_mouth_tokenid_vs_text_6seed.json` (currently a
placeholder -- `{"status": "queued, not yet run", ...}` -- overwritten by the real 6-seed result once
`research/queue/gpu.queue` runs it; see "The queued decisive run" below).

**One line.** Both `2026-08-27-open-ended-generation-time-honesty-PARTIAL.md` and
`2026-08-28-open-ended-generation-time-honesty-token-id-continuation-removes-retokenization-confound-PARTIAL.md`
named the SAME open rung -- "the decisive live-mouth multi-seed GPU comparison (does `token_id` continuation
raise the ON/LESIONED divergence rate above the text-continuation 1/3 baseline) is STAGED but NOT run" -- and
neither session ran it. This session (1) verified the comparison harness was ALREADY BUILT (not merely staged as
a plan) by prior commits, (2) confirmed it actually executes against a REAL Qwen with a bounded CPU sanity check,
and (3) QUEUES the real 6-seed GPU run for the first time. It does not itself land the decisive result.

## Verify-first (before building anything)

`bash tools/before_you_build.sh "generation-time honesty live-mouth token_id vs text continuation divergence
6-seed"` surfaced only the two parent findings plus
`2026-08-28-open-ended-generation-time-honesty-skip-and-continue-past-a-dropped-sentence-PARTIAL.md` (a THIRD,
independent extension in the same lane -- skip-and-continue past a dropped sentence, stacked behind its own
default-OFF `BRAIN_HONESTY_SKIP_CONTINUE` flag, `skip_continue=False` by default so it does not affect this
comparison). Reading `research/runners/_open_ended_gen_time_consensus_veto_derisk.py` (git log: `eeaa6fb7e`
2026-08-27 built the mechanism + controlled unit battery, `72b9f6fe6` 2026-08-28 added the `token_id` continuation
technique AND `run_battery`'s side-by-side `per_technique` A/B computation AND `main`'s `--seeds` multi-seed CLI,
`990de0d4b` 2026-08-28 added the unrelated skip-continue extension on top) showed **the decisive-comparison
harness this task asked to "build" already exists, complete, at HEAD** -- `run_battery` already runs BOTH
`continuation="token_id"` and `continuation="text"` per topic/seed on the identical prompt, and `main` already
computes `n_live_diverged` (token_id) vs `n_live_diverged_text` (text) and accepts `--seeds 42,43,44,100,101,102`,
loading the (expensive) off-bridge Qwen faculty once and reusing it across seeds. **This session's own
contribution is not writing that code -- it is RUNNING it**, after confirming it against a real (not fake) mouth.

Confirmed the existing recorded artifact (`research/findings/raw/_open_ended_gen_time_consensus_veto_derisk.json`)
is STALE for this claim: its provenance sidecar records `git_sha: 20b4b475c` (2026-08-27, before the token_id
technique existed) and `"seeds": null` (single-seed, pre-`--seeds`) -- `n_live_diverged_text` reads `null` in that
file because the "text" A/B measurement did not exist yet when it ran. No runner in `research/runners/` or
`webapp/` duplicates this comparison (grepped for `token_id.*divergence`, `live.mouth.*6.?seed`, `tokenid_vs_text`
-- no hits besides this lineage). `research/queue/gpu.queue` / `.done` / `.running` contained no prior job
referencing this runner's multi-seed comparison before this session queued one.

## The bounded CPU sanity check (real Qwen, real organs, reduced scope)

Before spending GPU queue time, this session ran the SAME, unmodified `build_consensus_chat` /
`generate_with_generation_time_veto` functions directly (not a fake tokenizer/model -- the real
`Qwen2.5-0.5B-Instruct` + the real spiking-op install/calibration pass + the real numpy Izhikevich consensus
organs) on `device="cpu"`, `T=2` (vs production `T=16`), `max_new_tokens=24`/`sentence_budget=24`/
`max_sentences=2` (vs production 160/64/6), topic `canada`, seed 42 only -- scoped down purely to confirm the
harness runs end-to-end without exception and to measure RSS before committing to the real sweep.

**Result: it ran clean, 0 exceptions, all 4 cells (token_id x text, ON x LESIONED) completed.**
  - `chat built` in 150.5s (RSS 0.90GB) -- the real numpy consensus organs (SIM_BRIDGE logs show genuine
    Izhikevich network builds: e.g. "installed 253440 synapses across 9 populations" for the buffer composer,
    "installed 884736 synapses across 3 populations" for the organ-B/C consensus network), not a stub.
  - Qwen loaded in 7.4s (`gen.fac.load_seconds=1.96`) -- **peak RSS 3.99GB**, right at this session's ~4GB
    budget on CPU/float32 (0.5B params in float32 plus the numpy organs already built). This is a concrete,
    measured reason the decisive run belongs on the GPU queue, not just for speed: on `device=cuda` the weights
    load in float16 into VRAM, not host RAM, so the real run's host RSS will sit well under this CPU figure --
    running the real sweep on CPU inline would not have this headroom for 3 topics x 6 seeds.
  - `token_id_ON` took 131.2s (first ON call -- pays the one-time consensus-network-build cost visible in the
    SIM_BRIDGE log between 16:17:06 and 16:18:42, ~96s); `token_id_LESIONED` 31.8s (no organ call at all --
    `lesion_coupling=True` returns `([], {})` without ever calling `two_organ_combine`/`three_organ_combine`,
    exactly as designed); `text_ON` 34.5s and `text_LESIONED` 32.8s (both reuse the already-built organ network
    from the first ON call within the same process -- consistent with the consensus buses' own module-level
    build-once caching, not a bug). This is useful, measured evidence for sizing the real 6-seed queue job: the
    organ-network build cost is paid ONCE per process, not once per topic/seed/technique.
  - All 4 cells produced the IDENTICAL text `"What are some of the key achievements in Canada?"` -- an
    UNDEFINED read for the live vary/lesion property at this reduced setting (nothing for either variant to
    suppress this run), consistent with the possibility both parent findings already disclosed ("the mouth's own
    greedy decode may still legitimately produce the SAME text ON vs LESIONED when the coupling has nothing to
    suppress"). **This session does NOT read this as evidence about the real comparison** -- `T=2` on CPU is a
    heavily degraded spiking-op regime relative to production `T=16`/cuda/float16, and `max_new_tokens=24` capped
    generation to one short sentence before any of the store's border/capital/continent content had a chance to
    appear. The sanity check's job was confirming the harness EXECUTES, not previewing the result.

## The queued decisive run

Queued via `bash tools/queue_add.sh gpu "..." "decisive-live-mouth-6seed-tokenid-vs-text-divergence-never-run-before"`
(the record-check matched the literal substring "cd" from this command's `cd /home/dant123/Projects/sim &&`
prefix rather than the runner name -- a false-positive on `queue_add.sh`'s own runner-name regex, not a missed
record; the actual runner was independently checked via `before_you_build.sh` and RAG above). Appended to
`research/queue/gpu.queue`:

```
cd /home/dant123/Projects/sim && SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._open_ended_gen_time_consensus_veto_derisk --seeds 42,43,44,100,101,102 --device cuda --out research/findings/raw/_open_ended_gen_time_live_mouth_tokenid_vs_text_6seed.json
```

`SIM_BACKEND=numpy` pins the lightweight consensus-organ half to CPU/numpy (the module's own `setdefault`, made
explicit here so an inherited queue-shell env cannot silently override it); `--device cuda` is the SEPARATE,
independent flag that puts the off-bridge Qwen mouth's own torch forward on the RTX 3090, exactly the same
GPU/CPU split `_grounded_lang_integration_derisk.py` already established as this project's precedent for the
identical numpy-organs + torch-CUDA-mouth combination. A placeholder
(`research/findings/raw/_open_ended_gen_time_live_mouth_tokenid_vs_text_6seed.json`,
`{"status": "queued, not yet run", "queue_file": "research/queue/gpu.queue", "argv": [...]}`) was committed at
the `--out` path so `tools/claim_check.py` finds the cited artifact present at commit time, matching the
established convention (`research/findings/raw/_mouth_readout_tuning/eprop_ntp20000_ep12_6seed.json`'s own
placeholder for the same reason) -- the queued job will OVERWRITE it unconditionally with the real result
(`main`'s existing, unmodified `json.dump(art, ...)`).

**What the controller will harvest:** once the queue reaches this job, `research/findings/raw/_open_ended_gen_time_live_mouth_tokenid_vs_text_6seed.json`
will carry, per the runner's own existing schema, `n_live_diverged` (token_id technique's live-mouth ON/LESIONED
divergence count out of 18 rows = 3 topics x 6 seeds) and `n_live_diverged_text` (the same measurement for the
original text-continuation technique, same seeds/prompts) side-by-side, plus the PRIMARY controlled-unit-battery
verdict (unaffected by either continuation technique, already GO per the parent findings) and the no-regression
safety-net counts for both techniques.

## Verdict framing (what the harvested result will mean -- not asserted here)

- **`n_live_diverged` (token_id) > `n_live_diverged_text` (text)**, i.e. materially above the 2026-08-27
  single-seed text-continuation baseline's 1/3: the retokenization-confound fix genuinely lets the honesty veto
  catch/repair MORE real, spontaneous fabrications from the live mouth than the original technique did on the
  identical prompts/seeds. This would be the first decisive (not opportunistic) evidence for promoting
  `continuation="token_id"` as the better default for gen-time honesty specifically on this axis -- reported, NOT
  autonomously flipped (`BRAIN_OPEN_ENDED_GEN_TIME_HONESTY` stays default-OFF; per `docs/TERMS.md` this mechanism
  is `wired (default-off)`, never `closed`/`integrated`, regardless of this result).
- **`n_live_diverged` (token_id) ~= `n_live_diverged_text` (text)**, or lower: a first-class NO-GO on the
  hypothesis that removing the retokenization confound raises the live-mouth catch rate -- reported honestly. The
  retokenization-confound FIX itself (kept-sentence continuation is provably byte-identical-to-one-shot, proven
  by the CPU-only wiring-sanity battery already GO in the 2026-08-28 finding) stands on its own correctness
  merits independent of this outcome; a flat or lower live-mouth divergence rate would mean the confound was not
  the (or not the only) reason the 2026-08-27 baseline saw only 1/3 -- i.e. the mouth's own spontaneous
  fabrication rate on these 3 topics/6 seeds may simply be lower than 1/3 needs, an honest scope/sample-size
  finding about the topic set, not a defect in either continuation technique.
- Either way, the PRIMARY, decisive evidence for the mechanism itself remains the controlled unit battery (both
  parent findings' GO, deterministic, topic-complete, continuation-technique-independent) -- this comparison is
  scoped SPECIFICALLY to the SECONDARY, opportunistic live-mouth confirmation rung, exactly as both parent
  findings disclosed.

## Honest scope

**Not run this session:** the decisive 6-seed GPU comparison itself -- QUEUED, not landed (see "The queued
decisive run"). **The CPU sanity check's own text output is NOT informative about the real result** (see above --
T=2/CPU/1-sentence is not the production regime). No `sim/` edit; no change to
`clause_filter_sentence`/`sentence_contradicts`/`consensus_facts_for_topic`/`generate_with_generation_time_veto`/
the string safety net -- this session's only code artifact is a throwaway, uncommitted sanity script exercising
the existing, unmodified functions.

NEXT: harvest `research/findings/raw/_open_ended_gen_time_live_mouth_tokenid_vs_text_6seed.json` once
`research/queue/gpu.queue` runs this job, and land the verdict per the framing above. If GO, the skip-and-continue
extension (`2026-08-28-...-skip-and-continue-...-PARTIAL.md`) still has its OWN separate decisive live-mouth GPU
comparison staged and un-run -- a natural next queue entry on the same pattern this finding establishes.
