# Opt #2 — wire the sequencer vocab-shrink into production (`OneBrainComposer`) (2026-06-22)

**Scope:** the latency-arc RANK-2 optimization from the megakernel-revisit audit
(`research/findings/2026-06-22-megakernel-revisit-optimization-audit.md`, §5 row #2 — "the cheapest
sub-lever" of the per-query `integrated_loop` sequencer bottleneck). Deploy the validated de-risk
(`research/findings/2026-06-21-seq-vocab-shrink-derisk.md`, `bf217eb3`, GO byte-identical) into the production
`OneBrainComposer`. **Runner-side only — NO `sim/` edit.** Commit `487df1a1` (both remotes).

## The bottleneck (audit §3)
The `integrated_loop` spiking K-way sequencer (the brain-based replacement for the host first-match `_scan`
cue-match) was built over the composer's **full** vocab `self.V` — at production V=320, K=32 that is an
~837K-neuron Izhikevich control bridge driven 80 full simulation steps **per query**. The region math makes
`2·V` cue word-lines + `K·2·V` decoded word-lines + `K·2·V` gated-match lines, i.e. one word-line per word in
the entire vocabulary — but a who/what cue can only ever be a **stored** agent or action (every other cue is
abstained BEFORE the sequencer by the absent-word check).

## The fix (answer-identical)
Build the sequencer over only the **distinct stored agents** (role A = V'_A) and **distinct stored actions**
(role X = V'_X) instead of the full V. At production V=320 / K=32 (≤32 distinct agents, 8 distinct actions),
that is **~34.6× fewer sequencer neurons** (218,830 → 6,330 in the de-risk's accounting). Answer-identical
because:
- The **cue** is remapped global→reduced (`mapA`/`mapX`); a cue whose agent or action is not a stored filler
  abstains BEFORE the fabric (== no block matches in the full-V build — the moat).
- Each **decoded line** is remapped global→reduced; a spurious near-tie decoded word outside the reduced vocab
  (a low-fidelity-code artifact) is **dropped** == a closed/absent line no battery cue ever drives (in the
  full-V build that line's per-word gate is opened only by its own cue word firing, and no battery cue is that
  spurious word — so it is gated closed either way).
- A **cross cue** (both fillers stored, but never together) still runs the fabric and abstains there (no block
  has both the cued agent gate AND the cued action gate open) — byte-identical to full-V.

## The wiring (`research/runners/one_brain_composer.py`)
Reuse-by-import of the de-risk's reduced builder/runner/cue-vocab functions (`_seq_imports` adds
`build_sequencerK_reduced_bridge`, `reduced_cue_vocab`, `run_sequencerK_reduced_with_drive`):
- **`_ensure_sequencer`** (when `enable_seq_vocab_shrink`, default ON): compute `(V'_A, V'_X, mapA, mapX)` from
  the store's first-K facts, build `build_sequencerK_reduced_bridge(VA, VX, K)`, and cache `mapA`/`mapX`.
  Rebuild the reduced fabric when K grows **OR** a cue-vocab **signature** `(tuple(V'_A), tuple(V'_X))` changes —
  so an in-place reconsolidation that rewrites a fact's agent/action is handled, not just store growth.
- **`_seq_block`**: add the moat-preserving abstain (`agent not in mapA or action not in mapX` → `None`) then
  call `run_sequencerK_reduced_with_drive` with the cue WORDS + `mapA`/`mapX`.
- The **score bridge + per-block drives stay full-V** (the reduced runner remaps the global-indexed drives), so
  only the sequencer control fabric shrinks.

Default ON; `enable_seq_vocab_shrink=False` is the full-V escape. Only active on the `integrated_loop=True`
(brain-based spiking sequencer) path — the default `integrated_loop=False` host-oracle path is structurally
unreachable by these edits (`_seq_block` returns on the host branch before `_ensure_sequencer`).

## Gates
- **Answer-identical (HARD):** new `tests/test_onebrain_integrated_loop_fold.py::test_seq_vocab_shrink_answer_identical_and_smaller`
  (2 seeds) — the reduced sequencer selects the SAME block as the full-V sequencer on the whole who/what +
  abstain + cross battery, with a strictly smaller fabric. **PASS.**
- **No regression / oracle parity:** the existing fold suite (`integrated_loop=True` == the host `_scan`
  oracle, multi-seed, now running ON the shrink path by default) + the verdict-logic suites — **55 passed**
  (`bp6lbsvw4`, 0:06:35 CPU). The default OFF-path byte-identity tests in the same file confirm the host path
  is unchanged.
- **No-confab moat: 0 false-accepts** — preserved by construction (the abstain cases are identical to the
  full-V build; an absent-word cue is caught before the sequencer; a cross cue abstains in the fabric).
- **`sim/` edit:** NONE (composer-layer reuse-by-import).

## Speed
Per-query full-V vs reduced 80-step run at K=32 (`_seq_vocab_shrink_derisk --time`, GPU): a separate wall-clock run
is in flight (the full-V K=32 build is slow). The load-bearing number is the neuron-count reduction below: the
per-query sequencer is 80 full simulation steps over the fabric, so wall-clock tracks the neuron count -- a
~34.6x-smaller bridge runs a proportionally cheaper 80-step settle.
Neuron-count reduction at production V=320/K=32: ~34.6× (218,830 → 6,330; de-risk accounting). The win scales
with V (the full-V fabric is O(V·K); the reduced is O((V'_A+V'_X)·K), and V'_A+V'_X ≪ V at production).

## Sibling lever — opt #4 (the drive-seed cleanup-codebook cache)
The sequencer's per-block drive seed (`block_cleanup_scores`, the audit's §5 row #4) rebuilt its cleanup-codebook
connection list from scratch for every one of the K blocks (~3.9M tuple constructions at K=32). But that cleanup
list is **block-INVARIANT** — it depends only on the concept codebook + the fixed single-block `c_base`/`q_base`
layout; only the UNBIND wiring is block-specific. New `OneBrainComposer._seq_cleanup_conns()` builds it ONCE and
reuses it across the K per-block reads; `_ensure_sequencer` invalidates the cache at the start of each drive
rebuild, so a store / reconsolidation / regrounded concept is always picked up (the moat is never served a stale
cleanup). The drive seed recurs on the first query after every store, so this is an **interactive-latency** win
(per store→query), not a one-time one. Reuse-by-import, NO `sim/` edit. Gate: the fold sequencer answer-identity
tests (which route through `block_cleanup_scores`) pass.

## Sibling lever — opt #3 (vectorize the sequencer gate-coupling loop — the one `sim/` edit)
The sequencer's per-step transmission-gate update (`_apply_gate_couplings`, `sim/bridge.py`) ran a Python loop over
all ~K·V' couplings on every one of the 80 settle steps per query, computing each coupling's EMA + threshold + gate
value individually (the audit's §5 row #3). With `enable_vectorized_gate_couplings` (which the K-way sequencer opts
into) the RATE was already a segment-sum (one device reduction), but the EMA + threshold + gate-select stayed a
Python per-coupling loop. opt #3 batches that into ONE numpy op (`new_ema = alpha*rates + (1-alpha)*ema`;
`new_value = where(new_ema >= threshold, open_value, 0)`), writing ONLY the gates whose value flipped. Byte-identical
to the scalar path (the same per-element EMA arithmetic; the segment-sum rate == per-coupling `.mean()` for boolean
firing; a NaN-init `last_value` reproduces the first-step "write every gate"; the scalar path is provably unchanged).
The ONE `sim/` edit in the latency arc (owner-approved): a cached `_gate_coupling_state` array structure (rebuilt
only on a coupling-count change, preserving existing state) + a rewritten `_apply_gate_couplings`. Gate: the fold
sequencer answer-identity tests 8 passed (the sequencer == the host oracle with the vectorized gate loop), moat 0-FA.

## Provenance
De-risk: `research/findings/2026-06-21-seq-vocab-shrink-derisk.md` + `research/runners/_seq_vocab_shrink_derisk.py`
(`bf217eb3`). Audit: `2026-06-22-megakernel-revisit-optimization-audit.md` (§5 row #2). Wiring + test:
`research/runners/one_brain_composer.py` + `tests/test_onebrain_integrated_loop_fold.py` (`487df1a1`, both
remotes). Sibling lever opt #1 (the `elaborate` read-batch): `2026-06-22-opt1-elaborate-read-batch.md`
(`0bba215c`, 30/30). Both of the audit's top-2 latency levers are now in production.
