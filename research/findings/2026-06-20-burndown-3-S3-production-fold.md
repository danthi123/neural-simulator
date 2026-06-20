# Burndown #3 Stage S3 — production fold scope verdict: DEEP REWRITE, not a clean fold

**Date:** 2026-06-20
**Task:** S3 — fold the validated on-bridge K-way sequencer into the production `OneBrainComposer`
query path so the host `_scan` control flow is retired for the common case.
**Verdict:** **DEEP REWRITE (NOT a clean reuse-by-import fold). STOP + report.** Per the task's explicit
guard ("If it's a deep rewrite (not a clean fold), STOP and report — #3 stays validly at the S2
mechanism-level characterized-partial close; do NOT force it"), #3 remains at its S2 close. **NO production
edit made. NO `sim/` edit. The moat is untouched.**

This is a valid, deliverable scope outcome (a scope verdict was the first task item; an honest deep-rewrite
finding was named an acceptable deliverable).

---

## What S3 was asked to do

Route the host `_scan` (`research/runners/one_brain_composer.py:492`) — and the four call sites that share its
cue-match + first-match-priority semantics (`query_agent`:567, plus the inlined loops in `query_patient`:558,
`ask_yes_no`:572, `render_fact`:584) — through the on-bridge spiking K-way sequencer validated in S0–S2, for
n_facts ≤ 16, with the host `_scan` as the fallback for n_facts > 16, behind an opt-in flag (default OFF =
byte-identical).

The validated mechanism (S2, commit `a7899182`, `_phaseB_onebrain_sequencerK_k32_margin.json`): on-bridge K-way
match cascade + first-match-priority WTA + `input_divisive_norm`-driven decoded-line drive. **`k_star=16`**:
D=128, K∈{2,4,8,16} GO (==host, moat 0-FA, peak-robust); **K=32 NEGATIVE** (the routing-margin boundary), so
the production target is the on-bridge path to K≤16 with the host `_scan` fallback above 16.

---

## What is ALREADY on-substrate in the composer (so S3's surface is narrow)

The composer's READ is already fully spiking. Every query path calls `_read_blocks()` (`one_brain_composer.py:485`),
which (via the batched `_read_all_blocks`, A5 lever 1) does the whole K-way op ON the composer's resonate-and-fire
bridge: fire every stored trigger → block-diagonal complex-synapse unbind of all roles in parallel → block-diagonal
matched-filter cleanup → read. The winner-pick within cleanup is also spiking when `enable_spiking_cleanup=True`
(the NEF WTA `_spiking_select`).

**The residual HOST control flow** S3 targets is exactly and only:
- `_scan` (lines 492–496): `for got in self._read_blocks(): if all(got.get(role)==want ...): return got.get(answer_role)` → `return None`.
- The four inlined equivalents (558/572/584 + `count_facts`:681, `_calibrate`/reconsolidation reads at 650): the same
  `dict.get(role) == want` cue comparison + first-match `return` / abstain.

i.e. a Python dict-equality cue-match + first-match-priority + the no-confab abstain, run over the
already-spiking per-block read-out.

---

## Why routing this through the validated sequencer is a DEEP REWRITE, not a fold

The validated sequencer is **not** an operation on the composer's own bridge. It is a **second, large, separately
constructed spiking bridge** with a fundamentally incompatible configuration, fed by a **third** score bridge. The
three bridges cannot be merged by reuse-by-import:

| | Composer bridge (`build_coresident_bridge`) | Sequencer bridge (`build_sequencerK_bridge`) | Score bridge (`build_divnorm_score_bridge` / WTA) |
|---|---|---|---|
| `enable_brain_region_framework` | **False** (unset) | **True** | **True** |
| `connections_per_neuron` | **0** | n/a (framework pathways) | n/a (framework pathways) |
| Synaptic substrate | RF **complex** weights (`cp_rf_w_re/im`); no `cp_connections` wiring | real `cp_connections` Izhikevich + **transmission gates** + **gate-couplings** | `cp_connections` Izhikevich + `input_divisive_norm` |
| Topology | 4 fixed slices (parser / RF registers / store / cleanup) | **thousands** of word-line regions: `2V` cue + `2KV` decoded + `2KV` gated-match + `4K+1` pools + `K` inh; plus `2KV` transmission gates each with a gate-coupling EMA hook | `V` divnorm word-pools (+ `wta_inh` for retreat 2) |

At the de-risk's V=72, K=16 the sequencer alone is ≈ `144 + 2304 + 2304 + 65 + 16 = 4833` regions (~100K+ neurons)
plus `2·K·V = 2304` transmission-gate couplings. The **production vocab is V≈320+**, where these scale `K·V` — a
second 100K–500K-neuron framework bridge per composer.

Worse, the data flow is host-mediated by construction even in the validated path:
1. the composer's per-block cleanup membrane reads (`block_cleanup_scores(c, b)`) are pulled to host,
2. fed as `cp_external_input_current` decoded-line **drive** into the score bridge (divnorm/WTA) to pick each role's
   winner-line,
3. that drive feeds the sequencer bridge's `d{b}{role}_w` lines; the cue word-lines are driven; the whole match
   cascade settles (`run_sequencerK_with_drive`),
4. the host reads the `m{b}` spiking match-pool rates and applies the first-match production rule
   (`decision_to_block`).

So a faithful fold would have to: construct + persist two extra large framework bridges inside `OneBrainComposer`
(both at production V, keyed to the live store size K, rebuilt as `store` grows n_facts); thread the composer's
live cleanup scores out to the score bridge and the sequencer; implement the per-query reset/drain discipline
(`reset_sequencerK_state`'s gate-EMA + membrane drain) on every `query_*`; and reconcile that the production
default reads via the **batched** `_read_all_blocks` (A5 lever 1), whereas every S0–S2 de-risk ran with
`enable_batched=False, enable_rf_cudagraph=False` — i.e. the sequencer was validated against the per-block oracle
path, not the batched/megakernel production read.

That is a multi-bridge integration + a per-query control-bridge lifecycle + a re-validation of the
batched-vs-per-block interaction — a deep architectural rewrite, not the "clean reuse-by-import fold" S3 was gated
on. Forcing a fragile version risks the no-confab moat (the moat currently lives in the same `dict ==`/abstain the
sequencer would replace), which the task forbids trading.

---

## Honest standing of #3

#3 is at its **S2 characterized-partial close** and that close is sound:
- the on-bridge K-way spiking sequencer (match cascade + first-match WTA + on-bridge divnorm drive, the host
  `scores_to_drive` peak-read retired) is **GO ==host, moat 0-FA, peak-robust, lesion-safe, permuted-inverts** at
  D=128, K∈{2,4,8,16}, 3 seeds — `k_star=16`;
- K=32 is the honest, characterized **routing-margin boundary** (the shared-action EXTRA), with the host `_scan`
  fallback above K* (the `--host-fallback-above 16` path) the documented partial conversion.

The mechanism is validated and the boundary is characterized. What S3 would add — wiring that mechanism into the
production composer's live query path — is the deep rewrite above, deferred (not forced).

If/when the multi-bridge sequencer fold is prioritized, the design is: a `SequencerControl` companion owned by the
composer that (a) lazily builds + persists the sequencer + score bridges keyed by (V, K), (b) exposes
`route(cue) -> block_idx | None` consuming the composer's per-block cleanup scores, (c) is opt-in
(`on_bridge_scan=True, host_fallback_above=16`) and default-OFF byte-identical, (d) is re-validated against the
**batched** production read, not only the per-block oracle. That is its own staged arc with its own gates — not a
fold.

---

## Gates

No production fold was made, so the S3 fold gate table is N/A (the fold was not attempted past scope). The
upstream mechanism gates remain as recorded in S0–S2:
- S2 (`_phaseB_onebrain_sequencerK_k32_margin.json`): per-K `{2:GO, 4:GO, 8:GO, 16:GO, 32:NEGATIVE}`,
  `k_star=16`, OFF==byte-identical guard PASS, moat 0-FA at every K.

NO `sim/` edit. Stayed on `main`. Touched only this finding (the `#3` path). The no-confab moat is untouched.
