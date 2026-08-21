---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-period2-read-noise-snapshot-isolated-read
lane: integration
integration_faculty: d5-live-consolidation
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_stabilized_read_derisk.py — measures, per graded read (depth_rest/depth_hold/soft) ×
  {LIVE, snapshot-ISOLATED}, the repeated-read noise floor (K reads, NO consolidation) vs the consolidation effect (the
  REAL continuous_engine.consolidate_used_memory learn-through-use loop) → SNR = |effect|/noise_std, plus monotone /
  faithful(formation-lesion) / moat-byte-id. Reuses step-6's validated weak-encode selection + GradedEpisodicDapMemory;
  the ISOLATED read is step-6's handler_read (restore a clean-rest snapshot + inject the current weights). NO sim/ edit.
runner: research/runners/_d5_stabilized_read_derisk.py
external: NO-EXTERNAL-NEEDED — an in-repo characterization of the D5 surfaced-read noise + a lower-variance read; no
  literature question. The read is the SAME cp_v_apical the production recall reads.
artifacts:
  - research/findings/raw/_d5_stabilized_read/summary_6seed.json
  - research/findings/raw/_d5_stabilized_read/seed42.json
  - research/findings/raw/_d5_stabilized_read/seed44.json
  - research/findings/raw/_d5_stabilized_read/seed101.json
  - research/findings/raw/_d5_stabilized_read/seed102.json
---
# The D5 surfaced-read noise is a PERIOD-2 limit cycle a complete-reset (snapshot-isolated) read removes ENTIRELY (deterministic, 6/6) — unblocking the crosstalk decidability — but the conversation-visibility rise to 6/6 is NOT unblocked by it: a DISTINCT, DETERMINISTIC saturating-tail wobble caps it (NEGATIVE on the combined GO; two residuals disentangled)

Artifact: `research/findings/raw/_d5_stabilized_read/summary_6seed.json` (6-seed; go_overall=False) + the per-seed JSONs.

**One line.** The D5 learn-through-use consolidation was blocked on ONE stated residual — "the surfaced dendritic-depth
read is too noisy at the mV scale" — said to block BOTH the memory-separator crosstalk verdict (#73) AND the
learn-through-use conversation-visibility. This de-risk DIAGNOSES that noise (a stationary period-2 limit cycle from an
incomplete reset), builds the lower-variance read the task asked for (a complete reset to clean-rest = a snapshot-isolated
read, which is DETERMINISTIC, std=0), and finds the residual is actually TWO things: the read noise (period-2) IS removed
by isolation → the crosstalk read becomes decidable; but the conversation-visible RISE to 6/6 is NOT unblocked by removing
the noise — it is capped by a SEPARATE, deterministic saturating-tail wobble in the weight→read curve. So the combined GO
(decidable AND monotone-rising on >4/6 seeds) is NOT met by any single read; honest NEGATIVE, banked with the SNR table
and the disentangled next levers.

## The noise is a PERIOD-2 LIMIT CYCLE, not jitter and not drift (diagnosed, decisive)
<!--derived-->
The production recall reads the LIVE bridge with only `hard_silence`+`_reset_apical_latch` between reads — an INCOMPLETE
reset that does not return the network to clean-rest baseline, so residual state carries from one read into the next.
Repeated LIVE reads of the SAME memory (byte-identical weights, NO consolidation) therefore ALTERNATE high/low: seed 42's
`depth_hold.live` noise across 8 reads is `[0.0, 15.97, 0.0, 15.97, 0.0, 15.97, 0.0, 15.95]` mV (from
`seed42.json`) — a textbook period-2 cycle (the reply literally flips between "dog completes" and "dog does not"),
std ≈ 7.98 mV. Across the 6 builds the `depth_hold.live` std ranges 0.0–7.98 mV (per-seed `[7.98, 1.71, 0.0, 2.51, 5.37,
0.42]`); a build where it reads 0.0 all 8 times (seed 44) is one where the incomplete reset locked the cycle to its
DOWN phase — the live read misses dog's completion entirely there.

## The lower-variance read: a COMPLETE reset (snapshot-isolated) is DETERMINISTIC (std = 0, 6/6)
<!--derived-->
The CLAUDE.md wall-reframe ("what companion process did we replace with a constant?"): the missing process is the
network's RETURN TO CLEAN-REST BASELINE between recalls (real recalls are time-separated; the network settles). The
incomplete reset approximated it. The faithful implementation — a complete reset: `snapshot_state`/`restore` to clean
rest with the current weights injected, then the SAME spiking apical-dAP recall — is DETERMINISTIC: repeated isolated
reads of the same memory are byte-identical (seed 42 `soft.iso` noise across 8 reads = `[0.5515]×8`, std 0.000000), so
`noise_std = 0` on 6/6 seeds for every isolated read (`summary_6seed.json` tally: depth_rest.iso / depth_hold.iso /
soft.iso all `decidable 6/6`). It is ADDITIVE: the binary moat gate (`apical_cue`, `in_memory`) stays the LIVE read,
byte-identical to production `_apical_up_read` (`moat_byte_id` 6/6); only the surfaced STRENGTH becomes the isolated read.
Because the isolated read is noise-free, "did A's consolidation move neighbour B?" becomes DECIDABLE (any B-shift IS the
effect) — which is exactly the te=40 crosstalk regime (#73) where the period-2 noise (a few mV) exceeded the ~1.79 mV
neighbour effect.

## But the rise to 6/6 is NOT unblocked by removing the noise — a SEPARATE saturating-tail residual caps it
<!--derived-->
The 6-seed SNR table (`summary_6seed.json`, weak-usable store per seed, cupy; read_go = decidable ∧ rises ∧ monotone ∧
faithful ∧ moat-byte-id):

| read.source     | noise_std | decidable | rises | strict-monotone | read_go |
|-----------------|-----------|-----------|-------|-----------------|---------|
| soft.live       | small     | 6/6       | 6/6   | 5/6             | **5/6** |
| depth_rest.iso  | **0.000** | 6/6       | 6/6   | 4/6             | 4/6     |
| depth_hold.iso  | **0.000** | 6/6       | 6/6   | 4/6             | 4/6     |
| soft.iso        | **0.000** | 6/6       | 6/6   | 4/6             | 4/6     |
| depth_hold.live | 0.0–7.98  | 6/6       | 6/6   | 3/6             | 3/6     |

Two facts the table forces. (1) At a weak-usable store (weight with rise headroom, w_dog ≈ 28→65 over 5 ticks) the
consolidation effect is LARGE (many mV), so the read noise is NOT the binding constraint on decidability — EVERY read,
live or isolated, is `decidable 6/6`. Removing the period-2 noise (isolation) therefore does NOT lift the rise reliability
here; the isolated reads are 4/6 strict-monotone, no better than the live baseline band. (2) The binding constraint on a
strict-monotone rise is a DETERMINISTIC saturating-tail wobble in the weight→read mapping: `depth_hold.iso` (std 0) rises
cleanly on 4/6 (e.g. seed 42 `[16.18, 16.71, 17.02, 17.20, 17.25, 17.27]`) but on seed 101 dips `15.32→14.97` and on
seed 102 reverses mid-trajectory `23.54→22.85 … 24.62→24.12` — the weight kept GROWING while the read dropped ~0.35–0.7
mV near the top. Because the read is deterministic (std 0) this is NOT read noise; it is the plateau-depth read saturating
non-monotonically in the weight — the same "saturating-tail" the flip-verify observed (strict-monotone 3/6 there). The
bounded `soft` read compresses the tail best (soft.live 5/6, the single best variant), but still does not reach 6/6.

## Verdict: NEGATIVE on the combined GO — but the residual is now correctly SPLIT
The task GO needs BOTH (a) noise clearly below the effect (decidable) AND (b) a monotone rise on MORE than the 4/6
depth_hold baseline (aim 6/6), from ONE additive read. No single read clears both: the deterministic isolated read gives
(a) trivially (std 0) but only 4/6 on (b); the best (b) is soft.live at 5/6, which is a bounded LIVE read, not the
lower-variance one. `go_overall=False`. What is now KNOWN, and was the point of the de-risk:
- **The read noise is a period-2 limit cycle from the incomplete reset, and a complete-reset (snapshot-isolated) read
  removes it entirely (deterministic, 6/6).** This is the lower-variance read; it is the path to a DECIDABLE crosstalk
  verdict at the saturated te=40 op-point (where the noise, not a large effect, was the blocker).
- **The conversation-visibility rise to 6/6 is a DISTINCT residual** — a deterministic saturating-tail non-monotonicity
  in the plateau-depth read, NOT the read noise. Lowering the variance does not touch it.

## Next levers (the two residuals, now separate)
- **Crosstalk decidability:** run the direct te=40 "A's consolidation moves neighbour B" experiment with the
  snapshot-isolated read (deterministic) vs the live read (period-2) — the isolated read should make the ~1.79 mV
  neighbour shift decidable where the ~5 mV period-2 noise buried it. (This de-risk proved the enabling property —
  std 0 — but measured decidability at the weak encode where the large effect made it non-binding.)
- **Rise to 6/6:** the saturating-tail wobble is the residual. Candidates: read the rise over the PRE-saturation window
  (fewer ticks / a weaker store), a bounded read (soft was best), or the flip-verify's relative-tolerance monotone (which
  banked 5/5 load-bearing by tolerating the sub-percent saturating ripple). Not a wall — a read-window / operating-point
  choice, not a substrate limit.

## Scope honesty
The isolated read's snapshot/restore is a host determinism guard — the SAME guard `consolidate_used_memory` already
declares — and biologically it is the return-to-rest between recalls, so it is a faithfulness improvement, not a new
cheat. The binary moat gate is unchanged (byte-identical, 6/6). The surfaced strength is a faithful spiking read, not a
phenomenal claim. seed 102 completed on te=8 (a build where the assembly did not self-ignite); the honesty gate abstains
on the self-igniting builds by construction.
