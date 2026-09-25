---
type: finding
status: no-go
lane: load-bearing (open-ended conversation; plan step S16, lane A3)
mechanism: open-ended-gated-turn (BRAIN_OPEN_ENDED_GATED, default OFF)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NOT-GO (UNDEFINED)
date: 2026-09-25
---

# Open-ended gated turn, Part A capability gate: NOT-GO (UNDEFINED) — seed 100 reproduces the standing a3 degenerate null on an independent instrument (2026-09-25)

Lane: open-ended conversation, plan step S16, lane A3. Pre-registration:
[`2026-09-24-open-ended-gated-turn-PREREGISTRATION.md`](2026-09-24-open-ended-gated-turn-PREREGISTRATION.md)
(committed `b4588775a`; Amendments 1 `98c4912e2`, 2 `7a39851cd`, 3 `6994e80f2`, 4 `e79031483`, all committed before
any gate-seed run). Governing rule: Amendment 3, section A3.1 (the reply-level CAT metric is the Part A verdict;
CONT is a manipulation check, never a GO). Instrument:
`research/runners/_open_ended_gated_turn_gate.py --score` (byte-identical between the pinned data revision
`b728777aec1287061eecc997fd3a32bbfc4c04a0` and this scoring pass's `HEAD` — `git diff` between the two on this file
is empty). This document scores Part A only. Part B (the three `lbf_rows/open_ended_gated.py` load-bearing rows) is
not addressed here; no Part B artifacts were in the raw census this document was asked to score. The seed-7 dev
smoke and the flag-off byte-identity check are reported separately in
[`2026-09-24-open-ended-gated-turn-seed7-smoke-and-flag-off-identity.md`](2026-09-24-open-ended-gated-turn-seed7-smoke-and-flag-off-identity.md)
and are not re-scored here.

## Verdict: **NOT-GO (UNDEFINED)**

Scored with the pre-registered command
(`--score --seeds 42,43,44,100,101,102 --out-dir research/findings/raw/_open_ended_gated/partA`) against all 42
governed session files (6 seeds x 7 files: 3x `intact` + 1x `intact_rebuild` + 3x `lesion`), exactly as the
prereg registers. Aggregate artifact (this scoring pass, committed alongside this finding):
`research/findings/raw/_open_ended_gated/partA/oe_gated_partA_aggregate.json`, `summary.verdict` =
`"NOT-GO (UNDEFINED)"`, `summary.GO` = `false`.

- **5/6 seeds DEFINED, not 6/6.** Seed 100 reads **UNDEFINED**: `noise streams never changed a reply within an arm
  -> the null is DEGENERATE` (`per_seed.100.CAT.reasons`, `summary.per_seed_reasons.100`). Both arms answered
  `deer` on 21 of 24 asks (the remaining 3 ABSTAIN in every session), identically in intact and lesion
  (`per_seed.100.CAT.hist_intact` = `hist_lesion` = `{"deer": 21, "ABSTAIN": 3}`). This is the SAME degenerate-null
  failure mode the standing 2026-09-23 a3 verdict hit at the identical seed, on a different (ungated) instrument
  and a different code revision — there too both arms answered `deer` on every ask
  (`2026-09-23-open-ended-production-turn-a3-6seed-harvest-NO-GO.md`). Seed 100's host weight vector here is
  `{"deer": 3.0, "cat": 2.0, "dog": 2.0, "rabbit": 2.0, "beetle": 1.0, "minnow": 1.0, "fish": 0.0, "memory": 0.0,
  "spikes": 0.0, "words": 0.0}` (`per_seed.100.CAT.host_weight_vector`) — `deer` dominates the next-highest
  candidates by 1.0 (a 50% margin over the runner-up), wide enough that neither the per-session OU noise stream
  nor the WTA's own stochasticity ever flips the argmax away from it, in 3 intact and 3 lesion sessions x 8 asks
  each. Two independent instruments now reproduce the identical failure at the identical seed: this reads as a
  property of seed 100's stored-facts weight vector, not of either runner.
- **The five DEFINED seeds' deltas do not clear the 0.10 floor on every seed either**, so even crediting seed 100
  most generously would not flip the verdict. Per-seed reply-level (CAT) delta, `attributable_to_host_weight_drive`
  and the descriptive exact permutation p:

| seed | CAT verdict | Delta_reply | perm p (descriptive) | attributable_to_host_weight_drive | MANIP (CONT) Delta | reason if UNDEFINED |
|---|---|---|---|---|---|---|
| 42 | DEFINED | 0.027778 | 0.35 | 0.037736 | 0.179472 | — |
| 43 | DEFINED | -0.027778 | 0.85 | -0.035088 | 0.177017 | — |
| 44 | DEFINED | -0.013889 | 1.0 | -0.018868 | 0.163470 | — |
| 100 | **UNDEFINED** | — | — | — | 0.198294 (DEFINED) | noise streams never changed a reply within an arm (degenerate null) |
| 101 | DEFINED | 0.138889 | 0.05 | 0.208333 | 0.146190 | — |
| 102 | DEFINED | 0.25 | 0.05 | 0.367347 | 0.175450 | — |

  Only 2 of the 5 DEFINED seeds (101, 102) clear the 0.10 floor; three (42, 43, 44) sit near zero, two of them
  negative. The registered rule needs Delta_reply >= 0.10 on EVERY seed AND all 6 DEFINED — both fail
  independently of each other (`summary.reply_every_seed_at_floor` = `false`, `summary.all_defined` = `false`).
  `summary.p_sign_test_reply` is `null` because the exact sign test over the full registered 6-seed set cannot run
  while one seed is UNDEFINED (the scorer's own construction, matching the a3 precedent).
- **The manipulation check (CONT, never a GO by the prereg's own labelling) is DEFINED on all 6 seeds and clears
  0.10 on every one** (0.146190–0.198294, see table; `summary.manipulation_check.delta_cont`), confirming the lesion reaches the
  reply-selecting spiking competition on every seed, including 100. `summary.manipulation_check.every_seed_at_floor`
  reads `false` only because the aggregate gates it on `all_defined` (which is `false` due to seed 100's CAT
  verdict) — the manipulation check's own per-seed numbers (0.146190–0.198294, see table) all individually clear
  the floor. <!--derived--> This is exactly the
  A3.1-predicted shape: the lesion visibly perturbs firing everywhere, while the reply it selects moves enough to
  clear the floor on only 2 of 6 seeds.
- **Pass-through (descriptive) confirms A3.1's design point: Part A does not exercise the gated routing.** Every
  ask in every arm and seed routed `hypothesis` and the BG race picked `SPEAK` on the non-abstaining asks (e.g.
  seed 42 intact: `route: {"hypothesis": 24}`, `bg_action: {"SPEAK": 21, "STAY_SILENT": 3}`,
  `n_replaced_by_gated_turn: 3` — the 3 holds, not a gated-turn substitution;
  `per_seed.42.pass_through.intact`). `summary.gated_routing_exercised` reads `true` for every seed/arm on the
  weaker "gated turn ran and left a trace" sense, consistent across all 6 seeds — but Part A's replies are, by
  design, the pipeline's own pass-through answers with the flag on, per A3.1.
- **Reply-identity against the a3 `default` sessions (descriptive, different code revision) matches on 6/8 or 7/8
  asks per session, every seed/arm** (`a3_reference_identity_descriptive`, e.g. seed 42: `[7, 8]` on every
  intact/lesion/rebuild session; seed 44: `[6, 8]`). The prereg attributes any mismatch to code drift since
  `eefdd666a`, not to the gated turn; this reading is consistent with that.

## Provenance

All 42 governed session files' `.prov.json` sidecars record the identical
`git_sha: "b728777aec1287061eecc997fd3a32bbfc4c04a0"`, `source_kind: "git_archive"`, `git_dirty: false` (checked
across every one of the 42 sidecars, not a sample). The runner file
(`research/runners/_open_ended_gated_turn_gate.py`) is byte-identical between that pinned revision and this
scoring pass's checkout (`git diff b728777aec1287061eecc997fd3a32bbfc4c04a0 HEAD -- research/runners/_open_ended_gated_turn_gate.py`
is empty), so the scoring logic run here is exactly the one the prereg registers, not a later edit. The runner's
own `--selftest` (pure, no brain) passes, including the negative-direction cases (a degenerate null must read
UNDEFINED; a wrong seed set must read NOT-GO (WRONG SEED SET); firing that moves without the reply moving must
read NO-GO). The aggregate artifact's own provenance sidecar
(`oe_gated_partA_aggregate.json.prov.json`) correctly records this as a scoring-only invocation (`git_dirty: true`,
reflecting the copied-in raw data in this worktree, `source_kind: null` — it is not a fresh experiment run) rather
than misrepresenting it as a `git_archive` data-generating run.

Completeness and liveness were checked directly, not assumed from the census note. All 42 partA session files (7
per seed x 6 seeds) are present with matching timestamps ending 2026-09-24 23:04:54 local. `pool.queue` (pending)
and `pool.queue.running` (in-flight) both show zero entries referencing `oe_gated`/`open_ended_gated`; `ssh` to all
four mini-PC pool nodes (`pool1`, `pool2`, `pool41`, `pool42`) at the time of this scoring pass shows no live
process matching `_open_ended_gated_turn_gate` or `lb_shard`, and no local process either. Nothing was still
running.

## What this does and does not show

- **Not a claim that the gated turn's spiking draw is load-bearing on the reply, under this registered design.**
  The registered GO rule (all 6 seeds DEFINED, the a3 sign-test-and-floor rule, Delta_reply >= 0.10 on every seed,
  the manipulation check >= 0.10 on every seed) is not met on two independent counts (an UNDEFINED seed and a
  floor miss on 3 of the 5 DEFINED seeds).
- **Not a claim that it is refuted either.** Two of five DEFINED seeds (101, 102) show a same-direction effect
  clearing the floor by a wide margin (0.138889, 0.25); the manipulation check confirms the lesion reaches the
  reply-selecting competition on every seed including the UNDEFINED one. The result is a NOT-GO on this
  registered method, not evidence the mechanism does not exist.
- **This is the reply-level pass-through path only** (A3.1): the ask used here ("what might a dog chase") routes
  `hypothesis` and is answered through the pipeline's ordinary spiking-draw path with the gated flag on; it does
  not exercise the gated routing itself (route/BG-race/marker), which Part B's `lbf_rows/open_ended_gated.py` rows
  measure separately and which this document does not score.
- **The lesioned edge is a HOST vector** (`BRAIN_SPIKING_DRAW_LESION=1` replaces the host co-occurrence-derived
  likelihood vector with `np.ones` before the host affine map into the spiking WTA's drive), declared in the
  prereg and repeated in the aggregate's own `lesioned_edge` field. This design shows whether that host vector,
  transmitted through the spiking draw, moves the reply — not whether the spiking draw itself is load-bearing
  absolute a host-oracle arm (the prereg explicitly declines that comparison for Part A; see A3.1 "why the
  review's other option ... was not taken").

## Next step, per THE LAW

This is a verdict on the registered METHOD (M=3 sessions/arm, per-seed floor 0.10, exact registered seed set), not
a license to close the capability question. Two independent instruments (a3's ungated turn at `eefdd666a`, and
this gated turn at `b728777ae`) have now hit the identical degenerate null at the identical seed (100), with the
same signature: one candidate's host likelihood dominates the runner-up by a wide margin, and 3 sessions of
per-session OU noise plus WTA stochasticity are not enough to ever flip the argmax at that seed. The standing a3
finding named the next lever as "a design that does not let one seed's frozen draw ... zero out a sixth of the
registered evidence"; this run confirms that lever is still open and now shows it reproduces across two different
turn implementations, so it is a property of the seed-100 stored-facts/weight-vector construction, not of either
runner. Concrete options for the next lever (banking this scoring method, not abandoning the capability): (a)
increase M sessions specifically where the intact-arm null is checked for degeneracy before scoring, so a
seed whose weight vector is this lopsided gets enough draws for the WTA's stochastic margin to matter; (b) widen
the per-session OU noise amplitude at seeds whose host weight vector's peak-to-runner-up margin exceeds a
declared threshold; or (c) redesign the per-seed KB/weight generation to bound that margin so no registered seed
can land in a construction where neither arm's WTA output can differ from the peak by chance. None of these
change what has already been measured; they are the next-lever candidates for a future amendment, not applied
here.

## Declared host shortcuts

Unchanged from the prereg and the aggregate's own `host_shortcuts` field: teach KB + ask prompt (world);
histogram/permutation statistic and the per-session noise-stream assignment (instrument); the co-occurrence matrix
P, weight vector w = `_weight_partner`, and the affine drive map (host, the lesioned input); argmax over the
bank's firing counts (host read-out); hypothesis role induction, the SVO template, the RF-composer moat verify,
and the prompt routers (host); the read->salience transduction and the route label (host, gated-turn residuals);
the warm Qwen faculty stub (FORM not measured — the gate never loads Qwen).
