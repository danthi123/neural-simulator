---
status: live
type: finding
lane: scaffold-retirement backlog rank-8 (research/coordination/scaffold_retirement_backlog.md)
date: 2026-09-05
integration_faculty: gnw-organ-combination-bus
---

# GNW congruence spiking read — the N-organ bus's organ-B/organ-C "does this match" check retires its host `==` for a spiking `pred_k->mm_k` match-veto read reusing the already-6/6-GO'd swap-intention ignition circuit. 6/6-seed GO on parity with the host decision (a real organ-B/organ-C battery), the trigger-lesion collapses discrimination on every genuine mismatch, and a SEPARATE production-hook verify confirms flag-off byte-identical + real-match parity + lesion-via-flag reverts the FULL `bus_combine` committed decision. De-risked + wired behind a DEFAULT-OFF flag (`BRAIN_GNW_CONGRUENCE_SPIKING`).

**Date:** 2026-09-05 · **Backend:** CPU (numpy) · **Verdict:** **GO** (circuit + production-hook level, 6/6 seeds each) · **No `sim/` edit** (`git diff sim/` empty) · FUNCTIONAL correlate only; NO phenomenal claim.

**Files:** `research/runners/_gnw_congruence_spiking_read_derisk.py` (NEW — `SpikingCongruenceReader`, `build_battery`, `evaluate_seed`), `webapp/gnw_congruence_spiking.py` (NEW — the default-OFF production glue: `congruence_spiking_enabled`, `congruence_lesion_on`, `get_congruence_reader`, `spiking_congruent`, RNG isolation), `webapp/gnw_bus_shadow.py` (one additive flag-branch inside `_organ_reads`, guarded by `_congruence_spiking_enabled()`; `seed` threaded through `_bus_combine_inner`), `research/runners/_gnw_congruence_spiking_hook_verify.py` (NEW — the dispatch/byte-identical/lesion-reverts verification). **Artifacts:** `research/findings/raw/_gnw_congruence_spiking_read_6seed.json` (the circuit GO gate), `research/findings/raw/_gnw_congruence_spiking_hook_verify.json` (the production-dispatch verification).

**Builds on / reuses:** `research/runners/_gnw_neural_swap_intention_derisk.py` (6/6-seed GO — the `pred_k->mm_k` MATCH VETO circuit, its biology citations, and its Izhikevich operating point, reused verbatim, NOT re-researched or re-calibrated); `research/runners/_gnw_coincidence_integrator_derisk.py` (`_assign_slots`, the SAME first-seen-order content->slot addressing every coincidence-integrator finding in this repo already uses); `webapp/gnw_bus_shadow.py` (the LIVE default-installed N-organ ignition bus this finding's target lives inside — its consensus-ignition/WTA and the STOP this bus's answer can feed are UNTOUCHED). Scoped by `research/coordination/scaffold_retirement_backlog.md` rank-8 ("GNW congruence host string-id").

## What this retires — organ B/C's congruence, not the ignition consensus

`webapp/gnw_bus_shadow.py::_organ_reads` is the LIVE production organ-combination read (installed by default since
the 2026-08-13 flip/retirement — `webapp/server.py::brain_reply` runs it on every turn). Two of its three organs
decided "does this second read CORROBORATE the first" with a bare host comparison:

```python
cand_B = cand_A if composer.query_patient(agent, action) == cand_A else None     # organ B: VERIFY re-check
cand_C = cand_A if composer.query_agent(action, cand_A) == agent else None        # organ C: reverse-binding
```

Both reads (`query_patient`, `query_agent`) are genuinely spiking FHRR unbinds, and the downstream consensus
(`norgan_hop`'s coincidence-ignition + shared-inhibition WTA) is already genuinely neural — but by the time a vote
reaches that consensus, whether organ B/C's vote exists AT ALL had already been decided by a raw Python `==`. This
finding targets ONLY that congruence decision; the ignition/WTA consensus and organ A's forward recall are
untouched, exactly as the backlog scoped it.

## The retirement mechanism — reusing the swap-intention circuit's OWN match-veto, not a new one

`research/runners/_gnw_neural_swap_intention_derisk.py` (6/6-seed GO) already contains a spiking circuit whose
entire purpose is "does a proposed content MATCH the held content": `pred_k` (driven by whichever pattern is
established/ignited) inhibits `mm_k`'s ability to fire on a proposal for the SAME slot k. `mm_k` fires (a real
population-rate signal) IFF the proposal targets a slot that is NOT currently held. `SpikingCongruenceReader`
reuses this circuit UNCHANGED (`build`, `run_intention_swap`, `MultiLoopSTD`, `SALIENT_PA` — reuse-by-import, no
`sim/` edit, no re-derivation) as a stateless pairwise reader: establish `held` as the incumbent, propose
`proposed`, and read the verdict off `run_intention_swap`'s own spiking output (`held`, `swapped`, `mm_peak`) —
"the same populations already firing for ignition," never a host `k_held == k_proposed`.

**Addressing vs. deciding (the honest distinction this rests on).** `held`/`proposed` still need a content->slot
address before any neuron can be driven — `_assign_slots` (imported unchanged, the same addressing every 6/6-GO'd
coincidence-integrator finding in this repo already uses) supplies it. This is a UNARY lookup: each string's slot
depends only on itself and prior registrations, never on the other operand in the SAME call — the same class of
"wiring" the codebase already accepts as legitimate (`ThoughtSwapWorkspace._slot_for`, `_ExpandedSurpriseOrgan.
_block_for`). What moved from host to spiking is the VERDICT: nowhere in this module, nor in the reused
`run_intention_swap`, is `k_held == k_proposed` computed and used to decide the outcome. The verdict is read from
`pred_k -> mm_k`'s temporal, threshold-crossing, lesionable population dynamics, which discriminate "same slot"
from "different slot" only because of anatomical wiring (`pred_k` projects only to `mm_k`; `pattern_k` drives only
`pred_k`) plus the drive protocol — the same "wiring + threshold dynamics realize the comparison" property the
swap finding itself already earned its GO on.

## GO GATE — the circuit (6 seeds 42/43/44/100/101/102, ALL hold)

A REAL organ-B/organ-C-shaped battery (32 pairs/seed: 8 chains x 4 pairs — organ-B match/mismatch + organ-C
match/mismatch) built from the SAME `CHAINS`/`RFPhasorComposer` fixture `_gnw_two_distinct_organs_derisk.py` itself
uses — genuine `query_patient`/`query_agent` reads on real stored facts, with mismatches drawn from a REAL
patient/agent in a DIFFERENT chain (not a synthetic string).
<!--derived-->

| seed | pairs (match/mismatch) | parity vs host `==` | intact mismatch acc | lesioned mismatch acc | lesion collapses | attributable | determinism |
|---|---|---|---|---|---|---|---|
| 42  | 32 (16/16) | 32/32 | 1.00 | 0.00 | yes | 100% | yes |
| 43  | 32 (16/16) | 32/32 | 1.00 | 0.00 | yes | 100% | yes |
| 44  | 32 (16/16) | 32/32 | 1.00 | 0.00 | yes | 100% | yes |
| 100 | 32 (16/16) | 32/32 | 1.00 | 0.00 | yes | 100% | yes |
| 101 | 32 (16/16) | 32/32 | 1.00 | 0.00 | yes | 100% | yes |
| 102 | 32 (16/16) | 32/32 | 1.00 | 0.00 | yes | 100% | yes |

`parity 192/192 (32 x 6 seeds) · lesion_collapses 6/6 (96/96 mismatch pairs flip under lesion) · determinism 6/6`.
GO gate (per seed): 100% parity vs the host `==` ground truth, the trigger-lesion collapses EVERY genuine-mismatch
pair to a false "congruent" reading, the battery contains a genuine mismatch arm (>=4 pairs, not a vacuous
all-match battery), and build-twice determinism holds. `attributable_to` credits 100% of the intact-vs-lesion
mismatch-discrimination difference to the `pred_k->mm_k` circuit on every seed (the lesion silences ONLY mm's
proposal drive; the addressing/wiring is untouched by it, so a changed verdict is attributable to the circuit's

**The boolean parity above saturates at 1.00 on every seed — a ceiling, not evidence on its own** (flagged
correctly by `gates/discriminating_power` as advisory). The CONTINUOUS signal underneath it is on record and
genuinely varies: mean match `mm_peak` 0.0007-0.0545, mean mismatch `mm_peak` 0.270-0.306, margin 0.225-0.287
across the 6 seeds (`per_seed[].mean_match_mm_peak` / `mean_mismatch_mm_peak` / `mm_peak_margin` in the cited
6-seed artifact) — a real, seed-varying physical margin the threshold clears on every seed, not a degenerate
test where the boolean read has no headroom to fail.
OWN firing, not the addressing). <!--derived-->

## Load-bearing — the trigger-lesion collapses discrimination (reused unchanged from the swap finding's own lever)

`trigger_lesion=True` (mm's proposal drive silenced — the SAME lever `_gnw_neural_swap_intention_derisk`'s own
"TRIGGER-LESION DISSOCIATION" test uses, reused verbatim, not re-derived) makes mm never fire regardless of
content, so the incumbent never self-evicts and EVERY proposal — match or genuine mismatch — reads "congruent." On
every seed, all 16 genuine-mismatch pairs (which correctly read "not congruent" intact) flip to a false
"congruent" reading under lesion (0.00 accuracy), while the 16 match pairs are unaffected (a match already reads
"congruent" intact, so the lesion cannot move them further). This is the anti-host-if-else: a python `==` hiding
behind the addressing would be untouched by silencing a spiking population's drive; the observed collapse proves
the verdict is read from `mm`'s firing.

## GO GATE — the production-hook dispatch (6 seeds, `_gnw_congruence_spiking_hook_verify.py`)

The circuit's own GO gate is not the same claim as "the production wire-in behaves correctly" (TERMS.md: `wired`
requires a call path from the production entry point, and `byte-identical` must be asserted in the data). A
second, narrower verification exercises `webapp.gnw_bus_shadow._organ_reads`/`bus_combine` directly (no `sim/`
edit; both take a bare `composer`, no fake `ChatBrain` needed) on the SAME fixture:

| seed | flag-off byte-identical vs FROZEN pre-edit reference | flag-on real-match parity vs flag-off | organ B/C: intact withholds / lesion falsely corroborates (manufactured mismatch) | `bus_combine`: intact matches host / lesion wrongly commits |
|---|---|---|---|---|
| 42–102 (all 6) | yes (8/8 queries) | yes (8/8 queries) | yes / yes | yes / yes |

`flag_off_byte_identical 6/6 · real_match_parity 6/6 · lesion_reverts 6/6`. The flag-off path is compared, per
query, against a FROZEN literal copy of the pre-rank-8 `_organ_reads` (embedded in the verify script's data, never
imported by production) — a tuple compare, not an inferred diff-read. Because this fixture's clean, unpermuted
facts never disagree naturally, the lesion-reverts claim uses a wrapper composer (`_ForceWrongSecondRead`) that
forces organ B's re-read / organ C's reverse-binding to a REAL patient/agent from a different chain on one probe
each — a genuine, manufactured mismatch. Intact: organ B and organ C each correctly withhold (matching the host's
own correct withhold on the identical manufactured mismatch), and `bus_combine`'s committed decision matches the
host's (both abstain — 2/3 organs voting is sub-quorum under the bus's own unanimity calibration). Lesioned: organ
B and organ C each falsely corroborate (`cand_A` instead of `None`), and on the organ-C probe this lets all three
organs agree -> `bus_combine` WRONGLY COMMITS the fact the host correctly withheld — the load-bearing collapse at
the full production-decision level, not just the isolated circuit. <!--derived-->

## Contract (additive, DEFAULT-OFF, reversible)

`BRAIN_GNW_CONGRUENCE_SPIKING` unset/0/false/off/no (DEFAULT) -> `_organ_reads` runs its ORIGINAL, unmodified host
`==` congruence checks for organ B and organ C; `webapp.gnw_congruence_spiking` is never imported (the flag check
is a bare env-var read inside `gnw_bus_shadow.py`). An explicit 1/true/on/yes -> organ B/C's congruence is decided
by `spiking_congruent` (the reused `pred_k->mm_k` match-veto) instead. `BRAIN_GNW_CONGRUENCE_LESION=1` silences
mm's proposal drive in every read this turn, exactly as the de-risk's own `trigger_lesion` lever does standalone.
The reader runs on its own private RNG timeline (snapshotted/restored around every read, the #77/#85/gnw-global-
stop pattern) so enabling it cannot perturb any OTHER RNG-dependent organ in the same turn. This is a DE-RISK +
WIRED-BEHIND-A-FLAG landing, not a default-on flip — the parent decides whether/when to flip
`BRAIN_GNW_CONGRUENCE_SPIKING` on, per this codebase's standing de-risk -> flip-soak -> flip sequence.

## Honest residuals (named, not claimed closed)

1. **The content->slot ADDRESSING is still a host dict lookup (`_assign_slots`)** — a UNARY, per-string operation
   (never compares `held` to `proposed`), the same convention every 6/6-GO'd coincidence-integrator finding in
   this repo already treats as legitimate wiring (see "Addressing vs. deciding" above), but it is still host
   bookkeeping, not a learned/self-organized address. Named, not hidden.
2. **N_PATTERNS=3** (inherited unchanged from the reused swap-intention circuit) bounds how many distinct
   concepts a SHARED, cross-call reader could address before LRU-style reuse; this finding sidesteps the concern
   by resolving both operands of one `congruent()` call TOGETHER (at most 2 slots needed, well under 3) rather
   than relying on cross-call address stability — correct for THIS use (a stateless pairwise check), but a reader
   asked to hold many concepts' identities simultaneously would need a larger reused substrate.
3. **Organ B/C remain dichotomous corroborate-or-withhold organs** (never propose an independent alternative
   patient/agent) — unchanged from the pre-existing production design; only the corroboration DECISION moved.
4. **This lands DEFAULT-OFF.** The already-shipped N-organ bus (default-ON since 2026-08-13) is completely
   unaffected while the flag is off (verified 6/6 at the production-hook level) — this finding does not change the
   production ledger row's `scaffold_retired`/`on_by_default` status for the bus's organ-combination row, which
   correctly still names organ B/C's congruence check as a residual until the owner reviews a flip.
5. **The scalar/boolean read-out convention** (`held and not swapped` -> a plain bool) discards the graded
   `mm_peak` the circuit actually produces; production only ever needed the boolean, matching the pre-existing
   `cand_B`/`cand_C` contract (`cand_A` or `None`), so nothing downstream currently reads the graded signal.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._gnw_congruence_spiking_read_derisk \
    --six-seed --json research/findings/raw/_gnw_congruence_spiking_read_6seed.json

SIM_BACKEND=numpy python -u -m research.runners._gnw_congruence_spiking_hook_verify \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_congruence_spiking_hook_verify.json
```
