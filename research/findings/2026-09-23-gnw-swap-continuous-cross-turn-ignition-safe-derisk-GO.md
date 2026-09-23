---
type: finding
status: complete
date: 2026-09-23
mechanism: continuous (no-restore) cross-turn ignition for the GNW held-topic swap workspace
  (webapp/gnw_thought_swap.py::ThoughtSwapWorkspace.observe, de-risk
  research/runners/_gnw_swap_continuous_recency_derisk.py) -- board #77/#85's own named next rung.
lane: laneC (GNW / integration-to-production)
verdict: GO on the SAFETY + MECHANISM claim (6/6 seeds); NOT-YET-LOAD-BEARING named honestly on the reply-level
  behavioral claim (1/6 seeds, existence-only, NOT gated into the GO above).
artifacts:
  - research/runners/_gnw_swap_continuous_recency_derisk.py
  - research/findings/raw/_gnw_swap_continuous_recency_smoke.json
  - research/findings/raw/_gnw_swap_continuous_recency_6seed.json
  - webapp/gnw_thought_swap.py (continuous_enabled() + the isolate= thread-through, +2 lines of production glue)
verification: |
  SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u -m research.runners._gnw_swap_continuous_recency_derisk --six-seed \
    --json research/findings/raw/_gnw_swap_continuous_recency_6seed.json
  -> verdict=GO seed_go 6/6, swap 6/6, branch 6/6, carry 6/6, blind 6/6, no_regression 6/6, det 6/6
     [diagnostic near-threshold dissociation 1/6, NOT gating]. Ran LOCALLY (CPU, ~80s total, a 960-neuron toy
     substrate -- "unit test / tiny smoke", not a full-brain build; tools/mem_ok.sh not required per CLAUDE.md's own
     compute-lane rules). An independent pool reproduction is staged (see "Compute" below) for cross-machine
     confirmation, not because the local result is in doubt.
---

# GNW swap CONTINUOUS cross-turn ignition: a real, 6/6-seed-verified, SAFE-to-enable synaptic recency mechanism -- honestly NOT YET load-bearing on the reply

**One line.** Board #77/#85's held-topic workspace restores its substrate to a clean snapshot every turn
(`isolate=True`); this finding builds, de-risks (6/6 seeds), and wires (default-off) the genuinely CONTINUOUS
alternative those findings named as "the next rung" — and reports, without rounding up, exactly how far that rung's
own behavioral consequence does and does not yet reach.

## Why this is the genuine next rung, not a re-derivation

`bash tools/before_you_build.sh "GNW continuous cross-turn ignition recency swap workspace no restore"` surfaced
`2026-08-19-gnw-swap-into-chat-GO.md:73` directly — its own honest-limit #3 (renumbered #1 in the current source):
*"A truly continuous cross-turn ignition (no restore) is the named next rung."* `git log --all --grep` and a scan of
`research/findings/*gnw*`/`*swap*`/`*thought-swap*` confirmed the SURROUNDING capability — making the swap DRIVE the
live reply — is already merged to `origin/main`, `DEFAULT-ON` (`_SWAP_DRIVES_DEFAULT_ON=True`,
`webapp/server.py`), and independently re-verified byte-identical against today's code by
`2026-09-05-rank11-topic-swap-scaffold-backlog-item-already-integrated.md`, which also confirms it is already a row
in `research/runners/load_bearing_fraction.py`'s `FACULTY_LESIONS` battery (`swap-drives-response`). Building that
again would duplicate shipped work — the task's own verify-first rule pivots to the next rung in that case, and the
next rung is named, in the shipped module's own words, above.

## The mechanism (reuse-by-import; NO `sim/` edit)

`research/runners/_gnw_neural_swap_intention_derisk.py::run_intention_swap` already exposes `isolate=False` ("a
CONTINUOUS run, 0 restore calls") and its own `run_two_swap` already uses it for a two-swap A→B→A reversibility
headline (part of the ALREADY-GO'd 6-seed de-risk). This finding is the first use of that existing continuous-mode
plumbing as a genuinely multi-turn, production-shaped conversation, and the first to characterize what continuity
alone buys once wired to the production `ThoughtSwapWorkspace`.

Protocol per seed: establish topic A (one necessary cold-start `isolate=True` — there is no prior turn to be
continuous with), then swap continuously (`isolate=False`) A→B. This evicts A via the SAME recurrence-weakening STD
the shipped mechanism already uses (Tsodyks-Markram short-term depression on the recurrent E→E loop); A's loop is
left with a depleted resource variable `x_A < 1`. Two controlled arms then branch from this IDENTICAL point (two
independent builds at the same seed reach a byte-identical branch state, confirmed on 6/6 seeds — `branch_identical`):
**RECENT** re-proposes A (the topic just left); **FRESH** proposes C, a topic never held this session (`x_C == 1`).

## What is robustly TRUE (the GO, 6/6 seeds, at the shipped production drive strength `SALIENT_PA`)

1. **The carryover is real, not a label.** `x_A` at the branch point measures 0.73–0.78 across all six seeds
   (never ≥0.95) — continuous mode genuinely carries synaptic state across the HTTP turn boundary.
2. **Restore mode is PROVABLY blind to it**, not just typically blind: calling the exact reset (`std.reset()`)
   `isolate=True` performs, on a substrate that has JUST come out of a real continuous swap carrying the ~0.75
   debt, wipes it to *exactly* 1.0 for every pattern, every seed (`restore_blind`, 6/6, code-level not statistical).
3. **Safe: no regression.** At the shipped production drive strength, every turn's swap-vs-hold VERDICT under
   continuous mode is identical to the restore-mode default — RECENT and FRESH both swap correctly, and their
   post-window firing rates match to 1e-6 (`no_regression_at_production_pa`, 6/6). A live in-process check of
   `ThoughtSwapWorkspace` (7-turn board-#77-shaped conversation: establish/same-topic/topic-change ×2/no-topic/
   topic-change) reproduces the IDENTICAL swap/hold pattern with `BRAIN_GNW_SWAP_CONTINUOUS` on and off.
4. **Determinism** (build-twice Izhikevich-parameter hash) holds 6/6.

This is what makes the new `BRAIN_GNW_SWAP_CONTINUOUS` flag (default-off, additive, byte-identical when unset —
`webapp/gnw_thought_swap.py::continuous_enabled()`) a low-risk addition: it is not merely unmeasured-but-hoped-safe,
it is measured-safe.

## What is honestly NOT YET true (named, quantified, not claimed closed — `docs/TERMS.md`)

The production swap decision is deliberately supra-critical/robust by design (the point of the original GO'd
mechanism), so at the shipped drive strength the ~25% STD carryover measured in (1) above does **not** change which
turns swap — a clean null on the behavioral question at that operating point, reported as a null, not omitted.

A hand-swept weaker re-proposal drive (`--near-threshold-pa`, 1500 pA, ~30% of `SALIENT_PA`) **does** produce a
qualitative recency-driven failure on seed 42 (RECENT fails to re-ignite, `swapped=False`; FRESH succeeds cleanly,
`swapped=True`, under the identical drive) — this is an **existence proof**, not a lever: the same fixed drive gives
both-fail on seed 102 and both-succeed on seeds 43/44/100/101 (`near_threshold_diagnostic.dissociation`, 1/6). A
per-seed calibration sweep (5000/3000/2000/1500/1200/1000/800 pA, seed 42 only) located the narrow band where the
effect exists at all; per-seed Izhikevich heterogeneity shifts each pattern's OWN ignition margin near that band
enough to dominate a fixed constant's outcome. **This diagnostic is recorded in the artifact and NEVER gates
`seed_go`/`pooled_go`** — the six-seed GO verdict above rests entirely on claims (1)-(4), which hold unconditionally.

**Consequently: `swap-continuous-recency` is NOT added to `load_bearing_fraction.py`'s `FACULTY_LESIONS` battery in
this finding.** That battery's `neural-lesion` rows assert a reproducible state→reply dependency (`docs/TERMS.md`'s
`load-bearing`); adding a row here, even with an honest note, would put a mechanism this finding explicitly could
NOT reproduce 6/6 into the "robust core" count. The neural lesion tool (`std.deps[A].x[:] = 1.0`, forcibly wiping
ONLY A's carryover while leaving every other continuous-mode state variable untouched) is built and exercised in the
diagnostic arm, ready for whichever future session calibrates a per-seed (or genuinely graded, non-fixed-PA)
operating point — that calibration, not a retry at a different constant, is the actual next-next rung.

## Anti-cheats

- **Controlled fork, not two independent runs:** RECENT and FRESH are built from the SAME seed through the
  IDENTICAL establish+A→B sequence; `branch_identical` (6/6) confirms the two builds reach a byte-identical state
  before the one proposal that differs between them.
- **The lever moved:** `x_A_at_branch < 1.0 - 0.05` is void-checked (`tools.lab.void_if`) before anything downstream
  is interpreted — a seed where the carryover failed to materialize would void that seed's comparison, not silently
  count as a null result.
- **Restore-blindness is a code-level check, not a threshold call:** exact float equality to 1.0 (1e-12), not "close
  to."
- **No sim/ edit:** `git diff sim/` is empty; every primitive (`build`, `run_intention_swap`, `MultiLoopSTD`,
  `SALIENT_PA`) is reused-by-import from the already-6/6-seed-GO `_gnw_neural_swap_intention_derisk` module.
- **Production wiring is additive and reversible:** `continuous_enabled()` unset/0 → `isolate=not continuous_enabled()`
  evaluates to `isolate=True`, the PRE-EXISTING hardcoded value — provably byte-identical by construction, confirmed
  live (7-turn in-process comparison, identical swap/hold pattern on/off).

## Compute

Ran locally (CPU, `SIM_BACKEND=numpy`, a 7-region/960-neuron toy substrate — the same scale as the reused de-risk
module's own `--six-seed` mode, which this codebase's own runners document running directly, not via the pool):
smoke (1 seed) + the full six-seed sweep together took ~80s wall. Per CLAUDE.md's compute-lane rules ("Unit tests /
tiny smokes (no full brain) — run locally, freely"), no `mem_ok.sh`/`memcap.sh`/pool routing was required for this
scale. An independent pool reproduction is staged anyway (see the task's structured result) for cross-machine
confirmation, not because the local six-seed GO is in doubt.

## Files

Full per-seed data: `research/findings/raw/_gnw_swap_continuous_recency_6seed.json` (the six-seed GO summary +
per-seed `go_gate`/diagnostic records) and `research/findings/raw/_gnw_swap_continuous_recency_smoke.json` (the
seed-42 single-seed record). `research/runners/_gnw_swap_continuous_recency_derisk.py` (new). `webapp/gnw_thought_swap.py`: `continuous_enabled()`
(new function, ~20 lines of docstring + 2 lines of logic), the `isolate=not continuous_enabled()` thread-through in
`ThoughtSwapWorkspace.observe`'s held-topic branch, an additive `"continuous"` key in that branch's info dict, and an
update to honest-residual #1 in the module docstring. `webapp/swap_drives_chat.py` is unchanged — it calls
`gnw_thought_swap.observe_turn`, which already threads the new flag through, so board #85's production reply-driving
path inherits `BRAIN_GNW_SWAP_CONTINUOUS` for free (still default-off, still byte-identical when unset).
