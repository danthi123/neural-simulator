# Production-wiring NAV CHUNK (Chunk N) — the merged "one brain" navigation runs fully-spiking-by-default (2026-06-21)

**Status: COMPLETE — all 5 items committed item-by-item, both remotes; all gates GREEN.** This is the
LAST chunk of the production-wiring execution plan
(`docs/plans/2026-06-22-production-wiring-execution-plan.md` §1.3 / Chunk N). It wires the validated nav
brain-shortcut closures (#6 log-polar, #5b grid front end, #5b determinism, B4 op-point, the B2/B3/B4
limbic core) into the production merged-nav default so the merged nav+conversation "one brain"
navigation cognition runs fully-spiking-by-default. **NO `sim/` edit anywhere** (every item is
runner/builder-side, reuse-by-import). **The no-confab moat was NEVER weakened** (the nav cascade is
array-disjoint from the conversational composer; re-asserted by the merged tests' `is None` no-confab
assertions on every gate run).

GPU (`SIM_BACKEND=cupy`, RTX 3090). PATHSPEC commits (narrow `git add`), each item its own commit,
pushed to both remotes (origin + gitea).

---

## The 5 items — per-item status + gate result

| # | item | flip | status | commit |
|---|---|---|---|---|
| 1 | #6 log-polar | `run_moving_goal_episode` `log_polar_retina` default `False`→`True` (library) | **FLIPPED** | `91442e0b` |
| 2 | #5b grid front end | promote the probe monkeypatch → a first-class `nav_critic_grid_frontend` builder flag (the ONE genuinely-new wiring) | **FLIPPED (opt-in, honest-scoped) — SUPERSEDED 2026-06-21: now the MergedNavConvAgent PRODUCTION default; #5b CLOSED, host-Gaussian retired (`2026-06-22-shortcut5b-CLOSED-grid-default.md`, `c32be6cc`)** | `822d3359` |
| 3 | #5b determinism | add `deterministic_read` kwarg holding `deterministic_transpose_matvec` ON through value-train + δ-read | **FLIPPED (kwarg, default-OFF)** | `55389e06` |
| 4 | B4 op-point | merged cue-shift de-risk defaults → the B4 COOLED op-point (`td_stdp_w_max=40`, `n_train=15` + the supporting strong-derivative regime) | **FLIPPED** | `dcca9aac` |
| 5 | limbic core | `MergedNavConvAgent` `co_resident_nav_critic` production-default ON (with mutual-exclusivity auto-yield) | **FLIPPED (agent default, GREEN_INERT)** | `bc060020` |

**The per-item HARD gate (all five hold on every item):**
- **Moat 0-FA (HARD):** `tests/test_nav_conv_merged_agent.py` (8) + `tests/test_nav_conv_step2b_coresident.py`
  (7) PASS on GPU, incl. the three `is None` no-confab assertions
  (`what_does`/`elaborate`/`describe`). NEVER weakened. (Plus `tests/test_merged_rf_composer_coresident.py`
  5/5 CPU = the third nav-merged gate.)
- **Nav not regressed:** the `nav_on_merged_smoke` PASSES (nav navigates + the parser weights stay
  BYTE-FROZEN under the live nav reward-STDP stressor, gains==0, nnz unchanged) — GREEN_INERT preserved.
- **CPU-portable + revertible:** every flip keeps a default-OFF/escape; GPU-only stays opt-in.

**sim/-edit flag: NONE.** Every item is runner/builder-side reuse-by-import — the deterministic-scatter
branch, the graded plateau, the spiking critic, the grid reference helper, the log-polar render all
already ship.

---

## Item 1 — #6 log-polar default-on (`91442e0b`)

**Flip:** `run_moving_goal_episode`'s `log_polar_retina` library default `False`→`True`
(`g11_bg_runner.py:~3708`). The low-level `render_egocentric_goal(log_polar=False)` util default stays
(callers pass explicitly). Added `--log-polar-retina` / `--no-log-polar-retina` CLI flags (default ON) +
the SC_LOG_POLAR env var as the explicit override.

**Why it is byte-inert except on the spiking-SC path:** the egocentric SC eye-drive render that consumes
the flag is GATED on `enable_spiking_sc and "sc_retina" in region_indices_cp` (`g11_bg_runner.py:~7104`),
and `enable_spiking_sc` itself defaults False. So flipping the library default regresses NO documented
standalone benchmark (none enable the spiking SC by default) — the `--readout-source motor`/`thal`
host-argmax ORACLE and every standalone run reproduce byte-identically. It only takes effect on the
spiking-SC orienting path (the merged-nav default where `nav_critic_spiking_sc=True` builds the SC),
making the biology-faithful log-polar render the merged-nav default there (5/6-GO,
`2026-06-22-shortcut6-log-polar-render-derisk.md`).

**Library-vs-merged scope:** the library `run_moving_goal_episode` default is flipped; the merged-nav
episode (`nav_on_merged_smoke` → `run_moving_goal_episode`) INHERITS it (documented at the call site).
The merged-nav STEP-2a byte-identity smoke runs the host-argmax cascade WITHOUT the spiking SC, so
log-polar is inert there → GREEN_INERT byte-identity preserved.

**Gate:** moat 8/8 + 7/7 (0-FA) · nav-not-regressed (byte-inert; `--no-log-polar-retina` reproduces the
linear render byte-identically) · CPU-portable + revertible.

---

## Item 2 — #5b grid front end default-flag (`822d3359`, the ONE genuinely-new wiring)

**Flip:** promoted the `_n5_grid_frontend_onbridge_probe` grid_cells→place mechanism from a probe
monkeypatch to a first-class builder flag `nav_critic_grid_frontend` on `build_bg_brain_regions` /
`run_moving_goal_episode` / `build_merged_nav_conv_bridge` / `MergedNavConvAgent` (+ the
`--nav-critic-grid-frontend` CLI). When ON (requires `nav_critic_place_selforg`, ASSERTED), the
`place_sensors` afferent is the DECORRELATED spatial-phase grid metric (`_n9_make_grid_code` at the
agent's own `(x,y)`, promoted VERBATIM from the de-risk reference helper) instead of the
locally-degenerate landmark render → the self-org place pool carves locally-SELECTIVE fields (place
value V n/f 4.5–12.3× vs the render's 1.0× R1-cap, R1 GO 3/3). `place_sensors` is sized to the grid dim
(`grid_n_modules*grid_n_per_module = 198`, the validated config). The grid reads ONLY `(x,y)` (structural
anti-cheat).

**HONEST SCOPE (conservative-default + production-opt-in) — SUPERSEDED 2026-06-21 (see the UPDATE below):**
this delivers the R1 **selective-afferent** win. The `nav_critic_place_selforg=True` self-org path already
retires the host-Gaussian `vs_place_context` position injection (the brain-shortcut closure). At the time of
this chunk the host-Gaussian's FULL retirement-BY-DEFAULT was framed as gated on the **δ-readout
stabilization** — a precisely-characterized DEEPER boundary: the grid graded-plateau READ conflates the
place code's structural near/far MAGNITUDE asymmetry with learned value
(`2026-06-22-shortcut5b-volley-normalization-close.md`). So the grid front end shipped as a **first-class
flag, OPT-IN** (default OFF = byte-identical, the landmark render).

**UPDATE 2026-06-21 — #5b CLOSED, the grid front end is now the MergedNavConvAgent PRODUCTION default
(`2026-06-22-shortcut5b-CLOSED-grid-default.md`, `c32be6cc`):** the TD-read de-risk
(`2026-06-22-shortcut5b-td-read-derisk.md`) RESOLVED the "gated on the δ-readout stabilization" disposition.
**#5b closes on R1 grounds** (the grid front end produces a genuinely-neural, value-gradable place code —
afferent selectivity + learned near/far value, both 3/3); the close does NOT depend on the δ/TD read. The
residual value-READ structural/learned separation is the **CHARACTERIZED DENDRITIC FRONTIER** (a point-neuron
limit, NOT a host shortcut, NOT a blocker — the existing graded-plateau read stays). So `MergedNavConvAgent`
now flips `nav_critic_place_selforg` + `nav_critic_grid_frontend` to the production default ON (the
`None`-sentinel auto-yield, mirroring Item 5), retiring `vs_place_context` for the production merged-nav
bridge. The function-level / low-level builder defaults stay `False` (standalone-CLI reproducibility +
research-runner config). Gates: moat 8/8 + 7/7 (0-FA) with the flip ON; the production bridge inventory
confirms `vs_place_context` is ABSENT.

**Validated:** CPU smoke — grid adjacent-cell afferent cos **0.61** (decorrelated) vs the landmark
render's **0.9954** (degenerate, the exact R1-cap); the grid reads only `(x,y)`. GPU — the grid-frontend
bridge builds with `place_sensors` sized **198** + the self-org place pool runs on the grid metric
(standalone `diff-loc cos=0.577` + merged); the guard fires (grid-frontend without place-selforg asserts).

**Gate:** moat 20/20 (8 + 7 + 5, 0-FA, default-OFF byte-inert) · nav-not-regressed · CPU-portable +
revertible.

---

## Item 3 — #5b determinism `deterministic_read` (`55389e06`)

**Flip:** a new default-OFF `deterministic_read` kwarg on `run_moving_goal_episode` (+ `--deterministic-read`
CLI) holds `cfg.deterministic_transpose_matvec` ON THROUGH the value-train + the graded-plateau δ-read,
instead of restoring it OFF after STEP-1 self-org (`g11_bg_runner.py:~5524`/`~5562`). The
deterministic-scatter SpMV already ships at the 5 critic-path matvec sites in `sim/bridge.py` (gated on
the flag, numerically allclose), so this is the runner-side **1b deploy scope** the determinism-close
finding named — NO `sim/` edit. It pins the read-time `Wᵀ@prev_firing` scatter ORDER → each seed's critic
rate is reproducible run-to-run → a seed-stable place→value volley.

**HONEST SCOPE:** determinism ALONE holds the SNc-burst δ **2/3** (`2026-06-22-shortcut5b-determinism-deltabar-close.md`);
3/3 needs the volley-normalization, AND the δ that then passes is the SAME structurally-influenced read as
Item 2 (the graded-plateau READ conflates the structural near/far magnitude asymmetry with learned value —
a characterized deeper boundary, equally true on the 2/3 determinism-only baseline). This item is exactly
the "keep the matvec flag ON through value-train + δ-read" deploy scope the finding named; the
clean-learned-δ readout redesign is the recorded next frontier, NOT this flip.

**Validated:** the kwarg path runs ("DET-READ held ON" prints; the flag stays ON through the read).
Default False = the documented STEP-1-only determinism is byte-identical (the read window restores to
`_saved_detmv`).

**Gate:** merged-agent moat 8/8 (0-FA, byte-inert by default) · CPU-portable + revertible.

---

## Item 4 — B4 op-point defaults (`dcca9aac`)

**Flip:** the merged TD cue-shift consolidation de-risk
(`_merged_td_cueshift_consolidation_derisk.py`) CLI defaults → the B4 COOLED op-point
(`2026-06-22-shortcut-B4-oppoint-r07-3of3.md`), so running it with NO extra flags reproduces the
documented strict Schultz-signature migration **r < −0.7 on 3/3 seeds** co-resident. The TWO headline
cooling levers: `td_stdp_w_max` 60→40 (cool the critic → the cue-burst snap moves to ~trial 9) + `n_train`
30→15 (read over the cooled convergence window → centered step). The supporting four (FS-clamp 30/20,
gabab_prop 0.04, derivative_gain 2, slow_tau 250) are the B4 op-point they cool FROM (the
strong-derivative regime required for the migration to reach the cue) — flipped to defaults too so the
two headline levers are meaningful (clip40+nt15 ALONE do NOT give the GO without the strong-derivative
regime; the finding's §2 landscape). The pre-B4 "standalone GO" legacy defaults are supply-explicit →
revertible.

**Validated (GPU, seed 42, no extra flags):** migration **r = −0.787** (< −0.7, `migration_r` True, dir
True) — byte-matches the documented seed-42 cooled result. The "PARTIAL" verdict = `late@cue`/`omit-dip`
graded on seed 42 (the documented HS98 graded-transfer regime, 2/3 seeds); the headline r<−0.7 is the GO
criterion and passes. The TD error stays 100% neural (a weight-BOUND + a measurement-window, not a host
value/reward computation).

**SCOPE:** this is the cue-shift DE-RISK path (`co_resident_td_cueshift`, a separate DA modulator over
`[td_snc]`), NOT the default production merge (`co_resident_nav_critic`, Item 5; mutually exclusive). The
two are not conflated. Moat preserved by construction (the td regions are array-disjoint, zero out-edges
to conversational slices; the B4 finding AC3).

---

## Item 5 — limbic core `co_resident_nav_critic` production-default ON (`bc060020`)

**Flip:** `MergedNavConvAgent`'s `co_resident_nav_critic` constructor default `False`→`None`, where
`None` = "production default ON unless a MUTUALLY-EXCLUSIVE critic was explicitly requested". The
production "one brain" agent now brings up the spiking limbic core by default — US→SNc reward burst +
striosome_value MSN-D1 value critic + the scope=all `dopamine` modulator over `[snc]` — so the merged-nav
cognition (reward, value, RPE) is spiking-by-default (brain-based purity).

**The mutual-exclusivity blocker + the auto-yield resolution (why NOT a builder-default flip):**
`co_resident_nav_critic` is mutually exclusive (asserted in the builder) with `co_resident_limbic` (the
4-region minimal organ) and `co_resident_td_cueshift` (the A-CSC TD cue-shift slice) — each registers its
OWN scope=all DA broadcast, so only ONE critic can be co-resident. Flipping the LOW-LEVEL
`build_merged_nav_conv_bridge` default to True would CRASH the many research runners that pass a
mutually-exclusive flag explicitly (the limbic-validate runner `co_resident_limbic=True`, the B4
cue-shift runner `co_resident_td_cueshift=True`, the DA-salience-gate smoke). So:
- The **low-level `build_merged_nav_conv_bridge` default STAYS `False`** (conservative — the research
  runners that compose their own critic config keep the assert protecting a genuine double-request).
- The **production `MergedNavConvAgent` default flips ON** via the `None` sentinel + auto-yield: an
  EXPLICIT `co_resident_limbic=True` / `co_resident_td_cueshift=True` YIELDS the production default (the
  explicit research-config request wins → no mutual-exclusivity crash); an EXPLICIT
  `co_resident_nav_critic=False` opts out (legacy no-critic). Verified: `co_resident_limbic=True`,
  `co_resident_td_cueshift=True`, `co_resident_composer+co_resident_limbic`, and explicit
  `co_resident_nav_critic=False` all build without the mutual-exclusivity crash (the research runners keep
  working).

**GREEN_INERT CAVEAT (documented, NOT hidden):** the nav value/RPE is BEHAVIORALLY INERT on the
orient-solvable immediate-reward gridworld (the #9 lesson / the merged-nav-critic BOUNDARY finding) —
flipping this is a brain-based-purity default, NOT a navigation behavior win; the limbic core is validated
spiking but its δ does not change the navigation score.

**Gate:** moat 8/8 + 7/7 with the default now building the critic-ON config (15/15, 0-FA, incl. the three
`is None` no-confab assertions — the scope=all DA broadcast, threshold-0/neutral-at-rest, does NOT perturb
the frozen conversational comprehension, confirming the B4 finding AC3). CPU-portable + revertible.

---

## The nav-not-regressed result (GREEN_INERT)

The `nav_on_merged_smoke` (the STEP-2a byte-identity gate) PASSES with ALL FIVE items' edits in place:
- (A1) nav+conv regions co-reside on ONE bridge ✓
- (A2) the parser (conversational) weights stay **BYTE-IDENTICAL** across the episode (frozen under the
  LIVE nav reward-STDP + dopamine stressor; gains==0; nnz unchanged) ✓
- (A3) the parser parses post-episode (`'dog go north'` → agent=dog/action=go/patient=north) ✓
- the merged bridge NAVIGATES ✓

The conversational populations are array-disjoint from the nav cascade (`cp_connections` /
`cp_membrane_potential_v` / `cp_firing_states` vs the composer's complex `cp_rf_w_*`), so the nav burst
leaves them byte-frozen — the moat is preserved by construction and re-asserted on every gate run. The
log-polar default flip is byte-inert in this gate (no spiking SC); Items 2/3 are default-OFF; Item 4 is a
separate de-risk path; Item 5's limbic core is GREEN_INERT.

---

## Scope-resolution summary (conservative-default + production-opt-in, where it applied)

Two items diverged from a blanket-flip and were resolved with honest scope (the pattern the
composer-`integrated_loop` + agent-`enable_learned_assoc` chunks established):
- **Item 2 (grid front end):** the R1 selective-afferent win is delivered as a first-class flag, OPT-IN
  (default OFF). A blanket default-on would overclaim a clean learned-δ host-Gaussian retirement the
  substrate does not deliver (the δ-readout boundary). The genuinely-new wiring is built + validated; the
  full host-Gaussian retirement-by-default awaits the δ-readout resolution (the recorded next frontier).
- **Item 5 (limbic core):** the production AGENT default flips ON; the low-level BUILDER default stays
  conservative (the mutual-exclusivity with the research runners). The auto-yield makes the production
  default ON for the agent without crashing any caller.

Items 1, 3, 4 are clean flips (default-inert escapes retained; revertible).

---

## Provenance / files
- `research/runners/g11_bg_runner.py` — items 1 (`log_polar_retina` default + CLI), 2 (`_n9_make_grid_code`
  helper + `nav_critic_grid_frontend`/`grid_*` kwargs + the `_n9_render` re-point + place_sensors sizing +
  CLI), 3 (`deterministic_read` kwarg + the restore-skip + CLI).
- `research/runners/nav_conv_merged_bridge.py` — items 1 (the merged-nav inherit note), 2
  (`nav_critic_grid_frontend` threaded through the builder + agent + the requires-place-selforg assert),
  5 (`co_resident_nav_critic` `None`-sentinel production default + auto-yield).
- `research/runners/_merged_td_cueshift_consolidation_derisk.py` — item 4 (the B4 cooled op-point CLI
  defaults).
- Findings: `2026-06-22-shortcut6-log-polar-render-derisk.md` (#6),
  `2026-06-22-shortcut5b-R1-grid-frontend-derisk.md` + `2026-06-22-shortcut5b-volley-normalization-close.md`
  (#5b grid + the δ-readout boundary), `2026-06-22-shortcut5b-determinism-deltabar-close.md` (#5b
  determinism), `2026-06-22-shortcut-B4-oppoint-r07-3of3.md` (B4).
- Plan: `docs/plans/2026-06-22-production-wiring-execution-plan.md` §1.3 / Chunk N.

---

## Verdict

**Chunk N COMPLETE — the merged "one brain" navigation cognition runs fully-spiking-by-default.** The
spiking limbic core (reward/value/dopamine) is the production agent default; the biology-faithful
log-polar SC retina is the library default (active on the spiking-SC path); the deterministic-read and
grid-frontend and B4-cooled mechanisms are wired as first-class flags. NO `sim/` edit, the no-confab moat
0-FA on every gate, nav-not-regressed (GREEN_INERT). The two honestly-scoped items (the grid front end's
δ-readout boundary, the limbic core's GREEN_INERT behavioral inertness) are documented, not hidden — the
host-Gaussian's full retirement-by-default and the clean-learned-δ are the recorded next frontier. The
next step (controller-run) is the COMBINED-CONFIG moat validation (§4 of the plan — all flips ON together).
