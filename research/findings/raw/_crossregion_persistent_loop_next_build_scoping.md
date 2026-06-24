# Cross-region persistent integrated spiking loop — the next-smallest BUILD (SCOPING, read-only)

**Date:** 2026-06-24. **Role:** read-only scoping subagent (NO edits, NO runs, NO webapp — :8765 live).
**Trigger:** the composer-INTERNAL persistent loop is DONE (CYCLE 526 / `_persistent_loop_flat_derisk.json`:
`persistent_loop=True` wired into `OneBrainComposer`, flat who/what byte-identical GO, clause path already
on-substrate). The remaining persistent-loop gap is the **cross-region hand-offs** — make the nav↔conv influences
spike through synapses by DEFAULT, with no host round-trip BETWEEN regions.
**North-star:** owner #1 — the whole pipeline as ONE persistent interacting spiking loop
(`project_one_brain_integrated_pipeline_and_cleanup`, `feedback_move_everything_to_shared_spiking_substrate`).

> **HEADLINE (load-bearing for the controller).** The cross-region wiring is **already BUILT and GO** — it is
> just **default-OFF on the deployed agent**, and ONE of the four routes still crosses the code via a **host
> projection `M`**. Verified in current code:
> - **Route A (language→action)** — `MergedNavConvAgent.command_move` wired (CYCLE 518 `975b8eb4`), the
>   parser's action-role FIRING opens the `command_route` transmission gate (a 0/1 gate STATE, NOT a value),
>   lesion-load-bearing, nav Δ=0, moat 0-FA. **Default-OFF** (`co_resident_command_route=False`,
>   `nav_conv_merged_bridge.py:541` builder / `:1470` agent).
> - **Route B (perception→compose)** — `MergedNavConvAgent.perceive_and_ground` wired (same commit), GO 6-seed
>   (`navigate_to_compose_then_answer`). **Default-OFF** (`co_resident_perception=False`,
>   `nav_conv_merged_bridge.py:540` builder / `:1470` agent) **AND it carries the residual HOST round-trip**:
>   `composer.concepts[o] = grounded_phases(rate, proj)` = `angle(M @ to_host(live cortex_it rate))`
>   (`navigate_to_compose_then_answer.py:212-213`; the agent builds `M` at `nav_conv_merged_bridge.py:1718`).
>   This is the **only genuine cross-region host cheat left** (I-4-resid / N-4 / H-2).
> - **Route C (DA/limbic→composer, read-side)** — `enable_da_salience_gate=True` is the **PRODUCTION DEFAULT**
>   (`nav_conv_merged_bridge.py:1471`), and the DA SOURCE (`co_resident_nav_critic`) resolves True by default
>   (`:1573-1575`). **The limbic core ALREADY reaches the conversational cortex by default** (I-4-a / I-7-a DONE,
>   CYCLE 512). The WRITE-side (I-7-b encoding hook) is wired default-OFF + inert on the numpy-kb composer.
> - **Route D (parser→composer comprehension hand-off, I-5)** — the parser FIRES the role synaptically, but
>   WHICH word→WHICH bind is a Python `{role:word}` dict inside `OneBrainComposer.hear` (the `RFPhasorComposer`-
>   based `MergedRFComposer` on the merged bridge is even further from synaptic — its kb is numpy). De-risked
>   GO standalone (`hear_synaptic`), NOT ported to the merge.
>
> **⇒ The next-smallest BUILD is a DEFAULT-FLIP + one host-`M` replacement, NOT new science.** The cheapest
> bankable move is to **flip Routes A+B on by default** (regression-gated). The genuine residual-closure is
> **replace the host `M` in Route B with the already-GO `gen_perception→gen_concept` Hebbian convergence**
> (spikes-only grounding) — the mechanism is GO (`2026-06-16-generalization-graded-propagation.md`), it is built
> into the builder (`co_resident_generalization`), it is just NOT exposed on the agent NOR wired to ground the
> composer code from `gen_concept` SPIKES.

---

## 1. RANKED cross-region hand-offs (leverage × de-risk-done × smallness)

> For each route: is the SPIKE hand-off de-risked (cite GO)? WIRED into production or default-off (cite
> file:line)? smallest BUILD to make it no-host-round-trip spiking?

### RANK 1 — Route B: perception→compose, CLOSE THE HOST `M` (I-4-resid / N-4) + flip default-on
**Leverage HIGHEST (it is the ONLY genuine cross-region HOST cheat left). De-risk-done HIGH. Smallness MEDIUM.**

- **Spike hand-off de-risked?** TWO halves, both GO:
  - the perception→compose BEHAVIOR: `navigate_to_compose_then_answer` 6-seed GO
    (`2026-06-16-navigate-to-compose-then-answer.md`) — held-out compose 1.000 ≫ mem-floor 0.444, lesion
    collapses, moat 6/6. BUT it grounds via host `M`.
  - the SPIKES-ONLY grounding replacement: `gen_perception→gen_concept` learned Hebbian convergence FIRES
    category-correct SPIKES for a held-out novel percept — `2026-06-16-generalization-graded-propagation.md`
    (3-seed GPU GO, real `cp_firing_states`, NMDA-integrated, derangement collapses, moat intact) +
    `-onsubstrate-convergence.md` (held-out cat-acc 0.92 on spikes). The capstone
    (`2026-06-16-generalization-capstone-verbalize.md`) closed perceive-novel→generalize→recall end-to-end via
    the spiking concept keying the validated composer.
- **Wired into production?** **NO — default-OFF + host round-trip present.** Builder
  `co_resident_perception=False` (`nav_conv_merged_bridge.py:540`); agent default `co_resident_perception=False`
  (`:1470`). The grounding is HOST: `composer.concepts[o] = grounded_phases(rate, proj)`
  (`navigate_to_compose_then_answer.py:212-213`), `M = _projection(D, it_size, seed)`
  (`nav_conv_merged_bridge.py:1718` / `navigate_to_compose_then_answer.py:145`). The `gen_*` convergence stack
  IS built (`co_resident_generalization`, builder `:543/:781/:1101/:1154/:1214`, `train_convergence` import
  `:438`) but is **NOT exposed on `MergedNavConvAgent`** (zero matches) and **NOT wired to ground the composer
  from `gen_concept` spikes**.
- **Smallest no-host-round-trip BUILD:** (1) expose `co_resident_generalization` on `MergedNavConvAgent`;
  (2) add a fixed route `gen_concept → rf` (or read `gen_concept`'s firing into the composer code) so the
  percept fires `gen_perception` → the LEARNED convergence fires `gen_concept` → a FIXED projection grounds
  `composer.concepts[o]` from `gen_concept` SPIKES (not `M @ rate`); (3) default-flip the route on after the
  spikes-only provenance gate passes. **This retires the last cross-region host cheat AND makes Route B
  interact by default.** See §2.

### RANK 2 — Route A: language→action, flip default-on
**Leverage HIGH (the "spoken command steers the body" interaction — the cleanest one-brain story). De-risk-done HIGH. Smallness HIGHEST.**

- **Spike hand-off de-risked?** GO 6-seed — `spoken_instruction_nav` (`2026-06-10-spoken-instruction-nav-GO.md`):
  the parser's action-role FIRING opens `COMMAND_GATE` on the LEARNED `language_input→cortex_X` route; the
  command is the only goal ⇒ load-bearing + lesion-confirmed + SCRAMBLE-regressed + ISOLATED-NAV/CONV both fail.
- **Wired into production?** **Built as a method, default-OFF.** `MergedNavConvAgent.command_move`
  (`nav_conv_merged_bridge.py:1729`), `_couple_command_gate` (`:1721`, `couple_gate_to_indices` over the parser
  action block — the in-substrate 0/1 gate-from-firing), `lesion_command_route` (`:1752`); validated CYCLE 518
  (`975b8eb4`: lesion-load-bearing, nav Δ=0, moat 0-FA). Default `co_resident_command_route=False` (builder
  `:541` / agent `:1470`).
- **Smallest BUILD:** a **DEFAULT-FLIP** — set `co_resident_command_route` default-on (regression-gated). The
  route is ALREADY spiking (the gate is a firing STATE, no host value crosses); provenance is already asserted
  (`spoken_instruction_nav.provenance_facts` — "no parser-derived value written to any nav drive"). **No
  host-round-trip to remove; pure consolidation.** *Reuse-by-import, NO `sim/` edit.*

### RANK 3 — Route C: DA/limbic→composer (read-side DONE; encoding-side wire-up)
**Leverage MEDIUM (it is mostly DONE — the limbic core already reaches the cortex). De-risk-done HIGH. Smallness HIGH.**

- **Spike hand-off de-risked?** Read-side GO (`_da_composer_salience_cleanup_derisk`, 6-seed precision +
  lesion-clean; Vijayraghavan/Arnsten D1 inverted-U). Encoding-side GO
  (`_burndown_I7_dopamine_encoding_deploy_derisk` + `2026-06-19-dopamine-encoding-gain-derisk.md`, Lisman-Grace
  D.16; the stored block magnitude == g, lesion-confirmed, moat 0-FA).
- **Wired into production?** **Read-side = PRODUCTION DEFAULT-ON.** `enable_da_salience_gate=True`
  (`nav_conv_merged_bridge.py:1471`); DA source `co_resident_nav_critic` default-resolves True (`:1573-1575`).
  Byte-identical at rest (DA==baseline ⇒ g_eff=g0 floor), engages only on a salient/high-DA turn; moat-safe by
  construction (a higher gate only TIGHTENS abstention). **The limbic core reaches the conversational cortex by
  default — Route C is the one route that already INTERACTS in the deployed default.** Encoding-side
  (`enable_da_encoding_gain`) default-OFF (`:1472`) AND **behaviorally INERT** on the merged default composer
  (`MergedRFComposer`/`RFPhasorComposer` numpy-kb stores PHASES = magnitude-invariant; the hook is load-bearing
  only on a MAGNITUDE-storing `OneBrainComposer`/substrate-store — the agent docstring `:1530-1535` flags this).
- **Smallest BUILD:** the encoding-side is **not a flip on the current default composer** — it needs the
  `OneBrainComposer` (magnitude-storing) co-resident on the merged bridge to be load-bearing (it is inert on the
  numpy-kb `MergedRFComposer`). ⇒ this is **coupled to the composer-substrate consolidation** (swap
  `MergedRFComposer` → an `OneBrainComposer` co-resident), which is a larger, separate arc. The read-side is
  DONE; the write-side is a deploy-smoke ONLY AFTER the magnitude-storing composer is co-resident. **Lower
  priority** (the high-value piece — limbic reaches cortex — is already live).

### RANK 4 — Route D: parser→composer comprehension hand-off (I-5)
**Leverage MEDIUM (purity — the WHICH-word-to-WHICH-bind selection). De-risk-done MEDIUM. Smallness LOW (on the merge).**

- **Spike hand-off de-risked?** GO standalone — `unified_brain_bridge.hear_synaptic` + `_op_synaptic`
  (`:447/:509`): per-role `role_route_<R>` transmission gate coupled to the parser ensemble routes the firing
  → the composer role bank topographically (comprehend→latch→act). CYCLE 512 logged I-5-a GO.
- **Wired into production?** **NO.** `OneBrainComposer.hear` selects the role→word map via a Python dict; the
  merged path's `MergedNavConvAgent.hear` (`nav_conv_merged_bridge.py:1854`) delegates to `self.parser.parse`
  → `self.composer.store(roles[...])` (string labels). The parse FIRES synaptically; the bind ASSIGNMENT is host.
- **Smallest BUILD:** port `hear_synaptic` onto the merge for the co-resident composer. **The snag (honest
  owner-steer):** `hear_synaptic` was built for the `CoreSimComposer` ±1 Hadamard role bank; the merged composer
  is the `MergedRFComposer` (RF phasor) / `OneBrainComposer` — the role drive must enter the **RF bind**, not a
  Hadamard AND bank. This is a genuine NEW wiring (re-express the role route for the RF composer), not a
  pure default-flip. ⇒ **defer behind Routes A+B** (it is purity polish on an already-synaptic-comprehension
  parser; the residual is the host *assignment* dict, not a host *quantity*).

---

## 2. THE RECOMMENDED NEXT BUILD — Route B host-`M` closure + Routes A+B default-flip

**Recommendation:** do the cheapest bankable consolidation **first** (flip Routes A+B on by default —
regression-gated), then the genuine residual-closure (**replace Route B's host `M` with the GO
`gen_concept`-spikes grounding**). Both can be one coordinated arc; the default-flip is the floor, the host-`M`
replacement is the ceiling.

### 2.1 Exact file:line targets

**The host round-trip to replace (the genuine cross-region cheat):**
- `navigate_to_compose_then_answer.py:212-213` —
  `phases = grounded_phases(rate, proj)` ; `composer.concepts[obj_word] = phases`
  (i.e. `composer.concepts[o] = angle(M @ to_host(live cortex_it rate))`).
- the projection built at `nav_conv_merged_bridge.py:1718` (`self._grounded_proj = _projection(_D, it_size, seed)`)
  and `navigate_to_compose_then_answer.py:145` (`proj = _projection(D, it_indices.size, seed)`).
- the provenance assertion that currently CONFIRMS the host round-trip (must be REPLACED with a spikes-only one):
  `navigate_to_compose_then_answer.py:380` (`assert np.allclose(composer.concepts[obj_word], ground_phases)`).

**The default-OFF flags to flip (the consolidation floor):**
- `nav_conv_merged_bridge.py:540` — `co_resident_perception: bool = False` (builder).
- `nav_conv_merged_bridge.py:541` — `co_resident_command_route: bool = False` (builder).
- `nav_conv_merged_bridge.py:1470` — `co_resident_perception=False, co_resident_command_route=False` (agent).
  (Pattern to follow: the `co_resident_nav_critic=None`-sentinel production-default-ON at `:1573-1575`, which
  already flipped the DA route on while keeping a revertible explicit-False escape.)

**The GO machinery to wire in (the spikes-only grounding):**
- `co_resident_generalization` builder kwarg (`nav_conv_merged_bridge.py:543`) + the built stack
  (`gen_perception`/`gen_concept`/`gen_fact` regions, the plastic `gen_perception→gen_concept` convergence,
  the FIXED `gen_concept→gen_fact`, `:781/:1101/:1154/:1214`; `train_convergence` `:438`). Constants
  `GEN_PERCEPTION`/`GEN_CONCEPT` (`:64-65`). **Currently NOT exposed on `MergedNavConvAgent`** — expose it,
  then ground `composer.concepts[o]` from `gen_concept`'s firing (a FIXED projection read off the spikes),
  not from `M @ rate`.

### 2.2 The de-risk it builds on
- Route B behavior: `2026-06-16-navigate-to-compose-then-answer.md` (6-seed GO).
- Spikes-only grounding: `2026-06-16-generalization-graded-propagation.md` (held-out novel percept → `gen_concept`
  SPIKES, 3-seed GPU GO) + `-onsubstrate-convergence.md` (cat-acc 0.92 on spikes) +
  `-capstone-verbalize.md` (perceive-novel→generalize→recall end-to-end, hybrid 0.92).
- Route A: `2026-06-10-spoken-instruction-nav-GO.md` (6-seed GO).
- The default-flip pattern + the regression-gate precedent: the DA-route default-ON
  (`2026-06-24-closeout-audit-default-on.md` / CYCLE 512) — `test_nav_conv_merged_agent` 8/8 +
  `test_nav_conv_step2b_coresident` 7/7 all GREEN under the flipped default.

### 2.3 The BYTE-IDENTITY / behavioral-equivalence GATE
- **Default-flip (Routes A+B) — byte-identity gate:** `test_nav_conv_merged_agent` 8/8 (incl. the three
  `is None` no-confab assertions) + `test_nav_conv_step2b_coresident` 7/7 must pass under the new default; the
  navigation score is **byte-identical (Δ=0)** to the pre-flip build (the routes are array-disjoint from the
  composer's `cp_rf_w_*` and the parser slice — same guarantee the CYCLE-518 wiring already proved). The
  standalone runners `spoken_instruction_nav` + `navigate_to_compose_then_answer` still GO under the default build.
- **Host-`M` → `gen_concept`-spikes replacement — behavioral-equivalence gate:** the held-out perceived-object
  compose recovers **≫ memorization floor** (the GO bar, ~1.000 vs ~0.444) with the grounding read off
  `gen_concept` SPIKES (the new provenance), **lesion** (sever the `gen_perception→gen_concept` convergence)
  collapses it, **moat 0-FA** on unstored. (CPU cheap-first first, then GPU 3/6-seed — the de-risk's bars.)

### 2.4 ANTI-CHEATS (the integration must be SYNAPTIC)
1. **PROVENANCE — no host quantity smuggled across regions (the load-bearing one for this build).** Grep the
   integrated path: there must be **no** `composer.concepts[o] = host_fn(to_host(<cortex_it/gen_*>))` and no
   `cp_external_input_current[<B>] = f(to_host(<A>))`. The ONLY legitimate host writes are the environment
   rendering the object into `cortex_it`/`gen_perception` (sensory render) + the body moving on the
   motor/sel winner. The cross-region coupling must be a **fixed synaptic route** (the `gen_concept→rf`
   grounding) or a **0/1 gate STATE from firing** (`command_route`), never a value copy. **The current
   `navigate_to_compose_then_answer.py:380` assertion CONFIRMS the host round-trip — it must be REPLACED with a
   spikes-only one** (the grounded code is read from `gen_concept`'s `cp_firing_states`, not `M @ rate`). This
   is the exact anti-cheat the scoping flags (I-4-resid / N-4 / H-2).
2. **LESION = the interaction vanishes.** Sever the `gen_perception→gen_concept` convergence (Route B) / never
   open `command_route` / lesion its synapses (Route A): the compose must collapse to the mem-floor/chance, the
   commanded move must collapse to chance. Both routes already carry this control (Route A `lesion_command_route`
   `:1752`; Route B the convergence-derangement collapse).
3. **BOTH-BRAINS-REQUIRED (isolated controls fail).** Route A: ISOLATED-NAV (no parser drive → gate closed →
   chance) + ISOLATED-CONV (no body) each fail. Route B: ISO-perception (no body, no navigation) grounds
   nothing → compose at floor; the `navigate_to_compose_then_answer` 6-seed already showed ISO-perception
   grounds 0.
4. **MOAT-PRESERVED — HARD.** No abstention the host returned may become a false-accept on any flipped/grounded
   path. Every regression re-runs the three `is None`/"unknown" assertions; the spikes-only grounding must keep
   moat 0-FA on unstored cues. Per `feedback_moat_not_hard_lossy_memory_ok` the moat is kept-where-free; here it
   is free (the grounding change is upstream of the cue-match abstention, which is unchanged).
5. **(Route A) SCRAMBLE.** Permuting word→direction must regress accuracy-vs-commanded (the agent follows what it
   COMPREHENDS) — already in `spoken_instruction_nav`.

### 2.5 `sim/`-edit flag
- **Routes A+B default-flip + the `gen_concept`-spikes grounding = reuse-by-import, NO `sim/` edit.** All
  primitives are public/borrowable: `couple_gate_to_indices` + `set_transmission_gate` (`sim/bridge.py:3179-3229`)
  for Route A; the `co_resident_generalization` stack + `train_convergence` + a FIXED route for the grounding;
  the `gen_concept` spikes are read via the existing `cp_firing_states` / a fixed projection (no new kernel).
  The NMDA-on-`gen_concept` propagation already works through the framework's per-region NMDA mask
  (`sim/bridge.py:1212-1221`, already exercised by `co_resident_generalization`). **No protected-`sim/` change.**

---

## 3. HONEST: reuse-by-import vs new mechanism, + owner-steer flags

**Reuse-by-import (de-risk done → just wire it):**
- **Route A default-flip** — pure consolidation; `command_move`/`_couple_command_gate`/`lesion_command_route`
  already on the agent (CYCLE 518). Flip `co_resident_command_route` default-on, regression-gate. **NO `sim/`
  edit. The cheapest bankable win.**
- **Route B default-flip** — pure consolidation; `perceive_and_ground` already on the agent. (But it still uses
  host `M` until the next item lands.) Flip `co_resident_perception` default-on, regression-gate.
- **Route C read-side** — ALREADY default-ON (the limbic core reaches the cortex by default). Nothing to build;
  confirm-only.

**Reuse-by-import but a real (small) wiring job, NOT a pure flip:**
- **Route B host-`M` → `gen_concept`-spikes grounding** — the MECHANISM is GO (`gen_concept` fires
  category-correct spikes for held-out percepts), the STACK is built (`co_resident_generalization`), but it is
  NOT exposed on the agent and NOT wired to ground the composer from spikes. The new work: expose the kwarg,
  add the FIXED `gen_concept → rf`/codebook grounding route, replace the host write + its provenance assertion.
  **NO `sim/` edit; the genuine residual-closure that retires the last cross-region host cheat.**

**New mechanism / deferred (NOT this build):**
- **Route D (parser→composer synaptic assignment)** — `hear_synaptic` is GO but built for the ±1 Hadamard role
  bank; re-expressing the role route for the RF composer is a genuine NEW wiring. Defer behind A+B.
- **Route C write-side (encoding hook load-bearing)** — needs a magnitude-storing `OneBrainComposer` co-resident
  on the merged bridge (the numpy-kb `MergedRFComposer` makes the hook inert). That is the **composer-substrate
  consolidation** arc (swap `MergedRFComposer` → co-resident `OneBrainComposer` with `persistent_loop=True`),
  separate and larger.

**Owner-steer flags:**
1. **Scope of the default-flip** — flip BOTH Routes A+B on by default now (regression-gated, the recommendation,
   matching the DA-route precedent that already flipped C on), vs. keep them opt-in and only land the host-`M`
   closure. The recommendation is to flip A immediately (zero residual) and flip B together with the host-`M`
   closure (so the deployed-default Route B is spikes-only, not host-`M`).
2. **The composer-substrate consolidation** (`MergedRFComposer` → co-resident `OneBrainComposer`/`persistent_loop`)
   — this is the bridge that (a) makes Route C's write-side load-bearing and (b) brings the proven composer-
   internal persistent loop ONTO the merged bridge. It is larger (re-host the merged composer) and is the
   natural FOLLOW-ON after A+B. An explicit owner call on whether to fold it into this arc or sequence it after.
3. **Route D** (synaptic comprehension assignment for the RF composer) — purity polish; an owner call on
   priority vs. the artificial-life horizon (Rank-2 next-frontier).

---

## 4. BOTTOM LINE

The cross-region "true one brain" is **already built and GO** — three of the four routes are wired (A
`command_move`, B `perceive_and_ground`, C the DA salience gate), and **Route C (limbic→cortex) is already the
production default**. The remaining persistent-loop gap is **(i) the two GO routes A+B are default-OFF on the
deployed agent**, and **(ii) Route B still grounds via a host projection `M`** — the one genuine cross-region
host cheat left.

**Recommended next BUILD:** flip **Route A (language→action) on by default** (the cheapest bankable
consolidation — zero residual, NO `sim/` edit), then close **Route B's host `M`** by grounding
`composer.concepts[o]` from the already-GO `gen_perception→gen_concept` Hebbian-convergence SPIKES (exposing
`co_resident_generalization` on `MergedNavConvAgent` + a fixed `gen_concept→rf` grounding route) and flip Route B
on with the spikes-only provenance. **File:line:** flip `nav_conv_merged_bridge.py:540-541` (builder) / `:1470`
(agent); replace the host write at `navigate_to_compose_then_answer.py:212-213` + its provenance assertion
`:380`; wire the GO stack at builder `:543/:781/:438`. **De-risk:** `2026-06-10-spoken-instruction-nav-GO.md`
(A) + `2026-06-16-navigate-to-compose-then-answer.md` (B behavior) +
`2026-06-16-generalization-graded-propagation.md` (B spikes-only grounding). **GATE:**
`test_nav_conv_merged_agent` 8/8 + `test_nav_conv_step2b_coresident` 7/7 + nav Δ=0 (byte-identity) for the flip;
held-out compose ≫ mem-floor with grounding off `gen_concept` SPIKES + lesion-collapses + moat 0-FA for the
host-`M` closure. **Anti-cheats:** PROVENANCE (replace the host-`M` assertion with a spikes-only one — no host
quantity across regions), LESION-vanishes, BOTH-BRAINS-required, MOAT-PRESERVED-HARD, SCRAMBLE (Route A).
**`sim/`-edit:** NONE (reuse-by-import). **Owner-steer:** the default-flip scope (A now, B with the host-`M`
closure) + the composer-substrate consolidation (`MergedRFComposer` → co-resident `OneBrainComposer`, which
unlocks Route C's write-side + the composer-internal persistent loop on the merge) + Route D priority.
