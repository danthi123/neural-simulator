# Route B (perception->compose) on the ONEBRAIN path — seam scoping (2026-06-25)

**Purity backlog #1 (the genuine cross-region residual per the biology + close-out audits): the
perception->compose host-`M` seam on the onebrain path.** The close-out audit
(`research/findings/raw/_closeout_to_full_capacity_audit.md`, the Route-B row + rank-4 item) flagged
that `co_resident_perception=True` + `co_resident_composer_kind='onebrain'` RAISES a guard
(`nav_conv_merged_bridge.py:1837-1840`) — i.e. Route B's perception-grounding is NOT wired through the
onebrain composer, only through the `rf` oracle composer.

**READ-ONLY scoping.** No edits, no runs, no webapp. Diagnosis -> portability -> ranked options ->
anti-cheats + de-risk + GO bars -> verdict.

---

## (1) DIAGNOSIS — exactly what the host `M` does, and why the onebrain path guards it out

### 1a. What Route B's grounding does (both modes)

Route B makes a PERCEIVED object composable. The agent navigates, ARRIVES at an object's cell, the
environment RENDERS the object's identity into a perception region (sensory render), and the agent reads
the object's LIVE spiking response OFF THE MERGED BRIDGE and writes a unit-phasor "grounded code" into
the composer's codebook so the FHRR bind/unbind/cleanup algebra can treat the percept as a concept.

There are TWO grounding modes (the cross-region host-`M` closure, 2026-06-24, finding
`2026-06-24-crossregion-onebrain-routeA-routeB-6seed-GO.md`):

- **`gen_spikes` (the DEFAULT, spikes-only).** The object is rendered as its structured-perception set
  into `gen_perception`; the LEARNED rate-Hebbian `gen_perception -> gen_concept` convergence fires the
  NMDA-integrated `gen_concept` assembly; the grounded code is a FIXED read-projection of `gen_concept`'s
  `cp_firing_states` (REAL spikes). The load-bearing percept->concept transform is the LEARNED SYNAPTIC
  convergence; the read-projection only FORMATS the concept spikes into a phasor (same legitimacy class as
  `rf_read_phases`). NO host quantity crosses regions.
  (`navigate_to_compose_then_answer.py:read_gen_concept_spikes`/`gen_grounded_phases`.)
- **`host_m` (the legacy, revertible A/B escape).** `composer.concepts[o] = angle(M @ cortex_it_rate)`,
  where `M` is a host-DESIGNED fixed random complex projection (`D x |cortex_it|`). This is the LAST
  genuine cross-region host quantity — retired as the default per
  `feedback_spiking_structure_must_self_organize` because `M` carries host-designed (not self-organized)
  structure. (`_step3_grounded_codes_production_composer_derisk._projection`/`grounded_phases`.)

The host `M`, then, is just the legacy `host_m` grounding's projection. It is NOT used by the default
behavioral runner (which defaults `gen_spikes`). **The seam is NOT "host `M` is still the default" — the
runner already closed that.** The seam is that the AGENT's `perceive_and_ground` only implements
`host_m`, AND that even `host_m` is wired to the wrong attribute on the onebrain composer.

### 1b. The two distinct sub-problems hiding behind the one guard

**(i) The wrong-attribute write (the literal cause of the guard).** The agent's `perceive_and_ground`
(`nav_conv_merged_bridge.py:2010-2035`) delegates to the standalone
`navigate_to_compose_then_answer._perceive_and_ground`, whose final write is
`composer.concepts[obj_word] = phases` (`navigate_to_compose_then_answer.py:363`).

- On the **rf** composer (`MergedRFComposer` <- `RFPhasorComposer`), `concepts` is the codebook the read
  path consults (`rf_phasor_composer.py:_filler_phases` line 259 reads `self.concepts[filler]`;
  `_cleanup_conj`/`_cleanup` read `self.concepts[w]`). So `composer.concepts[obj]=phases` lands exactly
  where the binds/cleanups read it. Route B works on the rf composer (validated 6-seed).
- On the **onebrain** composer (`CoResidentOneBrainComposer` <- `OneBrainComposer`), the actual codebook
  lives on the INNER `self.comp` (an `RFPhasorComposer`). Every store/cleanup goes through
  `self.comp`: `_compose_phases` binds `comp._filler_phases(fillers[i])` =
  `comp.concepts[filler]` (`one_brain_composer.py:405`), and `_cleanup_conj` reads
  `comp.concepts[concept_word]` (`one_brain_composer.py:460`; also `:840`, `:1133`, `:1136`, `:1165`,
  and the agent's own `_role_cleanup_scores` at `nav_conv_merged_bridge.py:2121` reads
  `comp.concepts[w]`). **There is NO `concepts` attribute on the `OneBrainComposer` wrapper** — so
  `composer.concepts[obj]=phases` creates a STRAY attribute that NOTHING reads. The grounded code is
  silently dropped; binds/queries use the object's ORIGINAL random/grounded-codes-init code. The compose
  would NOT reflect the percept (and `query_*` over a perceived object would behave as if it were never
  grounded). That is exactly the "not validated, guard it" situation the comment at
  `:1833-1836` describes ("`concepts` lives on `comp` for OneBrainComposer, so the host-M grounding would
  write a stray attr the reads never see").

**(ii) The agent has no gen_spikes path at all (the deeper residual).** Even fixing (i), the agent's
`perceive_and_ground` is HARD-CODED to `host_m` (`nav_conv_merged_bridge.py:2031` `h = {"grounding":
"host_m", ...}`), and the agent constructor never co-residents the generalization stack. The
spikes-only `gen_spikes` grounding (the actual host-`M` CLOSURE) lives ONLY in the standalone
`navigate_to_compose_then_answer.build_compose_bridge` (which passes `co_resident_generalization=True`
and builds the gen read-projection). The agent's `build_merged_nav_conv_bridge` call
(`:1879-1890`) does NOT pass `co_resident_generalization`, so there is no `gen_perception`/`gen_concept`
on the agent's merged bridge and no `handles["gen"]` to read concept spikes from.

**So "the onebrain path guards Route B out" decomposes into:**
- the LITERAL guard fires because of (i) — a one-line attribute mismatch that would SILENTLY corrupt the
  grounding (worse than crashing), so guarding is correct until it's wired;
- the PURITY residual is (ii) — the agent's grounding is host-`M`, not the spikes-only `gen_spikes`
  closure, regardless of composer kind. The runner closed (ii) for the `rf` composer on the behavioral
  surface; it is NOT closed on the agent for EITHER composer kind.

### 1c. Why this matters NOW (the default moved under the guard)

As of Closure 1 (2026-06-25, `nav_conv_merged_bridge.py:1634`), the agent's DEFAULT
`co_resident_composer_kind="onebrain"`. So the guard isn't an edge case — `MergedNavConvAgent(
co_resident_perception=True)` crashes on the PRODUCTION-DEFAULT composer. Route B is reachable on the
agent ONLY by explicitly downgrading to the `rf` oracle (`co_resident_composer_kind="rf"`), AND only via
host-`M`. The consolidated one brain (onebrain composer, the flagship) currently CANNOT perceive-and-
compose at all.

---

## (2) PORTABILITY — is the rf-path Route-B closure portable to the onebrain path?

**Yes for the grounding mechanism; the only real porting work is the codebook WRITE redirect + surfacing
the gen stack on the agent. The onebrain composer does NOT need a different grounding seam.** Three
reasons, each verified in code:

1. **The codebook is the SAME object.** `CoResidentOneBrainComposer.comp` IS an `RFPhasorComposer` —
   the very class Route B's grounding already writes to on the rf path. The grounded code is the same
   `D`-length phase vector. The only difference is one level of indirection: write `composer.comp.concepts[obj]`
   instead of `composer.concepts[obj]`. The bind/unbind/cleanup downstream is then byte-for-byte the rf
   path (the onebrain store binds `comp._filler_phases` -> `comp.concepts`).

2. **`grounded_codes` is an ALREADY-EXPOSED injection seam — the runtime write is its live twin.**
   `OneBrainComposer.__init__(grounded_codes=...)` (`one_brain_composer.py:114`, forwarded at `:266` to
   `RFPhasorComposer(grounded_codes=...)`, applied at `rf_phasor_composer.py:154-157`) is the SAME
   mechanism the production conversation uses to inject the 320 stream-learned codes. A perceived-object
   grounding is just that write happening LIVE (in-episode) for one word instead of at construction. So
   the onebrain composer is ALREADY designed to consume externally-supplied concept codes via `comp.concepts`
   — Route B grounding is a sanctioned use of an existing seam, not a new mechanism.

3. **The gen_spikes grounding is composer-agnostic.** `read_gen_concept_spikes` /
   `gen_grounded_phases` read `gen_concept` SPIKES off the merged bridge and FORMAT them into a `D`-phasor;
   nothing about that depends on whether the codebook lives on `composer` or `composer.comp`. The lesion
   (`lesion_gen_convergence`, sever the `gen_conv_mask` edges) and the spikes-only provenance assertion
   are likewise composer-agnostic — they operate on the bridge's `gen` handles, not the composer's storage
   layout.

**Caveat (the genuine porting nuance, not a blocker):** the onebrain composer caches CSRs
(`enable_csr_cache`, `_store_csr`, `_csr_cache`). A grounded code is consumed at STORE time
(`_compose_phases` reads `comp.concepts` fresh each store -> `comp._filler_phases`), so a code written
BEFORE the store is picked up correctly. But the cleanup-codebook conjugate phasors
(`_cleanup_conj` over `comp.concepts`) are used by `_read_block`/`_cleanup`; if any of those build a
cached codebook stack, a code RE-WRITTEN after a store (e.g. the lesion's re-ground) must invalidate that
cache. The rf path has no such cache. This is a "confirm the re-ground invalidates the cleanup cache"
de-risk item (anti-cheat 1 below), NOT a different seam. (Inspection: the onebrain cleanup builds the
codebook stack per-read from `comp.concepts`; the `_csr_cache`/`_store_csr` caches are keyed on
`store_conns`/bind structure, not the concept codebook — so a re-ground is expected to be seen. The
de-risk must confirm this empirically.)

**Conclusion:** the rf-path closure IS portable. The onebrain path needs (a) the codebook write redirect
to `comp.concepts`, and (b) the gen stack surfaced on the agent + a `grounding` selector — both reuse of
existing machinery, no new grounding seam, no `sim/` edit.

---

## (3) RANKED cheap-first options to close the seam (spikes-only on the onebrain path)

All reuse the validated cross-region machinery (`navigate_to_compose_then_answer`'s gen_spikes grounding +
lesion + provenance; `build_merged_nav_conv_bridge`'s `co_resident_generalization`). NO `sim/` edit in any
option (the gen stack, the masked RF ops, and the composer are all already built; this is runner-layer
wiring).

### Option 1 (RECOMMENDED) — redirect the codebook write + thread `co_resident_generalization`/`grounding` through the agent; default `gen_spikes` on the onebrain path

The full, principled close. Three small runner-layer changes:

- **1a. The codebook write redirect (fixes diagnosis (i)).** In the grounding write, target the composer's
  ACTUAL codebook: write `comp = getattr(composer, "comp", composer); comp.concepts[obj] = phases`
  (i.e. `composer.comp.concepts` for OneBrainComposer, `composer.concepts` for RFPhasorComposer). One
  branch in `navigate_to_compose_then_answer._perceive_and_ground` (or in the agent's `perceive_and_ground`
  wrapper) covers both composer kinds. Mirror the same redirect in the provenance assert
  (`_provenance_check` reads `composer.concepts[obj_word]` at `navigate_to_compose_then_answer.py:545/557`).
- **1b. Surface the gen stack + `grounding` on the agent (fixes diagnosis (ii)).** Add a
  `co_resident_generalization`/`grounding` kwarg to `MergedNavConvAgent.__init__`, pass
  `co_resident_generalization=(grounding=="gen_spikes")` into the `build_merged_nav_conv_bridge` call
  (`:1879`), capture `handles["gen"]`, build the gen read-projection (the agent already builds the host_m
  `_grounded_proj`; add the `_gen_read_projection` alongside), and make `perceive_and_ground` pass
  `grounding=self._grounding` + the gen handles to the standalone `_perceive_and_ground`. Default
  `grounding="gen_spikes"` when perception is on (the spikes-only closure).
- **1c. Replace the guard with the wired path.** Once 1a+1b land, `co_resident_perception=True` +
  `onebrain` is supported -> delete the `:1837-1840` `raise`. (Keep an assert that
  `co_resident_generalization` is on when `grounding=="gen_spikes"`, mirroring the runner.)

**Effort:** ~1 runner file (`nav_conv_merged_bridge.py`) + a 1-line branch in
`navigate_to_compose_then_answer.py`. The gen stack, the masked RF ops, the lesion, and the provenance are
all reused verbatim.
**Closes:** the genuine purity residual — the consolidated one brain (onebrain composer) grounds perceived
objects via SPIKES (the learned convergence), not host-`M`, on the agent default.

### Option 2 (CHEAPER, partial) — codebook write redirect only; keep agent grounding `host_m`, support `onebrain`+perception via host_m

Do only 1a (the attribute redirect). This makes `co_resident_perception=True` + `onebrain` WORK (no
crash, grounding actually lands) but still via host-`M`. Lets the onebrain composer perceive-and-compose
immediately, but does NOT close the purity residual (host-`M` is still the agent's grounding).

**Effort:** ~5 lines. **Closes:** the literal guard / the silent-drop bug. **Does NOT close:** the
host-`M` purity residual (the actual backlog #1). Useful only as a stepping stone or if the gen-stack
co-residence on the agent proves expensive; on its own it does not satisfy
`feedback_spiking_structure_must_self_organize`. **Recommend only as the first commit of Option 1**, not
as a terminal state.

### Option 3 (the "no agent change" alternative) — leave the agent guarded; treat the standalone behavioral runner as the Route-B-on-onebrain surface

Port gen_spikes-on-onebrain into `navigate_to_compose_then_answer` (its `build_compose_bridge` builds
`MergedRFComposer`; add a `--composer onebrain` that builds `CoResidentOneBrainComposer` + the codebook
redirect) and validate there, leaving the `MergedNavConvAgent` guard in place. This mirrors how Route B's
default-flip already lives at the behavioral-runner surface, not the agent constructor (the close-out
audit's "Route B's default-flip is at `build_compose_bridge`, not the agent").

**Effort:** ~1 runner (`navigate_to_compose_then_answer.py`) + the 1a redirect. **Closes:** demonstrates
+ validates Route-B-on-onebrain (spikes-only) end-to-end; the agent-constructor exposure (Option 1b)
remains a follow-on. **Trade-off:** the FLAGSHIP deployed agent still can't perceive-and-compose; the
capability is proven on the runner but not in the production agent. Good as the VALIDATION vehicle for
Option 1 (run the gate here first, then wire into the agent).

### Recommended path

**Option 3 as the gate -> Option 1 as the wire-in.** Validate gen_spikes-on-onebrain in the standalone
runner (cheapest place to get the 6-seed GO, where the gen stack co-residence is already proven), then
thread `co_resident_generalization`/`grounding` + the codebook redirect into the agent and delete the
guard. This is the same "validate on the runner, then flip the agent" discipline the cross-region A+B arc
used.

---

## (4) ANTI-CHEATS + cheap-first DE-RISK + GO bars

The anti-cheats already exist in `navigate_to_compose_then_answer.run_seed` and are composer-agnostic;
they MUST all pass with the onebrain composer (no relaxation). The de-risk is: run that runner with the
onebrain composer (Option 3) and assert the GO verdict.

### The anti-cheats (carry over verbatim; the grounding-is-SPIKES + lesion + moat are load-bearing)

1. **Grounding is SPIKES, not host-`M` (PROVENANCE — the load-bearing one).** `_provenance_check`
   (gen_spikes branch, `navigate_to_compose_then_answer.py:540-554`) must pass UNCHANGED:
   `source_kind == "gen_concept_spikes"` (NOT `cortex_it_rate_host_M`); the grounded code equals
   `gen_grounded_phases(gen_concept_spikes, gen_proj)` (now read off `composer.comp.concepts[obj]`); the
   `gen_concept` assembly actually SPIKED (`source.sum() > 0`). Plus: NO `composer.concepts[o] =
   host_fn(cortex_it_rate)` anywhere on the path (the anti-smuggle guard). On the onebrain composer add the
   explicit check that `composer.comp.concepts[obj]` (not a stray `composer.concepts[obj]`) is the grounded
   code — i.e. the redirect actually reached the codebook the binds read.
2. **LESION collapses the grounding.** `lesion_gen_convergence` (sever the `gen_perception->gen_concept`
   convergence via `gen_conv_mask`) + re-ground -> the held-out compose collapses toward chance
   (`lesion_clean <= floor + eps OR <= chance + 0.10`). This proves the compose rode the LEARNED synaptic
   convergence's spiking output, not a code-agnostic algebra trick. **Onebrain-specific:** confirm the
   re-ground after lesion is SEEN by the cleanup (the CSR/codebook-cache caveat in §2) — i.e. the lesion
   actually degrades the read, not masked by a stale cached codebook. If the cache hides the re-ground, the
   fix is a `_store_dirty`/codebook-cache invalidation on a `comp.concepts` write (runner-layer; confirm
   whether needed).
3. **MOAT preserved (the no-confab abstention).** Every unstored `(agent, action)` query returns None;
   a stored fact retrieves (positive control). `moat_ok == moat_tot AND pos == 1`, 0 false-accepts. NEVER
   weakened (a breach = HARD STOP). The onebrain composer's NATIVE `confidence_gate` abstention is the
   same `min(margin(agent),margin(action)) < g` mechanism; perceived-object facts must abstain identically.
4. **Compose != recall (held-out generalization).** Held-out (never-composed) perceived-object facts
   unbind correctly >> a memorization-floor recall baseline (`clean >= 0.90 AND clean >= floor + 0.30`,
   chance `1/N_OBJECTS = 0.25`). Plus ISO-PERCEPTION (no body -> never arrives -> 0 grounded) and the
   byte-identity / co-residence asserts (`composer._merged is bridge`, `cp_rf_w_re is not None` after a
   store).

### Cheap-first de-risk (the ONE run that settles it)

`python -m research.runners.navigate_to_compose_then_answer --grounding gen_spikes --composer onebrain
--seeds 42` (Option 3 wiring) on GPU. ~1 seed, single merged-bridge episode (the same scale as the
existing 6-seed Route-B run, ~minutes/seed). This exercises: the codebook write redirect (1a), the gen
stack co-resident with the onebrain composer, the spikes-only grounding, the lesion, the moat, and the
held-out compose — all four anti-cheats, with the onebrain composer's synaptic store + spiking cleanup.

### GO bars (must ALL hold; identical to the rf 6-seed GO)

- held-out compose **>= 0.90** AND **>= mem-floor + 0.30** (compose generalizes; recall does not);
- LESION (sever convergence) collapses compose toward chance (**<= floor + eps OR <= chance + 0.10**);
- MOAT **moat_ok == moat_tot**, **0 false-accepts**, **pos_recall == 1** (no breach);
- PROVENANCE: `source_kind == gen_concept_spikes`, grounded code == gen-spikes-derived, `gen_concept`
  spiked, codebook write reached `composer.comp.concepts` (NOT a stray attr), NO host-`M` quantity;
- ISO-PERCEPTION grounded == 0; byte-identity asserts hold;
- **then** the standing 6-seed gate (42/43/44/100/101/102) before the agent wire-in claims "works".

### The honest-negative shape (what would make this a BOUNDARY, not a closure)

The rf path got `gen_spikes` compose **1.000** 6/6. The onebrain composer adds the synaptic store + the
spiking cleanup (noisier than the rf numpy-kb cleanup). If the spike-grounded codes (0.92 cat-acc, per
the graded-propagation de-risk) are too noisy for the onebrain spiking cleanup to recover the perceived
object >> floor, the honest verdict is a BOUNDARY (the substrate's spiking cleanup can't read the
spike-grounded code at this fidelity) — and the host-`M` is NOT smuggled back to rescue the number. That
is a documented substrate limit, not a failure to wire.

---

## (5) VERDICT — closeable, and how cheaply

**CLOSEABLE, and cheaply — this is a runner-layer wiring close, NOT a deep boundary.** The guard exists
for a real reason (a silent codebook-drop bug, not a crash), but the fix is small and reuses fully-built,
separately-validated machinery:

- The **grounding mechanism** (gen_spikes: render -> learned convergence -> gen_concept spikes -> phasor)
  is already built, GO 6-seed on the rf composer, and composer-AGNOSTIC.
- The **onebrain codebook** is the SAME `RFPhasorComposer.concepts` the rf path writes to, one indirection
  away (`composer.comp.concepts`), and the `grounded_codes` seam shows the composer is already designed to
  consume externally-supplied concept codes through exactly that attribute.
- The **anti-cheats + de-risk + GO bars** already exist in `navigate_to_compose_then_answer` and are
  composer-agnostic; closing the seam = making them pass with the onebrain composer.
- **NO `sim/` edit** — the gen stack, the masked RF ops, and the composer are all built; this is runner
  wiring (the codebook redirect + threading `co_resident_generalization`/`grounding` through the agent +
  deleting the guard).

**The cheapest path:** Option 3 (validate gen_spikes-on-onebrain in the standalone runner, ~1 GPU seed,
all four anti-cheats) -> if GO, 6-seed -> Option 1 (thread the kwargs + the codebook redirect into the
agent, delete the `:1837-1840` guard). The single live risk is the spiking-cleanup fidelity on the
spike-grounded codes (the documented 0.92 cat-acc noise); if it doesn't clear the held-out bar that is an
HONEST BOUNDARY (a substrate cleanup limit), not a smuggle-host-`M`-back situation. The codebook-cache
invalidation on a live `comp.concepts` re-write (the lesion's re-ground) is the one onebrain-specific
detail to confirm in the de-risk; it is runner-layer if needed.

**Net:** backlog #1 (perception->compose host-`M` on the onebrain path) is a cheap, well-characterized
runner-layer close gated on a single GPU de-risk run — exactly the "validated-opt-in-but-not-on-the-
default-path" shape the close-out audit flagged, and closeable to the consolidated one brain without a
`sim/` edit.

---

### Key file:line references

- The guard: `research/runners/nav_conv_merged_bridge.py:1837-1840`.
- The agent default composer (now `onebrain`): `nav_conv_merged_bridge.py:1634`.
- The agent's `perceive_and_ground` (host_m only, delegates to the standalone):
  `nav_conv_merged_bridge.py:2010-2035`.
- The agent's host_m grounding-projection setup (no gen stack): `nav_conv_merged_bridge.py:1958-1968`.
- The standalone grounding write (the wrong attribute for onebrain):
  `navigate_to_compose_then_answer.py:363` (`composer.concepts[obj_word] = phases`).
- gen_spikes grounding (composer-agnostic): `navigate_to_compose_then_answer.py:read_gen_concept_spikes`
  (`:157`), `gen_grounded_phases` (`:197`), `lesion_gen_convergence` (`:205`), `_provenance_check`
  gen branch (`:540`).
- The onebrain codebook lives on `comp`: `one_brain_composer.py` `_compose_phases` (`:405`,
  `comp._filler_phases` -> `comp.concepts`), `_cleanup_conj` (`:460`, `comp.concepts[concept_word]`);
  `RFPhasorComposer._filler_phases` (`rf_phasor_composer.py:259`, `self.concepts[filler]`).
- The `grounded_codes` injection seam (the construction-time twin of a live grounding):
  `one_brain_composer.py:114/:266`, applied at `rf_phasor_composer.py:154-157`.
- The gen stack builder flag (exposed on the builder, NOT threaded to the agent):
  `nav_conv_merged_bridge.py:548` (`co_resident_generalization`), built at `:786/:796/:1116/:1169/:1229`;
  `handles["gen"]` at `:1230`; `gen_conv_mask` at `:1177`.
- The 6-seed rf Route-B GO: `research/findings/2026-06-24-crossregion-onebrain-routeA-routeB-6seed-GO.md`.
