# Tier-3 Option 3 "cross-modal one animal" — deep-research scoping (READ-ONLY)

**Date:** 2026-07-01 (autonomous loop; owner-directed Tier-3 follow-on scoping)
**Type:** Deep-research + reference-catalog scoping. **READ-ONLY — NO code / `sim/` / GPU edit.** This doc isolates
the single genuine residual for Option 3 and recommends the cheapest brain-based de-risk. It is the standing
"deep-research FIRST at a new direction" opening move ([[project_actual_goal_artificial_life_brain_analogue]],
[[project_post_conversational_roadmap_tiers]] Tier 3, [[feedback_deep_research_at_roadblocks]]).
**Predecessors (do NOT re-derive):** the capstone scoping `2026-06-30-tier3-artificial-life-capstone-scoping.md`
§4 Option 3 (which flagged this as the "Phase-3.1" cheap FOLLOW-ON, not a life); the two CLOSED synthesis slices
`2026-06-30-tier3-live-and-remember-first-slice.md` (Option 1, 6/6 GO) +
`2026-06-30-tier3-option2-develop-with-a-body-first-slice.md` (Option 2, 6/6 GO). Also grounds on the already-built
DA→composer routes `2026-06-30-tier2-6-limbic-to-composer-scoping.md` (roadmap #6, A/B/C) and the drive de-risks
`2026-06-17-homeostatic-spiking-drive-mechanism-GO.md` / `2026-06-20-tier3-spiking-living-loop-derisk.md`.

**Terms (defined once).** *Moat* = the no-confab abstention: the conversational agent returns `None` on a cue that
matches no stored fact — 0 false-accepts is a HARD standing rule. *DA / dopamine* = the shared spiking
`dopamine` neuromodulator concentration on the merged bridge (0–2, tonic baseline 0.5). *Moat-gate* =
`MergedNavConvAgent._da_confidence_gate` (`nav_conv_merged_bridge.py:2136`), which maps DA onto the composer's
cue-role confidence gate `g_eff` — CLAMPED so it can only RAISE the gate (tighten abstention), never lower it.
*Drive / hunger* = the co-resident 2-pool spiking hunger organ (`drive_agrp` = AgRP hunger / `drive_pomc` = POMC
satiety), driven by an interoceptive deficit current and read by `drive_agrp` firing rate.

---

## 1. TOP-LINE — is Option 3 largely done + cheap? YES. The genuine residual is ONE production-rule line.

Applying the SURPASS practice (isolate + QUANTIFY the genuine residual; don't accept a vague "it needs a hunger→DA
mechanism"):

**Almost the entire property is already built, and moat-safe by construction.** Of the four pieces Option 3 needs —
(i) a shared spiking DA, (ii) a hunger drive on the same brain, (iii) a moat-gate that reads DA and can only
tighten, (iv) a link making hunger raise DA — **three of the four already exist and are validated on the merged one
brain**, and the fourth (the link) is a **one-line runner-layer neuromodulator-config addition that needs NO `sim/`
edit** because the required production-rule machinery already ships.

| piece | status | evidence (file:line) |
|---|---|---|
| **(i) shared spiking DA** on the merged bridge | **EXISTS, default-ON** | the `dopamine` modulator over `[snc]` (or `[limbic_snc]`) — `nav_conv_merged_bridge.py:1046`, driven by SNc firing `from_region_firing_signed` (`:1049`); `co_resident_nav_critic` resolves to `True` by production default (`:1801-1803`) |
| **(ii) hunger drive** co-resident, spiking, deficit-tracking | **EXISTS, 6/6 GO** | `drive_agrp`/`drive_pomc` (`:882-889`); corr(deficit, `drive_agrp` firing) **+0.98** on the merged bridge (Option-1 gate 2, window=40) |
| **(iii) moat-gate** reading DA, clamped-to-tighten | **EXISTS, default-ON, wired into the read path** | `_da_confidence_gate` (`:2136`) → `da_to_gate` = `clip(g0, g_cap, g0+k·(DA−baseline))` (`_da_composer_salience_cleanup_derisk.py:216`); called live in `what_does`/`who_does`/`is_it_true`/`describe` (`:2342/2353/2375/2383`); `enable_da_salience_gate=True` default (`:1643`); GO 6-seed roadmap-#6 |
| **(iv) hunger→DA link** | **MISSING — the genuine residual** | `drive_agrp` has **ZERO out-edges** (`:890`) and feeds **NO** modulator; the `dopamine` production rule reads `[snc]`, not `[drive_agrp]` (`:1049`) |

**The genuine residual, quantified:** it is **exactly the missing arrow from (ii)→(i)** — one additional
`ProductionRule(rule_type="from_region_firing", source_regions=["drive_agrp"], …)` appended to the existing
`dopamine` `NeuromodulatorConfig.production_rules` list (the manager sums a modulator's production rules each step —
`sim/neuromodulators.py:264-265`). Everything downstream of DA (the moat-gate, its clamp, its wiring into the read
path) is already built and already moat-safe. **This is not a new mechanism class; it is wiring a validated signal
(hunger firing) into a validated modulator (DA) through a production-rule type that already exists and is already in
use for the SNc.** ⇒ Option 3 is **largely done + cheap** — the smallest, most self-contained Tier-3 slice remaining.

---

## 2. DIAGNOSIS — the exact hunger→DA gap + the ranked link options

### 2a. The exact gap
The co-resident drive slice is deliberately **maximally nav-inert**: `drive_agrp`/`drive_pomc` are built with
`internal_density=0` and **no pathways at all** (`nav_conv_merged_bridge.py:878-890`, comment at `:864-867`: "driven
by external interoceptive current + READ by firing rate, so they need NO internal pathways and NO neuromodulator").
The hunger is consumed today only by the survival reward gate (`run_moving_goal_episode(homeostatic_hook=…)`:
`reward *= hunger`). It never touches the `dopamine` concentration. So a hungry brain's DA is UNCHANGED from a sated
brain's DA, and the moat-gate `g_eff` therefore does not shift with hunger. **The hunger→DA arrow is the one and
only missing edge for Option 3.**

Note the DA source is present on the merged one brain by the production default: `co_resident_nav_critic=None`
resolves to `True` unless a mutually-exclusive critic (`co_resident_limbic` / `co_resident_td_cueshift`) is explicitly
requested (`:1801-1803`). The Option-1 runner (`_tier3_live_and_remember_derisk.py:421-424`) passes only
`co_resident_composer=True, co_resident_drive=True` with no critic flag → the nav critic (hence the `dopamine`
modulator over `[snc]`) is live. So Option 3's base bridge already carries a shared DA; only the drive→DA arrow is
absent.

### 2b. The ranked link options

**(b) — a second `from_region_firing` production rule over `[drive_agrp]` on the existing `dopamine` modulator — CHEAPEST + MOST BRAIN-BASED + NO `sim/` EDIT. ★ RECOMMENDED.**
- **Does the subsystem support a firing-driven production rule?** **YES — and it is already the mechanism the SNc
  uses.** `sim/neuromodulators.py` ships `from_region_firing` (one-sided, `:736-772`) and
  `from_region_firing_signed` (two-sided, `:774-817`): each reads the mean firing fraction across `source_regions`
  off `bridge.cp_firing_states` + `region_manager.indices`, maintains an EMA over `window_ms`, and produces
  `sensitivity·(rate_ema − threshold)`. This is NOT the `from_novelty` stub (which is RESERVED and emits 0, `:732`).
  The `dopamine` modulator on the merged bridge already carries a `from_region_firing_signed` rule over `[snc]`
  (`nav_conv_merged_bridge.py:1049`). Because a `NeuromodulatorConfig` holds a *list* of production rules and the
  manager SUMS them each step (`neuromodulators.py:264-265`), a SECOND rule over `[drive_agrp]` can be appended —
  the DA concentration then = the SNc term (unchanged) + a hunger term. Use the ONE-SIDED `from_region_firing`
  (hunger can only ADD to DA, `max(0,·)` built in) with `threshold` set to `drive_agrp`'s tonic/quiescent firing
  fraction so a sated (low-`drive_agrp`) brain contributes ~0. **This is a runner-layer edit** (`nav_conv_merged_bridge.py`
  lives in `research/runners/`, NOT `sim/`) — no protected-code change. **NO `sim/` edit predicted.**
- **Why brain-based:** the arrow is exactly the hypothalamus→midbrain-DA projection biology posits for incentive
  motivation (§3): AgRP/lateral-hypothalamic drive raises the reward/salience DA signal. The link is realized as
  spikes-read-firing (`drive_agrp`'s `cp_firing_states`) → a modulator concentration, the same read the SNc uses —
  no host quantity is invented.
- **Cost:** SMALL. One additive default-off builder kwarg (e.g. `drive_to_da=False`) appends the rule when both
  `co_resident_drive` and a DA source are present; default OFF = byte-identical to Option-1's 6/6 GO. A `sensitivity`
  + `threshold` are the only two tunables (their scale is bounded — DA is capped at 2.0 and the moat-gate is clamped
  at `g_cap`, so an over-strong link cannot loosen the moat, only saturate the tightening).

**(a) — a small additive `drive_agrp → snc` (SNc/dopamine-region) EXCITATORY `RegionPathway`.**
- **Mechanism:** a runner-layer `BrainRegion` pathway (like every other co-resident slice's internal edges), so
  hunger firing directly excites the SNc pool → the *existing* `from_region_firing_signed` rule over `[snc]` reads
  the elevated SNc rate → DA rises. Also a runner-layer edit, NO `sim/`.
- **Why NOT the pick:** it BREAKS the drive slice's validated "maximally nav-inert, ZERO out-edges" invariant
  (`:867-868`) — it adds a `cp_connections` edge from the drive into the nav/critic subgraph, which the co-residence
  anti-cheat and the byte-isolation of the frozen conversational slice are designed to EXCLUDE. It also entangles the
  hunger signal with the SNc's RPE dynamics (a plastic/excitatory afferent onto a dopamine pool that is
  simultaneously computing reward-prediction error), risking a confound the moat-safety argument would then have to
  re-establish. Option (b) keeps the drive slice pristine and adds the influence purely at the neuromodulator
  concentration, where the moat-safety is structural. (a) is a valid FALLBACK if a later slice wants the hunger→DA
  coupling itself to be synaptic-and-plastic — but that is not needed for the property demonstration.

**(c) — route the interoceptive deficit current into the existing SNc/limbic so hunger CO-DRIVES the DA the critic produces.**
- **Mechanism:** inject the same deficit current that drives `drive_agrp` ALSO into the SNc/limbic pool, so hunger
  raises the critic's DA output.
- **Why NOT the pick:** this makes hunger a HOST current injected straight into the DA pool, bypassing the
  `drive_agrp` neurons entirely — the hunger signal would no longer be the SPIKING `drive_agrp` firing (the validated,
  corr-0.98 brain-based signal) but a host current, weakening the "the SAME spiking drive touches both halves" claim
  (the whole point is that it is the drive's SPIKES, read the same way the survival half reads them). It also
  directly perturbs the SNc's operating point (the merged critic's f-I is finicky — see the limbic-core-lift lesson
  `:840-846`). Least brain-faithful of the three.

**Verdict: (b).** Cheapest, most brain-based (spikes→modulator via the SNc's own rule type), keeps the drive slice's
nav-inert invariant intact, moat-safe by construction downstream, and needs **NO `sim/` edit**.

---

## 3. THE BIOLOGY (catalog-first, cited)

Read from `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.

- **O.10 Incentive Motivation Theory — "deficiency adjusts reward value"** (`:4863`). Berridge/Toates: *deprivation
  does not generate behaviour directly — it AMPLIFIES the reward value (incentive salience) of the relevant goal
  stimuli.* "Sated mouse with AgRP optogenetic stim → behaves as fasted." **This is the exact framing for Option 3.**
  The catalog's sim-status for O.10 is "missing — flagship reward is fixed external scalar; agent state doesn't
  modulate per-stimulus reward weights" (`:4869`). Option 3 realizes O.10's mechanism in the conversational half:
  hunger (AgRP/`drive_agrp` firing) raises the shared DA — the incentive-salience signal — which then re-weights the
  brain's *conversational* precision (via the moat-gate). It is O.10 applied not to food-approach but to how
  conservatively the SAME brain speaks, which is why it demonstrates "one drive touching both halves." **Yes, O.10
  is the right biological framing.**
- **O.05 Hypothalamic Homeostatic Architecture** (`:4803`, ⭐) — sensor→integrator→effector loops; and **O.06 Arcuate
  POMC/AgRP/MC4R feeding loop** (`:4815`, ⭐) — the antagonistic AgRP(hunger)/POMC(satiety) tug-of-war = the
  `drive_agrp`/`drive_pomc` pools. O.06's sim-status flags the actionable next step verbatim: "if hunger were modeled
  as a slow-changing state variable that *modulates the reward value* of … inputs (incentive motivation, see O.55/
  O.10), this would naturally produce richer … behaviour than fixed external reward" (`:4821`). Option-3's link is
  precisely that modulation, now reaching the conversational read.
- **O.11 Drive Reduction Theory** (`:4875`, ⭐) — the survival-reward half Option 1 already uses (hunger is aversive;
  eating relieves it → the intrinsic drive-reduction reward). Cited here to place Option 3 as the *other* face of the
  same drive: O.11 = why the animal ACTS; O.10 = why the drive re-weights the value/salience of stimuli (and, in
  Option 3, of conversational precision).
- **C.22 Dopamine Reward-Prediction-Error** (`:907`, project-critical) + **C.32 Two-component DA — detection/salience
  (Component 1) precedes utility-RPE (Component 2)** (`:610`) + **C.34 DA codes economic utility** (`:635`). These
  ground WHY raising a single shared DA signal is a legitimate way for hunger to reach the conversational half: DA's
  Component-1 (salience/detection) term "amplifies learning rate and downstream sensory GAIN on any potentially
  important event" (C.32, `:615`) — i.e. DA is exactly the broadcast signal that re-weights downstream precision.
  Hunger raising DA → higher salience/gain → the moat-gate reads it as "be more careful." Note the merged bridge's DA
  is a single scalar concentration (C.32's two components are not separated here — a documented simplification, not
  load-bearing for Option 3).

**Catalog verdict:** O.10 (incentive motivation) is the correct primary framing; O.05/O.06 supply the drive; C.22/
C.32/C.34 supply the "DA is the salience/value broadcast that re-weights downstream gain" justification for using DA
as the cross-modal carrier. All are catalog-cited.

---

## 4. THE SINGLE RECOMMENDED cheap-first de-risk

**Build `_tier3_cross_modal_one_animal_derisk.py`** — a new runner, extending the validated Option-1 base
(`_tier3_live_and_remember_derisk.py`), adding the ONE link (option b) and asserting the property. **NO `sim/` edit
predicted** (a runner-layer neuromodulator-config addition, gated by an additive default-off builder kwarg).

### Runner design (concrete)
1. **Bridge:** build the merged one brain exactly as Option 1 —
   `MergedNavConvAgent(seed, vocab=VOCAB, co_resident_composer=True, co_resident_drive=True, …)` — which already
   brings up the shared `dopamine` modulator (nav critic default-ON) + the moat-gate (`enable_da_salience_gate=True`)
   + the spiking hunger drive.
2. **The link (option b):** via a new additive default-off `build_merged_nav_conv_bridge(drive_to_da=False)` kwarg
   (forwarded by `MergedNavConvAgent`), append to the existing `dopamine` `NeuromodulatorConfig.production_rules`:
   `ProductionRule(rule_type="from_region_firing", sensitivity=<s>, threshold=<tonic drive_agrp rate>,
   window_ms=200.0, source_regions=["drive_agrp"])`. Default `False` = byte-identical to Option-1's 6/6 GO. Only
   valid when both `co_resident_drive` and a DA source are present (assert; else it is a documented no-op).
3. **Two conditions on the SAME persistent brain, borderline cue:** pick a "borderline" query — a cue whose stored
   fact's cue-role cleanup margin sits just above `g0` (the band the salience gate is designed to abstain on;
   construct it as the roadmap-#6 de-risk does — a low-margin stored fact under mild read noise, plus a bank of
   never-stored cues for the moat). Run the SAME agent (a) **SATED**: inject a small/zero deficit → `drive_agrp` near
   tonic → DA ≈ baseline → `g_eff ≈ g0`; (b) **HUNGRY**: inject a large deficit → `drive_agrp` fires → DA rises →
   `g_eff > g0`. Read the borderline cue and the never-stored cues under each condition (many query reps, matched
   read-noise seed across conditions — the ONLY difference is the hunger state).

### The decisive checks (GO / BOUNDARY / NEGATIVE)
1. **HUNGRY IS MORE CONSERVATIVE (the property):** on the borderline cue, the HUNGRY brain ABSTAINS more often than
   the SATED brain (`abstain_rate(hungry) > abstain_rate(sated)`, materially, across reps). GO = the shift is present
   and tracks the direction hunger→higher-gate.
2. **THE MOAT NEVER LOOSENS — structurally guaranteed AND asserted byte-unchanged:** `moat_false_accepts == 0` at
   BOTH hunger levels (HARD), AND `false_accepts(hungry) ≤ false_accepts(sated)` (monotone). This is guaranteed by
   `da_to_gate`'s lower clamp at `g0` (DA can only RAISE `g_eff`) — assert it structurally (the map cannot return
   `g_eff < g0`) AND empirically (0-FA both levels). Also assert the conversational (parser/composer) SYNAPSES are
   byte-identical across the run (the moat mechanism is unperturbed — the DA only shifts the read-side gate scalar,
   it does not touch stored facts). A single false-accept at either level = HARD STOP.
3. **THE SHIFT TRACKS THE DRIVE (validate-by-function):** a **drive-lesion** control (zero the interoceptive current
   → `drive_agrp` silent → the hunger term is ~0 regardless of energy) → NO abstention shift between "would-be-hungry"
   and "would-be-sated" (the two collapse). This is the load-bearing anti-cheat: it proves the shift is the DRIVE's
   doing, not an artifact of the two query batches. (Mirror of the roadmap-#6 lesion, which severs DA→`g_eff = g0`.)
4. **PERMUTED/SATED control:** a **sated-both** control (both conditions sated) → NO shift (the conditions must
   differ only by real hunger); and a **permuted-hunger** control (a shuffled hunger signal decorrelated from the
   injected deficit, matched marginal — the Option-1 yoke pattern) → the abstention shift no longer tracks the
   deficit. Both isolate "the shift is caused by the genuine drive state."
5. **REWARD/PROVENANCE (honest label):** assert the hunger term entering DA is read from `drive_agrp`'s
   `cp_firing_states` (via `from_region_firing`), NOT a host deficit scalar injected into DA — the cross-modal signal
   is the drive's SPIKES.

### The ladder
- **1-seed GPU smoke** (`--smoke`): the property mechanics — the link raises DA when hunger is injected (assert DA
  rises), `g_eff` rises accordingly, the borderline cue's abstention rate is higher hungry-vs-sated, moat 0-FA both
  levels, drive-lesion collapses the shift. Mechanics check only.
- **6-seed GPU** (seeds 42/43/44/100/101/102): the standing robustness bar for the "hungry-abstains-more" +
  "shift-tracks-drive" claims and 0-FA both levels every seed (the moat is HARD).

### Predicted `sim/` edit
**NONE.** The `from_region_firing` production-rule type already ships (`sim/neuromodulators.py:736`); the addition is
a runner-layer config-list append behind an additive default-off kwarg. If the smoke reveals the hunger term needs a
production-rule shape the subsystem lacks (it does not — `from_region_firing` is exactly the shape), that would be
flagged; not expected.

**Honest expected caveat (a valid deliverable):** the moat-gate's operating band is narrow (`g0=0.06`, `g_cap=0.25`),
so the abstention SHIFT will be visible only on genuinely-borderline cues (a decisive read passes at both levels; a
never-stored cue abstains at both — as it must). The de-risk must construct the borderline band explicitly (as the
roadmap-#6 de-risk does), or the effect will look absent because every cue is either clearly-answerable or
clearly-abstained. If, at the tuned link strength, the effect is real but small, that is an honest characterization
of "how much a hunger state can re-weight conversational precision on this substrate" — a valid result, not a
failure. The moat can never loosen regardless (structural).

---

## 5. HONEST SCOPE

- **This is a PROPERTY demonstration, not a new life.** It shows the SAME shared spiking drive touches BOTH halves of
  the one brain (the acting half via the survival reward gate — already done in Options 1/2 — and the conversing half
  via the DA moat-gate — the new arrow). It is the capstone scoping's **"Phase-3.1" cheap FOLLOW-ON** to Option 1
  (`2026-06-30-tier3-artificial-life-capstone-scoping.md` §4 Option 3: "SMALL, but it is a *property demonstration*,
  not a life — a cheap FOLLOW-ON to Option 1, NOT the first slice"). It does not add a continuous life, a new
  perceivable set, or a new capability — Options 1/2 already deliver the living/developing agent.
- **The moat can ONLY tighten.** By construction (`da_to_gate`'s lower clamp at `g0`), no hunger level, no link
  strength, can lower the gate below the floor — the no-confab guarantee is structural, not a tuning outcome. The
  0-FA assertion is belt-and-suspenders.
- **Single-scalar DA.** The merged bridge's DA is one concentration; C.32's Component-1/Component-2 split is not
  modeled (a documented simplification, not load-bearing for the property).
- **Deferred (unchanged):** the learned spatial policy = the Tier-4 dendrite wall (off the critical path); true
  `cp_connections` synaptic persistence of the merged bridge (the `LivingState`/`DevelopState` JSON re-instate
  suffices).

---

## 6. VERDICT for the owner — Option 3 vs Option 2B vs Option 4

All three are cheap, reuse-by-import, and predicted `sim/`-edit-free. Recommendation and relative cost:

| next slice | what it delivers | cost | `sim/` edit? |
|---|---|---|---|
| **★ Option 3 (cross-modal one animal)** | the SAME hunger drive measurably touches BOTH halves — a hungry brain converses more conservatively; the moat can only tighten | **SMALLEST** — one production-rule line + a borderline-cue de-risk (all downstream machinery built + moat-safe) | **NONE predicted** |
| **Option 2B (24/7 develop harness)** | promotes the validated Option-2 develop-with-a-body into the crash-proof, pausable, resumable `develop_gpu`/`develop_loop_supervisor` 24/7 loop — the *watch-a-brain-develop* north-star made durable | **SMALL–MEDIUM** — an additive default-off `per_day_agent_factory` seam on `develop_gpu` + harness wiring; more moving parts than Option 3, but a bigger north-star payoff (a brain you can watch develop over real time + load per-day bundles) | NONE predicted |
| **Option 4 (lived consolidation)** | event-triggered SWR replay (N.17 awake-replay at pauses / D.23 misplace-novelty) so consolidation fires on a LIVED event, not a scripted phase | **MEDIUM** — needs a novelty/mismatch read-out wired to the replay trigger (a genuinely new read-out, not just wiring) | possibly (a novelty read-out) |

**Recommendation: do Option 3 FIRST** (it is the cheapest, the most self-contained, closes the "one drive touches
both halves" claim that is the literal statement of the TRUE-ONE-BRAIN property, and is predicted `sim/`-edit-free
with a structurally-guaranteed moat), **then Option 2B** (the durable watch-a-brain-develop harness — the
higher-payoff north-star deliverable, and the natural home to eventually run an Option-3-enabled agent 24/7). **Defer
Option 4** (it introduces a new read-out, so it is the least cheap and the least "wiring-only" of the three). This
ordering matches the SURPASS/cheap-first discipline: bank the one-line, moat-safe property demonstration, then invest
in the harness that makes the developing brain watchable, before building the one genuinely-new mechanism (lived
consolidation).

**One-line owner summary:** Option 3 is largely done — the moat-gate is built and moat-safe, the drive is built and
6/6-validated; the only residual is a single runner-layer `from_region_firing` production rule over `[drive_agrp]`
appended to the existing `dopamine` modulator (NO `sim/` edit), de-risked by a hungry-abstains-more-than-sated check
on a borderline cue with the moat asserted 0-FA at both hunger levels (structurally it can only tighten), 1-seed
smoke → 6-seed. Recommend Option 3 now, Option 2B next, Option 4 last.
