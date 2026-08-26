---
type: finding
status: live
date: 2026-08-26
mechanism: value-critic
---

# Value-driven choice — production wire-in GO: the brain COMMITS BY VALUE where it used to arbitrate by first-match (2026-08-26)

## Result — GO (organ built + wired + flag-OFF byte-identical + ON load-bearing PROVEN; DEFAULT-OFF, awaiting the pool soak)
The RANK-1 value-critic GO (`2026-07-23-value-critic-closure-RANK1-GO.md`, 6/6 seeds) is now WIRED into the live chat
gate behind a NEW env flag `BRAIN_VALUE_CHOICE` (**default OFF** — the parent flips default-on after the 6-seed pool
soak passes). On a `>=2`-distinct-patient `(agent, action)` recall — the ambiguity the GNW deliberation keystone today
resolves by an ARBITRARY FIRST-MATCH (verified live: the onebrain composer returns "dog chases cat" for stored
`dog->chase->{cat, ball}`) or by a halt-if-unsure abstain — the faculty instead COMMITS the higher-VALUE patient. The
value is the brain's OWN LEARNED spiking `striosome_value` critic (DA-gated STDP), the commit is a spiking value-WTA
race (Wang-2002 biased competition). This is the owner's directive "make the brain COMMIT [by value] instead of
abstain[ing]/guessing."

**Honest correction to the wire-in sketch's premise:** the sketch stated today's pipeline ABSTAINS on the
`>=2`-competing case; empirically (live, onebrain composer) it commits an ARBITRARY first-match. The faculty therefore
targets the `>=2`-competing case whether the pipeline abstains OR first-matches, and on decline (lesion / non-decisive
value) reverts to EXACTLY the inner pipeline result (the first-match, or the abstain) — so the lesion oracle holds in
both regimes.

## What was built (ADDITIVE, reuse-by-import, NO `sim/` edit)
- **Organ**: `research/runners/value_choice_production_organ.py` — `ValueChoiceProductionOrgan` (build the merged
  one-brain bridge + value-train the `striosome_value` critic + the spiking value-WTA) + `install_value_choice(chat)`
  (wraps `ChatBrain.gate`). Reuses `_merged_navcritic_valuetrain` (the learned spiking value) and
  `_navcloseout_R5_value_driven_choice` (`SpikingValueChoice`, `_drives`, `make_salience_bias` — the spiking decision).
- **Wiring**: `webapp/server.py` `brain_chat` — a guarded `BEGIN/END`-marked block installs the wrapper OUTSIDE the
  GNW deliberation gate (INSIDE the multistep gate, so chase-form questions keep precedence). DEFAULT-OFF: with the
  flag unset the wrapper is NOT installed, so `chat.gate` is the pure existing chain. A guarded prewarm-warm block
  pre-builds the organ when the flag is on (so the first triggered turn does not stall on the ~4-min value-train).
- **Soak**: `research/runners/_value_choice_flip_soak.py` — the 6-seed no-regression gate (below).

## The wire-in adapter (the honest residual, identical in kind to the GO's own construction)
The GO reads V at nav PLACE cues on the goal->far diagonal (near = high learned V). Here each candidate patient carries
an ENGAGEMENT/reward CONTEXT scalar `e in [0,1]` (fact recency + the discourse-WM referent — the "prior reward/
engagement/DA context" the sketch names), mapped to a cue `pos = near + (1-e)*(far-near)`: a more-engaged candidate
sits nearer the goal, so the LEARNED critic reads a higher V for it. The critic's learned V (a real `cp_firing_states`
read) + the spiking WTA then do the work. The G_LESION / G_UNTRAINED anti-cheats below prove the LEARNED SPIKING VALUE
is load-bearing — the host engagement ordering ALONE (without the trained critic) does NOT produce the commit. This is
the SAME host boundary the GO itself carries (its cue positions are host-supplied; the value + choice are spiking).

## Load-bearing PROVEN (organ level, seed 42; `research/findings/raw/_value_choice_prodflip/organ_loadbearing_seed42.json`)
Two candidate patients (`fish`, `ball`); the value-train grew the plastic `vs_place_context->striosome_value` weight
**0.19 -> 3.98 (20.9x)** — a real learned value gradient (goal cue ~31 Hz >> far cue ~18 Hz).

- **VARY (the faculty CHANGES the output)**: engagement favouring candidate 0 -> V `[30.97, 17.92]` -> the spiking WTA
  commits **`fish`**; engagement favouring candidate 1 -> V `[17.64, 30.69]` -> commits **`ball`**. The committed
  patient FLIPS with the value context.
- **LESION (the surface change VANISHES + reverts to flag-off behavior)**: pin each candidate's learned V to the MEAN
  (the R5b G_LESION) -> the fed value gradient is 0 -> the WTA drives are equal (counts 80/80, margin 0) -> the organ
  DECLINES. At the organ level `choose()` returns `None`; at the gate level the wrapper then returns the INNER
  pipeline result verbatim (the first-match, or the abstain) -> the value-driven change VANISHES and the turn reverts
  to EXACTLY the flag-OFF behavior. The lesion oracle is the finding's own G_LESION.
- **G_UNTRAINED (the SUBSTRATE'S LEARNING is load-bearing, not a wired prior)**: score with the UNTRAINED critic (no
  value-train) -> flat/anti V -> the commit no longer tracks engagement (both orderings pick `fish`) -> the trained
  engagement-advantage VANISHES.

Verdict: `LOAD_BEARING_PROVEN`. A flip whose lesion did nothing would be a hollow checkbox; this one's lesion collapses
the commit back to the inner result, and the untrained critic has no advantage.

## Load-bearing PROVEN LIVE too — through the REAL ChatBrain gate (onebrain composer, stub renderer, seed 42)
`research/findings/raw/_value_choice_prodflip/soak_real_onebrain_seed42.json` — the faculty installed on the real
ChatBrain (deliberation keystone + value-choice, webapp order). Trigger "what does dog chase" (stored
`dog->chase->{cat, ball}`):
- OFF (flag unset): `"The dog chases cat."` — the pipeline's ARBITRARY first-match.
- ON, engagement favouring `cat`: `"The dog chases cat."` ; ON, favouring `ball`: `"The dog chases ball."` — the
  committed patient FLIPS with the value context, through the real gate (the live VARY).
- ON + LESION: `"The dog chases cat."` — reverts to the inner first-match. The value-driven change VANISHES.
- Ordinary turns (confident recall, untaught abstain, self): byte-identical OFF vs ON.
This is the live-handler-level load-bearing proof (stub renderer per the wedge-avoidance directive — no Qwen warm; the
committed SVO is the coupling, prose fluency is downstream and unchanged).

## Flag-OFF byte-identical
By CONSTRUCTION: the server block installs the wrapper only `if value_choice_enabled()`, so with `BRAIN_VALUE_CHOICE`
unset `chat.gate` is untouched (the pure existing GNW chain) -> byte-identical to today. Confirmed additionally at the
wrapper level (mock panel + the soak's real-ChatBrain ordinary panel, `soak_real_rf_seed42.json` /
`soak_real_onebrain_seed42.json`): with the wrapper INSTALLED but the flag off, every turn returns the inner gate's
result verbatim (a `<2`-candidate turn — confident single recall, single-patient/untaught abstain, self — is ALWAYS
returned verbatim, even ON); only the `>=2`-distinct-patient turn changes when ON.

## The 6-seed no-regression SOAK (the gate the parent runs before flipping default-on)
`research/runners/_value_choice_flip_soak.py` runs the SAME panel twice (flag OFF vs ON) on the same real ChatBrain
(deliberation keystone + value-choice installed, mirroring the webapp order) across 6 seeds. The HARD no-regression
gate: every ORDINARY (`<2`-candidate) turn is byte-identical OFF vs ON. The load-bearing confirmation on the triggered
`>=2`-patient turn: VARYing the engagement through the REAL gate FLIPS the committed patient (favour cat -> commits
cat; favour ball -> commits ball; both STORED -> the moat holds), and the LESION reverts to the inner pipeline result.
The harness `--smoke` passes; a real 1-seed rf run confirms the ordinary panel byte-identical; the parent runs the full
6-seed CPU soak (`--composer onebrain`) on the pool.

## Honest scope
- The **value CONTEXT** (per-candidate engagement) is host-supplied (fact recency + discourse referent) — the SAME
  host boundary the R5b GO carries (host cue positions). The VALUE and the DECISION are spiking; the lesion/untrained
  anti-cheats prove the learned spiking value is what commits. A fully-spiking engagement/DA-tag source is a documented
  next residual, not this wire-in.
- Verified at the ORGAN + gate-WRAPPER level (stub renderer; no live Qwen warm, per the wedge-avoidance directive) —
  the load-bearing coupling is the gate commit, which the organ-level test exercises decisively; prose fluency is
  downstream of the committed SVO and unchanged.
- DEFAULT-OFF: the parent runs the pool soak, then flips `BRAIN_VALUE_CHOICE` default-on.
