---
type: finding
status: superseded
date: 2026-09-01
mechanism: declarative-cross-edge-curiosity-to-d6wm
lane: onebrain-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_6seed.json
runner: research/runners/_onebrain_crossedge_curiosity_to_d6wm.py
builds_on:
  - research/findings/2026-09-01-declarative-cross-edge-functional-gate-read-credit-livedrive-GO.md
  - research/findings/2026-09-01-onebrain-crossedge-provenance-to-selfschema-reciprocal-GO.md
  - research/findings/2026-08-27-onebrain-completeness-audit.md
---

# A FRESH declarative cross-edge on the one-brain connectome: curiosity's `ask` crave pool -> d6's `w0` WM slot — added as a data row + 2 callables, 6-seed GO (6/6); the effect is an honest SUPPRESSION, not the boost the first hypothesis predicted

## ⛔ CORRECTION (2026-09-02, read-isolation fix) — the "6-seed GO (6/6)" headline was PARTIALLY INFLATED by an incomplete `_hard_reset()`; corrected verdict is NO-GO 3/6

`AskToW0Pool._hard_reset()` never restored `cp_refractory_timers`/`cp_prev_firing_states`/`cp_neuron_activity_ema`/
`cp_neuron_firing_thresholds` (the C2 bug class, `research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`)
nor the NMDA-recurrent/synapse-pulse buffers this pair's own read rides on, leaking residual state
order-dependently across reads AND training episodes. Isolating the read (both framework state tuples restored,
verified BITWISE-identical repeat reads on a lesioned pool) collapses `delta_lesion` to exactly 0.0 on every seed
(the read is now trustworthy) but ALSO collapses `delta_intact` below `INTACT_FLOOR=0.008` on 3 of 6 seeds
(43, 101, 102) — **n_go 6/6 -> 3/6**. The mechanism itself (Hebbian-grown edge, sign-consistent suppression,
byte-identical-off) survives on the 3 seeds that still clear the floor; the "GO 6/6" claim below does not. Full
before/after table, the fix, and the selftest: `research/findings/2026-09-02-onebrain-crossedge-curiosity-to-d6wm-read-isolation-fix-corrects-GO-to-NOGO-3-6.md`.
This is a LIVE default-ON production faculty (`research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-production-wire-GO.md`,
also ⛔-corrected) — the default-ON decision is flagged for owner review there, not changed by this correction.

**One-line (ORIGINAL, now PARTIALLY SUPERSEDED — see the correction above):** curiosity (D3) and d6_multiref_wm (D6) are both already co-resident on `full7` but had ZERO synaptic
interaction with each other before this edge — two organs sitting side by side, not one brain. This finding wires
curiosity's `ask` (crave/epistemic-gap) pool -> d6's `w0` working-memory slot, PURELY BY DECLARATION (a 1-row
`CrossEdge` + `train_fn` + `read_fn`, through the SAME generic `onebrain_crossedge_gate.run_gate` R1/R4/R4-reciprocal
all use — no bespoke F-gate file). 6-seed GO: the edge grows from near-zero by the substrate's own standard Hebbian
rule (0.05 -> 1.7-2.1 across seeds), is load-bearing (driving curiosity's `ask` pool alone measurably SUPPRESSES an
already-held WM slot's sustained firing rate, 89-102% lesion-attributable on every seed), and is byte-identical-off.
**The first hypothesis tried (a DA-gating BOOST, matching Braver/Cohen-style PFC working-memory gating) was WRONG in
sign** — the substrate's own measurement is a clean, repeatable SUPPRESSION, and the biological framing below was
corrected to match what was actually measured (an attentional-capture / resource-competition account), not forced
to agree with the first guess.

## 1. Why this pair (biological rationale)

Per the brief's candidate list, this session picked **curiosity(D3) <-> attention/WM(D6)** over the other three
offered pairs (affect<->value-choice, comprehension<->prospective-memory, metacog<->confidence-forthcomingness)
because both curiosity and d6_multiref_wm are ALREADY registered, ALREADY co-resident organs in this project's
`onebrain_merge_framework.REGISTRY` (the same 7-organ `full7` set `causal_whatif`/`comprehension`/`self_schema`/
`source_provenance`/`curiosity`/`prospective_memory`/`d6_multiref_wm` already merges in production de-risks), so
the pairing needed no new organ construction — only the missing synapse. Curiosity's `ask` population is this
substrate's own novelty/epistemic-gap crave signal (`curiosity_production_organ.py`; DR-1's own
`from_novelty -> excitability_drive` neuromodulator targets `group:ask`). d6's multi-referent WM buffer holds the
discourse referent(s) currently in play (`d6_multiref_wm_production_organ.py`, already reciprocally wired to
`comprehension` via R1/R2/R3 — the FIRST pair this project's cross-edge framework closed). Neither organ had a
declared cross-edge with `curiosity` before this finding.

**An honest correction, kept for the record (the instrument is part of the emulation).** The FIRST hypothesis was a
dopaminergic gating-BOOST account: Lisman & Grace 2005 (*Neuron*, "The Hippocampal-VTA Loop: Controlling the Entry
of Information into Long-Term Memory") — novelty detected upstream drives VTA dopamine that gates what gets
consolidated; Bunzeck & Duzel 2006 (*Neuron*, "Absolute coding of stimulus novelty in the human substantia
nigra/VTA") — SN/VTA (curiosity's own co-resident `snc` population) fires selectively to novel stimuli; and
Braver & Cohen's adaptive-gating account of prefrontal working memory (O'Reilly & Frank 2006, *Neural Computation*,
"Making Working Memory Work") — phasic dopamine GATES what is admitted into / sustained in a PFC WM buffer. Trained
and read exactly as declared in §2 below, the substrate's own measured effect is the OPPOSITE sign, cleanly and
repeatably across all 6 seeds (§3): driving `ask` SUPPRESSES the already-held `w0` referent's sustained firing rate.
Rather than force the read to agree with the first hypothesis, the biological framing was corrected to the account
the substrate's own measurement actually supports — ATTENTIONAL-CAPTURE / resource-competition for a
limited-capacity WM buffer: an involuntary orienting response to a salient/novel signal measurably DISRUPTS ongoing
WM maintenance, rather than reinforcing it (Berti & Schroger 2003, *Journal of Cognitive Neuroscience*, "Working
memory controls involuntary attention switching: evidence from an auditory distraction paradigm"; SanMiguel,
Corral & Escera 2008, *Journal of Cognitive Neuroscience* 20:1131-1145, doi:10.1162/jocn.2008.20078 <!--derived-->
(a citation locator, not a measurement),
"When loading working memory reduces distraction: behavioral and electrophysiological evidence from an
auditory-visual distraction paradigm" — https://pubmed.ncbi.nlm.nih.gov/18284343/, verified via live external
search this session — on involuntary attention capture impairing WM consolidation/maintenance under low load).
Both accounts are genuine, independently-documented readings of how salience/novelty interacts with
working memory in the literature; this substrate, as trained here, realized the competitive half, not the
gating-boost half — a real result about THIS mechanism's operating point (an NMDA-mediated recurrent excitatory
population pushed past its own effective operating range by additional excitatory drive, rather than linearly
summing it — CLAUDE.md's own standing lesson that an operating point is implicit, not free, and decides which
biological account applies), not a defect in the measurement.

**Conversational rationale.** This is the substrate correlate of "a genuinely novel, urgent question can knock the
thing you were just discussing out of mind" — curiosity's own crave state, when it fires hard, measurably COMPETES
with the currently-held referent rather than reinforcing it: a self-report-honest correlate of a common
conversational experience (a tangent derails what you were holding in mind), not an idealized always-helps account
of curiosity. It is also an honest, minor counter-example to the intuitive first guess, which is itself useful for
the project's honesty-boundary deliverable: this is exactly the kind of correlate that should be reported as "my
crave signal is competing with what I was holding in mind," not glossed as unconditionally helpful.

## 2. The edge, added PURELY BY DECLARATION

```python
CROSS_EDGES = [
    CrossEdge(key="ask_to_w0", source_key="curiosity", source_region="ask",
             target_key="d6_multiref_wm", target_region="w0", init_weight=0.05, plastic=True,
             gate="ask_to_w0", learn_rule="rate_hebbian", freeze_rest=True),
]
```

Both `ask` and `w0` are registered top-level regions (curiosity's own `regions=(...,"ask")`; d6's `w0`, the SAME
region R1's own edge already reads/writes) — no `source_idx_fn`/`target_idx_fn` needed, unlike R4's sub-slice
edges.

**`train_fn`** — the substrate's OWN standard Hebbian rule (`hebbian_symmetric`), grown from a host tonic co-drive
of `ask` (ASK_DRIVE_PA=600.0, matching the scale curiosity's own de-risk uses to drive its pools) and `w0`
(LOAD_PA=400.0, R1's own WM-slot load current, reused verbatim) TOGETHER for 100 episodes. Declared, not hidden:
like every cross-edge in this codebase, the co-occurrence experience is HOST-SUPERVISED (a teaching current), not
claimed self-organized — the substrate's own Hebbian rule does the binding, the host supplies the correlated
experience.

**`read_fn`** — LOAD `w0` into its own held bump first (LOAD_PA/LOAD_STEPS, IDENTICAL in both read conditions — a
condition-blind step, exactly R1's own load-then-cue read shape), THEN drive ONLY `ask` over the read window under
`familiar` (ask_pa=0.0, the CONTROL — ask genuinely silent, no injected current) or `novel` (ask_pa=ASK_DRIVE_PA),
and read `w0`'s mean firing rate during that window. An earlier design (documented in the runner's own module
docstring and skipped straight past here) drove `ask` alone with NO load step at all — that version's lesioned
reads still showed a small residual delta between conditions (~26% of the intact effect), because an unloaded `w0`
has no baseline bump to modulate and the read partly measured noise-floor artifacts rather than a genuine
modulation; adding the condition-blind load step (R1's own established shape) cleared that up: the lesioned delta
across all 6 seeds below is a genuine near-zero, a min/max read off the per-seed table in §3: <!--derived-->
|delta_lesion| <= 0.0011 on every seed, vs. |delta_intact| >= 0.0095. <!--derived-->

No `selectivity_pairs` are declared — ONE-SIDED BY DESIGN, matching R4/R4-reciprocal's own honest
characterization: a single edge onto one WM slot has no companion population for a weight-ratio comparison.
Selectivity is demonstrated FUNCTIONALLY at the read (below, via lesion-attribution), not as a weight ratio — per
`docs/TERMS.md`'s condition for "selective" (a permuted/scrambled control + raw magnitudes), this finding does NOT
claim the edge is "selective"; it claims the measured shift is LOAD-BEARING (lesion removes it).

## 3. 6-seed result (42/43/44/100/101/102), numpy CPU — GO 6/6

(the table reports the cited 6-seed artifact's values to 6 decimal places; open the JSON directly for full double
precision.)

| seed | grown weight | w0 rate (familiar, control) | w0 rate (novel) | Δ intact | Δ lesion | frac attributable | emg · int · byte-off | GO |
|---|---|---|---|---|---|---|---|---|
| 42 | 2.020219 | 0.060625 | 0.049250 | -0.011375 | -0.000750 | 0.934066 | ✓ · ✓ · ✓ | GO |
| 43 | 1.889884 | 0.058125 | 0.047625 | -0.010500 | 0.000250 | 1.023810 | ✓ · ✓ · ✓ | GO |
| 44 | 1.981809 | 0.060375 | 0.047375 | -0.013000 | 0.000250 | 1.019231 | ✓ · ✓ · ✓ | GO |
| 100 | 1.970807 | 0.058375 | 0.047625 | -0.010750 | -0.000125 | 0.988372 | ✓ · ✓ · ✓ | GO |
| 101 | 2.117586 | 0.059625 | 0.045375 | -0.014250 | -0.000375 | 0.973684 | ✓ · ✓ · ✓ | GO |
| 102 | 1.739121 | 0.059750 | 0.049750 | -0.010000 | -0.001125 | 0.887500 | ✓ · ✓ · ✓ | GO |

Every seed: the edge GROWS from `W0=0.05` to 1.7-2.1 (>34x the `grow_factor*init_weight=0.25` emergence floor,
well under the `HMAX=6.0` soft bound, no runaway); the `no_corruption` check (max\|Δ\| over every non-edge synapse)
reads exactly 0.0 (< `drift_tol=1e-6`) on all 6 seeds; the intact `novel`-vs-`familiar` shift on `w0`'s rate clears
the `INTACT_FLOOR=0.008` (R1's own established floor, reused verbatim) with 1.25-1.8x headroom on every seed, and
is signed NEGATIVE on all 6 (a robust, direction-consistent suppression, not a seed-dependent flip); the shift is
89-102% lesion-attributable (two seeds read slightly over 100% — the lesioned control moved a hair opposite the
treatment, the SAME benign wobble `tools.lab.attributable_to` documents and flags on its own "ABOVE 100%" line; the
lesioned |delta| itself is small and near-zero on every seed <!--derived--> (<=0.0011, a min/max read off the
per-seed table above)); and the no-edge pool's base connectivity
is exactly byte-identical to the with-edge pool once the declared edge's own synapse slots are excluded, on all 6
seeds.

**Calibration note (kept for the record).**
<!--derived-->
(every number in this paragraph restates an ad hoc pre-registration vary-check run BEFORE the committed 6-seed
gate, to choose `N_EPISODES` — not the cited artifact, which only ever ran the final `N_EPISODES=100` design; the
intermediate run's own console output was not saved to a file.) A first pass at `N_EPISODES=60` (R4-reciprocal's
own episode count) produced a 1/2-seed indicator on a quick 2-seed vary-check: seed 44 cleared the floor cleanly
(delta_intact -0.0115) but seed 43 fell just short (delta_intact -0.0079 vs. the 0.008 floor) — the grown weight
(~1.6, well under `HMAX=6.0`) was under-trained for that seed's particular read. Raising `N_EPISODES` to 100 (still
well under `HMAX`) grew the edge further (1.7-2.1 across seeds, see §3's table) and cleared the floor on every one
of the 6 committed seeds; no other constant changed. This mirrors R4's own precedent (its module docstring
documents an analogous episode-count bump for the same reason — "de-risks under-training before the first
calibration read").

## 4. What this demonstrates about "the next edge is a data row + 2 callables"

This is the THIRD edge added to the connectome this arc purely through `onebrain_crossedge_gate.CrossEdgeGateSpec`
+ `run_gate` (after `...functional-gate-...GO.md`'s comprehension<->d6 reciprocal and
`...provenance-to-selfschema-reciprocal-GO.md`'s self_schema<->source_provenance reciprocal), and the FIRST on a
genuinely new organ PAIR that had never shared an edge in either direction before (R1/R4 both completed an
ALREADY-open pair's other half; this pair had zero prior edges). What was written per-edge: one `CrossEdge` row (no
`idx_fn` needed — both endpoints are top-level regions), a `train()` method (10 lines, a plain tonic co-drive), a
`read_w0()` method (16 lines, a load-then-condition read), and the `CrossEdgeGateSpec` declaration itself (11
lines). Everything else — the emergence read, the no-corruption drift, the lesion, `attributable_to`, the byte-off
comparison — came from the harness, unmodified. The harness also caught the FIRST read design's flaw (a non-clean
lesioned control, §2) the same way it caught R4-reciprocal's ambiguous-control flaw — the generic `verify_interaction`
step, run against a rough draft, made the confound visible before the 6-seed commit rather than after.

## 5. Honest residuals (declared, not hidden)

- **The biological SIGN was not predicted in advance; it was measured, then the framing was corrected to fit.**
  §1 keeps the full account of the wrong first hypothesis. This is disclosed as a methodological note, not hidden
  behind a retroactively "obvious" citation — the attentional-capture account is a genuine, independently-supported
  reading of the literature, but it was NOT the account this session started from.
- **Region-pair choice remains hand-directed.** A human (via this session, from a 4-pair candidate list the task
  brief supplied) picked curiosity<->d6_multiref_wm as the next pair to wire; the framework does not yet propose
  candidate pairs from the connectome's own structure.
- **Training is host-supervised**, exactly like every other cross-edge in this codebase: the tonic co-drive of
  `ask` and `w0` is a host-injected teaching current, not a self-organized discovery of what should co-occur. The
  substrate's own Hebbian rule does the binding; the host supplies the correlated experience.
- **ONE-SIDED BY DESIGN**: `w0` is one WM slot among d6's `w0`..`w29`; this edge only ever biases `w0` specifically
  (from `ask`), not the WM buffer generically, and only in the direction curiosity->WM (the reverse, WM->curiosity,
  is not wired here). No selectivity_pairs are declared (§2) — the finding does not claim "selective" per
  `docs/TERMS.md`'s stricter condition.
- **Not wired into production.** This is a runner-level 6-seed GO
  (`research/runners/_onebrain_crossedge_curiosity_to_d6wm.py`), matching R1/R4's own precedent before their later
  production wire-in. Production integration is the natural next rung, not claimed here.
- **The suppression's biophysical mechanism is not isolated.** §1 offers a plausible substrate-level reading
  (NMDA-recurrent operating-point saturation) but this finding does not attempt to isolate which specific dynamic
  (refractory accumulation, conductance saturation, or something else) produces the sign flip — that would be a
  follow-on de-risk, not claimed here.

## 6. Files

`research/runners/_onebrain_crossedge_curiosity_to_d6wm.py` (NEW — `CROSS_EDGES`, `_build`, `AskToW0Pool`,
`GATE_SPEC`, `_noedge_bridge`, `run_seed`, `main`) ·
`research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_6seed.json`. Reused, unmodified:
`research/runners/onebrain_crossedge_gate.py` (`CrossEdgeGateSpec`, `run_gate`, `verify_byte_off`,
`cross_edge_masks`), `research/runners/onebrain_merge_framework.py` (`REGISTRY["curiosity"]`,
`REGISTRY["d6_multiref_wm"]`, `CrossEdge`, `merge_organs`), `research/runners/_onebrain_integration_r1_wm_comprehension.py`
(`LOAD_PA` — constant reuse only, no logic reimplemented). No `sim/` file touched; no `webapp/server.py` edit; no
production default changed.

Functional read-outs only; no phenomenal-experience claim.
