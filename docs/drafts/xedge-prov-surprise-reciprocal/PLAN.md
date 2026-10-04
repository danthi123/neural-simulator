# PLAN — xedge source_provenance -> surprise reciprocal

DRAFT ONLY. Output = PROVISIONAL finding at best; never a GO claim; production wire-in is Claude's.
New file in a draft worktree: `research/runners/_xedge_prov_surprise_reciprocal.py` (copy-and-reshape of the
2026-09-01 reciprocal runner's shape). sim/, registered runners, scorers and gates: UNTOUCHED.
Written 2026-10-04 by the local-model driver per runbook §5 step 3 from one MiMo Pro delegate's plan
(deleg_15ee3fb0); a second delegate refutation round is required before any run.

## 1. Question

Does the brain's own source/provenance verdict feed back into its prediction-error read? On the SAME physical
input, does a claim the substrate's source-monitoring opponent circuit reads as externally-sourced
('asserted'/'perceived') drive a HIGHER windowed surprise-pool rate than one it reads as internally-generated
('generated') — and is that difference carried by the reciprocal CrossEdge source_provenance -> surprise pool?

**The pair's closed-loop claim (what this edge does that surprise->prov_generated cannot):** the existing edge
lets surprise gate what gets ENCODED into provenance (R3's novelty-gated encoding gate). It is structurally
incapable of letting a verdict ALREADY IN MEMORY change the live 'am I surprised?' read — that requires the
reverse anatomical path. Together: experience (mismatch) shapes the provenance trace; the provenance trace then
modulates the PE read on later inputs — the Lisman-Grace loop's feedback limb, at the synapse level.

**Probe (exact):** one fixed D2 mismatch trial — cue block c (CUE_PA operating point) + asserted patient block
c'!=c — read = windowed mean firing rate of `cp_firing_states[surprise]` over the trial window. delta_intact per
seed = read(perceived-verdict) - read(generated-verdict). **Trace-level load-bearing means** (TERMS.md 09-29: unqualified 'load-bearing' now defaults to reply-level;
ours is a trace field): removing exactly the declared
edge's synapses (harness lesion) collapses that delta on the same input:
|delta_lesion| < lesion_ratio(0.34)*|delta_intact|, frac_attributable ~ 1.

## 2. Biology it follows

- Johnson, Hashtroudi & Lindsay 1993 — CONFIDENT on: source monitoring is inferential, not a one-way read-out.
  The 'continuously-updated loop, each side informing the other' framing is this project's argued extension (the
  2026-09-01 reciprocal GO's rationale section, which recruits Northoff & Bermpohl for the self-model limb) —
  NOT a claim JHL itself states.
- Lisman & Grace 2005 (Neuron) — hippocampal-VTA loop: hippocampal novelty/subicular signal drives the VTA/DA
  (PE-gain) side; the design doc's R3 entry (phase-DESIGN ~line 220) names exactly this loop around
  surprise->provenance encoding. Confident on the paper; exact-tract detail (unverified).
- Sensory attenuation / efference copy: self-generated content LESS surprising, foreign MORE. Blakemore,
  Wolpert & Frith forward-model attenuation of self-produced sensations — confident on the line of work, exact
  citations (unverified). Wolpert & Flanagan 2001 'Motor prediction' (unverified year/venue). Reduced
  auditory-cortical responses to self- vs externally-produced speech — replicated finding class (specific paper
  unverified).
- Rao & Ballard 1999 (predictive coding); Friston 2005 predictive coding (unverified exact ref).
- Plasticity: the repo's plain TWO-factor symmetric Hebbian rule — grounding: `sim/config.py` hebbian_symmetric note
  (~lines 869-873: Kandel 6e Ch 54; Marr 1971 autoassociator). Fremaux & Gerstner 2016 (`docs/biology.md:284`)
  grounds the THREE-factor (pre x post x dopamine) family this rule is the plain two-factor cousin of — not the
  citation for what this draft trains.

**SIGN CONSTRAINT (design finding, load-bearing):** cross-edge weights grow positive only (excitatory). The
anchor's attenuation half (self-generated -> LESS surprise) cannot ride a direct excitatory edge onto the
surprise FIRING region — a literal `prov_generated -> surprise` would make self-generated content MORE
surprising, the anchor's opposite. The draft therefore declares the amplification half as the plastic edge
(`prov_perceived -> surprise`: 'foreign content MORE surprising', the Lisman-Grace excitatory sign) and reads
the attenuation half as the matched CONTROL. The pre-declared variant that mechanizes attenuation directly
(`prov_generated -> patient_expected`, the surprise circuit's subtractive PREDICTION neurons = efference-copy
anatomy) is section 7's follow-up — a one-row change, not a new harness.

## 3. What it must not claim

No phenomenal anything: never 'feels surprising', 'experience of agency', 'the brain notices'. Only honest
functional read-outs: 'the substrate's windowed surprise-pool rate shifts with its own source-monitoring
verdict'. Not GO — this is a DRAFT; the strongest possible outcome is a PROVISIONAL finding; production
wire-in (default-ON) is Claude's call, never this draft's. Not gate evidence: dev seeds 7/1-15 ONLY — never
42/43/44/100/101/102, and dev-seed results never support a generalization claim. Not self-organized: training
co-occurrence is a HOST-SUPERVISED teaching current (world/teacher role), the substrate's own Hebbian rule does
the binding — declared, not hidden. Not a complete loop in any brain sense: only the two declared edges of this
organ pair. One-sided: the edge can only amplify surprise under an external verdict; 'self-generated attenuated'
is the matched-control READ, not a demonstrated attenuating mechanism. Not a claim about the LLM mouth,
transformer, or any host computation: every effect between verdict and surprise must travel synapses in the ONE
merged spiking pool.

## 4. Design (edges / conditions / probes)

Pool (unchanged shape from `_onebrain_integration_surprise_episodic_crossedge.py`):
`merge_organs([SURPRISE_LITE, source_provenance], wire=True, cross_edges=[...])` — same CONFIG RECONCILIATION
mandatory: `SURPRISE_LITE = replace(SURPRISE, config={**SURPRISE.config, 'enable_hebbian_learning': False,
'hebbian_rate_window': False})` (the R4 hebbian_rate_window hazard), enable_hebbian_learning flipped LIVE only
around this edge's training window, STANDARD (non-rate-window) symmetric rule.

    CROSS_EDGES = [
        CrossEdge(key='provper_to_surprise', source_key='source_provenance',
                 source_region='prov_perceived',
                 target_key='surprise', target_region='surprise', init_weight=0.05,
                 plastic=True, gate='provper_to_surprise', learn_rule='rate_hebbian',
                 freeze_rest=True),
    ]

(`prov_perceived` is a registered top-level region — no idx_fn, same as `prov_generated` in the precedent.)

- **train_fn** — host-supervised episodes (teaching current, DECLARED): co-drive a heard CONTRADICT experience —
  the mismatch trial (cue c + asserted c'!=c, surprise fires) while source_provenance's ctx_perceived + episode
  content drive makes prov_perceived win its opponent read. N episodes; the substrate's own standard symmetric
  Hebbian rule binds; trajectory recorded by the harness. Weight must GROW from 0.05.
- **read_fn** — probe of section 1. Verdict drive: recall one fixed battery exemplar (content drive alone; ctx
  lines silent at recall — the opponent mechanism's own recall rule) encoded purely under 'perceived' (test) or
  'generated' (CONTROL) contexts, held THROUGH the trial window. Conditions: condition_order=('generated',
  'perceived'), control='generated', expected={'perceived': {'sign': +1, 'floor': INTACT_FLOOR}},
  credit_signal='rate_hebbian'.
- **Intact vs no-edge pair:** (a) intact trained pool — the two conditions; (b) in-pool LESION re-read of the
  same conditions (harness lesion_cross_edges, exactly the declared slots); (c) a structurally no-edge pool
  (cross_edges=None, identical otherwise) for verify_byte_off, whose perceived-generated delta must be ~0 — the
  pre-registered negative-control probe. Per the precedent: read directly off the two organ-build-time battery
  exemplars every arm already shares; no per-arm extra patterns (its byte-off asymmetry bug's lesson).
- Everything else — emergence, drift, lesion, attributable_to, byte-off — comes from
  `research/runners/onebrain_crossedge_gate.run_gate` UNMODIFIED. Per-edge code = 1 CrossEdge row + train_fn +
  read_fn + the spec (~40 lines).

## 5. Outcome-to-reading table (pre-registered — written BEFORE any run)

| # | Outcome (pre-registered BEFORE any run) | What it DOES show | What it DOES NOT show |
|---|---|---|---|
| 1 | Edge grows (0.05 -> > grow_factor*0.05) + delta_intact in predicted sign clears floor + 100% lesion-attributable + byte-off PASS + clean-zero control, ALL 6 dev seeds | PROVISIONAL-supports: the substrate's source verdict shifts the windowed surprise-pool rate (honest functional read-out) through the declared edge; the pair surprise->prov (encoding gate) + prov->surprise (read modulation) forms a reciprocal feedback loop at the synapse level — NOT a 'closed' capability (default-OFF; wire-in is Claude's) | Not GO, not gate-seed evidence, not phenomenal, not self-organized (host-supervised episodes), not production wiring |
| 2 | Grows, delta sign correct but below floor on some seeds (heterogeneous) | Wiring works; effect size is operating-point dependent — supports the mechanism weakly | Not trace-level load-bearing at this operating point; do NOT claim the verdict is load-bearing (even qualified); next lever is drive/operating-point recalibration (CUE_PA class), not a new mechanism |
| 3 | Grows, but delta does NOT collapse under lesion (delta_lesion large / frac_attributable low) | Only that the shift is NOT carried by the declared edge (candidates: base-wiring leak, co-drive artifact, unmodeled path — this cell cannot distinguish them) | NOT trace-level load-bearing; run the instrument checks FIRST and repair only one that actually fails; if the instrument is clean, bank the honest negative (edge exists, does not carry the effect) and iterate the episode recipe per no-defer — never presume instrument fault |
| 4 | Does NOT grow (grown <= grow_factor*init) | Verdict on the TRAINING RECIPE: the co-drive episodes do not produce the co-occurrence the substrate's rule needs | NOT a verdict on the biology or the pair; per no-defer, change the episode recipe and re-run — never a license to drop the capability |
| 5 | Grows + trace-level load-bearing, but the no-edge/lesioned arm ALSO moves the read beyond tolerance, or byte-off FAIL | CONFOUND: effect not attributable to the declared edge | NO biological reading is permitted from this cell; instrument bug (leaky graded control, merge-wiring leak, read-isolation, threshold drift) — fix and re-run |
| 6 | Sign REVERSED (generated > perceived) | The edge implements self-contradiction AMPLIFICATION, not foreign-content amplification | A directional negative on the anchor; report it, never flip expected sign post hoc; the pre-declared variant (prov_generated -> patient_expected) becomes the next draft |
| 7 | Byte-off PASS but clean-zero control FAILS (prov_perceived active under the 'generated' exemplar) | The control is leaky (graded opponent trace) — measurement artifact class already seen | No reading of rows 1-6 valid until the control exemplar is replaced with a genuinely-generated one (precedent's calibration lesson) |

## 6. Seeds + checks + queue sizing

6 dev seeds: [7, 1, 4, 9, 12, 15]. Seed 7 is the smoke seed (its smoke cell is CALIBRATION ONLY — never
evidence — and is re-run clean inside the 6-seed batch). 1/4/9/12/15 spread across 1-15 (bracketing endpoints
1 and 15, odd/even mix) to avoid adjacent-RNG correlation clusters. NEVER 42/43/44/100/101/102 (gate seeds —
a draft may not touch them).

Per-seed checks (the precedent's 6-seed pattern, unmodified harness): (1) Hebbian growth — grown mean weight
over the declared mask > grow_factor * init_weight (0.05), trajectory recorded; (2) no-corruption —
frozen_maxdrift over every NON-edge synapse < drift_tol=1e-6 after training (also the rate-window-hijack
detector); (3) trace-level load-bearing — delta_intact per seed in the predicted sign, clearing the frozen
floor, and |delta_lesion| < lesion_ratio(0.34)*|delta_intact| with frac_attributable ~ 1 after lesioning
exactly the declared slots — with the persistence-at-measurement check: the declared (pre,post) slots read
EXACTLY 0 at the moment of every lesion re-read (enable_hebbian_learning OFF for all re-reads; the TERMS
'lesion' code condition held at measurement); (4) byte-identical-off — the cross_edges=None pool's base connectivity is byte-identical to the
intact pool minus the declared (pre,post) slots (verify_byte_off), and the no-edge pool's perceived-generated
delta must be ~0 (the pre-registered negative control); (5) clean-zero control — measured prov_perceived
activity under the 'generated' control exemplar must sit near zero (the precedent's leaky-graded-trace lesson).

Queue sizing: 1 cell = 1 seed = 1 job (2 pool builds + train + probes), numpy CPU, ~3-6 min/cell expected;
smoke = 1 job, batch = 6 jobs -> `tools/sweep_pool.sh` / workstation queue per RULING compute-placement (never
>10 min on this server; if a smoke cell exceeds 10 min, halve N_EPISODES before queueing). Total ~7 jobs,
~30-40 min wall. Build as NEW `research/runners/_xedge_prov_surprise_reciprocal.py` in a DRAFT WORKTREE;
sim/, registered runners, scorers and gates stay untouched (small additive helpers live inside the new file
only).

## 7. Risks and failure modes

1. **Windowed-rate timing of the feed** — the surprise read is a windowed mean; the verdict drive must be
   present across the whole trial window or the delta dilutes. Holding episode drive through the trial risks
   merge-wiring leak into the surprise circuit — it cancels in the delta ONLY if both conditions hold identical
   drive (exemplar-swapped); the lesion + no-edge arm is the attribution instrument.
2. **hebbian_rate_window hazard (R4)** — SURPRISE's registered config hijacks co-resident edges onto an
   uncalibrated rule; the SURPRISE_LITE reconciliation is mandatory, and frozen_maxdrift < 1e-6 is the detector.
3. **Leaky graded provenance baseline** (the precedent's root-cause note) — the opponent trace is GRADIENT;
   ambiguous/dual-context exemplars drive both lines partway and make a leaky control. Control must be a
   genuinely GENERATED exemplar with measured prov_perceived ~ 0 (row 7 of the table).
4. **Adaptive-threshold drift in the surprise pool** — MANDATORY Port-A reset: every between-condition reset
   goes through `MergedPool._PER_NEURON_STATE` (`onebrain_merge_framework.py:254-259`: restores
   cp_neuron_firing_thresholds, cp_refractory_timers, cp_prev_firing_states, cp_neuron_activity_ema — the
   2026-09-02 C2 read-isolation fix; the 15 hand-copied `_hard_reset` versions that omitted it are the recorded
   FAILURE_LOG class). Smoke check: thresholds byte-restored at re-read. Read order: COUNTERBALANCED IN THIS
   RUN (or order-invariance proven at smoke) — never deferred to a follow-up.
5. **Operating-point fragility** — the surprise->prov runner needed a 600->2000pA recalibration; the merged
   pool with one more edge may again. Smoke may recalibrate and FREEZE INTACT_FLOOR, all before the 6-seed
   queue (declared).
6. **Sign reversal risk** (table row 6) — if the learned effect amplifies self-generated surprise instead, the
   anchor is directionally wrong here; report honestly; variant B (prov_generated -> patient_expected) is the
   pre-declared next draft, never a post-hoc sign flip.
7. **One-sided** — no attenuation mechanism is built; 'self-generated LESS' is the control read only.
8. **Host supervision** — the co-occurrence experience is a teaching current; if growth only occurs under
   unrealistic co-drive, that bounds the recipe, not the biology (no-defer: change the recipe).

## 8. First smoke step

On dev seed 7 ONLY (calibration, never evidence): in the draft worktree, write the outcome table (section 5)
into the runner docstring FIRST, then run one cell end-to-end via the pool/workstation queue. Verify in order:
(a) the edge wires (harness mask assert non-empty); (b) config reconciliation holds (frozen_maxdrift ~ 0 — no
hebbian_rate_window hijack); (c) clean-zero control (measured prov_perceived under the generated exemplar
~ 0); (d) verdict drive present across the trial window (log per-condition prov-line rates + surprise window
(e) delta sign must be +1 as predicted; INTACT_FLOOR := 0.5 x smoke seed-7 delta_intact — the floor-setting
rule is PRE-DECLARED HERE: frozen verbatim into the runner docstring + WORKLOG before the 6-seed queue, never
revisited, never re-set from later seeds; then queue the 6 clean cells. If any
of (a)-(d) fails, fix the instrument before queueing — never read rows 1-6 with a broken instrument.
If (e) fails at smoke (sign not +1): do NOT freeze INTACT_FLOOR, do NOT queue — a negative floor would be
nonsense; report the sign-reversed smoke per row 6.
