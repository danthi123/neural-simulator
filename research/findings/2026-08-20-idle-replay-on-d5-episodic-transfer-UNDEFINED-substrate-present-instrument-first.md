---
type: finding
status: live
date: 2026-08-20
mechanism: idle-tick-replay-stabilization
lane: continuous-substrate
seeds: [42]
seed-waiver: A single-seed TRANSFER probe — does the synthetic-network emergent-replay mechanism run on the REAL D5 episodic bridge. The evidence is a within-run presence/absence of pattern-completion + specific write on one organ, plus a code-read of the D5 recurrent structure — not a stochastic effect size; a seed population measures nothing while the instrument precondition is failing.
instrument: research/runners/_idle_replay_on_d5_episodic_derisk.py — untargeted-noise replay + BTSP write on the UNCHANGED D5 EpisodicDapMemory (n_ca3=2000), vs a never-stored control
runner: research/runners/_idle_replay_on_d5_episodic_derisk.py
external: NO-EXTERNAL-NEEDED — an in-repo transfer probe of an in-repo mechanism onto an in-repo bridge; the next step is an instrument fix, not a literature question.
artifacts:
  - research/findings/raw/_idle_replay_on_d5_episodic/seed42.json
---
# UNDEFINED: emergent-replay onto the REAL D5 episodic bridge — the recurrent substrate IS present, but the instrument + noise-drive must be fixed before transfer can be measured

Artifact: research/findings/raw/_idle_replay_on_d5_episodic/seed42.json

**One line.** To wire LEARN-THROUGH-USE live, the 6-seed-GO emergent pattern-completion replay must run on the REAL
episodic memory. This de-risks the transfer onto the production D5 bridge (`EpisodicDapMemory`, n_ca3=2000). Two
results: (a) POSITIVE — the D5 CA3 already has the RECURRENT substrate the mechanism needs, and the store works; but
(b) the transfer MEASUREMENT is UNDEFINED — an instrument precondition failed AND the spontaneous untargeted-noise
replay produced no pattern-completion signal, so whether emergent-replay transfers is NOT yet answered.

## (a) POSITIVE, confirmed (de-risks feasibility)
- **The recurrent substrate exists.** By code-read (`_riii_ca3_coincidence_completion_derisk._build`): the D5 CA3 has
  a genuine dense CA3→CA3 `RegionPathway` (`ca3_recurrent_density`=0.5 in D5's GO defaults) routed through a
  coincidence detector, with within-assembly recurrent weights that BTSP-grow toward `btsp_w_max`=100 — a real,
  load-bearing recurrent net for pattern completion. So the emergent-replay mechanism is not structurally excluded.
- **The store works on the real bridge.** `store('dog')` potentiated dog's within-assembly weight 2.84 → 82.04
  (`reaches` precondition TRUE) — the encode the replay would strengthen is genuinely formed.

## (b) The transfer measurement is UNDEFINED (the runner's own verdict; the silent-failure discipline)
The runner's Verdict machinery returned UNDEFINED because a REQUIRE precondition failed — NOT a NO-GO (a failed
precondition means the effect is not cleanly attributable, never a validated negative):
- **Instrument precondition FAILED:** "D5's own intact / unstored / lesion / quiet-inert / recall-reproducible all
  hold" = FALSE. The instrument (D5's own baseline behaviour) did not validate, so nothing measured on top of it is
  trustworthy yet.
- **Pattern-completion did NOT fire:** `pc_apical_gap` = 0.0, `pc_apical_max_gap` = 0.0 — untargeted noise recruited
  NO apical-dAP gap between the formed (dog) and never-formed (cat) assemblies. The write also did not fire
  (`dw_dog` = `dw_cat` = 0.0). So the spontaneous noise dose, as configured, did not drive the recurrent completion
  on the real sparse (~1%) code.

## Why it doesn't transfer as-is (and connects to D5's OWN prior finding)
The synthetic emergent-replay de-risk drove ~18% of the pre-assembly DIRECTLY with noise and let recurrence complete
the rest. Here PURE untargeted noise into a ~1%-sparse 2000-cell CA3 (18% of the population, 900pA, 200 steps) crossed
the per-cell coincidence threshold (kt=8) for NO held cell of EITHER topic — dog and cat apical-UP tied at exactly
0.000, write tied at 0.000. This is not a fluke: D5's OWN precedent already found that reading this recurrent
attractor at the POPULATION level is non-specific at this assembly scale (the dapB soma-read was 0/6 across seeds,
`research/findings/raw/_gap5_dapB/dapB_6seed.json`), which is exactly WHY D5 abandoned population reads for an
intrinsic per-cell dAP latch driven by a HOST-KNOWN cue — nothing in D5's pipeline ever drives an untargeted random
CA3 subset. So untargeted-CA3-noise replay is the WRONG driver at this sparsity; the substantive transfer read is a
NEGATIVE (0/0), with the run's UNDEFINED coming from a secondary instrument fragility (recall() varied by ±1 held
cell between calls on a provably-inert state — a real, separate finding about the completion read's noise floor at
small held-cell counts).

## The next mechanism (biologically-correct: drive the AFFERENTS, not CA3)
Not "more CA3 noise" (that risks the same excitability-not-completion non-specificity dapB already hit). The right
driver is the DG/EC AFFERENT layer: the mossy-fiber DETONATOR already concentrates activity onto whichever CA3 cells
were selected for a stored pattern (per `emergent_assemblies`' own mechanism), so a content-blind afferent drive is
STRUCTURE-aware — it seeds the stored assembly without a host-known cue, which is exactly how biological sharp-wave-
ripple (SWR) replay INITIATES. So the bridging step is: replace untargeted-CA3-noise with DG/EC afferent drive, re-run
the transfer probe, then (if the dog-vs-cat gap appears) add the metaplastic starting-weight gate, 6-seed, and wire
under the continuous idle tick default-off. Also fix the secondary instrument fragility (average the completion read
over a few draws, or raise the held-cell count) so the baseline validates. Not wired live — transfer is a NEGATIVE
as-configured, with a biologically-grounded next driver named. The recurrent substrate being present + BTSP-formed is
the load-bearing good news: the mechanism is not excluded, it was driven at the wrong locus (CA3 soma vs the
afferents). (Agent-built + launched; parent read the runner's OWN verdict + the agent's afferent-drive analysis, and
banked it honestly.)
