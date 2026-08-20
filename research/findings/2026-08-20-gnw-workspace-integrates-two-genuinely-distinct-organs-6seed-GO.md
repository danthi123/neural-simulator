---
type: finding
status: live
date: 2026-08-20
mechanism: gnw-workspace
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gnw_two_distinct_organs_derisk.py — 2-organ coincidence-ignition where the two subthreshold reads come from GENUINELY DIFFERENT organs (FHRR composer recall + the production SurpriseProductionOrgan spiking read), 2-hop re-entrant, full anti-cheat battery
runner: research/runners/_gnw_two_distinct_organs_derisk.py
external: NO-EXTERNAL-NEEDED — an in-repo generalization of the 2026-08-12 GNW GO to two distinct in-repo organs; the next step is wiring, not a literature question.
artifacts:
  - research/findings/raw/_gnw_two_distinct_organs/summary.json
---
# GO (6-seed): the workspace bus integrates TWO GENUINELY DISTINCT organs — closing the last "both reads came from the composer" caveat

Artifact: research/findings/raw/_gnw_two_distinct_organs/summary.json

**One line.** The 2026-08-12 GNW GO (the substrate COMBINES two subthreshold organ reads via coincidence-ignition, an
AND not a host if/else) carried one honest-scope caveat: both reads came from the SAME organ (the composer under two
relations). This de-risks the load-bearing generalization — does coincidence-ignition integrate two GENUINELY DIFFERENT
organs? **6-seed GO. The bus is organ-agnostic — the wiring prerequisite is closed.**

## The two organs are genuinely distinct (not two composer reads)
- **Organ A** = `rf_phasor_composer.query_patient(EAT)` — the FHRR composer's recall of a stored edge.
- **Organ B** = `surprise_production_organ.SurpriseProductionOrgan.read_surprise(cp_firing_states[surprise])` — the
  PRODUCTION spiking surprise organ, holding its OWN independent expectation and reading its OWN `cp_firing_states`.
  Organ B is a real spiking read, not a host boolean, and it DISCRIMINATES: mean surprise firing is 0.13 Hz when the
  two organs AGREE vs 2.74 Hz when they DISAGREE (all 6 seeds `organ_b_discriminate=True`) — the surprise organ fires
  low on a confirmed read and high on a conflicting one, exactly the convergent-evidence signal the workspace needs.

## The 6-seed verdict (all_go=True, 6/6)
<!--derived-->
`mean_coincidence_2hop_acc = 1.000` (= mean_query_chain_2hop_acc 1.000, host parity). EVERY anti-cheat collapses to
**0.000** across all six seeds: `a_only` and `b_only` (each organ subthreshold ALONE cannot ignite — the anti-host-
if-else keystone), `disagree` (conflicting reads → consensus-veto, no broadcast), `shuffle` (organ B off-target),
`onecycle` (a single pass can't reach the 2-hop conclusion — re-entry is load-bearing), `organ_b_lesion` (remove organ
B → collapse), `workspace_lesion` (no workspace → no ignition). `single_hop_reflex_acc = 1.000` (the dissociation
keystone — the single-hop path survives), `mutual_exclusion_frac = 1.000` (single-content access at every hop),
`all_moat_ok = True`. Preconditions hold on every seed (`all_precondition_ok=True`, `all_d_sub_in_window=True`:
d_sub=1400 solo-subthreshold, 2×d_sub ignites).

## Backend finding (a real instrument constraint, recorded)
The decisive run is on **numpy (float64)**, not cupy. cupy's float32 breaks the surprise organ's GABA_A cancellation —
`confirm` fires 2.18 Hz instead of ~0.08, destroying organ B's discrimination — which matches the surprise organ's own
test convention. So numpy is the REQUIRED instrument for any de-risk that reads the surprise organ; this is a
substrate-precision constraint to respect when wiring organ B, not a defect of the mechanism.

## Significance + next step
Coincidence-ignition is now proven organ-AGNOSTIC: ANY two organs that write subthreshold drive to the shared workspace
integrate identically (AND + consensus-veto + re-entry), whether they are two relations of one organ or two structurally
different organs. That is the load-bearing generalization that makes the GNW bus a real conversational faculty (combine,
e.g., a recall read + a surprise/familiarity read + an affect read into one broadcast conclusion). It closes caveat #1
of [[2026-08-12-gnw-coincidence-integrator-substrate-combines-two-organ-reads]]. NEXT (the finding's own path-to-wiring):
wire the bus into production `webapp/server.py` / `brain_chat_tui.py` — each organ writes its read as a subthreshold
drive into a persistent shared workspace region, the host per-turn combine step is replaced by ignition, a re-entrant
cycle feeds the ignited partial conclusion back, and metacog/conflict gates an extra cycle. Not wired live. (Agent-built
+ launched; parent verified all 6 seeds' anti-cheat means + the organ-B discrimination from the artifact before banking.)
