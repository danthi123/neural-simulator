---
type: result
status: qualified
date: 2026-11-02
mechanism: bg-live-seam-retirement, amendment 2 (preset-centred, bounded, sub-stepped cells at stn200_gpe100) with the spiking commit stage (BRAIN_BG_COMMIT)
prereg: research/findings/2026-11-01-bg-live-seam-gate-AMENDMENT-stn200-gpe100.md
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO as registered (G1, G2a, G2b, G3, G4 amended, G5 pass; every precondition holds), commit-with-caveat - G3 passes by one event and the credit is narrower than the amendment's sentence; no default changes
runner: research/runners/host_share_bg_seam.py
artifacts:
  - research/findings/raw/_bg_live_seam/commit_gate/e017eb0_preset/score.json
  - research/findings/raw/_bg_live_seam/commit_gate/e017eb0_tonic0/score.json
  - research/findings/raw/_bg_live_seam/commit_gate/e017eb0_hybrid/score.json
  - research/findings/raw/_bg_live_seam/commit_off_ablation/e017eb0_preset/score.json
---

# BG live-seam gate on stn200_gpe100: GO as registered, with a narrow credit

Date: 2026-11-02 · prereg: `research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md`, amended by
`research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md` and
`research/findings/2026-11-01-bg-live-seam-gate-AMENDMENT-stn200-gpe100.md` (+ its `-REPIN.md` addendum) · pin
`e017eb00e38e69d230a7d930e181049d7822160b` · scorer of record: `research/runners/host_share_bg_seam.py` AT THE PIN
(exported with `git archive e017eb00e`), `--score <OUT> --seeds 42 43 44 100 101 102 --commit --out <OUT>/score.json`.

host-decision: none changed. This is a measurement of the gated build behind its flags; no default moves.

## 1. Verdict

**GO.** The gate arm (`research/findings/raw/_bg_live_seam/commit_gate/e017eb0_preset/score.json`) reads `verdict: GO`, label GATE,
`production_seam_class: HOST-OWNED (replaceable)`, with every gated key true and every precondition holding:

| gate | measure (pooled, six seeds) | bar | value |
|---|---|---|---|
| G1 production host owns the reply | F | >= 0.8 | 0.9583 (23/24) <!--derived--> |
| G2a neural seam holds empty turns | empty_held (seam_b) | >= 0.75 | 0.8333 (10/12) <!--derived--> |
| G2b neural seam never holds a question | content_held (seam_b `hsb_q`) | = 0 | 0.0 (0/6) <!--derived--> |
| G3 the value route moves the reply | ns | >= 0.3 | 0.375 (9/24) <!--derived--> |
| G4 (amended) commit read invariant at x0.5 / x2 | commit_thr_change | = 0 | 0.0 (0/24) <!--derived--> |
| G5 the bus argmax does no work | bus_unique_agree | >= 0.95 | 1.0 (12/12) <!--derived--> |

Preconditions: sham null 1.0, membrane in [-150, +100] mV on every step (observed -90.9 to +40.0 mV) <!--derived-->,
commit engaged 1.0 on 72/72 neural-seam events with exactly one per seam-arm turn, `commit.op == 4` and
`commit.organ_seed` == chat seed on all 72, every gated measure n >= 6. The reported-only old-latch G4 is false
(0.125), which is the point of the commit stage.

The amendment's sentence for a GO is: "Behind the flags, the live seam's admission, score and commit are neural."
Section 3 of the amendment limits it further: a GO credits the spiking commit stage and the value route on
preset-centred cells, not the tonic STN/GPe drive. The verification below narrows it again (section 3).

## 2. Report-only ablations (never part of the verdict)

| arm | score file | verdict | what it says |
|---|---|---|---|
| A-tonic0 (`BRAIN_BG_TONIC_DRIVE=0`) | `research/findings/raw/_bg_live_seam/commit_gate/e017eb0_tonic0/score.json` | GO | Prediction MET: committed winners equal the gate arm on 24/24 events; GPe/STN go to 0 Hz yet GPi barely moves. The tonic drive is inert by wiring (STN does not reach GPe; GPe does not reach GPi). No pallidal credit. | <!--flag-not-on-main: the gate code stays on research/bg-gate-amend-1101 and is reached through the pinned revision e017eb00e (amendment section 9); no default flips-->
| A-hybrid (`BRAIN_BG_PRESET_CELLS=0`) | `research/findings/raw/_bg_live_seam/commit_gate/e017eb0_hybrid/score.json` | NO-GO | Prediction MET: the committed winner differs on 17/24 events; empty_held 0.0, content_held 0.3333. The preset cell types carry the result. The knob also changes the production arms (F 0.0), so the comparison is not confined to the seam. | <!--flag-not-on-main: the gate code stays on research/bg-gate-amend-1101 and is reached through the pinned revision e017eb00e (amendment section 9); no default flips-->
| A-commit-off (no `--commit`) | `research/findings/raw/_bg_live_seam/commit_off_ablation/e017eb0_preset/score.json` | NO-GO (09-29 scorer) | Prediction FAILED (latch_thr_change 0.0, predicted > 0.10), but the arm is CONFOUNDED: without the commit flag the organ seed is pinned to 42 for every chat seed (`research/runners/bg_action_selection_production_organ.py` at the pin, the `seed = 42 if seed is None` branch), so latch rates are identical across the six seeds. It is one organ, not six, and its losing latch rate is exactly 0.0 on every event (a floor). It neither supports nor refutes "the commit stage makes the read invariant". The usable paired evidence is inside the gate arm: old latch read 0.125 vs commit read 0.0 on the same trials. |

## 3. Verification (verify-go, five independent refuters; none refuted the registered GO, all five narrowed it)

Lenses: gate-cheat, control integrity, per-seed reproducibility, attribution/mass, instrument trust + where-ran.

- **Gate-cheat.** Every precondition is computed from the shards and invoked in `verdict()` at the pin; every bar
  recomputed by hand from the raw JSON matches. Latent scorer gap (not triggered here): a missing or errored
  production arm leaves `sham_ok` = 1 vacuously. Two of 24 seam_b events (s42 and s102 `hsb_empty2`) carry no
  committed winner and pass `commit_engaged` / `commit_thr_change` vacuously; they are the two G2a misses, and
  excluding them flips no bar.
- **G3 is one event from NO-GO.** 9/24 against a bar met at 8/24. Per-seed ns is 0.25, 0.75, 0.5, 0.5, 0.0, 0.25
  (s42, s43, s44, s100, s101, s102) <!--derived-->; dropping s43 lands exactly on the bar. Only seeds 43 and 100
  clear every bar on their own (the prereg gates the pooled value, so this does not change the verdict).
- **What G3's numerator is.** 6 empty-turn hold losses, 2 content-turn wrong holds (s43, s44 `hsb_q`), 1 one-word
  spoke->held (s43 `hsb_hi`). Of the 6 hold losses, 3 are the lesioned organ committing SPEAK and 3 are the lesioned
  organ not committing at all, so the host falls through and answers "I don't know about that." Counting only events
  where the lesioned organ itself commits the other action gives 6/24 = 0.25 <!--derived-->, below the bar.
- **The value lesion is not selective.** On s42 and s100 it silences all motor output on all four turns. It closes
  heard->SPEAK and hold->SILENT together, so G3 cannot separate a value/content signal from the general drive that
  lets the motor pools reach the commit threshold.
- **The "70% in the control" warning is mostly pooling.** By turn class: empty turns held 0.8333 intact vs 0.3333
  lesioned; content 0.0 vs 0.3333; one-word 0.0 vs 0.1667. The magnitude-free discrimination (held empty minus held
  content) is 0.8333 intact and 0.0 lesioned <!--derived-->: in the commit-on organ ALL of the empty-vs-content
  discrimination needs the value afferents. Paired empty-turn flips: 6 held->spoke, 0 spoke->held. Four of ten intact
  empty holds survive the lesion; they carry no class information (the same lesioned organ also holds questions).
- **G4 is saturated, not a tested margin.** On every committed event the losing slot count is 0 against 76-92 for the
  winner, so no x0.5/x2 read can flip it. G4 shows a wide separation.
- **Readout permutation never engaged** (alt winner == latch winner on 24/24); `readout_share` 0.0 says nothing.
- **Answer rate.** Questions are answered 6/6 in both seam_b and prod_a; `hsb_hi` abstains 6/6 in both. The seam turns
  2 of 12 empty-turn holds into "I don't know about that." (s42, s102 `hsb_empty2`, the two no-commit events). No
  content answer is lost.
- **Where it ran.** `tools/where_ran.py` over the gate arm: 42 files, pool = 21, workstation = 21; s42/s43/s100 on
  pool nodes, s44/s101/s102 on the workstation, each seed's seven arms on one host. Production answers reproduce
  across hosts. ns by host class is 0.50 (pool) vs 0.25 (workstation), fully confounded with seed.
- **Instrument.** The scores first read at S main's scorer (`de84f2cc7`, which does not contain the pin) gave the same
  gated JSON; the scorer of record is the pin's, whose `score.json` files are the artifacts above. All 175 provenance
  sidecars: `git_sha` e017eb00e, `git_dirty` false, manifest verified at start and exit. Gate seed s42 ran as the pool
  canary before the dev-seed s7 preflight finished (ordering, not validity).
- **Faithfulness flag.** GPi reads 170-330 Hz in every arm, far above the 40-80 Hz physiological band the amendment
  cites. STOP 4 (report-only): `str_d1_0` goes below its rest bound (-85 mV) on 19 of 72 events, v_min -90.9 mV
  <!--derived-->.

## 4. What this GO credits, stated honestly

In the commit-on, preset-cell organ, the value afferents (heard/hold -> D1) are NECESSARY for the seam to tell empty
turns from questions, and the commit is a spiking latch whose read is invariant across a wide separation. It does
NOT show that the value route carries the decision's content as distinct from general drive, it credits nothing to the
tonic STN/GPe drive, and it says nothing about the commit stage's necessity (the commit-off ablation is confounded).
Everything is wired, not learned (learned-content fraction 0), the organ is its own bridge (not on the one-brain
substrate), and no flag is on by default.

## 5. Next methods (no defer; filed the same day)

1. **Commit-off ablation, unconfounded.** Fix the runner so the organ seed follows the chat seed with or without
   `--commit`, then re-run A-commit-off on the six seeds (report-only). FAILURE_LOG line lands with the fix.
2. **A selective value-route lesion.** Lesion heard->SPEAK and hold->SILENT separately, so G3-type evidence can tell a
   content signal from general drive. Report-only, dev seeds first.
3. **Pallidal ladder (amendment section 8), step 2 before step 1.** GPi at 170-330 Hz is a faithfulness defect that
   sits under every BG number; recalibrate `gpi_tonic_pA` against the preset f-I curve and the 40-80 Hz band, then
   build the STN <-> GPe loop with GPe -| GPi.
4. **Integration.** A default-on flip of the neural seam is NOT proposed from this gate: it passes by one event,
   adds two empty-turn abstentions, and rests on an unphysiological GPi. The flip packet waits for items 2 and 3 and a
   flip battery with flag-off arms and an answer-rate read.

## 6. Honesty boundary

Every number here is a functional read-out of the organ and the registered scorer (held fractions, spike counts,
rates, membrane steps). Nothing here asserts phenomenal experience.
