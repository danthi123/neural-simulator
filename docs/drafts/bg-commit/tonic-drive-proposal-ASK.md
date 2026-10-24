---
type: preregistration
status: DRAFT
date: 2026-10-19
mechanism: bg-live-seam-retirement, amendment 2 (tonic pallidal drive) — DRAFT for Claude's review
amends: research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md
also-covers: ldc-gpe-tonic (RULING ldc-synzero-1014; no separate prereg — the LDC ladder owns the tonic question)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION AMENDMENT DRAFT only (pre-data). No dev seed 9-16 and no LDC probe of the tonic build has run when this is drafted.
runner: research/runners/host_share_bg_seam.py
organ: research/runners/bg_action_selection_production_organ.py
flag: BRAIN_BG_TONIC_DRIVE (default OFF)
---

# Amendment 2 (DRAFT): tonic pallidal drive — one investigation for bg-commit-next and ldc-gpe-tonic

**Status: DRAFT for Claude's review; pre-data.** To be merged/registered by Claude, never by the driver. This draft is
the ONE flag-gated proposal required by RULING bg-commit-next-1014 step 3 and by RULING ldc-synzero-1014 step 3 (the
[ldc-gpe-tonic] ASK is not filed separately: both rulings name the same missing mechanism, so both run as ONE
investigation — one shared flag, one shared reading, one amendment).

bar-sanity: docs/drafts/bg-commit/bar_sanity_record_tonic_drive_amendment2.json (created at registration; bars unchanged)

## Why one investigation

RULING bg-commit-next-1014 step 2 (the wall): real basal ganglia hold posture/action by DEFAULT through tonic
inhibition of thalamus/cortex by GPi/SNr, itself driven by tonically active GPe and STN pacemaking neurons; producing
an action requires positive release (the direct pathway / D1 counteracts the tonic hold). RULING ldc-synzero-1014: a
717-synapse str_d2->gpe cut is bit-for-bit silent in every DA/spike scalar across 20 trials, and the gpe-observability
probe records 0 GPe spikes in the reward window in BOTH arms while str_d2 fires 667/630 and DA is bit-identical —
"GPe silent = missing tonic pacemaking, not a dead cut." Same missing mechanism on both sides: without tonic pallidal
drive, GPe/STN are quiescent, the D2->GPe cut has nothing to remove, and the BG "hold" has no default to release. ONE
mechanism addition therefore answers the bg-commit-next wall question and the ldc-gpe-tonic question together.

## The one mechanism addition (flag-gated, default OFF)

**Flag: `BRAIN_BG_TONIC_DRIVE` (default absent/0 = OFF; `--tonic-drive` on any runner that takes CLI flags).** With the
flag ON, the organ (`research/runners/bg_action_selection_production_organ.py`) gives the GPe populations (`gpe_0`,
`gpe_1`) and the STN population a constant depolarizing pacemaking current for the whole trial (drive, hold, the
W_RESET inter-turn window, washout): a regular tonic firing mode so they fire at a baseline rate outside (and inside)
the reward window instead of sitting at 0 spikes. Minimal biophysical sketch: one constant current per population
(I_tonic, pA) — the pacemaking analogue of the real GPe/STN autonomous regular-spiking mode; no new synapses, no
learned content, no host computation. **Every existing weight and read parameter is untouched**: W_XG 30, W_GG 20,
W_GGI 60, W_GIG 60, W_GS 12, W_SS 60, W_GIS 60, W_RESET 16 (16.0 on every sweep row — not a swept dimension),
densities, READ_STEPS 30, READ_FRAC 0.03, `BG_COMMIT_OP = 4` and its frozen row
(`tests/test_bg_commit.py::test_frozen_row_4`). I_tonic is the only new degree of freedom; its magnitude is a
dev-swept operating point, declared in the worklog before any gate seed.

## What is unchanged

The question, the turns (`hsb_empty1` "...", `hsb_q` "what does the dog chase", `hsb_empty2` "?", `hsb_hi` "hi"), the
seven arms and their environments, the scorer's 09-29 measures, and **every existing bar, unchanged**: G1 F >= 0.8,
G2a >= 0.75, G2b = 0, G3 >= 0.3, G4 (amendment-1 commit read) = 0, G4 old-latch <= 0.10, G5 >= 0.95. Every 09-29
precondition (the sham null on every target event; the membrane in [-150, +100] mV on every neural-seam trial; at
least 6 events per gated measure) and every amendment-1 precondition and construction guard stands. The 09-29 latch
stays built and its read is recorded from the same trial, never used. The LDC side needs NO new prereg: its rung
ladder owns the tonic question and keeps its own bars.

## What changes

1. **Build (flag ON only).** The three seam arms (`seam_b`, `seam_readout`, `seam_lesion_value`) add
   `BRAIN_BG_TONIC_DRIVE=1`; the prod arms are untouched. The pacemaking current above is the entire change to the
   organ. Nothing else in `webapp/server.py`, the scorer, or the trial schedule changes.
2. **Identity (construction guard).** With the flag OFF the build is byte-identical to the amendment-1 commit build,
   in the same flag-off byte-identity style as `tests/test_bg_neural_seam.py` / `tests/test_bg_commit.py` (hashes
   captured at the amendment-2 commit). A failure is a bug to fix before any scoring. No default flip anywhere.
3. **Report-only diagnostics (ADDED; no bar changes).** (a) GPe baseline firing rate in the reward window, per arm and
   per trial (spikes/neuron/ms over the window the gpe-observability probe read; expected > 0 with the flag ON, 0 with
   it OFF). (b) STN baseline firing rate, same window. (c) Tonic-hold level at the GPi/SNr-equivalent output of the
   organ if such a population exists there; where it does not, the nearest downstream tonic-hold proxy at the
   motor/commit read (reported as a proxy, not as GPi/SNr). (d) The str_d2->gpe cut's effect with the flag ON (the LDC
   reading: does the cut stop being bit-for-bit silent once GPe has a baseline to remove?). (e) Carry-over diagnostics
   from bg-commit-next step 1: per-empty-turn G-side drive and slot counts surviving the W_RESET = 16.0 inter-turn
   window, read alongside the re-settling S latch (the carried value-input state behind the seed asymmetry 42 T/F,
   43 T/F, 44 T/T, 100 T/F, 101 F/T, 102 T/T).
4. **Shared reading.** One read across both sides: GPe engages (nonzero tonic baseline in the reward window) on the bg
   dev runs AND on the ldc dev probe; the bg bars hold unchanged on the flag-ON build; the flag-off identity check is
   green. The LDC rung-2 ladder (400-trial cells) resumes ONLY after the probe confirms GPe engages.

## Dev plan (dev only; no gate seed in this draft)

- **bg side: dev seeds 9-16 first** (RULING bg-commit-next-1014 step 3), organ level, on the existing commit-build dev
  harness (`research/runners/_bg_commit_derisk.py`, or a tonic-specific dev runner added beside it), raw under
  `research/findings/raw/_bg_commit/` (a `dev_tonic/` sibling). Per seed: the live four-turn sequence plus trials per
  message class on one warm organ; report the new diagnostics and the same dev-table columns the amendment-1
  operating-point sweep used. The I_tonic operating point is chosen on these dev seeds only, by a rule declared in
  `docs/drafts/bg-commit/WORKLOG.md` before any gate seed.
- **ldc side: dev only** (RULING ldc-synzero-1014 step 1): read-only GPe-rate measurement on the intact arm dev 7, A/B
  with the flag OFF/ON, 1 trial per phase, `gpe_0`/`gpe_1` already on the instrument candidate list. The rung-2 ladder
  resumes only after this probe confirms GPe engages under the flag.

## Preconditions and risks (A0.2-style)

- **R1 tonic firing changes baseline latch dynamics** (the G/S latch race, the empty-turn hold). Mitigation: default
  OFF + flag-off byte identity (change 2); dev seeds 9-16 read the latch race before any gate seed; bars unchanged, so
  a changed race surfaces as a bar failure, never as a moved bar.
- **R2 the tonic drive interacts with the W_RESET window.** Step 1 found the reset clears carry bookkeeping without
  fully clearing the carried drive; a current persisting through the window may amplify or suppress that carry-over.
  Mitigation: change 3(e) reports carry-over per empty turn; any change of the empty1/empty2 asymmetry pattern under
  the flag is reported before gate seeds.
- **R3 GPe/STN engagement perturbs DA and the value read** (G3's lesion bar; the value-lesion asymmetry where 7 of 24
  lesioned events still held). Mitigation: the same report-only diagnostics on the dev trials; G3's bar stays 0.3 and
  is read on gate data by Claude.
- **R4 operating-point selection risk.** I_tonic chosen after seeing dev data can lean toward incumbent behavior (the
  amendment-1 row-4 precedent). Mitigation: the selection rule is declared in the worklog before any gate seed and
  stated plainly in the registration.
- **R5 scope creep into the frozen row.** Any change to `BG_COMMIT_OP`, row 4, or the trial schedule voids the
  amendment (amendment-1 item 1); the pacemaking current must not touch any weight. Guarded by
  `tests/test_bg_commit.py::test_frozen_row_4` and by change 2.
- **R6 "GPe engages" is necessary, not sufficient.** A nonzero baseline does not guarantee the D2->GPe cut stops being
  inert; change 3(d) reads the cut's effect directly before the ladder burns 400-trial cells.

## Driver vs Claude

**Driver may (after approval):** implement the flag-gated mechanism behind `BRAIN_BG_TONIC_DRIVE` (default OFF), run
the flag-off identity check, run dev seeds 9-16 and the ldc dev 7 probe, write results to
`docs/drafts/bg-commit/WORKLOG.md`; before approval, file the ASK and commit draft docs only. **Claude only:**
merge/register this amendment (amendment 2) against the 09-29 prereg, create the bar-sanity record, run gate seeds
42/43/44/100/101/102, and rule the verdict.

## Outcome -> verdict

GO iff every precondition holds, the flag-off identity check is green, and every bar of the 09-29 prereg as amended by
amendment 1 passes at its unchanged value on the flag-ON build, with the shared reading showing GPe engaged. NO-GO iff
preconditions hold and any bar fails. UNDEFINED otherwise (a precondition failure is never NO-GO). A GO means: the
tonic pallidal drive is a built, measured, default-OFF mechanism whose addition moves no bar and gives GPe/STN the
baseline the biology requires. It does not turn any flag on by default and does not by itself retire the live seam —
that remains the 09-29 capability, gated by the same unchanged bars.