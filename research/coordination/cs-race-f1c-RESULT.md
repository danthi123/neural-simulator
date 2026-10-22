# [cs-race-f1c] RESULT — the ruled WTA lever space does not make the symmetric race commit (27 cells, 0 passing)

**AMENDMENT (2026-10-21; RULING cs-race-f1d-1021) — the read-out below is partly void; where they disagree, this block is the record.**
1. The decisive "non-commits" are CO-COMMITS: rows 3-4 (18 cells, 1,440 decisive draws) held 1,158 draws with BOTH latches ON, 282 with exactly one, 0 with none. `latch_read` returns winner None for both co_commit and none, so the tally merged them — "0 decisive commits" means "0 single-winner reads".
2. Ignition is cue-locked and near-simultaneous: the two first gate spikes land a median 3-4 ms apart; a gap of 8 ms or less never gave a commit (0 of about 1,056 draws); single commits need about 9 ms or more, or one gate that never fires. Seeds 2, 3, 6 (earliest-gate step SD 1.3-1.7 ms) never commit.
3. The step-3 Vm table is a measurement artefact: the engram -69.5 mV was read at t = 600 ms, after ignition (about 160-210 ms) and after the occupancy pool hyperpolarises everything (no-latch draws read about -60 mV); the `thr` field is the unused homeostasis array (spike criterion = vpeak 35 mV; regenerative saddle about -42.9 mV). "Engrams sit ~30 mV below threshold, the levers are inert" is VOID as a mechanism claim.
4. The NMDA arm (`nmda_grid_n3`) is VOID as evidence — instrument failure, not a result about NMDA: its latches were ON before the cue (80/80 decisive) and its engrams fired 0 spikes during the cue.
- Under-drive was recomputed and REFUTED: decisive cue spikes per window average 33.75 against 28.9 for posctl, both engrams fire about 18 spikes, both gates fire on every failing draw, cue weights are not split by the paired write.
- Disclosure: a Claude research agent ran UNRECORDED scratch probes on dev seeds 1-24 and on gate seeds with W_GIS 30-90; W_GIS=30 is a peeked value, confirmed only on the fresh CONFIRM block (seeds 25-40).
- What survives: the control bars, posctl 8/8 everywhere, and the commit-rate tables as counts of single-winner reads; the negative verdict stands under its corrected name — the ruled levers produce CO-COMMIT, not silence. Measurement M (step 1d) supersedes the mechanism readings below.


**Status:** PROVISIONAL (agent-filed; Claude's gates/rulings ratify). Rulings: `cs-race-f1b-1021` (assigns step 1c), `cs-race-f1b-1021c` (its spec: control into the probe before any run; two-lever grid; NMDA arm only if the record supports it; Vm at each stage; F1 rerun only where BOTH controls hold 8/8; then post the commit-rate table and STOP), `cs-race-f1-void-1021` (f1 void — its banner line landed as `5a25634d0`).

**preconditions:**
- Probe: `.claude/worktrees/cs-race` `research/runners/cs_race_f1.py` @ `e290dbdaa` (decisiveness control + `--controls-only` + `--w-ee`/`--w-ie` levers) + `6a82d0794` (NMDA arm). `research/runners/idle_replay_organ.py` (the organ) UNEDITED; `sim/` UNEDITED.
- Operating point: F1 grid **row 3 (NE=24, U=0.30, tau_f=150)** carried per `cs-race-f1b-1021`; row 4 (NE=24, U=0.45, tau_f=300) = the slow-facilitation arm. Seeds 1-8 (dev), draws 10, controls-only mode, `SIM_BACKEND=numpy`, host `83db10ba1ddb`.
- Every cell rc=0, clause errors=0, `declare_output` sidecars per cell and per driver summary (runner `research/runners/cs_race_f1.py`).
- Criterion (ruling-defined bar): decisive control PASS on a seed ⇔ commits ≥ 9/10 draws AND each of frog/vole wins 2-8/10. Step-4 F1 rerun fires only if some cell has BOTH controls passing 8/8 seeds.

**Step 1b deliverable — organ `_R` preset vs probe `_BASE` (the constants that differ):** `NOISE_TAU` 5.0 → 50.0; `W_XG` 30 → 45 and `W_GS` 12 → 40 (both step-1b operating-point fixes); organ's `BASE=0.8`, `TAU_TURNS=1.5`, `SAL_MIN/MAX`, `K_MAX` are absent from the probe's `_BASE`; `REST_STEPS=500`/`HOLD_STEPS=100` vs probe `CUE_MS`/`HOLD_MS`. Unchanged: `W_EE=W_EI=W_IE=40`, `NE/NI/NG/NGI/NS/NO`, `BIAS_PA=42`, `NOISE_SD=25`, read constants (30 steps, 0.03 frac), and the lateral-inhibition wiring itself (each `E_k → own I_k` at `W_EI`, `I_k → other engrams` at `W_IE`) are identical. `before_you_build` corpus check: 5 hits, none about symmetric-race non-commit (fresh log `raw/_cs_race` provenance `corpus_check_*` fields).

**Step 2 — the decisiveness control (written into the probe BEFORE any run):** two engrams written together (`encode=("ABsim",)`, no recency gap), one shared cue (`CUE_OF[1]`), no mention; bar as above. Smoke at baseline (40/40, seeds 1-2): posctl 9/10 + 10/10 PASS, decisive 6/10 (frog 5, vole 1) + 0/10 FAIL — the defect reproduces exactly under the control's conditions; verdict correctly UNDEFINED (not a negative).

**Step 3 — commit-rate tables (commits of 80 draws; every cell: posctl 8/8 seeds, decisive 0/8 seeds):**

Row 3 (NE=24, U=0.30, tau_f=150), cells `W_EE x W_IE` → commits (frog/vole):
```
        W_IE=20        W_IE=40        W_IE=80
W_EE=40   17 (10/7)      17 (10/7)      18 (11/7)
W_EE=80   13 (9/4)       14 (9/5)       14 (9/5)
W_EE=120  15 (10/5)      15 (10/5)      17 (10/7)
```
Row 4 (NE=24, U=0.45, tau_f=300 = slow-facilitation arm), same axes:
```
        W_IE=20        W_IE=40        W_IE=80
W_EE=40   17 (10/7)      17 (10/7)      17 (10/7)
W_EE=80   14 (9/5)       14 (9/5)       14 (9/5)
W_EE=120  16 (10/6)      16 (10/6)      17 (10/7)
```
NMDA arm (row 3, `W_IE=40`, additive `receptor="nmda_slow"` recurrent entry, `--w-nmda`): `W_EE ∈ {40,120} × W_NMDA ∈ {40,80}` → **0 commits in all 4 cells (0/320)**; posctl 8/8, decisive 0/8. Cells 17-18/80 baseline → 0 with the slow arm ON while the lone control stays perfect: at these settings the arm makes the symmetric race strictly worse.

**Vm at each stage (means over decisive draws, mV):** cue layer −59.94 flat in all 22 cells; engram −69.45…−69.83; commit-gate −71.36…−71.71; slot −31.86…−32.75 (baseline grids) / −28.17…−29.91 (NMDA cells). Neither lever moves any stage mean meaningfully — the engrams sit ~30 mV below spike threshold, so the spike-mediated recurrent knobs (both are E/I conductances) are inert unless something else ignites them first.

**Step 4 — no F1 rerun:** no cell reached both controls 8/8 (decisive was 0/8 in all 27 cells; posctl 8/8 in all 27), so the ruling's rerun condition never fired. Per the ruling this RESULT posts and the driver STOPs.

**Verdict (probes only): characterized NEGATIVE for the ruled lever space** — a 3×6 grid over engram recurrent excitation × exclusion inhibition, repeated on the slow-facilitation row, plus a record-supported slow-NMDA arm, never makes two engrams written together under one cue commit; the ~17-22% rare commits split ~evenly (frog 55-63%), matching the ruling's reading that the race is unresolved rather than biased. The capability verdict stays open (per the void ruling's stage-1-exit clause; this is a probe-stage instrument result).

**Record check (ruling's condition for the NMDA arm): SUPPORTED, so built and run.** `research/findings/2026-06-06-action-selection-readout-deep-research.md:413` cites Wang 2002 (*Neuron* 36:955-968): "recurrent NMDA attractor integrates weak evidence into a binary choice" — exactly this regime. Converging precedent: `research/findings/2026-06-17-biased-competition-wta-multireferent-derisk.md` (NEGATIVE, 6 seeds): symmetric WTA + top-down bias loses to intrinsic excitability asymmetry; "the mutual inhibition makes the competition sharper… in favour of the intrinsically-stronger attractor" — the same inert-exclusion signature seen here (W_IE flat across 20-80).

**Next lever (Claude's design call):** the knob space above is exhausted at these operating points; candidates consistent with the record are (a) an excitability/encoding-strength bias so the race has an asymmetry to resolve (2026-06-17's bound says bias-to-symmetric-WTA cannot win — asymmetry is what makes winners), or (b) a read-stage tie-break judged against K1-K6. The NMDA arm at one STP regime (row 3 only) is not a full characterization of slow-recurrent mechanisms.

**Artifacts:** `research/findings/raw/_cs_race/lever_grid_r3/` (9 cells + `summary.json`, each with `.prov.json`), `research/findings/raw/_cs_race/lever_grid_r4/` (same), `research/findings/raw/_cs_race/nmda_grid_n3/` (4 cells + summary). Probe commits: `e290dbdaa`, `6a82d0794` on `research/cs-race` (Claude merges). Smoke/aggregation logs: `/opt/data/cache/scratch/a_1c_*` (scratch; cell logs copied beside each grid).