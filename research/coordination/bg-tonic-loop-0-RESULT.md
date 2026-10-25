# [bg-tonic-loop-0] RESULT - step 0 (four-arm seam x isolated cells) 2026-10-24

Status: PROVISIONAL (agent-submitted; verify-go / Claude review pending).
Packet: docs/drafts/bg-tonic-loop/PACKET.md. Rulings: bg-tonic-loop-1020 (0a GO),
bg-tonic-loop-partA-1020b (step-0b spec + this RESULT's table shapes), -1020a (rule 2
bounded cond), parallel-builds-1021 (build order), wrc-frame-0-1021/-1021b.
Seeds 9-16 (dev). Revisions: Part A 1b96700be, Part B tip 8e3eafd6f, Part C 71efc4a5c.

## 0. Entry gate (read this before anything else)

The gate module research/runners/_bg_loop_entry_gate.py (branch research/bg-tonic-loop)
was copied byte-identically (sha256 vs git blob verified) and run at home:

```
entry gate: compared=96/96 missing=0 mismatched=0
gate PASS: A0 reproduces the refile exactly on all 96 records
```

Self-test (refile against itself): PASS 96/96.

Operational note: the a0/s11 cell was dispatched 2026-10-21 23:56 on wspool and died
without output; requeued verbatim once (2026-10-24 01:27, FRONT=1, canary on record);
landed 7/7 artifacts. No other cell death.

## 1. What step 0 decides (packet sec 3, fixed before the data)

**MAIN RESULT: the seam does not survive the move to real cells at today's constants.**
A0p (preset-cell params, nothing bounded) differs from A0 (hybrid) on the committed
winner, this turn's reply, or abstention on **all 8 seeds** (and on read_invariant on
s10, s13). Per the packet's first decision clause, that is the main result and
**the recalibration becomes the method** (step 1 onward, spec written by Claude).

- A5 (hybrid + bounded) vs A0: same failure shape (winner/answer/abstained all 8 seeds;
  read_invariant s9, s10) -- bounding alone does not restore the hybrid seam.
- A5p (preset + bounded) vs A0: closest arm -- read_invariant identical on all 8 seeds,
  abstained differs on 3 (s11, s14, s16), answer on 7, winner on 8 -- still a fail under
  the packet's any-seed rule, but the bounded preset arm recovers most of the behavior.
- Sub-reversal steps below E_inh - 5 mV (Part A, summed over 21 regions x 8 seeds):
  A0=0, A0p=8319, A5=0, A5p=0. The rule FIRES: every later step runs with
  BRAIN_BG_BOUNDED_COND=1 (as ruling -1020a already specified). A5p leaves no such steps,
  so the report-regions clause does not engage.
- Nothing in step 0 picks a current, a weight or a mechanism (packet sec 3).

## 2. Part B record-level decision read (seam arms x seeds 9-16)

Comparator fields: committed winner, read_invariant, latch onset and latch rates,
gate counts (drive/hold), motor counts (hold), tonic_rates_hz, and per-turn
answer + abstained. Coverage: 0 files missing (8 seeds x 4 arms x 7 files).

MISSING files: 0

a0p vs a0: decision-field diffs=228 table-field diffs=670
  DECISION abstained: seeds [9, 10, 11, 12, 13, 14, 15, 16]
  DECISION answer: seeds [9, 10, 11, 12, 13, 14, 15, 16]
  DECISION read_invariant: seeds [10, 13]
  DECISION winner: seeds [9, 10, 11, 12, 13, 14, 15, 16]
  table fields: {'gate_counts_drive': 96, 'gate_counts_hold': 96, 'latch_onset_ms': 96, 'latch_rates': 96, 'motor_counts_hold': 96, 'slot_counts': 94, 'tonic_rates_hz': 96}
  -> a0p DIFFERS from a0 on decision fields (winner/read_invariant/reply/abstain)

a5 vs a0: decision-field diffs=199 table-field diffs=667
  DECISION abstained: seeds [9, 10, 11, 12, 13, 14, 15, 16]
  DECISION answer: seeds [9, 10, 11, 12, 13, 14, 15, 16]
  DECISION read_invariant: seeds [9, 10]
  DECISION winner: seeds [9, 10, 11, 12, 13, 14, 15, 16]
  table fields: {'gate_counts_drive': 96, 'gate_counts_hold': 96, 'latch_onset_ms': 96, 'latch_rates': 96, 'motor_counts_hold': 96, 'slot_counts': 91, 'tonic_rates_hz': 96}
  -> a5 DIFFERS from a0 on decision fields (winner/read_invariant/reply/abstain)

a5p vs a0: decision-field diffs=73 table-field diffs=634
  DECISION abstained: seeds [11, 14, 16]
  DECISION answer: seeds [9, 10, 11, 12, 13, 14, 16]
  DECISION winner: seeds [9, 10, 11, 12, 13, 14, 15, 16]
  table fields: {'gate_counts_drive': 89, 'gate_counts_hold': 89, 'latch_onset_ms': 93, 'latch_rates': 90, 'motor_counts_hold': 88, 'slot_counts': 89, 'tonic_rates_hz': 96}
  -> a5p DIFFERS from a0 on decision fields (winner/read_invariant/reply/abstain)

=== per-seed decision matrix (arm: n_decision_diffs) ===
s9 a0p:28  a5:28  a5p:7
s10 a0p:31  a5:20  a5p:10
s11 a0p:28  a5:25  a5p:7
s12 a0p:27  a5:20  a5p:8
s13 a0p:27  a5:29  a5p:12
s14 a0p:30  a5:27  a5p:10
s15 a0p:27  a5:26  a5p:3
s16 a0p:30  a5:24  a5p:16

## 3. Part A - all-region table, param asserts, sub-reversal input

## Part A coverage: 64 of 64 runs

## Part A all-region table (mean over seeds 9-16; rate range = min rate_min..max rate_max; silent = summed; isi_cv/burst = medians; subrev = summed v_steps_below; v_min = min)
| current | arm | region | rate_mean | rate_range | silent | isi_cv_med | burst_med | subrev | v_min |
| stn40 | A0 | commit_0 | 0.88 | 0.0..12.0 | 210 | nan | nan | 0 | -65.6 |
| stn40 | A0 | commit_1 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -70.8 |
| stn40 | A0 | commit_fs_0 | 1.62 | 0.0..20.0 | 112 | nan | nan | 0 | -61.4 |
| stn40 | A0 | commit_fs_1 | 0.00 | 0.0..0.0 | 128 | nan | nan | 0 | -56.3 |
| stn40 | A0 | gpe_0 | 33.86 | 9.0..77.0 | 0 | nan | nan | 0 | -66.5 |
| stn40 | A0 | gpe_1 | 29.48 | 8.0..72.0 | 0 | nan | nan | 0 | -65.2 |
| stn40 | A0 | gpi_0 | 157.90 | 82.0..237.0 | 0 | nan | nan | 0 | -50.0 |
| stn40 | A0 | gpi_1 | 164.86 | 99.0..250.0 | 0 | nan | nan | 0 | -50.0 |
| stn40 | A0 | motor_0 | 3.01 | 0.0..36.0 | 210 | nan | nan | 0 | -68.6 |
| stn40 | A0 | motor_1 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -63.0 |
| stn40 | A0 | practice_arousal | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.0 |
| stn40 | A0 | proposal_0 | 0.00 | 0.0..1.0 | 479 | nan | nan | 0 | -66.3 |
| stn40 | A0 | proposal_1 | 0.00 | 0.0..0.0 | 480 | nan | nan | 0 | -66.2 |
| stn40 | A0 | selector_reset | 0.00 | 0.0..0.0 | 160 | nan | nan | 0 | -55.0 |
| stn40 | A0 | selector_stn | 21.59 | 0.0..120.0 | 30 | nan | nan | 0 | -66.1 |
| stn40 | A0 | str_d1_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A0 | str_d1_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A0 | str_d2_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A0 | str_d2_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A0 | thal_0 | 0.66 | 0.0..13.0 | 173 | nan | nan | 0 | -60.0 |
| stn40 | A0 | thal_1 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -55.4 |
| stn40 | A0p | commit_0 | 234.38 | 0.0..500.0 | 120 | nan | nan | 4000 | -228.1 |
| stn40 | A0p | commit_1 | 237.27 | 0.0..500.0 | 120 | nan | nan | 4000 | -238.1 |
| stn40 | A0p | commit_fs_0 | 250.00 | 0.0..500.0 | 64 | nan | nan | 0 | -55.0 |
| stn40 | A0p | commit_fs_1 | 250.00 | 0.0..500.0 | 64 | nan | nan | 0 | -55.0 |
| stn40 | A0p | gpe_0 | 86.37 | 58.0..125.0 | 0 | nan | nan | 0 | -50.1 |
| stn40 | A0p | gpe_1 | 84.77 | 48.0..125.0 | 0 | nan | nan | 0 | -50.1 |
| stn40 | A0p | gpi_0 | 278.04 | 200.0..334.0 | 0 | nan | nan | 0 | -50.0 |
| stn40 | A0p | gpi_1 | 286.38 | 250.0..334.0 | 0 | nan | nan | 0 | -50.0 |
| stn40 | A0p | motor_0 | 11.78 | 0.0..334.0 | 226 | nan | nan | 0 | -70.5 |
| stn40 | A0p | motor_1 | 16.98 | 0.0..334.0 | 211 | nan | nan | 319 | -87.1 |
| stn40 | A0p | practice_arousal | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.0 |
| stn40 | A0p | proposal_0 | 0.00 | 0.0..1.0 | 479 | nan | nan | 0 | -66.3 |
| stn40 | A0p | proposal_1 | 0.00 | 0.0..0.0 | 480 | nan | nan | 0 | -66.2 |
| stn40 | A0p | selector_reset | 0.00 | 0.0..0.0 | 160 | nan | nan | 0 | -55.0 |
| stn40 | A0p | selector_stn | 0.03 | 0.0..4.0 | 159 | nan | nan | 0 | -64.9 |
| stn40 | A0p | str_d1_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A0p | str_d1_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A0p | str_d2_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A0p | str_d2_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A0p | thal_0 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -61.9 |
| stn40 | A0p | thal_1 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -62.4 |
| stn40 | A5 | commit_0 | 29.11 | 0.0..85.0 | 60 | nan | nan | 0 | -69.9 |
| stn40 | A5 | commit_1 | 9.43 | 0.0..57.0 | 142 | nan | nan | 0 | -70.9 |
| stn40 | A5 | commit_fs_0 | 44.61 | 0.0..110.0 | 32 | nan | nan | 0 | -64.0 |
| stn40 | A5 | commit_fs_1 | 14.01 | 0.0..73.0 | 72 | nan | nan | 0 | -63.4 |
| stn40 | A5 | gpe_0 | 33.86 | 9.0..77.0 | 0 | nan | nan | 0 | -66.5 |
| stn40 | A5 | gpe_1 | 29.48 | 8.0..72.0 | 0 | nan | nan | 0 | -65.2 |
| stn40 | A5 | gpi_0 | 157.90 | 82.0..237.0 | 0 | nan | nan | 0 | -50.0 |
| stn40 | A5 | gpi_1 | 164.86 | 99.0..250.0 | 0 | nan | nan | 0 | -50.0 |
| stn40 | A5 | motor_0 | 65.62 | 0.0..157.0 | 60 | nan | nan | 0 | -73.4 |
| stn40 | A5 | motor_1 | 23.78 | 0.0..114.0 | 120 | nan | nan | 0 | -74.0 |
| stn40 | A5 | practice_arousal | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.0 |
| stn40 | A5 | proposal_0 | 0.00 | 0.0..1.0 | 479 | nan | nan | 0 | -66.3 |
| stn40 | A5 | proposal_1 | 0.00 | 0.0..0.0 | 480 | nan | nan | 0 | -66.2 |
| stn40 | A5 | selector_reset | 0.00 | 0.0..0.0 | 160 | nan | nan | 0 | -55.0 |
| stn40 | A5 | selector_stn | 21.56 | 0.0..120.0 | 30 | nan | nan | 0 | -66.1 |
| stn40 | A5 | str_d1_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A5 | str_d1_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A5 | str_d2_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A5 | str_d2_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A5 | thal_0 | 24.03 | 0.0..63.0 | 4 | nan | nan | 0 | -60.0 |
| stn40 | A5 | thal_1 | 18.07 | 0.0..62.0 | 8 | nan | nan | 0 | -60.0 |
| stn40 | A5p | commit_0 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -61.5 |
| stn40 | A5p | commit_1 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -61.3 |
| stn40 | A5p | commit_fs_0 | 0.00 | 0.0..0.0 | 128 | nan | nan | 0 | -55.0 |
| stn40 | A5p | commit_fs_1 | 0.00 | 0.0..0.0 | 128 | nan | nan | 0 | -55.0 |
| stn40 | A5p | gpe_0 | 86.37 | 58.0..125.0 | 0 | nan | nan | 0 | -50.1 |
| stn40 | A5p | gpe_1 | 84.77 | 48.0..125.0 | 0 | nan | nan | 0 | -50.1 |
| stn40 | A5p | gpi_0 | 278.04 | 200.0..334.0 | 0 | nan | nan | 0 | -50.0 |
| stn40 | A5p | gpi_1 | 286.38 | 250.0..334.0 | 0 | nan | nan | 0 | -50.0 |
| stn40 | A5p | motor_0 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -61.4 |
| stn40 | A5p | motor_1 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -62.4 |
| stn40 | A5p | practice_arousal | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.0 |
| stn40 | A5p | proposal_0 | 0.00 | 0.0..1.0 | 479 | nan | nan | 0 | -66.3 |
| stn40 | A5p | proposal_1 | 0.00 | 0.0..0.0 | 480 | nan | nan | 0 | -66.2 |
| stn40 | A5p | selector_reset | 0.00 | 0.0..0.0 | 160 | nan | nan | 0 | -55.0 |
| stn40 | A5p | selector_stn | 0.03 | 0.0..4.0 | 159 | nan | nan | 0 | -64.9 |
| stn40 | A5p | str_d1_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A5p | str_d1_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A5p | str_d2_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A5p | str_d2_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn40 | A5p | thal_0 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.4 |
| stn40 | A5p | thal_1 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.9 |
| stn50 | A0 | commit_0 | 0.88 | 0.0..12.0 | 210 | nan | nan | 0 | -65.4 |
| stn50 | A0 | commit_1 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -70.7 |
| stn50 | A0 | commit_fs_0 | 1.69 | 0.0..21.0 | 112 | nan | nan | 0 | -61.8 |
| stn50 | A0 | commit_fs_1 | 0.00 | 0.0..0.0 | 128 | nan | nan | 0 | -56.3 |
| stn50 | A0 | gpe_0 | 33.86 | 9.0..77.0 | 0 | nan | nan | 0 | -66.5 |
| stn50 | A0 | gpe_1 | 29.48 | 8.0..72.0 | 0 | nan | nan | 0 | -65.2 |
| stn50 | A0 | gpi_0 | 157.93 | 82.0..237.0 | 0 | nan | nan | 0 | -50.0 |
| stn50 | A0 | gpi_1 | 164.91 | 99.0..250.0 | 0 | nan | nan | 0 | -50.0 |
| stn50 | A0 | motor_0 | 3.05 | 0.0..35.0 | 210 | nan | nan | 0 | -69.5 |
| stn50 | A0 | motor_1 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -63.0 |
| stn50 | A0 | practice_arousal | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.0 |
| stn50 | A0 | proposal_0 | 0.00 | 0.0..1.0 | 479 | nan | nan | 0 | -66.3 |
| stn50 | A0 | proposal_1 | 0.00 | 0.0..0.0 | 480 | nan | nan | 0 | -66.2 |
| stn50 | A0 | selector_reset | 0.00 | 0.0..0.0 | 160 | nan | nan | 0 | -55.0 |
| stn50 | A0 | selector_stn | 31.52 | 4.0..124.0 | 0 | nan | nan | 0 | -65.9 |
| stn50 | A0 | str_d1_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A0 | str_d1_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A0 | str_d2_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A0 | str_d2_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A0 | thal_0 | 0.64 | 0.0..13.0 | 174 | nan | nan | 0 | -60.0 |
| stn50 | A0 | thal_1 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -55.4 |
| stn50 | A0p | commit_0 | 234.38 | 0.0..500.0 | 120 | nan | nan | 4000 | -228.1 |
| stn50 | A0p | commit_1 | 237.27 | 0.0..500.0 | 120 | nan | nan | 4000 | -238.1 |
| stn50 | A0p | commit_fs_0 | 250.00 | 0.0..500.0 | 64 | nan | nan | 0 | -55.0 |
| stn50 | A0p | commit_fs_1 | 250.00 | 0.0..500.0 | 64 | nan | nan | 0 | -55.0 |
| stn50 | A0p | gpe_0 | 86.37 | 58.0..125.0 | 0 | nan | nan | 0 | -50.1 |
| stn50 | A0p | gpe_1 | 84.77 | 48.0..125.0 | 0 | nan | nan | 0 | -50.1 |
| stn50 | A0p | gpi_0 | 278.05 | 200.0..334.0 | 0 | nan | nan | 0 | -50.0 |
| stn50 | A0p | gpi_1 | 286.37 | 250.0..334.0 | 0 | nan | nan | 0 | -50.0 |
| stn50 | A0p | motor_0 | 11.78 | 0.0..334.0 | 226 | nan | nan | 0 | -70.5 |
| stn50 | A0p | motor_1 | 16.98 | 0.0..334.0 | 211 | nan | nan | 319 | -87.1 |
| stn50 | A0p | practice_arousal | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.0 |
| stn50 | A0p | proposal_0 | 0.00 | 0.0..1.0 | 479 | nan | nan | 0 | -66.3 |
| stn50 | A0p | proposal_1 | 0.00 | 0.0..0.0 | 480 | nan | nan | 0 | -66.2 |
| stn50 | A0p | selector_reset | 0.00 | 0.0..0.0 | 160 | nan | nan | 0 | -55.0 |
| stn50 | A0p | selector_stn | 0.24 | 0.0..5.0 | 145 | nan | nan | 0 | -66.0 |
| stn50 | A0p | str_d1_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A0p | str_d1_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A0p | str_d2_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A0p | str_d2_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A0p | thal_0 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -61.9 |
| stn50 | A0p | thal_1 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -62.5 |
| stn50 | A5 | commit_0 | 29.30 | 0.0..84.0 | 60 | nan | nan | 0 | -69.2 |
| stn50 | A5 | commit_1 | 9.33 | 0.0..56.0 | 149 | nan | nan | 0 | -70.8 |
| stn50 | A5 | commit_fs_0 | 44.89 | 0.0..110.0 | 32 | nan | nan | 0 | -62.7 |
| stn50 | A5 | commit_fs_1 | 13.69 | 0.0..71.0 | 80 | nan | nan | 0 | -63.6 |
| stn50 | A5 | gpe_0 | 33.86 | 9.0..77.0 | 0 | nan | nan | 0 | -66.5 |
| stn50 | A5 | gpe_1 | 29.48 | 8.0..72.0 | 0 | nan | nan | 0 | -65.2 |
| stn50 | A5 | gpi_0 | 157.94 | 82.0..237.0 | 0 | nan | nan | 0 | -50.0 |
| stn50 | A5 | gpi_1 | 164.90 | 99.0..250.0 | 0 | nan | nan | 0 | -50.0 |
| stn50 | A5 | motor_0 | 66.19 | 0.0..157.0 | 60 | nan | nan | 0 | -71.9 |
| stn50 | A5 | motor_1 | 23.49 | 0.0..114.0 | 134 | nan | nan | 0 | -74.2 |
| stn50 | A5 | practice_arousal | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.0 |
| stn50 | A5 | proposal_0 | 0.00 | 0.0..1.0 | 479 | nan | nan | 0 | -66.3 |
| stn50 | A5 | proposal_1 | 0.00 | 0.0..0.0 | 480 | nan | nan | 0 | -66.2 |
| stn50 | A5 | selector_reset | 0.00 | 0.0..0.0 | 160 | nan | nan | 0 | -55.0 |
| stn50 | A5 | selector_stn | 31.54 | 4.0..124.0 | 0 | nan | nan | 0 | -65.9 |
| stn50 | A5 | str_d1_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A5 | str_d1_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A5 | str_d2_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A5 | str_d2_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A5 | thal_0 | 23.98 | 0.0..63.0 | 4 | nan | nan | 0 | -60.0 |
| stn50 | A5 | thal_1 | 18.03 | 0.0..62.0 | 8 | nan | nan | 0 | -60.0 |
| stn50 | A5p | commit_0 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -61.5 |
| stn50 | A5p | commit_1 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -61.3 |
| stn50 | A5p | commit_fs_0 | 0.00 | 0.0..0.0 | 128 | nan | nan | 0 | -55.0 |
| stn50 | A5p | commit_fs_1 | 0.00 | 0.0..0.0 | 128 | nan | nan | 0 | -55.0 |
| stn50 | A5p | gpe_0 | 86.37 | 58.0..125.0 | 0 | nan | nan | 0 | -50.1 |
| stn50 | A5p | gpe_1 | 84.77 | 48.0..125.0 | 0 | nan | nan | 0 | -50.1 |
| stn50 | A5p | gpi_0 | 278.05 | 200.0..334.0 | 0 | nan | nan | 0 | -50.0 |
| stn50 | A5p | gpi_1 | 286.37 | 250.0..334.0 | 0 | nan | nan | 0 | -50.0 |
| stn50 | A5p | motor_0 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -61.4 |
| stn50 | A5p | motor_1 | 0.00 | 0.0..0.0 | 240 | nan | nan | 0 | -62.4 |
| stn50 | A5p | practice_arousal | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.0 |
| stn50 | A5p | proposal_0 | 0.00 | 0.0..1.0 | 479 | nan | nan | 0 | -66.3 |
| stn50 | A5p | proposal_1 | 0.00 | 0.0..0.0 | 480 | nan | nan | 0 | -66.2 |
| stn50 | A5p | selector_reset | 0.00 | 0.0..0.0 | 160 | nan | nan | 0 | -55.0 |
| stn50 | A5p | selector_stn | 0.24 | 0.0..5.0 | 145 | nan | nan | 0 | -66.0 |
| stn50 | A5p | str_d1_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A5p | str_d1_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A5p | str_d2_0 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A5p | str_d2_1 | 0.00 | 0.0..0.0 | 288 | nan | nan | 0 | -80.0 |
| stn50 | A5p | thal_0 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.4 |
| stn50 | A5p | thal_1 | 0.00 | 0.0..0.0 | 192 | nan | nan | 0 | -60.9 |

## Part A param_assert
- ('A0', 'stn40'): asserted=False four_params=None symdiff=None
- ('A0', 'stn50'): asserted=False four_params=None symdiff=None
- ('A0p', 'stn40'): asserted=True four_params=['cp_izh_a', 'cp_izh_b', 'cp_izh_d_increment', 'cp_izh_C'] symdiff=[]
- ('A0p', 'stn50'): asserted=True four_params=['cp_izh_a', 'cp_izh_b', 'cp_izh_d_increment', 'cp_izh_C'] symdiff=[]
- ('A5', 'stn40'): asserted=False four_params=[] symdiff=[]
- ('A5', 'stn50'): asserted=False four_params=[] symdiff=[]
- ('A5p', 'stn40'): asserted=False four_params=['cp_izh_a', 'cp_izh_b', 'cp_izh_d_increment', 'cp_izh_C'] symdiff=[]
- ('A5p', 'stn50'): asserted=False four_params=['cp_izh_a', 'cp_izh_b', 'cp_izh_d_increment', 'cp_izh_C'] symdiff=[]

## Step-0 decision input: sub-reversal steps (sum v_steps_below over 21 regions x seeds, per arm/current)
- stn40: A0=0  A0p=8319  A5=0  A5p=0
- stn50: A0=0  A0p=8319  A5=0  A5p=0
## 4. Part C - isolated-cell f-I, rebound, citations, positive control

## Part C isolated-cell f-I (canary run bg_loop_part_c_71efc4a5c_pool41 20261021T022651, rc=0)

### IZH2007_FS_CORTICAL_INTERNEURON.json  ref=no source found cite=
| pA | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 |
| rate_hz | 0.0 | 199.2 | 273.9 | 324.2 | 356.1 | 420.8 | 468.7 | 499.9 | 499.9 | 500.0 | 500.0 | 500.0 | 500.0 |
| isi_cv | - | 0.06 | 0.05 | 0.03 | 0.05 | 0.06 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
- rebound: hyper -200.0 pA / 200.0 ms; rebound_spikes/rep=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

### IZH2007_GPE_PACEMAKER.json  ref=30-80 (PV+ prototypic, in vivo tonic) cite=PBR-160 ch 7 p 112 (feature-catalog.md line 237); ~50 Hz spontaneous PV+ (line 366)
| pA | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 |
| rate_hz | 0.0 | 86.1 | 126.5 | 152.2 | 177.4 | 198.1 | 218.8 | 237.2 | 246.5 | 274.1 | 283.3 | 308.1 | 316.4 |
| isi_cv | - | 0.02 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
- rebound: hyper -200.0 pA / 200.0 ms; rebound_spikes/rep=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

### IZH2007_GPI_OUTPUT.json  ref=40-80 (tonic in vivo) cite=feature-catalog.md lines 132/250 (Part V Ch 30-39; A.04 supplemental)
| pA | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 |
| rate_hz | 0.0 | 64.0 | 113.0 | 144.2 | 167.4 | 188.3 | 208.3 | 227.3 | 239.8 | 258.3 | 279.0 | 286.9 | 311.8 |
| isi_cv | - | 0.07 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 |
- rebound: hyper -200.0 pA / 200.0 ms; rebound_spikes/rep=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

### IZH2007_STN_BURST.json  ref=5-15 (autonomous, slice) cite=PBR-160 ch 10, Bevan et al. pp 175-176 (feature-catalog.md line 273)
| pA | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 |
| rate_hz | 0.0 | 10.6 | 19.1 | 28.3 | 37.0 | 45.6 | 55.0 | 63.7 | 72.3 | 80.2 | 88.1 | 95.5 | 102.5 |
| isi_cv | - | 1.58 | 1.52 | 1.33 | 1.23 | 1.10 | 0.90 | 0.70 | 0.52 | 0.39 | 0.29 | 0.24 | 0.21 |
- rebound: hyper -200.0 pA / 200.0 ms; rebound_spikes/rep=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

### IZH2007_STRIATAL_MSN_D1.json  ref=no source found cite=
| pA | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 |
| rate_hz | 0.0 | 0.0 | 0.0 | 7.7 | 18.7 | 29.5 | 39.8 | 49.4 | 58.8 | 67.0 | 75.6 | 83.6 | 91.4 |
| isi_cv | - | - | - | 0.10 | 0.08 | 0.07 | 0.06 | 0.05 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 |
- rebound: hyper -200.0 pA / 200.0 ms; rebound_spikes/rep=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

### IZH2007_STRIATAL_MSN_D2.json  ref=no source found cite=
| pA | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 |
| rate_hz | 0.0 | 0.0 | 6.3 | 17.4 | 27.5 | 36.7 | 45.4 | 53.7 | 61.1 | 68.8 | 75.8 | 83.2 | 89.7 |
| isi_cv | - | - | 0.12 | 0.10 | 0.08 | 0.07 | 0.06 | 0.06 | 0.06 | 0.07 | 0.07 | 0.08 | 0.09 |
- rebound: hyper -200.0 pA / 200.0 ms; rebound_spikes/rep=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

### IZH2007_THALAMIC_RELAY.json  ref=no source found cite=
| pA | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 |
| rate_hz | 0.0 | 1.3 | 12.3 | 21.4 | 31.1 | 40.0 | 48.2 | 55.7 | 63.5 | 70.4 | 76.9 | 82.8 | 89.4 |
| isi_cv | - | 0.12 | 0.13 | 0.12 | 0.09 | 0.08 | 0.07 | 0.06 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 |
- rebound: hyper -200.0 pA / 200.0 ms; rebound_spikes/rep=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

## Part C positive control (hybrid bridge STN params, seed 9, 20 cells)
- n_cells: 20
- rheobase_measured_pA: {'range': '25..35', 'per_cell': [35.0, 30.0, 30.0, 35.0, 30.0, 30.0, 35.0, 25.0, 30.0, 35.0, 30.0, 35.0, 35.0, 35.0, 30.0, 30.0, 35.0, 30.0, 30.0, 30.0]}
- rheobase_closed_form_pA: {'range': '23.7..33.7', 'per_cell': [30.191284458556623, 28.938658504742424, 28.643836217519038, 30.096714193366136, 28.383616522707502, 27.943186366907195, 31.56066173953002, 23.65663724461798, 29.821035719851505, 32.11
- rate_at_40pA_hz: {'range': '6..77', 'per_cell': [12.0, 10.0, 10.0, 10.0, 13.5, 45.0, 9.5, 77.0, 6.0, 14.0, 10.0, 13.5, 22.0, 11.0, 17.0, 12.5, 7.0, 35.5, 12.0, 12.0]}
- reference_bg_tonic_stn_1020f: {'rheobase_pA': [23.5, 33.7], 'rate_at_40pA_hz': [18.0, 49.0]}
- note: mismatch is reported, never tuned (partC-1021)
## 5. Provenance and calendar reading (packet sec 5)

- Part C artifacts come from the pool-canary run
  bg_loop_part_c_71efc4a5c_pool41_20261021T022651 (rc=0, complete 7 presets x 13
  currents): no full Part C queue line was ever queued (claims file: 0 part_c lines).
  Stated here because provenance, not volume, is what the record checks.
- Part A: 64/64 runs landed. Part B: 224/224 files landed after one requeue.
- Calendar reading for Claude: the BG commit seam gate window (stage-2
  11-03..11-10) now depends on the recalibration, and step 0 says recalibration IS
  the method. Per packet sec 5, Claude rules on the calendar in this RESULT's reading;
  step 1 (recalibration spec) is the next method under priorities-1018.
