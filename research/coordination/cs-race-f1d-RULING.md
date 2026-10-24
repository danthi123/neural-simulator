# RULING cs-race-f1d-1021 (Claude, 2026-10-21): measure before the next lever

Source: the research-gate round Claude ran on 10-21 (three readers: probe mechanics and stored data, the project record, the
biology; then a design and an adversarial critique). Step 1c was the second lever against one defect, so no third lever is
built until this measurement is read. All work is probe-local on `research/cs-race`, CPU, numpy backend, through the CPU pool
under `tools/memcap.sh`. Gate seeds 42/43/44/100/101/102 are NOT run in this ruling. K1-K6 and both control bars are unchanged:
posctl commits >= 9/10 on every seed; decisive commits >= 9/10 AND frog 2-8/10 AND vole 2-8/10 on every seed.

## What the stored records already establish (recomputed from `raw/_cs_race/lever_grid_r3`, `lever_grid_r4`, `nmda_grid_n3`)

1. The decisive non-commits are CO-COMMITS, not silence: rows 3-4 (18 cells, 1,440 decisive draws) have 1,158 draws with both
   latches ON, 282 with one, 0 with none. `latch_read` returns winner None for both co_commit and none, so the tally merged them.
2. Ignition is cue-locked and nearly simultaneous: the two first gate spikes land a median 3-4 ms apart; a gap of 8 ms or less
   never gave a single commit (0 of about 1,056 draws); single commits need about 9 ms or more, or one gate that never fires.
   Seeds whose earliest-gate step barely varies across draws (SD 1.3-1.7 ms: seeds 2, 3, 6) never commit.
3. Two step-1c readings are measurement artefacts. The engram Vm of -69.5 mV is read at t = 600 ms, after ignition (about
   160-210 ms) and after the occupancy pool has hyperpolarised everything; draws with no latch read about -60 mV. The `thr` field
   is the unused homeostasis array: the spike criterion is vpeak 35 mV and the regenerative saddle is about -42.9 mV.
4. The NMDA arm is VOID (instrument failure, not evidence against NMDA): its latches are ON before the cue (80/80 decisive) and
   its engrams fire 0 spikes during the cue.
5. Under-drive is refuted: decisive cue spikes per window average 33.75 against 28.9 for posctl (analytic drive ratio about 1.17),
   both engrams fire about 18 spikes and both gates fire on every failing draw. Cue weights are not split by the paired write.

Disclosure: a Claude research agent ran unrecorded scratch probes on dev seeds 1-24 and on the gate seeds with the single-slot
exclusion term W_GIS at 30-90. W_GIS = 30 is therefore a peeked value; it is confirmed below only on a fresh seed block.

## Step 0: correct the record (no runs)
- Put an AMENDMENT block at the top of `research/coordination/cs-race-f1c-RESULT.md` stating points 1-4 above and the disclosure.
- One `research/FAILURE_LOG.md` row: the probe tallied co-commits as non-commits and read Vm after the decision, so a
  read-out defect was reported as "engrams subthreshold, levers inert". Name a gate or write `NOT-GATEABLE: <reason>`
  (suggested rule: a commit probe reports co_commit and none separately, and every Vm field states its read window).
- STOP condition: committed and `.venv/bin/python tools/check_docs.py` passes.

## Step 1: instrument the probe (additive; with no new flag every decision is unchanged)
a. Verdict fix (merge gate VN blocked `research/cs-race` on it): under `--controls-only` the positive control stays a
   `require`, and the decisiveness control is the DECIDED criterion (`v.decide(dec_ok)`), so a failed decisiveness control
   reads NO-GO, not UNDEFINED.
b. New flags, defaults equal to today's values: `--w-gis` (0), `--w-sgi` (0), `--cue-n` (8, cue slots only, mention slots stay 8),
   `--w-cue` (45), `--cue-noise-sd` (0) and `--cue-noise-tau` (5; private OU current on cue cells), `--w-xg` (45), `--w-gs` (40),
   and a steer: `--steer-pa`, `--steer-stage {engram,gate}`, `--steer-target k` (reuse the existing steer_pa path; label the record
   `steer_host_probe=true`: it is a measurement probe, never a mechanism).
c. New per-draw fields: per-ms spike counts of engram, gate and slot for each k over cue + hold; `t_ign[k]` = first ms at which
   engram k has >= 3 spikes in the trailing 5 ms (None if never); latch onset per k (>= 3 slot spikes in 5 ms); which gate's latch
   survived (the first or the last gate to burst) and each gate's burst duration; in the pre-ignition window
   [min(t_ign) - 50, min(t_ign) - 5] ms (for a draw with no ignition, [100, 400] ms): mean Vm (mV), mean TOTAL excitatory synaptic
   current per cell (pA, named so), the exact cue-only current computed from the cue raster x W_CUE x the conductance kernel (pA),
   and the engram spike count inside the window; the latch read over BOTH the first 30 ms of the hold (today's read) and the last
   30 ms (the spiking_commit contract), with each disagreement reported. Once per seed: the unitary cue EPSP (force one cue cell
   to spike from the quiescent snapshot, peak dV over 30 ms averaged over its engram targets, mV).
d. Identity checks, asserted in the record: (1) with no new flag, seed 1 `--controls-only --grid-row 3` reproduces
   `raw/_cs_race/lever_grid_r3/wee40_wie40.json` seed 1 per_draw winner, slot_counts, gate_counts and first_gate_step exactly;
   (2) for arms that change `--cue-n`, the engram, gate, GI, slot and occupancy heterogeneity arrays (a, b, d, C) and initial
   state equal A0's for the shared indices, and the per-seed mean cue-only current of A0 and A3 is recorded.
   If either fails, STOP and post the diff.

## Step 2: measurement M (all arms and values frozen here; nothing is chosen from results)
Common: `--grid-row 3`, W_EE 40, W_IE 40, `--controls-only`, 10 draws, paired noise seeds across arms (base_seed = seed*1000003 + d).
Two seed blocks: DEV = seeds 1-8 (reproduction and exploration); CONFIRM = seeds 25-40 (fresh; nobody has run them).

DEV arms:
- A0 baseline (it must reproduce 13-18/80 decisive commits; if not, STOP: the instrument changed the circuit)
- A1 W_GIS 30 · A1s W_SGI 30 · A1gs W_GIS 30 + W_SGI 30 · plateau W_GIS 20, 25, 35, 40
- A3 `--cue-n 48 --w-cue 7.5` (same mean cue drive on many weak synapses, unitary EPSP about 1 mV)
- A3n private cue noise `--cue-noise-sd 15 --cue-noise-tau 5` at W_CUE 45 x 8 cells
- A4 = A3 + W_GIS 30
- A5 gate gain back to the organ's integrated drive: `--w-xg 20 --w-gs 12` (commit gate still closed during encode), W_GIS 0;
  and A5 at W_IE 20 and at W_IE 80
- Ownership steer (decisive control only) in A0, A1, A4 and A5: engram-stage steer on frog at +1, 2, 4, 8 pA, and gate-stage
  steer on frog at the same values; one check at +4 pA on vole for each stage.
CONFIRM arms (seeds 25-40): A0, A1, A4, A5, and the +4 pA engram and gate steers on frog in each.

## Step 3: post `research/coordination/cs-race-f1d-RESULT.md`, then STOP for Claude's ruling
For each arm and block: decisive commits / co-commits / none; frog and vole wins per seed and the pooled frog share with a
binomial 95% interval; posctl commits; seeds passing the FULL decisive bar (8/8 dev, 16/16 confirm; no looser count); median
ignition gap and per-seed cross-draw SD of min(t_ign); cue-lock index minus its chance baseline (the same statistic at a random
offset inside the cue window); unitary EPSP; decisive/posctl ratio of the pre-ignition cue-only current; first-vs-last gate
survivor share; first-30 vs last-30 latch disagreements; the steer curves (engram vs gate).
Read each cause against these rules, fixed now:
- (iii) read-out co-commit: CONFIRMED iff an exclusion arm (A1, A1s or A1gs) passes the full decisive bar on 8/8 DEV and 16/16
  CONFIRM (A1 is the only one with a CONFIRM run) with posctl intact. If the passing W_GIS band on DEV does not include both 25 and
  35, the exclusion is a knife-edge balance: a FINDING that blocks adoption, not a pass.
- (iv) common-volley synchrony: CONFIRMED iff A3 or A3n raises the median ignition gap to >= 8 ms and the per-seed ignition SD to
  >= 10 ms on >= 6/8 DEV seeds AND cuts A0's co-commits by >= 50%.
- (v) hair-trigger gate: CONFIRMED iff A5 raises decisive commits by >= 30/80 over A0 with posctl >= 9/10 on 8/8 AND the W_IE
  sensitivity returns (A5 at W_IE 20 vs 80 differs by >= 15/80).
- Ownership, per arm: the engram stage owns the choice iff a +4 pA (or smaller) engram steer moves the latched winner to >= 8/10
  on >= 6/8 DEV seeds AND the same-pA gate steer moves it less. Agreement between the first-igniting engram and the winner is
  reported only as a description, with its coverage (one-sided ignitions as their own category).
Adoption is Claude's next ruling, not this step's. The rule it will apply: prefer the most upstream mechanism that passes the bar
AND the ownership test; a read-out exclusion that passes the bar but fails ownership is a latch-stage coin flip the content
carriers (recency, mention) could never steer, recorded as a finding and not adopted as the race.
Run the verify-go skill on the RESULT before committing it.

## Step 4: book-keeping
Update the board entry for the cs-race through the vikunja skill (next action: Claude's ruling on f1d). Claude merges
`research/cs-race` after step 3 (the VN fix in step 1a unblocks it).

## Out of scope here (kept on the ledger, not dropped)
The asymmetric carriers (clause a recency, clause b mention; their first step is `tools/before_you_build.sh`); fast shared
FS/PV inhibition (needs the RS-heterogeneity fix ON); NMDA recurrence after the pre-cue latch is fixed; a spiking urgency or
LC-like gain population (expected to worsen co-commit; relevant only if residual draws are none after exclusion works); the gate
seeds; production `BRAIN_CS_RACE` wiring.
