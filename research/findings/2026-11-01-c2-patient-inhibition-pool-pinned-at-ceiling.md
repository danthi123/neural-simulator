---
type: finding
status: qualified
lane: gap#4
date: 2026-11-01
seeds: [1, 2, 3, 4, 5, 6, 7]
---

seed-waiver: a dev-seed diagnostic over the C2 step-0 relay shards already on disk (dev seeds 1-7); no gate seed, no run, no capability claim.

# C2 patient inhibition: the shared FS pool fires at its ceiling on every block, so it supplies a constant inhibition, not a graded one (dev seeds 1-7)

## Answer in one paragraph

The patient-inhibition pool `pinh` (30 FS cells, research/runners/_c2_patient_inhib.py, COMPOSER_KNOBS) emitted 298 to 300
spikes per 20-step block in all 315 flag-on blocks of the C2 step-0 relay shards (dev seeds 1-7; 310 blocks at exactly 300).
300 is the ceiling at dt = 1 ms: every cell fires every second step, about 500 Hz, which is above the usual fast-spiking range.
Over the same blocks the summed patient drive varied 5.4-fold (3.14e6 to 1.70e7), so the pool's output carries no information
about its input. Selection was still correct in every block (exactly one pool fired, the decoded patient, 315/315). Read
together with the step-0b I->E sweep, the working mechanism is: the divisive stage ranks the word pools, and `pinh` adds a
fixed inhibitory level set by the I->E weight, which acts as a threshold. The "dose-graded" result in the step-0b finding
is graded in the I->E knob, not in the input.

## Why it saturates

The feed-forward current into `pinh` is ff_gain x input_gain x the summed patient slice: 3e-4 x 1.0 x (3.1e6 to 1.7e7), so
about 0.9 to 5.1 nA into 20 pF FS cells (block 0 of s7 infer: 1867 pA). A few hundred pA already drives such a cell hard.
The patient slice itself is read from the cleanup layer's membrane array, and its values are in the millions (the top
word of block 0, s7 infer: 2.41e6), the same phasor-state-in-the-membrane-array interaction the 2026-10-26 membrane census
recorded for the composer region (research/findings/2026-10-26-membrane-census-dev7.md, BridgeParser row). The word pools
receive that value times input_gain 1.0 as external pA, and the external-only divisive stage rescales it before the cells.

## What was read

- Artifact: research/findings/raw/_c2_pinh_ceiling/relay_inh_vs_drive_s1-7.json (every block below, with its source shard).
- Shards: the C2 step-0 relay shards for dev seeds 1-7, listed in that artifact's `sources` (flag BRAIN_C2_COMPOSER_PATIENT_INHIB on, the production relay
  `_seq_patient_drives`, RULING c2-relay-1031), every block with a `patient_relay` record: 315 blocks.
- Per block: `patient_relay.inh_spikes`, `patient_slice_nz` (summed for the total drive), `n_fired`, `fires_decoded_patient`.
- Counts: inh_spikes 300.0 x310, 299.0 x2, 298.0 x3. n_fired 1 in 315/315; fires_decoded_patient true in 315/315. The
  lowest-drive block (s1 amb1 block 3, total 3.14e6) still read 299.
- Not read here: the lesion line (_c2_step0_relay_lesion/s7) had not landed. The step-0b lesion arm already showed the
  inhibition is load-bearing (lesion != on); this finding is about its form, not whether it matters.

## What this does not claim

No capability claim, no default change, no gate seed. It does not say selection is wrong: it was right in 315/315 blocks.
It says the pool is not doing what its name implies (a competition whose strength follows the input) and runs at a
non-physiological rate. The open question is whether a graded operating point exists that still selects one correct
pool; RULING c2-ff-grade-1101 queues that grid (dev seeds only).
