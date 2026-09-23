---
type: finding
status: partial
lane: load-bearing
date: 2026-09-23
mechanism: DA-gated synaptic tagging-and-capture (webapp/da_tag_capture.py, default OFF) under a natural drive, 24 h recall -- v2 at the production composer block size D=128
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_da_encoding_natural_drive/seed42.json
---

# PRE-REGISTRATION v2 — the same gates at the production composer size D=128 (2026-09-23)

This addendum to `2026-09-23-da-encoding-natural-drive-24h-persistence-PREREGISTRATION.md` is committed on its own,
BEFORE any D=128 run. It is written AFTER seeing the first v1 seed, and says so.

## Why a v2 exists (disclosed: it was prompted by a v1 result)

The v1 runner copied the composer block size D=64 from the 2026-09-20 natural probe. The production chat composer is
built by `BrainConversationalAgent` with its default D=128 (`research/runners/brain_conversational_agent.py`). A probe
must match the deployed configuration.

<!--derived-->
The mismatch surfaced because v1 seed 42 read UNDEFINED on G2 (`research/findings/raw/_da_encoding_natural_drive/seed42.json`).
Immediate recall failed only in the unit-gain arms (lesion_da_encoding: 2 of 4 correct, 1 confabulation) at the
primary baseline ratio 1.0, and at the band point 2.0. With a baseline as large as a unit increment, a 64-synapse
block does not always carry a unit write. Every G3-G8 comparison on that seed went the predicted way, but under the
pre-registered rule the seed is UNDEFINED and stays UNDEFINED. The v1 run completes on all six seeds and is reported
as pre-registered.

## What v2 changes, and what it does not

- Changes: `--D 128` only.
- Unchanged: stimuli, arms, constants, band, gates G1-G8, the aggregate permutation null, the UNDEFINED rules. In
  particular G2 still requires 4/4 immediate recall in EVERY arm, including the unit-gain lesion arm, and the
  baseline ratio stays 1.0 (the band point 0.67 that passed on v1 seed 42 is NOT promoted to primary).
- v2 can fail. If G2 fails again at D=128, the reading is that the DA write gain has become load-bearing on the
  immediate read once a synaptic baseline exists. That would contradict Bethus 2010, where D1/D5 blockade left
  immediate recall intact. The next method would then move DA's effect from write magnitude to capture only.

## Command (one per seed, own output directory)

`.venv/bin/python -u -m research.runners._da_encoding_natural_drive_persistence --D 128 --seed <s> --out research/findings/raw/_da_encoding_natural_drive_D128/seed<s>.json`
then `--aggregate research/findings/raw/_da_encoding_natural_drive_D128`.
