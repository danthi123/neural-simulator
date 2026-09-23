---
type: finding
status: contributing
date: 2026-09-23
lane: D6-learn-and-grow
mechanism: in-conversation fact WRITE by a local phase-coupled Hebbian rule on the RF substrate (BRAIN_D6_HEBBIAN_STORE) + write-only plasticity-freeze lesion (BRAIN_D6_HEBBIAN_FREEZE), measured by a pre-registered 2x2 (use x plasticity) probe through the real /api/brain-chat handler
seeds: [42]
seed-waiver: a labelled 1-seed SMOKE of a pre-registered gate; the 6-seed runs (both variants) are staged on the mini-PC pool, paths below
verdict: NO-GO on the pre-registered gate at seed 42 (C3 fails; C1 C2 C4 C5 C6 C7 hold). The plasticity carries the RECALL; a host bookkeeping list carries a second use-trace into the reply.
runner: research/runners/d6_learn_through_use_lb.py
artifacts:
  - research/findings/raw/_d6_learn_through_use/d6_ltu_s42_smoke.json
  - research/findings/raw/_d6_learn_through_use/mechanism_6seed.json
  - research/findings/raw/_d6_learn_through_use/s42_USE_H.json
  - research/findings/raw/_d6_learn_through_use/s42_USE_H_REP.json
  - research/findings/raw/_d6_learn_through_use/s42_SHUF_H.json
  - research/findings/raw/_d6_learn_through_use/s42_FREEZE_H.json
  - research/findings/raw/_d6_learn_through_use/s42_USE_D.json
---

# D6 learn-through-use: the Hebbian fact write carries the recall, but a host list still carries familiarity (seed-42 smoke, NO-GO on C3)

## What was built (default-OFF, byte-identical when off, no `sim/` edit)

Charter D6 asks for continuous learning that is lesion-verified load-bearing: the brain changes from use,
and freezing the plasticity removes the change. Two defects stood in the way.

1. **The in-conversation write was not plasticity.** `OneBrainComposer._write_block` copied a host-read
   composite into the trigger->readout weights. That is a host-designed weight, the residual the ledger row
   `in-loop-learning` names ("no lasting per-turn BTSP/plasticity write").
   `research/runners/d6_hebbian_store.py` replaces it behind `BRAIN_D6_HEBBIAN_STORE=1`:
   - the composite stays live in the `acc` register and drives the new block's readout cells through a
     one-to-one instructive pathway;
   - the block's trigger (context) cell fires phase-locked to the rhythm;
   - each synapse accumulates its own pre x post phasor correlation `dw = eta * z_post * conj(z_pre)`;
   - the magnitude saturates at `w_max`.
   The phase is the stored content, and the phase is the rule's output.
2. **The only LEARN lesion check could not fail.** `_production_lesion_probe.py` tested for "bird" in an
   answer about a "deer", so it returned True unconditionally. The check is fixed and the failure is logged
   in `research/FAILURE_LOG.md`.

`BRAIN_D6_HEBBIAN_FREEZE=1` sets eta=0 only for writes made inside `ChatBrain._maybe_acquire`. The encode
turn still runs the same activity, and build-time facts and the read path are untouched.

**Mechanism check, 6 seeds** (`research/findings/raw/_d6_learn_through_use/mechanism_6seed.json`, from
`research/runners/_d6_hebbian_mechanism_probe.py`; numpy, 12-word vocab, D=128; plus
`tests/test_d6_hebbian_store.py`, 7 passed):
- On all 6 seeds the Hebbian block's phase error vs the direct copy is 0.01474-0.01713 rad (seed 42: 0.01534 and
  0.01541 rad), which is the spike-step quantization.
- Cleanup margins match the direct path. At seed 42 the deer margin is 0.7123 (Hebbian) vs 0.7108 (direct).
  Across seeds the largest per-role margin difference is below 0.008. <!--derived-->
- On all 6 seeds a frozen in-conversation block has max |w| = 0.0 and its fact abstains, while the
  build-time fact still recalls "cat".
- The engram read marks the built block held and the frozen block not held on every seed.
- The rule is deterministic, and leaving the flag unset is byte-identical to the direct copy.

<!--derived-->
An earlier version activated the context cell at rhythm counter 416 (2 cycles + 16 steps). That rotated
every learned weight by about 0.49 rad and cut cleanup margins by 12-30% (pre-fix scratch run, not kept).
Activating the cell at the read's own reference phase fixed it; no tuned constant is involved.

## The pre-registered gate and the seed-42 result

The gate (C1..C7) was written into the runner docstring before any result existed. Five fresh brains were
built at `BRAIN_CHAT_SEED=42`, each running session turns teach -> d1 -> d2 -> probe -> xprobe through
`webapp.server.brain_chat`.

| criterion | result | evidence |
|---|---|---|
| C1 learns | PASS | USE_H probe "what does the wolf hunt" recalls `[wolf, hunt, deer]` |
| C2 use changes the reply | PASS | SHUF_H (taught "the fox eats the berry") abstains on the same probe |
| C3 freeze removes it | **FAIL** | FREEZE_H abstains (`abstained`/`recalled_svo` identical to SHUF_H), but the `answer` text differs |
| C4 write-only lesion | PASS | the teach ack and d2 recall ("what does the dog chase" -> cat) are identical; learned mean\|w\| 1.80 plastic vs 0.0 frozen |
| C5 specific | PASS | SHUF_H xprobe recalls berry; USE_H xprobe abstains (double dissociation) |
| C6 deterministic | PASS | USE_H == USE_H_REP on every turn (null diffs 0; attributable_to = 1.0) |
| C7 no-regression | PASS | USE_D (production direct copy) is decision-identical to USE_H on teach/d2/probe/xprobe |

The lesion condition in TERMS.md asks whether the frozen weight still holds at measurement time. It was
read as 0.0 immediately after the teach turn. Every later turn is a question, so no write path runs. The
only other store-side process, Turrigiano scaling, is multiplicative, so a zero weight stays zero.

## Why C3 failed (post-hoc diagnosis — not a re-scoring of the gate)

Field-by-field diff of the FREEZE_H vs SHUF_H probe responses
(`research/findings/raw/_d6_learn_through_use/s42_FREEZE_H.json` vs `s42_SHUF_H.json`):

| field | FREEZE_H | SHUF_H |
|---|---|---|
| curiosity novelty | 0.0 (familiar) | 0.97 |
| curious | False | True |
| common-ground topic | `wolf`, grounded | none |
| thread swap / GNW stop | fire on `wolf` | do not fire |

So FREEZE_H answered "Setting the held thread aside — On wolf, then — I don't know about that." SHUF_H
answered "I don't know about that. My curiosity is piqued — I haven't learned about wolf yet: ...".

The cause is host bookkeeping, not synapses. `hear()` appends the heard fact to the host `kb` list whether
or not any synapse changed. `ChatBrain._refresh_facts` then builds the known-word sets (`agents_set` /
`actions_set` / `patients_set`) from that list, and curiosity novelty, common ground and the thread swap all
read those sets. The synaptic freeze removed the recall exactly. The familiarity of the taught word
survived in a Python list.

Measured split: the plasticity carries the RECALL (C3 holds on `abstained` + `recalled_svo`). The
host record carries a second use-trace into the reply's framing. Under the brain-based-only standard that
second channel is a shortcut: a list is doing the brain's remembering.

## Next method (built, staged; NO-DEFER)

`BRAIN_D6_ENGRAM_VOCAB=1` makes `_refresh_facts` keep only the facts whose engram reactivates on the
substrate (`d6_hebbian_store.engram_held`). It kicks the block's trigger cell and reads the mean |Z| over
the block's readout cells off the membrane. A block counts as held only if that activity clears the read's
own spike floor; a frozen block reads exactly 0.0.

The runner's `--variant engram` adds this flag to every Hebbian arm. USE_D stays pure production, so C7
now also checks that the full D6 configuration is decision-identical to production. The gate is C1..C7,
unchanged, and was pre-registered before any engram-variant result.

## Staged 6-seed runs (mini-PC pool, one queue line per seed)

- **base variant** — revision `3fd611b7c`, all 6 seeds dispatched 09:32-09:40.
  Node dir `~/derisk-pool/revisions/3fd611b7c10c6b08ad130ecd6149a37bf13cae77/research/findings/raw/_d6_learn_through_use/d6_ltu_s<seed>.json`
  plus the per-arm `s<seed>_<ARM>.json` files.
- **engram variant** — revision `bca675ea3`, queued.
  Node dir `~/derisk-pool/revisions/bca675ea369105bebde7d05884afdaad3678f9ec/research/findings/raw/_d6_learn_through_use_engram/d6_ltu_engram_s<seed>.json`
  plus the per-arm files.
- **scoring** (after the per-arm files are pulled into one dir):
  `.venv/bin/python -m research.runners.d6_learn_through_use_lb --score-only [--variant engram] --arm-dir <dir> --seeds 42 43 44 100 101 102 --json <dir>/d6_ltu[_engram]_6seed_verdict.json`

## Honest scope

- This covers one learning pathway: in-conversation declarative fact acquisition on the tiny-demo brain.
- The instructive pattern the rule stores comes from the composer's FHRR bind/bundle, the standing
  composer idealization.
- Which trigger cell a new fact takes is host bookkeeping.
- The rule is evaluated in the runner module from membrane state, not in a `sim/` kernel.
- Nothing here is flipped on by default. Production flips are owner-reserved.
- No felt or phenomenal claim is made. "Familiar" above means the curiosity organ's novelty read was 0.0.
