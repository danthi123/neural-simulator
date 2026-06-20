# Burndown #3 Stage S1 — divnorm-in-loop: drive the K-way sequencer from the on-bridge divisive normalization (host `scores_to_drive` read RETIRED)

**Date:** 2026-06-20
**Runner:** `research/runners/_phaseB_onebrain_sequencerK_divnorm_derisk.py`
**Backend:** CPU / numpy (`SIM_BACKEND=numpy`)
**Builds on:** S0 (`_phaseB_onebrain_sequencerK_derisk.py`, commit `44f4a166`, GO K∈{2,4,8,16}) + S5 (`_phaseC_S5_divnorm_derisk.py`, commit `960467b0`, the proven on-bridge `input_divisive_norm` closure)
**`sim/` edit:** NONE (reuse-by-import of the S0 K-way sequencer + the EXISTING `input_divisive_norm` primitive flipped on a runner-built score bridge)

---

## Verdict

**GO at K∈{2,4,8} (3/3 seeds, production D=128); K=16 is a cleanly-mapped boundary — the early S2 signal.** The
moat held HARD (0 false-accepts) at **every K at both dimensions**. The host `scores_to_drive` peak-read is GONE
from the drive path.

| Dim | K=2 | K=4 | K=8 | K=16 | **moat (every K)** | lesion | OFF-guard |
|---|---|---|---|---|---|---|---|
| **D=128 (production)** | **GO 3/3** | **GO 3/3** | **GO 3/3** | 2/3 (boundary) | **0-FA 3/3** | safe 3/3 | PASS |
| D=64 (small) | ==host 3/3 | ==host 2/3 | ==host 2/3 | ==host 2/3 | **0-FA 3/3** | safe 3/3 | PASS |

`GO` here = ==host (the right block answers) **and** moat 0-FA **and** lesion-fails-safe **and** permuted-rule-inverts
**and** the no-divnorm control fails (divnorm load-bearing). All seeds 42/43/44. S5 op-point carried forward verbatim:
`input_gain=1.0`, `sigma=1.0`, `gain=0.05`, the Izhikevich rheobase as the placed threshold. **No config search beyond
the S5/S0 op-points** (the owner's explicit instruction); the K=16 miss is REPORTED as the boundary, the moat was never
relaxed.

---

## What S1 changed (and what it did NOT)

S0's K-way sequencer drove each block's decoded word-lines from the **host** `scores_to_drive(block_cleanup_scores,
frac)` read — a per-query `thr = frac * scores.max()` peak-normalization computed in Python
(`_phaseB_onebrain_sequencer_derisk.py:scores_to_drive`, the line `thr = frac * s.max()`). That `scores.max()` is the
last host DATA read inside the K-way scan loop — the residual S5 host read.

**S1 replaces ONLY the drive source.** For each stored block, the block's cleanup scores (the op result, read via the
unchanged `block_cleanup_scores`) are driven through a divnorm-flagged Izhikevich score pool (S5's
`build_divnorm_score_bridge` + `onbridge_divnorm_drive`); which words FIRE — the on-bridge per-query peak-normalization
(`r_i = x_i/(sigma + gain·mean_j x_j)`) followed by the placed rheobase threshold — IS the per-block decoded-line drive.
That drive feeds S0's K-way sequencer (the SAME match cascade + first-match priority WTA + production rule, imported
verbatim). There is **no `scores_to_drive` / `s.max()` anywhere in the S1 drive path** (confirmed by `grep`: those tokens
appear only in docstrings; the priority anti-cheat panel reuses S0's `run_priority_check` which uses S0's own host drive
— that is the structural priority check, not the S1 divnorm parity battery, and is labelled as such in the code).

Everything else is byte-for-byte the S0 control fabric: `build_sequencerK_bridge`, `wire_sequencerK_couplings`,
`reset_sequencerK_state`, the K-way first-match production rule, and the moat/lesion/permute/priority helpers are all
imported, not re-implemented.

---

## The K=16 boundary — the mechanism (the S2 K=32 early signal)

At D=128 the on-bridge divnorm produces a **clean single-argmax 92–93% of role-reads** (0 MISS). The residual is two
modes, both harmless at small K:

| mode | what | rate D=128 | rate D=64 |
|---|---|---|---|
| **EXTRA** | a runner-up word also fires (the divnorm tolerates a sub-peak at `gain=0.05`) | 4/48 (K=8), 7/96 (K=16) | 12/48, 20/96 |
| **MISS** | a near-tie role (peak/runner-up ratio ≳ 0.85) is squeezed below rheobase, NOTHING fires | 0 | 1/48, 1/96 |

The **single** K=16 failure (D=128 seed-42) is an EXTRA becoming load-bearing under shared action words:
- cue `(leaf, run)`; the true block is 13 (`leaf run wolf`), and block 13 decodes cleanly (`leaf`, `run`).
- BUT block 1 (`cat run river`) has its **agent** decoded line spuriously lighting `leaf` (an EXTRA, in addition to
  `cat`), and block 1's action is **also** `run` (the word `run` appears in facts 1/5/9/13).
- So block 1 spuriously matches `(leaf, run)`, and **first-match priority** (block 1 < block 13) routes the answer to
  block 1 → `sub=river` (block 1's patient) instead of block 13's `wolf`.

This is a present-cue ROUTING precision error, **not** a confabulation: the host `scores_to_drive(frac=0.9)` does not hit
it because frac=0.9 lights ONLY the argmax (it is the divnorm's runner-up tolerance at `gain=0.05` that creates the EXTRA).
As the store grows and action words repeat, an extra-lit agent on a *lower* block sharing an action word causes a
wrong-block first-match. **This is precisely the K-way squeeze the S1 brief anticipated as the early S2 K=32 signal** —
the placed threshold (held at the S5 op-point) does not give the runner-up margin that the production K=32 + D=128
boundary test (S2) must characterize.

**The moat is orthogonal to this and held HARD (0-FA at every K, both dims):** absent/cross moat cues match NO block's
argmax, so an EXTRA on a present-fact word never produces a false-accept. The K=16 boundary is a within-present-set
routing-precision limit, not a moat breach.

### D=64 vs D=128

D=64 is fidelity-bounded: the cleanup has more near-ties (e.g. K=4 seed-42 block-2 action `see`, peak/runner-up = 0.88 →
MISS → that present cue abstains), which is the same small-dimension cleanup caveat S0's own `host_scan_block` docstring
documents. At D=128 (where S2 runs) those near-ties resolve and K∈{2,4,8} are a clean 3/3 GO. The moat is 0-FA at every
K at **both** dimensions.

---

## Anti-cheats (all behaved)

- **MOAT (HARD, 0-FA):** every absent-agent / absent-action / cross cue abstains, at K∈{2,4,8,16}, both dims, 3/3 seeds.
  **Never traded.**
- **NO-DIVNORM control (load-bearing):** a divnorm-OFF score pool driving the SAME placed rheobase threshold on the RAW
  un-normalized drive FAILS the battery (winner + runner-up both fire → the whole-row-lights wall → moat breach or
  ≠host) at D=128 3/3 every K → the divisive normalization is load-bearing. (At trivial K=2/D=64 the raw control passed
  on 1/3 seeds — a property of the 2-fact control at small D, not the divnorm path; at D=128 it fails 3/3 everywhere.)
- **OFF==byte-identical:** S5's `check_off_byte_identical` guard PASS — a divnorm-OFF score bridge has
  `cp_input_divisive_mask=None` (the per-step divide unreached) and steps byte-identically; an ON bridge's divide changes
  the dynamics. Confirms the `sim/` primitive is a guarded no-op when off (no `sim/` edit was made — this asserts the
  contract).
- **lesion-fails-safe:** severing the result→op conditioning (zero decoded-line drive) → abstain, 3/3 every K.
- **permuted-rule inverts:** a present cue for block b routes to `ans{(b+1)%K}` — the decision follows the cyclic-shift
  RULE, not a fixed scan order (3/3 where ==host holds; the K=16 seed-42 perm-fail is the SAME blk-1-vs-13 routing miss,
  not a rule-following failure).
- **per-block priority:** the degenerate two-block-shared-cue store answers the LOWER block == host first-match, 3/3.

---

## Reproduce

```bash
# production D=128 (the load-bearing GO at K in {2,4,8}; K=16 = the mapped boundary)
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_divnorm_derisk \
    --seeds 42,43,44 --dim 128 --ks 2,4,8,16

# small D=64 (fidelity-bounded; moat still 0-FA at every K)
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_divnorm_derisk \
    --seeds 42,43,44 --dim 64 --ks 2,4,8,16
```

Raw outputs: `research/findings/raw/_phaseB_onebrain_sequencerK_divnorm_d128.json` (production),
`_phaseB_onebrain_sequencerK_divnorm_derisk.json` (D=64).

---

## Bottom line

S1 wires the **proven S5 on-bridge divisive normalization into the K-way sequencer** and retires the host
`scores_to_drive` peak-read from the drive path, with **zero `sim/` edits**. The divnorm-driven K-way sequencer is
**== host + moat-0-FA at K∈{2,4,8}** (3/3 seeds, production D=128), with all anti-cheats behaving (no-divnorm control
load-bearing, OFF byte-identical, lesion-safe, permuted-inverts, priority correct). **K=16 is a cleanly-characterized
boundary** — the divnorm's runner-up tolerance at the fixed S5 op-point (`gain=0.05`) lets an EXTRA-lit agent on a lower
block sharing an action word win first-match — which is the **honest early signal for the S2 K=32 boundary test**, and is
NOT a moat breach (the moat held HARD throughout). The point: the host data read is gone; the residual is a routing-margin
limit S2 must measure at K=32, not a re-introduced shortcut.
