---
type: finding
status: design
claim_check: synthesis
date: 2026-09-09
mechanism: THIRD-ORDER (triple) S2.5 configural-binding conjunction units (a,b,c;Delta1,Delta2), --conj-order triple in _vision_lindiscrim_readout_derisk.py -- built as a cascade of two applications of the established pairwise coincidence-binding primitive, targeting the diagnosed representational ceiling of the pairwise PARTIAL landing
lane: vision (identity readout, D-perception configural binding)
seeds: [42]
verdict: MECHANISM BUILT + PRE-REGISTERED GATE + single-seed tiny-scale SMOKE ONLY (deterministic, runs end-to-end, anti-cheats compute) -- NO capability claim made at this scale. Decisive 6-seed eval QUEUED (tools/gpu_queue.sh), verdict PENDING.
artifacts:
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1152_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/vlin_triple_smoke.json
seed-waiver: single-seed tiny-scale smoke only, no capability verdict drawn from it; the decisive claim is deferred to a queued-but-not-yet-produced 6-seed run (path named in prose below, does not exist yet -- not listed as an artifact here for that reason)
---

# Third-order conjunction units — the next mechanism after the pairwise-binding PARTIAL, pre-registered before the decisive run

**Status:** this is a BUILD + PRE-REGISTRATION note, not yet a capability result. It picks up the named residual
from the PARTIAL landing (`conjbind_bindarm_n1152_heldoutpos_scramblenull_6seed.json`, verdict
`LINDISCRIM-READOUT-PARTIAL-beat0/6-lb4/6`, committed `c390e372e`): the FF-inhibition + temporal-integration
signed-linear-discriminant readout makes learning load-bearing on 4/6 seeds but does not clear the crossing floor
on any seed. The board's own next-mechanism note (`GAP_CLOSURE_MISSION.md`, same commit) named this residual as
"not yet attempted." Verified before building: `git log --all --grep=conjbind\|configural\|bindarm` and a
directory listing show no floor-clearing finding has landed since — this is genuinely the next lever, not a
re-derivation.

## The diagnosis that motivates THIS specific lever (not another readout retune)

<!--derived-->
From the PARTIAL artifact's own `reframe_means`/`headroom`, averaged over its 6 seeds:

<!--derived-->
| quantity | value | reading |
|---|---|---|
| `learned_spkwta_held` (the fully-spiking signed-discriminant readout) | 0.3281 | at, not above, the floor |
| `config_c_nogo_floor` | 0.34 | the #72/#75 fully-spiking NO-GO |
| `rate_lin_ceiling_held` (idealized non-spiking linear readout, SAME features) | 0.3403 | **the best possible LINEAR read of this front end sits AT the floor, not above it** |
| `headroom.learned_minus_nogo_floor` | -0.0119 | negative: the fully-spiking readout is not even leaving headroom relative to the rate ceiling |
| `spkport_cost` (rate-vs-spike gap attributable to the spike port itself) | 0.0017 | negligible -- the spike port is NOT the bottleneck |

<!--derived-->
The load-bearing readout class (FF-inhibition + temporal integration, already the fix for the two earlier
dead-ends: config-C centroid and R-STDP block-sum) is confirmed NOT the residual — its own ceiling (an idealized
non-spiking linear discriminant on the identical C2 features) is *already* at the floor. Width sweeps
(conj_n 1024→2304, 2026-09-03 finding "a fragile peak, not a plateau") and normalization sweeps (satdiv/alpha/z,
2026-09-01) were both exhausted at this same representational layer. **The wall-reframe question the project
requires ("what companion process did we replace with a constant?") points at the FRONT-END REPRESENTATION, not
the readout**: the frozen-random PAIRWISE conjunction bank's own linear-decodable content is capped, so no
readout-side lever (spiking or not) can lift it further.

## The mechanism: third-order (triple) conjunction units

This task's objects have `n_slots=3` (`_vision_hmax_hierarchy_derisk.py:131-174`): each of the 4 (of 3!=6)
classes is a distinct PERMUTATION of 3 oriented strokes across 3 relative slots. A pairwise unit
`AND(a@p, b@p+Delta)` can specify at most 2 of the 3 slots per unit. Two permutations of 3 elements that agree on
2 fixed positions are identical, so pairwise conjunctions are in principle *sufficient* — but the fixed-random
bank samples (a,b,Delta) uniformly from a combinatorial space of ~74k triples (n_s2=96, 8 offsets) while
allocating only ~1.1k units, so correct-pair coverage is thin and most units combine uninformative templates. A
unit that ANDs all THREE slot-relevant templates in one place is a strictly higher-SNR feature for this task's
structure — it fires only for its own fully-specified configuration, which a *linear* readout can exploit far
more directly than reconstructing the same specificity from several partial pairwise indicators (a linear
combination of pairwise-AND features cannot implement the AND of two independent pairwise events without already
having the conjoined feature as an input — the same reason XOR needs a hidden unit).

**Built as:** `_make_conjunction_bank_triple` + `_bind_conjunctions_triple`
(`research/runners/_vision_lindiscrim_readout_derisk.py`), wired via a new `--conj-order {pair,triple}` flag
(default `pair` = the exact prior code path, provably unchanged — the dispatch is
`bind_fn = _bind_conjunctions_triple if conj_order=="triple" else _bind_conjunctions`, an identical call to the
pre-existing one when `pair`). `triple` samples `(a,b,c,Delta1,Delta2)` quadruples once per seed from a stream
independent of the pairwise bank's, and computes `MAX_p AND(drive[p,a], drive[p+Delta1,b], drive[p+Delta2,c])`.

**Brain-based grounding — no new primitive claimed.** The triple AND is a CASCADE of two applications of the
SAME established pairwise primitive `research/biology/coincidence-binding.md` already grounds ("two signals in,
a supralinear conjunction out," Kandel PNS-6e NMDA-receptor Mg2+-block supralinearity) — `AND(AND(a,b),c)`, which
for both `min` and `prod` combination modes is associative and identical to a direct three-way AND. The
biological correlate is two dendritic branches each performing a local pairwise coincidence check, converging on
a shared integrative compartment for a second check (Poirazi, Brannon & Mel 2003, Neuron 37:989 — the two-layer
branch-then-soma dendritic model), with independent support that IT neurons conjunctively encode multi-part
shape arrangements rather than isolated pairs (Brincat & Connor 2004, Nat. Neurosci. 7:880). No new
`research/biology/` entry is registered because no claim beyond the existing established one is being made.

## Pre-registered GO gate (fixed BEFORE the decisive run; identical anti-cheats to the PARTIAL)

Everything downstream of the S2.5 stage is unchanged — same FF-inhibition signed-discriminant readout, same
temporal integration, same anti-cheats. Only the conjunction order changes.

- **task GO** (unchanged formula, `_summarize`): `beats_config_c_nogo` (per-seed
  `learn_spkwta_held >= nogo_floor(0.34) + beat_margin(0.10)`) **AND** `learning_load_bearing`
  (`learned - random >= beat_margin`), **each at >=5/6 seeds**, under `--heldout-position --scramble-null`
  (contiguous-block held-position extrapolation + the learned-readout pixel-scramble null, anti-cheats 5-6).
- **per-seed `capability_go`** additionally requires clearing the V1-direct/flat-pool floors, position pooled out
  of the class-population code, and label-shuffle-null at chance (unchanged `run_seed` formula).
- **Verdict bands, fixed in advance:** `beat>=5/6 & lb>=5/6` = GO. Anything with SOME (>0) beats/lb but short of
  5/6 = PARTIAL (an improvement over the pairwise arm's `beat0/6-lb4/6` is a partial win, still not a GO). `beat0
  & lb0` = NO-GO for this lever — bank it and take the next mechanism (recurrent binding stage, or an
  attention-gated readout, are the next named candidates); closure is not deferred either way.
- **Decisive command** (same op-point as the PARTIAL landing's own `conj_bind=fixed conj_n=1152` run, only
  `--conj-order triple` added), **QUEUED**, not yet run:
  ```
  OUTDIR=research/findings/raw/lanes/perception
  OUTFILE=conjbind_triple_n1152_heldoutpos_scramblenull_6seed.json    # does not exist yet -- QUEUED, not run
  SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
      --ridge 0.5 --conj-bind fixed --conj-order triple --conj-n 1152 --conj-offset-max 4 \
      --n-s2 96 --heldout-position --scramble-null --seeds 42 43 44 100 101 102 \
      --out "$OUTDIR/$OUTFILE"
  ```
  Queued via `tools/gpu_queue.sh add` (0 Claude tokens, sequential/VRAM-safe), running against this worktree
  (`.claude/worktrees/agent-ac8628a960479d44e`, branch `worktree-agent-ac8628a960479d44e`) since the mechanism is
  not yet on `main` — the artifact lands under that worktree's `research/findings/raw/lanes/perception/` and
  needs to be picked up (copied/committed) once the run completes, alongside the verdict write-up.

## What the smoke shows (sanity only — NOT a capability measurement)

<!--derived-->
`vlin_triple_smoke.json`, seed 42 only, tiny scale (`n_s2=24, conj_n=96, n_pos_total=4, n_ex=2, n_glimpses=1`, far
below the decisive op-point): `LEARNED_spkwta_held=0.3125`, `RANDOM_spkwta_held=0.25`,
`RATE_lin_ceiling_held=0.1875`, `scramble_null_pass=True`, elapsed 1.7s. Re-run byte-for-byte identical (checked
by diffing two independent invocations, only `--out` and `elapsed_seconds` differ) — the new code path is
deterministic under the file's existing seeding contract. This confirms the mechanism parses, runs end-to-end
through the FULL pipeline (spiking C1 → triple-bound S2.5 → LIF class-population readout → all 6 anti-cheats),
and the pairwise (`--conj-order pair`, default) path is unchanged. **No conclusion about whether triple
conjunctions clear the floor can be drawn from this tiny single-seed run** — the decisive claim is the queued
6-seed run above.

## Reproduce

```bash
# tiny smoke (seconds, sanity only):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --seeds 42 --n-s2 24 --conj-bind fixed --conj-order triple --conj-n 96 --conj-offset-max 2 \
    --n-pos-total 4 --n-ex 2 --n-glimpses 1 --heldout-position --scramble-null \
    --out research/findings/raw/lanes/perception/vlin_triple_smoke.json

# decisive 6-seed (queued, not yet executed inline; does not exist yet):
OUTDIR=research/findings/raw/lanes/perception
OUTFILE=conjbind_triple_n1152_heldoutpos_scramblenull_6seed.json
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --ridge 0.5 --conj-bind fixed --conj-order triple --conj-n 1152 --conj-offset-max 4 \
    --n-s2 96 --heldout-position --scramble-null --seeds 42 43 44 100 101 102 \
    --out "$OUTDIR/$OUTFILE"
```
