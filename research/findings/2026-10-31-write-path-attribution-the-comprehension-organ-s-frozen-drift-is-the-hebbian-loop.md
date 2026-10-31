---
type: finding
status: resolved
date: 2026-10-31
lane: Vikunja #203 follow-up (which write path actually moves the comprehension organ's frozen synapses) -- the pi lane
mechanism: the #203 merge (dc6abf989) gated SIX `cp_connections.data` write sites behind
  `cfg.enforce_plastic_mask_beyond_hebbian`. Instrumenting every write to that array during the production
  comprehension battery found that NONE of those six sites fires in this organ: the only writes are the Hebbian
  LTP potentiation (`sim/bridge.py:11045`) and the Hebbian decay (`sim/bridge.py:11071`), which the 2026-09-02
  fix gates behind the OTHER flag, `cfg.enforce_plastic_mask_in_hebbian`. The umbrella env var
  `BRAIN_ENFORCE_PLASTIC_MASK` arms both flags, so the env path was covered; the cfg-field path was not: arming
  only `enforce_plastic_mask_beyond_hebbian` left the Hebbian sites permissive. Fixed by making
  `_hebbian_plastic_mask_enforced` honour either field.
seeds: [7, 8] (dev seeds only -- 42/43/44/100/101/102 untouched by this instrument); the scenario rule and the
  byte-identity comparison run on all six generalization seeds
verdict: the organ's frozen-synapse drift is 100% attributable to the Hebbian LTP + decay writes: with the flag
  ON every measured site's frozen delta is exactly 0.0, `attributable_to` reports 1.0 for both sites, and the
  instrument's signed residual (writes recorded vs net coordinate drift) is 0.0 -- no write path escaped it.
  The pooled wave3 organ that the #203 reply arm ran makes ZERO weight writes and shows zero drift in either arm,
  which is why that arm found no reply change. On the standalone organ the flag changes the reply margins by at
  most 0.01875 (seed 7) and 0.0375 (seed 8) and the competence verdict on all 25 battery items not at all. Flag
  OFF is SHA-identical to the genuine pre-fix code (42/42 comparisons, 7 rules x 6 seeds). No default flip, no
  capability claim.
runner: research.runners.pi203_write_path_attribution (orchestrator, 16 cells = 2 seeds x 2 organs x 2 arms x
  tap on/off, CPU/numpy, 868.65 s total) over research.runners.pi203_write_path_tap (the instrument)
second runner: research.runners.pi203_frozen_synapse_battery (7 rules x 6 seeds x 2 arms = 84 runs, 9.5 s) and
  research.runners.pi203_prefix_comparison (42 byte-identity comparisons against `git archive 710883be5`)
---

# The comprehension organ's frozen drift is the Hebbian loop -- and none of the six gated sites fires there

**Question the #203 reply arm left open.** The merged freeze changes the production comprehension organ's reply
**zero** times on 6 seeds, yet the 2026-09-02 read-isolation audit
(`research/findings/2026-09-02-read-isolation-audit-29runner-followup-map-plus-a-bigger-adjacent-bug.md`) measured
max weight drift 13.8 after one read and 56.1 after 30 reads of one sentence on this very organ, on 1710 internal
recurrent `sel_* <-> sel_FS_*` synapses declared `plastic_internal=False`. Either the audit's number is stale, or
the drift comes from a write path the six fixes do not touch. It is the second one.

## The instrument, and why its zero means something

`research/runners/pi203_write_path_tap.py` wraps the live `cp_connections.data` and hooks **every** way that array
can change: in-place `__setitem__`, `__iadd__`, `__imul__`, `__isub__`; `csr.data = <new array>` (e.g.
`set_pathway_weights`, `sim/bridge.py:5766`); and `self.cp_connections = <new csr>` (formation, CSR rebuild,
deferred compaction). Each record carries the sim line that performed the write, the signed delta, and the
frozen/plastic split evaluated at the instant of the write. Nothing is written back into the simulation.

Raw records: `research/findings/raw/_pi203_frozen_synapse_residual/write_path_attribution_v2.json` (16 cells),
`research/findings/raw/_pi203_frozen_synapse_residual/write_path_attribution_aggregates.json` (the aggregates quoted below),
`research/findings/raw/_pi203_frozen_synapse_residual/verification_record_v2.json` (84 scenario runs) and
`research/findings/raw/_pi203_frozen_synapse_residual/byte_identity_record.json` (42 comparisons).

Two guards make the numbers trustworthy rather than asserted:

1. **Non-invasiveness**: tapped and untapped runs of the same cell produce byte-identical reply digests
   (8/8 tapped-vs-untapped pairs in both matrices). An observer that changes what it measures is not measuring.
2. **Completeness**: the tap sums every signed delta it sees; the coordinate snapshots (weights keyed by
   `(pre, post)`, not by position, so a CSR re-ordering cannot fake a zero) give the net change. When the two
   agree to 1e-6 the hooks covered every path. They do: signed residual 0.0 in every cell. (The absolute residual
   is nonzero -- potentiation and depression cancel -- which is the expected arithmetic, not a gap.)

## The paths, measured (dev seeds 7, 8; standalone organ; flag OFF vs ON)

| write site | what it is | frozen delta OFF | frozen delta ON | attributable to the flag |
|---|---|---|---|---|
| `sim/bridge.py:11055` | Hebbian LTP potentiation (`data[idx] += delta_weights`) | 26883.72 (seed 7, 3973 writes) / 31018.10 (seed 8, 4436 writes) | 0.0 | 1.0 |
| `sim/bridge.py:11081` | Hebbian global decay `data *= (1 - gated_decay)` | 3045.99 (seed 7, 5986 writes) / 3332.76 (seed 8, 5948 writes) | 0.0 | 1.0 |
| any of the six #203 sites | scaling, structural elimination, reward clip/update, remap, compaction, formation | **does not fire** in this organ | 0.0 | n/a |

`tools.lab.attributable_to` reports 1.0 (the ON arm is exactly 0) for both sites; the lever check
(`tools.lab.lever("BRAIN_ENFORCE_PLASTIC_MASK", before, after)`) moves the organ's total weight writes
9959 -> 0 (seed 7) and 10384 -> 0 (seed 8). The frozen populations that move are the organ's own internal
recurrent ones: `sel_agent -> sel_agent` (289 synapses, max drift 47.8 at seed 7), `sel_patient -> sel_patient`
(298 synapses, max drift 56.3 at seed 8), `sel_FS_agent -> sel_patient` (288 synapses, max 34.9 / 39.2) -- exactly the `plastic_internal=False` pathways the 2026-09-02 audit named.

**Why the six fixes changed nothing here.** The organ's config has `enable_structural_plasticity`,
`enable_reward_modulation`, `enable_homeostasis` and `enable_synaptic_scaling` all False. The six gated sites are
inert in this organ by construction. The 2026-09-02 audit's 13.8 -> 56.1 number was always the Hebbian loop, which
the 2026-09-02 fix had already gated -- but only behind `enforce_plastic_mask_in_hebbian`.

**Why the reply arm saw nothing.** `CO.get_organ(seed)` (the pooled wave3 one-brain pool, 531916 synapses,
391715 of them mask-frozen) makes **zero** writes to `cp_connections.data` during the 25-item, 11314-step
battery, and its coordinate snapshot shows zero drift for frozen AND plastic synapses in both arms. The reply arm
could not have detected anything: there is nothing to detect. The drift the audit measured lives in the
standalone `ComprehensionProductionOrgan(seed)` bridge (1032 neurons, 24764 synapses, 1724 frozen).

## Reply-level effect (standalone organ, seed 7, 25 items, tap ON)

Margins change, the verdict does not. Max absolute margin difference 0.01875 (seed 7) and 0.0375 (seed 8); `competent` is True for all 25
items in both arms on both seeds (zero competence flips); the item digests differ because the margins are in them. In the ON arm the margins sit at
the untrained uniform 1/3. So the drift is **not load-bearing for the reply** on this organ -- freezing it changes
nothing that the reply arm scores. That is reported as a negative, not as comfort: it means this organ's
comprehension read-out is not using those synapses as a learned signal yet.

## The code change

`sim/bridge.py`, `_hebbian_plastic_mask_enforced`: now returns True if the umbrella env var is set **OR**
`cfg.enforce_plastic_mask_in_hebbian` **OR** `cfg.enforce_plastic_mask_beyond_hebbian`. Both flags
default False, so the flag-OFF path is untouched. Two new tests:

- `tests/_plastic_mask_beyond_hebbian_scenario.py` rule `hebbian`: arms ONLY the umbrella cfg field (deliberately
  not the env var, because the env var arms both flags and the rule would then prove nothing). Six seeds: frozen
  drift 19.01-19.06 OFF, 0.0 ON, plastic control still moves (19.00).
- `tests/test_comprehension_frozen_flat.py`: the audit's own protocol (30 successive reads of one sentence) on the
  organ the audit measured, dev seeds 7/8. Flag ON: every mask-frozen synapse bit-unchanged, and no recorded
  write touches a frozen entry. Flag OFF: the drift is present (the zero is not vacuous). Plus instrument
  inertness: tapped == untapped replies.

## What this does NOT claim

- No default flip. `enforce_plastic_mask_beyond_hebbian` and `enforce_plastic_mask_in_hebbian` both stay False;
  the owner flip decision is governed by `research/findings/2026-09-24-plastic-mask-flip-PREREGISTRATION.md` and
  still needs cupy confirmation.
- No capability claim. Freezing the drift does not improve the comprehension reply on this organ; it stops
  1724 (seed 7) / 1700 (seed 8) declared-frozen synapses from drifting up to 55.6 / 58.5 weight units over 30 reads.
- This is CPU/numpy. The attribution ran on the numpy path by design (dev-seed smokes, no GPU contention); the
  gating code is backend-agnostic (`cp.where` / boolean masks, no backend-specific calls), but the GPU path has
  not been re-run for this follow-up.
- The pooled organ's zero-write behaviour is a measured fact about that organ under this battery, not a claim
  that the pooled brain never plasticizes.
