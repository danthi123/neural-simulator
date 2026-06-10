# Step 2a — navigation gate (a): the 6-seed campaign (IN FLIGHT)

> **Status: IN FLIGHT** (campaign `b1kpv0j99`, ~3.4 hr, 12 runs). The
> seed-42 preview below is decisive-looking (byte-identical); the
> 6-seed table + final verdict are filled when the campaign lands. The
> verdict is produced by the shipped, tested aggregator (commit
> `d4e03965`).

## What this gate asks

Roadmap step 2 consolidates the navigation brain and the conversational
brain onto **one** `SimulationBridge` (the owner's "one brain"
directive). Step 2a holds the conversational half **frozen** (its
synapses cannot change — their per-synapse learning rate is set to zero)
and on its **own block of neuron indices** (disjoint from navigation),
then asks the one question that could sink the whole consolidation:

> Does carrying that frozen conversational half **change how well the
> bridge navigates**?

If it does, merging is not free, and the cost has to be reported (an
honest negative is the deliverable). If it does not, navigation and
conversation can genuinely share one network.

The conversational acceptance gate (gate (b)) is already GREEN — the
full conversational behaviour passes verbatim on the merged bridge
(comprehension, fact memory, question answering, negation/yes-no,
embedded clauses, dialogue planning, generation, and the refusal to make
up an answer it doesn't know). See
`tests/test_nav_conv_merged_agent.py` (8/8, including the three
`is None` abstention assertions) and the prior step-2a findings. Gate
(a) is the **last** step-2a piece.

## Method

Each seed is run **twice**, identically, with `--deterministic` set
(`CUBLAS_WORKSPACE_CONFIG`):

- **standalone:** a navigation-only bridge (the biology-grounded
  flagship recipe — grid 32, moving goal, multi-goal schedule, G v2.5 +
  K v2: MSN lateral inhibition, D1/D2 asymmetry, striatal PV-FSI,
  cluster-A closed loop, cluster-E topography, dlPFC working memory,
  prefrontal NMDA, visual cortex with a 600-step action warm-up).
- **merged:** the same navigation bridge **plus** the frozen
  conversational regions (the sentence parser + the prefrontal
  dialogue-planning regions), each on its own block of neuron indices,
  navigated by the same recipe through the hybrid nav-episode
  integration (`research/runners/_nav_gate_merged_run.py --with-conv`).

**Navigation score** = the sum, over the four goal phases, of the
final-quarter mean Manhattan distance to the goal. **Lower is better**
(the agent is sitting closer to the goal at the end of each phase). The
score is **not** a stored field; it is computed from `phase_stats`.

**Aggregation + verdict** (the shipped, tested tool):

```bash
python -m research.runners.nav_gate2a_aggregate \
    --raw-dir research/findings/raw/nav_gate_2a \
    --out research/findings/raw/nav_gate_2a/_gate2a_verdict.json
```

reads `gate6_{standalone,merged}_seed{42..47}.json` and emits:

| Verdict | Condition | Meaning |
|---|---|---|
| **GREEN_INERT** | max matched-seed \|merged − standalone\| ≤ 0.05 | the conversational half is inert; merging is free (the intended result) |
| **GREEN_WITHIN_NOISE** | ≤ 0.7 (the deterministic run-to-run noise floor) | merging is within run-to-run noise |
| **REGRESS** | > 0.7 | carrying the conversational half measurably changed navigation — a real finding (the measured cost of merging), reported, not hidden |

## Why inertness is expected (the mechanism)

Two independent guarantees:

1. **Disjoint.** The conversational regions occupy their own contiguous
   neuron indices; no pathway runs from a conversational region into any
   navigation region. The merged build is genuinely larger (see the
   seed-42 preview: +4 regions, +2,166 neurons), so this is not a silent
   no-op — the conversational neurons are present.
2. **Frozen.** The conversational synapses' per-synapse learning rate is
   held at zero, so navigation's continuous reward-and-spike-timing
   learning cannot change them. The one ungated path — a global
   "keep weights in range" clip applied regardless of learning rate — is
   neutralized by widening the allowed range (`stdp_w_max` +
   `hebbian_max_weight` = 400) above the largest conversational weight
   (~300), proven byte-safe for navigation by the earlier cheap-first
   check (`2026-06-10-nav-gate-stdp-wmax-400-cheap-check-PASS.md`).

In step 2a the conversational half is **all standard (Izhikevich)
neurons** — the phase-based composer is still on its own separate bridge
here, so the neuron-model-coexistence question does not arise at this
gate (it is step 2b). This gate is purely: does a frozen, disjoint,
same-model population perturb navigation?

The nav-on-merged single-seed smoke already showed the frozen
conversational synapses stay **byte-identical** before/after a live
navigation episode that actively rewires the navigation half — the
freeze holds in vivo, through the post-initialization perception-weight
rebuild (`2026-06-10-nav-on-merged-smoke-PASS-hybrid-integration-works.md`).

## Seed-42 preview (2 of 12 runs)

| arm | regions | neurons | navigation score |
|---|---|---|---|
| standalone | 47 | 4,662 | **2.0000** |
| merged | 51 | 6,828 | **2.0000** |
| | | | **Δ = 0.0000** |

The merged bridge carries **+4 regions / +2,166 frozen, disjoint
conversational neurons** and navigates **byte-identically** — every one
of the four goal phases has the same final-quarter mean distance
(`[0.496, 0.504, 0.496, 0.504]`) in both arms. This is the strongest
possible inertness signal at a single seed: the deterministic navigation
computation is unchanged by the presence of the frozen conversational
half. On track for **GREEN_INERT** across all six seeds.

## Results (6-seed) — PENDING `b1kpv0j99`

<!-- Fill from: python -m research.runners.nav_gate2a_aggregate -->

| seed | standalone | merged | delta (m − s) |
|---|---|---|---|
| 42 | 2.0000 | 2.0000 | +0.0000 |
| 43 | _pending_ | _pending_ | _pending_ |
| 44 | _pending_ | _pending_ | _pending_ |
| 45 | _pending_ | _pending_ | _pending_ |
| 46 | _pending_ | _pending_ | _pending_ |
| 47 | _pending_ | _pending_ | _pending_ |

**Verdict:** _pending the aggregator on the complete campaign._

## On a GREEN verdict

Gate (a) GREEN + gate (b) already GREEN ⇒ **step 2a complete**:
navigation and conversation are capability-equivalent to the two
separate brains while sharing one network. Next is **step 2b** — moving
the phase-based composer onto the same bridge too, via the
owner-approved masked phase-neuron operations
(`docs/plans/2026-06-10-step2b-rf-coresident-implementation.md`,
trust-but-verified execution-ready). Step 3 (replacing the composer's
fixed binding algebra with a learned spiking cortex) stays a separate,
later effort.

## On a REGRESS verdict

A REGRESS would be unexpected (the conversational half is frozen and
disjoint), so it would point to an interaction the mechanism does not
predict — a genuine finding worth tracing (e.g. a shared global
normalizer, an index-aliasing bug, or a determinism leak), not something
to paper over. The cheap-first check and the single-seed smoke both
already came back inert, so the prior is strongly GREEN.

## Trail

- Construction increment: `2026-06-10` (commit `045ab7ae`)
- Conversational gate (b) GREEN: commit `e0dc8d2a`,
  `tests/test_nav_conv_merged_agent.py`
- Cheap-first `stdp_w_max` check PASS:
  `2026-06-10-nav-gate-stdp-wmax-400-cheap-check-PASS.md`
- Nav-on-merged smoke PASS:
  `2026-06-10-nav-on-merged-smoke-PASS-hybrid-integration-works.md`
- Aggregator + tests: commit `d4e03965`,
  `research/runners/nav_gate2a_aggregate.py`,
  `tests/test_nav_gate2a_aggregate.py`
- Architecture (plain language): `docs/ARCHITECTURE_nav_conv_merge.md`
