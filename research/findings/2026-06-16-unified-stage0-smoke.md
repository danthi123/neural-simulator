# Unified embodied agent — STAGE 0 CPU/numpy integration smoke: GO

**Date:** 2026-06-16
**Type:** cheapest-first CPU/numpy integration de-risk (§3 Stage 0 + §5 GATE of
`research/findings/2026-06-16-unified-embodied-agent-scoping.md`).
**Runner:** `research/runners/_unified_stage0_smoke.py` (`SIM_BACKEND=numpy`, single-seed, ~60 s/seed).
**Verdict:** **GO** (seed 42; corroborated seed 43). NO `sim/` edit.

---

## The question Stage 0 answers

Do the NEW integration pieces co-exist without regression on ONE small bridge —
- **H5** the generalization read (structured perception → NMDA concept assembly SPIKES for the right category),
- **H6** the option-(b) HYBRID recall (read which concept-category SPIKED → key the validated
  `RFPhasorComposer` recall of THAT category's stored fact),
- the no-confab **MOAT** (a no-category cue must ABSTAIN),
- AND a co-resident **PARSER** whose comprehension read is BYTE-STABLE beside the new regions
  (the 5a co-residence discipline, in miniature)?

This de-risks the merged-bridge GPU wiring (Stage 1) on CPU before any heavy build.

## How the two stacks co-reside on ONE `SimulationBridge`

Brain-region framework ON; FIVE disjoint regions, one bridge, one step loop (1006 neurons total at the smoke
config):

| Region | Size | Role |
|---|---|---|
| `perception` | 400 | structured-perception input (the controlled Option-B given) |
| `concept` (NMDA) | F·40 = 320 | the per-category concept assembly (the H5 spiking read) |
| `fact` (NMDA) | N_CAT·40 = 160 | the per-category fact-tag scaffold (convergent block from concept) |
| `parse_conj` | 6 | the parser's 6 (position×voice) conjunction units |
| `parse_role` | 3·40 = 120 | the parser's agent/action/patient role ensembles |

Small config owned by the runner so it stays a few-minutes numpy smoke: **4 categories × 2 exemplars = 8
concepts**, tiny D=48 composer, 12 convergence epochs, 60-step reads.

Wiring (one `inject_explicit_wiring` call — see the load-bearing gotcha below):
- `perception → concept`: all-to-all, plastic, near-floor 0.05 init — the **rate-Hebbian** convergence the setup
  pass LEARNS (STDP is the wrong rule for symmetric co-occurrence — the CYCLE-95 finding; reused verbatim).
- `concept → fact`: CONVERGENT block (every concept block of category *c* → fact-tag block *c*), FIXED.
- `parse_conj → parse_role`: all-to-all, init 0.5, plastic, tagged `parser_fixed` (the framework parser
  pathway, rebuilt into the same explicit injection).

**Co-residence discipline applied (5a, in miniature):**
1. The parser is trained (its own teacher-driven Hebbian co-firing on the **disjoint** parse slices), then its
   gate is frozen (`set_plasticity_gate("parser_fixed", 0.0)`).
2. The parser read is taken **BEFORE** the generalization training, and again **AFTER** — and asserted
   byte-stable. The convergence pathway is on slices disjoint from the parser, so the per-synapse Hebbian update
   touches each independently; the frozen gate + the disjoint slices keep them from drifting each other.
3. `hebbian_max_weight` raised to **400** (the merged-bridge / `BridgeParser` value) so no frozen/learned weight
   is clipped below where it needs to be (see the parser-cap finding below).

## Result (seed 42)

| Metric | Value | Gate (§5) | Pass |
|---|---|---|---|
| Parser byte-stable (before == after generalization train) | **True** | byte-stable | ✅ |
| Parser correct (voice-invariant SVO: `{agent:dog, action:chase, patient:cat}`) | **True** | correct | ✅ |
| H5 concept-category spike accuracy | **0.75** | > chance (0.25) + margin | ✅ (3×) |
| concept spikes/cue | 222 | spikes present | ✅ |
| fact spikes/cue | 404 | (propagates one synapse further) | ✅ |
| H6 hybrid category-fact recall | **0.75** | ≥ 0.50 | ✅ |
| MOAT: no-category cue | **ABSTAIN** (`novel_recall=None`) | **ZERO breaches (HARD)** | ✅ |
| MOAT familiarity contrast | held-out win-fire **1.75** vs novel **0.92** (gate 1.05) | held-out > novel | ✅ |

**==> GO.** All §5 gate conditions met, zero moat breaches.

**Corroboration (seed 43, not required by the single-seed gate):** H5 **1.00**, H6 **1.00**, parser
byte-stable + correct, moat ABSTAIN (held-out 1.91 vs novel 1.14). The GO is not seed-42-lucky.

## The one finding that mattered: the global Hebbian cap must be 400, not 20

First run (`hebbian_max=20`, the convergence de-risk's default) → **NEGATIVE**: the parser read collapsed (every
position read the default first role; all role-ensemble firing = 0 under conj-alone drive), H5 at chance.

Diagnosis (weight probe): the parser **learned the correct selective weights** — conj0→agent 13.96 (others
0.44), conj1→patient 19.76, conj2→action 19.81, … — but the readout role ensembles never spiked, because the
weights were **capped at 20**, below what a single conjunction neuron needs to drive a role ensemble over the
Izhikevich +30 mV threshold. The standalone `BridgeParser` and `nav_conv_merged_bridge` both use
`hebbian_max_weight=400` for exactly this reason.

Fix: raise the **global** Hebbian cap to **400** (now the runner default). The convergence's
**category-mean-over-spikes** read is robust to the higher cap (more concept spikes, still category-correct), so
one global cap serves both stacks. Measured: 20 → parser correct FALSE / H5 chance; 400 → parser
byte-stable+correct / H5 0.75 / moat intact = GO. This is a bounded config value carried into Stage 1, not a
wall.

## The load-bearing wiring gotcha (recorded for Stage 1)

`inject_explicit_wiring` **REPLACES** `cp_connections` entirely (`sim/bridge.py:2289`, "rebuilds
self.cp_connections from the explicit edges"). The framework-injected `parse_conj → parse_role` pathway would be
**wiped** if the explicit injection names only the generalization populations. The runner therefore rebuilds the
parser edges into the **same** injection dict (all-to-all, init 0.5, plastic, gated `parser_fixed`) so both
stacks are wired in one replacing pass. Stage 1 (which appends these regions to `build_merged_nav_conv_bridge`)
rides the framework's own combined-injection path and so does not hit this — but the gotcha is why the de-risk
runners that "fully overwrite" the framework wiring must include every population they want kept.

## Anti-cheat honesty

- The structured-perception generator scatters each concept's category-core + unique-tail across the whole
  region via a random permutation, so neither sits in a low-index block — killing the spurious monotonic
  index-bias that would let category 0 always win. H5 = 0.75 (not a suspicious 1.00) at seed 42 with a working
  familiarity contrast (1.75 vs 0.92) is consistent with **real structure-driven category transfer**, not an
  artifact.
- This is the **cheap §5 gate** (one structured arm + the moat + the parser co-residence). The full
  **flat-distinct + category-derangement** controls (visual structure load-bearing; the transfer is the LEARNED
  correspondence) are Stage 1's gate (§4), where they ride the already-validated `_genfrontier_*` de-risk
  machinery at the larger convergence dims.
- The moat rides the concept SPIKES (a no-category cue's best-category response 0.92 falls below the
  held-out-calibrated gate 1.05 → not familiar → the agent does **not** key the composer recall → `None`). The
  recall + abstention are the validated `RFPhasorComposer` mechanism; the moat is **never** weakened.

## `sim/` edit?

**NONE.** Reuse-by-import only: the generalization machinery (`_genfrontier_*` builders/training/read pattern),
the parser ports (`nav_conv_merged_bridge`: `parser_regions_pathways`, `train_parser_on_slices`,
`role_of_on_slices`, `parse_on_slices`), and the H6 hybrid + moat (`RFPhasorComposer`).

## Verdict + what it gates

**GO** → promote to **Stage 1**: add the generalization stack (`structured_perception` / `concept` / `fact`) to
`build_merged_nav_conv_bridge` as additive default-off regions appended LAST, route the concept-spike read into
the composer recall (H6 hybrid), and run the single-seed GPU gate (Option-B → A closes on the merged bridge for
a held-out novel shape through real Gabor/V1 + the existing co-residence battery still passes: nav
byte-identical, conversational 7/7, moat intact). Carry `hebbian_max_weight=400` (the merged bridge already uses
it).

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._unified_stage0_smoke --seed 42
# GO: parser byte-stable+correct | H5 0.75 (>chance 0.25) | H6 0.75 | moat ABSTAIN. ~60 s.
```
