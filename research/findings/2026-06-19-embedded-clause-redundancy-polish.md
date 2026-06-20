# Embedded-clause depth-1 parse — population-redundancy read-out polish (closes the 0.02 gap)

> **Follow-on to `2026-06-19-embedded-clause-parse-derisk.md`** (the near-GO: matrix clause 6/6, embedded clause mean
> 0.951, 2 seeds (43, 101) at **0.88 — 0.02 under** the ≥0.90 bar). That finding named the lever: **population
> redundancy on the embedded read-out**. This closes it. Runner (additive edit, NO `sim/` edit):
> `research/runners/_phaseB_embedded_clause_parse_derisk.py` (`--readout-redundancy N`).

## Verdict: GO (6/6 seeds, GPU + CPU) — the redundancy lever lifts the 0.88 marginal seeds to 1.000; depth-1 embedded-clause parsing is now a clean multi-seed GO

The two marginal seeds (43, 101) that landed at **0.88/0.875** at R=1 lift to **1.000** at R=3 (R = independent
phasor populations, majority-vote per embedded slot) on BOTH the GPU 6-seed run (all 6 seeds 1.000) and the CPU
check, with **every anti-cheat control still collapsed** (no-segmentation 0.500, scramble 0.000, head-attachment
0.000) and the **no-confab moat intact 6/6**. The prior near-GO (embedded mean 0.951, 4-of-6 strict) is now a clean
embedded 6/6 at 1.000. Recommend `--readout-redundancy 3` as the de-risk default for depth-1 embedded-clause parsing.

## The mechanism (the named lever; reuse-by-import, NO `sim/` edit)

`RedundantEmbeddedReadout` wraps **R independent `RFPhasorComposer` replicas** — same `D` + vocab, but each replica's
phasor codebook is drawn from a DISTINCT seed (`seed + 1000*i`), i.e. a different sub-population. `store()` writes the
same parsed nested fact into every replica; `query_patient()` decodes the embedded clause in EACH replica and
**majority-votes** the decoded `(agent, action, patient)` per word slot.

**Why it lifts the marginal seeds.** The embedded clause sits a SECOND unbind down (the matrix patient IS the embedded
composite), so the 2-level `query_patient → _render → _cleanup` path accumulates the phasor round-trip phase-jitter
TWICE before the cleanup argmax. On 2 of 6 seeds that jitter tips ONE of the 3 embedded slots past its codebook
neighbour → 0.875 (21/24). Because each replica's phase-noise is INDEPENDENT (independent codebooks), a per-slot
majority vote across R replicas averages the noise out and the marginal slot recovers. This is the documented
spike-native robustness rung **(c) "population redundancy + attractor cleanup"** — a redundant cortical read-out, NOT
a change to the binding algebra.

The lever is **additive and default-preserving**: `--readout-redundancy 1` is a pass-through to a single composer
(byte-identical to the prior path). The matrix-clause parse is UNCHANGED (it reads the parser, not the composer), so
redundancy only touches the embedded read-out. The **moat is preserved** by construction: a replica that abstains
(None) casts no vote, and if EVERY replica abstains the vote is None — a missing fact is never voted into existence.

## Results

### CPU/numpy — the decisive lift (faithful to the GPU result: 0.875 ≈ the documented 0.88)

The CPU numpy path reproduces the GPU `0.88` marginal dip as `0.875` (21/24) and the lift is unambiguous:

| seed | R=1 embedded | R=3 embedded | matrix (R=1=R=3) | no-seg (must fail) | scramble | head-attach | moat |
|---|---|---|---|---|---|---|---|
| 42  | 1.000 | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 | ✓ |
| 43  | **0.875** | **1.000** | 1.000 | 0.500 | 0.000 | 0.000 | ✓ |
| 101 | **0.875** | **1.000** | 1.000 | 0.500 | 0.000 | 0.000 | ✓ |

(seeds 43, 101 are the two documented marginal seeds; seed 42 confirms R=1 == R=3 where the read-out was already
clean. R=1 verdict on {43,101} = BOUNDARY → R=3 verdict = GO.)

### Multi-seed (GPU/cupy, seeds 42,43,44,100,101,102; 12 subj + 12 obj held-out relatives/seed), R=3 — GO 6/6

The CPU lift reproduces on the real GPU substrate: **all 6 seeds at 1.000**, both documented marginal seeds (43, 101)
lifted from 0.88 → 1.000, every control collapsed, moat intact.

| seed | embedded acc | matrix acc | no-seg (must fail) | scramble (must fail) | head-attach (must fail) | moat | secs |
|---|---|---|---|---|---|---|---|
| 42  | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 | ✓ | 391.8 |
| 43  | **1.000** (was 0.88) | 1.000 | 0.500 | 0.000 | 0.000 | ✓ | 388.7 |
| 44  | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 | ✓ | 387.5 |
| 100 | 1.000 (was 0.96) | 1.000 | 0.500 | 0.000 | 0.000 | ✓ | 399.6 |
| 101 | **1.000** (was 0.88) | 1.000 | 0.500 | 0.000 | 0.000 | ✓ | 421.8 |
| 102 | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 | ✓ | 410.2 |
| **mean** | **1.000** | **1.000** | 0.500 | 0.000 | 0.000 | 6/6 | — |

Verdict line from the run: `embedded roles ≥0.90: 6/6 seeds   matrix roles ≥0.90: 6/6 seeds`; NO-SEGMENTATION
baseline FAILS (all); scramble FAILS (all); permuted-head FAILS (all); moat intact (all); leakage clean (all);
**==> depth-1 GO**. (Compare the prior near-GO: embedded mean 0.951 / 4-of-6 strict at R=1; the redundancy lever
takes embedded to a clean 6/6 at 1.000.) Per-seed wall ~390–420 s at R=3 (GPU; R=3 triples the read-out cost). JSON:
`research/findings/raw/_embedded_clause_redundancy_multiseed.json`.

## The controls all still collapse (a "success" without these is an artifact)

1. **NO-SEGMENTATION baseline FAILS** (0.500 < 0.90) — the load-bearing control. UNCHANGED by redundancy: the
   redundant read-out only votes over the PARSER's segmentation; a flat (unsegmented) reader still mis-reads the
   object-relative half by luck-vs-structure (0.5).
2. **Scramble FAILS** (0.000). Redundancy CANNOT rescue a wrong parse: the parser feeds the SAME scrambled span to
   every replica, so all replicas vote the same wrong answer — the control stays collapsed (strictly: a scramble
   would have to fool a majority of independent replicas to "pass", which it cannot when they all see the same wrong
   span).
3. **Permuted-head-attachment FAILS** (0.000) — the matrix answer tracks the actual head (a wrong head → wrong
   answer); UNCHANGED (the matrix parse is not touched by the read-out redundancy).
4. **Held-out + leakage-asserted** — role assignment is by position-conjunction; test filler tuples disjoint by
   construction (leakage = 0).
5. **The path is NEURAL** — each replica's per-clause role read-out is the spiking `AttributedBridgeParser`; the
   suspended-head HOLD is the spiking `OrderedPositionWM`; the nested decode is the spiking resonate-and-fire 2-level
   unbind; the moat is the spiking familiarity/cue-match abstention. The redundancy is a population of these spiking
   read-outs voted together. Host is limited to the environment (token string + closed-class lexical tag) + the body
   (emit) + the majority tally of the per-replica neural read-outs.
6. **The no-confab MOAT is intact** — garbled stream → abstain; unknown token → abstain; never-stored cue → None;
   all-replicas-abstain → None. 0 false-accepts.

## Cost

R=3 triples the per-seed read-out cost (3 composer replicas instead of 1; CPU ~17 s/seed vs ~9 s at R=1; GPU
proportionate). This is the population-redundancy trade: 3× read-out compute buys the 0.02-gap closure. The lever is
opt-in (`--readout-redundancy`), so the cheap R=1 path remains available.

## Recommendation

Adopt **`--readout-redundancy 3`** as the depth-1 embedded-clause de-risk default. depth-1 embedded-clause parsing is
now a clean multi-seed GO (embedded ≥0.90 on 6/6 seeds, matrix 6/6, all controls collapse, moat intact). The
deprioritized follow-ons from the prior finding are unchanged: the production `parse_nested` opt-in (mirroring
`enable_attributed`/`enable_multiframe`); the fully-neural relativizer/verb detector; and depth-2 center-embedding =
the expected catalog G.12 boundary (the human ~2-level limit; NOT to brute-force).

## Files

- runner (additive edit, NO `sim/` edit): `research/runners/_phaseB_embedded_clause_parse_derisk.py`
  (`RedundantEmbeddedReadout`, `--readout-redundancy N`)
- GPU multi-seed JSON: `research/findings/raw/_embedded_clause_redundancy_multiseed.json`
- reused (NO edit): `rf_phasor_composer.py` (`RFPhasorComposer`, `Clause`), `attributed_parser.py`
  (`AttributedBridgeParser`), `ordered_position_wm.py` (`OrderedPositionWM`).
