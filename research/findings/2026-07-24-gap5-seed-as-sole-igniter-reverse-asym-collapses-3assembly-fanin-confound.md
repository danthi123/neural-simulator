# gap#5 SWR readout — seed-as-sole-igniter COLLAPSES reverse-asym (co-fire confound removed), but the 3-assembly chain's FAN-IN GEOMETRY is a NEW confound → residual is STORE ARCHITECTURE (2026-07-24)

**One-line:** With the SWR seed as the SOLE igniter (no co-firing background), the REVERSE-ASYMMETRY lesion finally
COLLAPSES (0.500 → 0.250) — the forward direction genuinely rides the learned weight asymmetry, a real advance over the
co-fire regimes. BUT the GO bar still fails, and the anti-cheats isolate a **new, real cause**: on a **3-assembly** chain
the forward order is dominated by **fan-in connectivity geometry** (SHUFFLE-INSENSITIVE) + a seed-0-start lock, on n=2–4
events. That is a **STORE-ARCHITECTURE** limit (chain length + adjacent-link dominance + generic connectivity), **not**
readout tuning. The SWR-state readout machinery is proven; the open question moved to the store.

## Context / the arc that led here
The gap#5 imaginative-replay READOUT is the surviving path for a 5-method-boundary capability (research gate
`2026-07-24-gap5-SWR-state-readout-research-gate-the-missing-biology.md`). This session built the SWR-state
E/I-transient envelope readout (`research/runners/_gap5_swr_envelope_replay_derisk.py`, reuse-by-import, **NO `sim/`
edit**) on the frozen 6/6-GO DECOUPLED forward-asymmetric store (within ~200, adj_fwd ~38, adj_rev ~5), and drove it
through four mechanisms with 9 anti-cheats each. The progression of readouts and what each anti-cheat revealed:

| readout | ignition | selectivity (NO-ENCODE) | forward order | REVERSE-ASYM lesion | verdict |
|---|---|---|---|---|---|
| Option 1 (cued completion op-point) | assembly-0 ignites (per_asm_frac 0.244) | — | no hand-off | — | ignition IS achievable → boundary was op-point/STATE |
| broad-exc envelope (env_exc=550) | co-fire [3,3,3] | **[10,10,10] — NON-selective** | 0.0–0.33 (flips w/ store non-determinism) | — | detonator, not selective |
| Mechanism #1 latch (whole-envelope, sub-detonator) | selective [3,3,3] | **[0,0,0] — SELECTIVE** ✓ | co-fire, forward not robust | — | 12/13 checks pass; only forward-ordering fails |
| Mechanism #3 seed-BIAS (in co-fire regime) | [3,3,3] | [0,0,0] | FWD-from-seed 0.60 (looks great) | **HELD at 0.5** | NO-GO — anti-cheat caught it: order = seed-position + co-fire, not the links |
| **seed-as-SOLE-igniter** (noise OFF, env_exc=0) | selective, seed-first | **[0,0,0] no-cascade** ✓ | FWD-from-seed 0.50 | **COLLAPSES 0.5→0.25** ✓ | NO-GO — new cause: fan-in geometry (below) |

## The seed-as-sole-igniter result (seed 42, GPU, n_ca3=2000)
Config: env_exc=0, **noise OFF** (seed is the only ignition source), a SYNCHRONOUS coincident seed volley (seed_dur=3,
seed_frac=0.8) onto ONE RANDOM assembly per envelope, whole-envelope latch (self_regen_ignite=0.15, ignite_frac=0.4 →
latch-to-ignite then release-to-hand-off), SFA d_abs=80, and **recall_k_thresh lowered 40 → 8/10/12** (the SWR up-state's
documented 3–5× excitability gain literally lowers the ignition threshold — Buzsáki *Rhythms of the Brain* L14452;
biology, not a fudge). Best point: rk=8, seed_pa=700, ignite_frac=0.4.

- **NO-ENCODE = [0,0,0] no-cascade at EVERY point** — clean cascade-selectivity (the un-encoded store, weights=0.5, never cascades).
- seed-first onset high, discrete, self-terminating.
- **FWD-from-seed 0.500 vs reverse 0.000 vs chance 0.167.**
- **⭐ REVERSE-ASYM-LESION COLLAPSES: 0.500 → 0.250** (vs mechanism #3, where it HELD at 0.5). Removing the co-fire made
  the forward DIRECTION ride the learned weight asymmetry — the diagnosis was correct, and this is real progress.

## Why it still fails — the NEW, quantified confound (3-assembly fan-in geometry)
1. **`not_just_seed0` = FALSE (k_fwd = 1):** forward cascades occur ONLY when the random seed lands on assembly **0**
   (the chain start). `by_seedpos = {0:(fwd 1, multi 2), 1:(0,0), 2:(0,0)}`. Root cause, quantified: a cascade to
   assembly **2** needs forward inputs from **both 0 and 1** to cross the coincidence threshold. At ca3_density=0.05 and
   assembly ~240, each cell has only ~**12** within-group / adjacent-forward inputs. Seeding 0 → 0 fires → 0→1 ignites 1
   → assembly 2 now receives 0→2 **+** 1→2 (~accumulated) → crosses → ignites. Seeding **1** → only the 1→2 link (~12
   inputs) → **does not cross** → no cascade to 2. Lowering rk to 8 did not fix it (12 spread inputs still short of a
   reliable coincident volley).
2. **SHUFFLED HOLDS (0.333, does NOT collapse):** permuting the between-assembly edge WEIGHTS leaves the forward order
   intact, because the feedforward **CONNECTIVITY geometry** (assembly 2 sits downstream of the 0,1 fan-in) carries the
   0→1→2 order **independent of the specific learned weights**. This is the decisive tell that the order is geometric,
   not a per-link learned cascade.
3. **Tiny statistics:** n_multi = **2–4** events per seed. FWD 0.50 vs reverse-asym 0.25 vs shuffled 0.33 are all within
   noise at n=2–4 — the "collapse" is not statistically robust.

## Verdict + reframe (the negative IS the finding)
Seed-as-sole-igniter **removed the co-fire confound** (reverse-asym collapses — genuine, real progress) but does **NOT**
produce a genuine per-link sequential cascade on a 3-assembly chain: the residual order is dominated by (a) a seed-0
start lock + (b) fan-in connectivity geometry (shuffle-insensitive), on too few events to be robust. **This is not
tuning-fixable — it is the chain's architecture.** The SWR-state readout machinery (latch selectivity + seed initiation
+ the forward-from-seed scorer with `by_seedpos` + reverse-asym / shuffled controls) is BUILT, proven, and reusable; the
open question is now the **STORE ARCHITECTURE**.

## The decisive next test (store architecture, not readout tuning)
A store where the order can ONLY ride the LEARNED WEIGHTS, removing the geometric shortcut:
1. **≥6 assemblies** → seed **interior** positions (k=2 AND k=3) and require the cascade to continue forward from EACH
   → `not_just_seed0` becomes a real multi-position test.
2. **GENERIC UNIFORM between-assembly connectivity** (NOT a hand-wired fan-in) + **adjacent-link-dominant LEARNED
   weights** (strong adj_fwd, weak adj_rev / skip) so a **single k→k+1 learned link alone** drives the next across the
   coincidence threshold → the order MUST ride the learned weights → **SHUFFLED must then COLLAPSE**.
   - Biology grounding for strong-adjacent-links: forward-asymmetric STDP / theta phase-sequence compression / BTSP
     encoding of a place-cell sequence all produce **strong nearest-neighbour forward** weights (Skaggs-McNaughton
     phase precession; Bi-Poo asymmetric STDP; Bittner BTSP). **Honest flag:** whether a *single* adjacent link should
     suffice to ignite the next (vs. real replay genuinely relying on convergent fan-in from several recent-past cells)
     is an open biological question — if real replay relies on fan-in, then the "shuffle must collapse" bar itself may
     need rethinking. To be reported alongside the result.
3. **Enough events** (longer rest / more envelopes) for a real FWD-vs-lesion separation — report n_multi + the statistic.

**Tightened GO bar (load-bearing):** SHUFFLED must COLLAPSE (the check that the fan-in artifact is gone — order rides
SPECIFIC learned weights) **AND** REVERSE-ASYM must COLLAPSE **AND** `not_just_seed0` = TRUE (forward from interior
seeds) **AND** NO-ENCODE=[0,0,0] **AND** discrete + self-terminating **AND** robust statistics. If a longer
generic-connectivity chain STILL can't carry a shuffle-sensitive per-link cascade even with strong learned adjacent
links → a deeper store-encode finding → full research gate.

## UPDATE (same day) — the 6-assembly adjacent-dominant test confirms: the FRAME is wrong, not the store
Followed the scoped next test: a **6-assembly** chain, GENERIC uniform connectivity, made **adjacent-dominant** by a
skip-suppression WEIGHT lesion (RAW BTSP profile adj_fwd 39.4 / **skip_fwd 98.6** — BTSP is inherently skip-DOMINANT
fan-out; POST-SUPPRESS adj_fwd 39.4 / skip_fwd 1.5 → **adj_dominance 7.87**), seed as SOLE igniter (noise off), INTERIOR
seeds, rest_steps=8000 (seed 42, `swr_longchain.json`). Result: **GO=False, and interior seeds STILL don't cascade** —
`k_fwd_interior = 0`, `by_seedpos = {0:(1,3), 1:(0,0), 2:(0,0), 3:(0,0), 4:(0,0), 5:(0,0)}` (forward only from seed-0,
STILL), REVERSE-ASYM **HELD** (0.500), n_multi=3 (not robust). ⇒ **even after fixing the store's own defect (the BTSP
fan-out) to a clean 7.87 adjacent-dominance, the discrete-assembly + discrete-ignition FRAME cannot produce a traveling
interior-seed cascade.** The residual is the **frame + metric**, not the store: a single adjacent link cannot ignite the
next DISCRETE assembly (the ~10-input coincident volley from one predecessor assembly is insufficient, and there is no
graded overlap to carry a moving bump). This is exactly what the follow-on research gate
(`2026-07-24-gap5-replay-sequence-encoding-shuffle-bar-research-gate.md`) predicts → REDIRECT to the Ecker-2022
graded-band / moving-bump frame (overlapping place fields + a decaying near-diagonal weight band + population-decode
readout), NOT discrete assemblies. Raw: `research/findings/raw/gap5_r4/swr_longchain.json`.

## Files
- Runner (all mechanisms + scorers, additive, NO `sim/` edit): `research/runners/_gap5_swr_envelope_replay_derisk.py`
  — Option 1/2, latch-then-release schedule, env_exc ramp, per-envelope random single-assembly seed (`env_seed_log`),
  `_score_forward_from_seed` (forward-FROM-seed + `by_seedpos` breakdown), NO-ENCODE ignition-selectivity gate.
- Raw: `research/findings/raw/gap5_r4/swr_sole_igniter.json` (+ `swr_envelope_seed42.json`, `swr_envelope_latch_seed42.json`,
  `swr_seeded_derisk.json`).
- Research gate: `research/findings/2026-07-24-gap5-SWR-state-readout-research-gate-the-missing-biology.md`.
