# Biased-competition WTA for multi-referent disambiguation — NEGATIVE (6 seeds)

**Date:** 2026-06-17
**Status:** **NEGATIVE, 6 seeds (0/6 all-four; the decisive ORDER-control 0/6).** Adding winner-take-all
mutual inhibition between the referent attractors PLUS a read-time salience bias does **not** let a bare
pronoun bind the *attended* referent: the winner is still decided by **intrinsic attractor strength**, not by
where the attention bias is applied. The mutual inhibition makes the competition *sharper* (a clean winner,
the loser fully suppressed) — but it sharpens in favour of the intrinsically-stronger attractor, not the
attended one. This is the **third converging NEGATIVE** on the same wall, and it bounds it precisely: on a
point-neuron attractor substrate, symmetric WTA + a top-down bias cannot overcome pattern-specific
excitability asymmetry.

## Context — the pre-registered next mechanism

Two prior NEGATIVES established the wall and named this fix:
- `2026-06-17-multireferent-disambiguation-NEGATIVE.md` — **recency** is NEGATIVE: which of two held referents
  dominates is seed-dependent attractor competition, not recency; the order-control never flips.
- `_phaseB_salience_pointer_derisk.py` — a **salience BOOST** (up to 4× write drive on the foregrounded
  referent) is ALSO NEGATIVE: boosting only ADDS activity; it never SUPPRESSES the competitor, and the
  stronger *intrinsic* attractor wins regardless of drive/order (the per-concept attractors are **independent**
  — no cross-referent coupling — so a boost can't win the competition).

The named fix (Desimone & Duncan 1995, *Annu. Rev. Neurosci.*, **biased competition**): attentional selection
is *competitive* — the attended referent must **suppress** the others via mutual (lateral) inhibition, not
merely out-drive them. This de-risk **builds** that and tests it under the NEGATIVE's exact protocol.

## Mechanism built (`_phaseB_biased_competition_wta_derisk.py`, no `sim/` edit)

A 3-region runner-side bridge (`cortex_ctx` ↔ `dlpfc_wm` attractor loop, as in the validated
`SpikingLoopContextBuffer`, **plus** a dedicated all-inhibitory pool `wta_inh`):

1. **Mutual inhibition between concepts.** `wta_inh` has `exc_fraction=0.0`, so the region framework flips all
   its neurons' trait to inhibitory (verified: `cp_traits[wta_inh] == 1`) → every synapse *from* a `wta_inh`
   neuron routes through the inhibitory conductance channel automatically. Wiring (via `set_pathway_weights`,
   `add_missing=True`):
   - each concept's **excitatory** pattern subset → `wta_inh` (excitatory feedforward; the inhibitory members
     of a concept pattern are filtered out so the drive *excites* the pool),
   - `wta_inh` → **every** concept pattern (inhibitory feedback).
   So any active concept recruits shared inhibition that suppresses ALL concepts → they now **compete**.
2. **Salience bias = a read-time top-down attention current** (`salience_pA`) on the foregrounded referent's
   pattern *during the read only* — the write is EQUAL for both concepts. The intent: mutual inhibition makes
   a modest bias *decisive* (the crucial difference from the failed write-time boost).

**Two knobs, tuned, physiologically modest:** `attractor_weight = 35.0` (a **graded** regime — at weight 50 the
attractors are bistable and pinned at the 0.5 firing ceiling, leaving zero dynamic range for competition);
`inhib_weight = 2.0` (the sweet spot: a modest mutual inhibition that sharpens to a clean winner + fully
suppresses the loser — `inhib_weight ≥ 8` overshoots, killing both then rebound-saturating both to 0.5);
`salience_bias = 300 pA` (the largest value where the empty-WM moat mostly holds — above ~300 the bias alone
starts to ignite a cold pattern). **The verdict is INSENSITIVE to the salience bias across 100–1200 pA** (swept
below) and to a bias-and-resettle read variant.

## Per-seed results (6 seeds, CPU/numpy; `attractor_weight=35`, `inhib_weight=2`, `salience_bias=300 pA`)

NATURAL = write cat,bird (equal) → read with bias on bird (attended). ORDER = write bird,cat (equal) → read
with bias on cat (attended); cat should now win. SUPPR = NATURAL competitor `cat` rate WITH inhibition vs the
SAME competitor in a no-inhibition baseline (plain buffer). NO-SPUR = empty WM + bias on bird → bird must stay
below the read threshold (0.05).

| seed | NATURAL (bird att. vs cat) | dom? | ORDER (cat att. vs bird) | flip? | SUPPR cat: w/inh ← no-inh | sup? | NO-SPUR empty-bird | holds? |
|---|---|---|---|---|---|---|---|---|
| 42  | 0.300 vs 0.004 (r80.0) | ✅ | 0.000 vs 0.315 | ❌ | 0.004 ← 0.172 (drop 0.169) | ✅ | 0.026 | ✅ |
| 43  | 0.000 vs 0.029 | ❌ | 0.000 vs 0.193 | ❌ | 0.029 ← 0.010 (drop −0.019) | ❌ | 0.046 | ✅ |
| 44  | 0.284 vs 0.036 (r7.8) | ✅ | 0.000 vs 0.393 | ❌ | 0.036 ← 0.019 (drop −0.018) | ❌ | 0.083 | ❌ |
| 100 | 0.315 vs 0.214 (r1.47) | ❌ | 0.000 vs 0.403 | ❌ | 0.214 ← 0.236 (drop 0.022) | ❌ | 0.033 | ✅ |
| 101 | 0.000 vs 0.005 | ❌ | 0.000 vs 0.001 | ❌ | 0.005 ← 0.225 (drop 0.220) | ✅ | 0.031 | ✅ |
| 102 | 0.000 vs 0.180 | ❌ | 0.000 vs 0.074 | ❌ | 0.180 ← 0.236 (drop 0.056) | ✅ | 0.039 | ✅ |

**Per-condition pass counts: NATURAL-dominance 2/6 · ORDER-flip 0/6 · SUPPRESSION 3/6 · NO-SPURIOUS 5/6 ·
ALL-four 0/6.** Reproducible run-to-run (numpy deterministic).

## Reading it honestly

- **The decisive failure is ORDER-flip 0/6.** A bare pronoun should bind whichever referent is *attended*;
  the order-control proves that by swapping which concept gets the bias and demanding the winner swap with it.
  It never does: **`bird` wins on every seed in the ORDER arm regardless of the bias being on `cat`** (and
  regardless of write order). The winner is fixed by the random patterns' intrinsic excitability, not by
  attention. This is the *same* failure the recency- and boost-negatives reported — now shown to **survive the
  addition of mutual inhibition**, which was the pre-registered fix.
- **The WTA inhibition does what it should mechanistically — it just doesn't help the attended one.** When the
  attended concept *happens to be* the intrinsically-stronger one (seed 42: NATURAL bird att. → bird 0.300,
  cat suppressed 0.172→0.004, a clean winner), the suppression is dramatic and correct. But that is the
  inhibition sharpening the **intrinsic** winner; it offers no purchase to attention on the seeds where the
  attended concept is intrinsically weaker (43/101/102 → the attended concept reads ≈ 0).
- **The salience bias is caught in an irreducible tension** (swept 100–1200 pA, and a bias-and-resettle read
  that applies the bias through a re-competition phase — both NEGATIVE): a bias small enough to respect the
  empty-WM moat (≤ ~300 pA, NO-SPURIOUS) is far too weak to flip the intrinsic winner; a bias large enough to
  even dent the competition (≥ ~800 pA) instead **manufactures** the attended referent from bias+noise on an
  empty WM (NO-SPURIOUS breaks ≥ ~300 pA). There is no value that flips the order *and* keeps the moat.
- **Why, mechanistically:** the concept attractors are **independent basins** (no learned cross-coupling), and
  symmetric mutual inhibition through a shared pool inhibits all of them *equally*. A top-down bias on one
  pattern raises its drive, but the inhibitory feedback is common-mode, so the **relative** advantage of the
  intrinsically-stronger basin is preserved — the bias cannot re-rank the basins. (This is the working-memory
  analogue of the rate-coded common-mode wall the project hit on the conversational composer: a symmetric
  shared-inhibition WTA removes a common amount from everyone; it does not invert a pre-existing asymmetry.)

## Verdict: NEGATIVE — and it maps the boundary precisely

Three converging NEGATIVES now bound multi-referent disambiguation on the plain spiking WM loop:
**recency** (NEGATIVE), a **salience boost** (NEGATIVE), and **biased-competition WTA + a salience bias**
(NEGATIVE, this doc). The honest conclusion: a *symmetric* mutual-inhibition WTA over *independent,
intrinsically-asymmetric* attractors plus a *top-down read-time bias* is **not sufficient** — the bias cannot
overcome pattern-specific excitability, and the moat caps how strong it may be.

**What the data says is actually needed (the precise next mechanism, if multi-referent dialogue is
prioritized):** the bias must enter the competition **asymmetrically** so it can re-rank the basins, not as a
common-mode add. Concretely, two candidates the negative points to:
1. **Disinhibition of the attended representation** (release the attended concept *from* the shared inhibition,
   rather than adding excitation to it) — a VIP→PV-style attentional gate (Pi-Kepecs 2013) that *lowers* the
   attended basin's inhibition while the others stay inhibited. This inverts the common-mode geometry that
   defeats the additive bias.
2. **Equalized basin depths** — the failure is downstream of the attractors having *different* intrinsic depth
   (random patterns × per-neuron excitability). A homeostatic / normalization step that equalizes basin depth
   before competition would make the modest bias decisive (consistent with the project's standing finding that
   per-pattern excitability asymmetry dominates seed-variable selection). This is the WM analogue of the
   read-out normalization already validated for the conversational pipeline.

A dedicated one-hot "attentional spotlight" population that *gates* the read (winner-take-all on the spotlight,
not on the concepts) is a third option, but it moves the selection off the WM attractors into a separate
circuit — worth scoping only if (1)/(2) also fail.

**Where this leaves multi-turn dialogue (unchanged from the prior negatives):** single-referent anaphora across
turns is and stays GO (production `MultiTurnAgent`). Multi-referent disambiguation remains a mapped boundary —
now with a third, sharper NEGATIVE that rules out symmetric WTA + additive bias and points to disinhibitory /
normalized competition as the specified next mechanism.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_biased_competition_wta_derisk --seeds 42 43 44 100 101 102
```

Knobs (defaults): `--attractor-weight 35 --inhib-weight 2 --salience-pA 300`. No `sim/` edit; reuse-by-import
of the `SpikingLoopContextBuffer` attractor-installation pattern + bridge builder helpers; the WTA wiring and
read-time bias are added runner-side (a new all-inhibitory `wta_inh` region + `set_pathway_weights`).
Raw: `research/findings/raw/_phaseB_biased_competition_wta.json`.
