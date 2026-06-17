# Multi-referent disambiguation — NEGATIVE on the plain WM loop (it needs an attention/salience pointer)

**Date:** 2026-06-17
**Status:** **NEGATIVE, 3 seeds — and it cleanly bounds the multi-turn capability.** The single-referent case is
GO (shipped: the `MultiTurnAgent`). When the working memory holds *several* referents, a bare pronoun ("it")
cannot be disambiguated by recency on the plain spiking loop — which referent dominates is seed-dependent
attractor competition, not recency. Multi-referent disambiguation needs an added salience/attention mechanism.

## The question

A bare pronoun usually binds the **most recent** salient referent ("the dog saw the cat. it ran." → it = the
cat). The multi-turn de-risk (`2026-06-17-multiturn-anaphora-derisk-GO.md`) flagged this as the honest next
stress. Does the spiking WM loop carry a **recency gradient** (the most-recently-written referent dominates the
read), so a bare pronoun binds it — or does it hold all referents as an equal **set** (the validated ≥3-concept
hold), leaving a bare pronoun genuinely ambiguous?

## Result (`_phaseB_multireferent_disambiguation_derisk.py`, 3 seeds, CPU)

| seed | NATURAL (write cat, then bird) recent=bird | ORDER-CTRL (write bird, then cat) recent=cat | REFRESH bird again |
|---|---|---|---|
| 42 | bird 0.33 ≫ cat 0.03 → recent wins | cat 0.00 vs bird 0.33 → **older wins** | bird 0.32 > cat 0.07 → recent |
| 43 | cat 0.42 > bird 0.22 → **older wins** | cat 0.11 < bird 0.28 → older wins | bird 0.275 ≈ cat 0.264 → tie |
| 44 | cat 0.315 ≈ bird 0.335 → tie | cat 0.32 ≈ bird 0.335 → tie | tie |

**NEGATIVE (0/3 natural, 1/3 refresh).** There is no reliable recency gradient: in NATURAL the recent referent
wins only on seed 42; on seed 43 the *older* one wins, on seed 44 it's a tie. The order-control does not flip the
winner (it should, if recency were the driver). So **which referent dominates is seed-dependent attractor
competition, not recency** — the loop holds both as a set (consistent with the validated multi-concept hold), and
re-driving the recent one once (REFRESH) creates the gradient on only 1/3.

## Reading it honestly

- **The single-referent multi-turn case is and stays GO** (the `MultiTurnAgent` resolves "it" when one referent
  is held, 3 seeds). This negative does not touch that.
- **Multi-referent disambiguation is the genuine wall for multi-turn dialogue on the plain loop.** A bare pronoun
  among several held referents cannot be resolved by the loop's own dynamics — the WM stores the *set*, not a
  ranked salience. This is mechanistically sensible: a content-addressable attractor memory holds *what* is in
  mind, not *which* is currently most salient.
- **The next mechanism is precise and biologically grounded (a buildable follow-on, not a dead end):** an
  explicit **salience / attention pointer** that tags the most-recent (or grammatically-foregrounded) referent —
  e.g. a transient gain boost on the salient attractor, or a separate one-hot "attentional spotlight" population
  that gates the read. Biology: attentional selection of a discourse referent (the brain does not bind a pronoun
  by raw memory strength; it uses attention/saliency). That is the right next de-risk if multi-referent dialogue
  is pursued.

## Follow-up — the salience-boost mechanism is ALSO NEGATIVE; the requirement is WTA inhibition

The natural next mechanism — an attentional **salience boost** on the foregrounded referent (drive it harder so
it wins) — was de-risked directly (`_phaseB_salience_pointer_derisk.py`, 3 seeds, boosts 1×/2×/4×). **Also
NEGATIVE.** Even a **4× drive boost** on the foregrounded referent does not reliably make it dominate: the
order-control never passes (e.g. seed 42, 4× boost: the boosted `cat` 0.31 still loses to the normal `bird` 0.33).
The decisive observation: **whichever concept has the stronger *intrinsic* attractor (seed-dependent random
pattern) wins regardless of recency OR drive** — `bird` wins on seed 42 whether it is written first or second,
boosted or not. The set-hold is robust to drive asymmetry because the per-concept attractors are **independent**
(no cross-referent coupling), so a boost only adds activity; it does not suppress the competitor.

⇒ The precise requirement is **winner-take-all lateral inhibition between the referent attractors** — *biased
competition* (Desimone & Duncan 1995, *Annu. Rev. Neurosci.*): the attended/salient referent must **suppress** the
others, not merely out-drive them. Biology: attentional selection is competitive (mutual inhibition among the
candidates), exactly the mechanism the plain independent-attractor loop lacks. That is a real WM-wiring build
(install inhibitory cross-connections between concept patterns + a salience signal that biases the competition),
not a parameter tweak — the clear, precisely-specified next mechanism whenever multi-referent dialogue is
prioritized.

## Where this leaves multi-turn dialogue

- **GO:** single-referent anaphora across turns (production `MultiTurnAgent`).
- **Mapped boundary (2 converging NEGATIVEs):** multi-referent disambiguation needs **winner-take-all biased
  competition** (lateral inhibition between referent attractors + a salience bias) — NOT recency (NEGATIVE) and
  NOT a salience boost alone (NEGATIVE). A real, biologically-grounded WM-wiring build, precisely specified.

Two honest negatives that converge on the exact missing mechanism are the deliverable.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_multireferent_disambiguation_derisk --seeds 42 43 44
```

No `sim/` edit. Reuse-by-import: `SpikingLoopContextBuffer`.
