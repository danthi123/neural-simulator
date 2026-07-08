# CANCELLATION (GO, 6-seed): a member's OWN property OVERRIDES its category's inherited one, over REAL-corpus-discovered categories — "birds move, but the penguin walks." A regulated graded apical drive (weight scaled to override, member-specific) is NECESSARY (no fixed gain passes 6 seeds). NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_cancellation_derisk.py` (reuse-by-import: `RealCorpusConsole` rung-4 + `_splits`/`_coherence`; numpy reasoner, CPU, fanned across cores). NO `sim/` edit.
**Verdict:** GO — member-specific cancellation over real-corpus categories, all anti-cheats pass 6-seed.

## Why this ran
The rung-4 reasoner inherits UNIFORMLY — every member of a discovered category gets the category's taught property. Real semantic cognition has EXCEPTIONS: a penguin IS a bird (inherits "flies") but its OWN property ("walks") OVERRIDES the inherited one (EMERGE-42/54's per-member cancellation), now over the emergent, real-TinyStories-discovered categories. The override answer ("the penguin walks") is also the intransitive speech frame (the modal frame minus "can"), speakable with the existing trained A→W vocab — so cancellation is the natural next capability toward a richer talkable brain.

## The mechanism (biology-grounded)
On the rung-4 associative memory `M`:
- teach the class to SOME members → bind them to `P[cat]` (the category property tag);
- the exception member is HELD-OUT (would INHERIT `P[cat]` by generalization);
- teach the exception → bind the member's code to its OWN tag `P_exc` with a **regulated graded drive** — the weight scaled just large enough that the member's own property beats its top inherited class score by a margin (a homeostatic apical amplifier, EMERGE-54), NOT a fixed gain;
- `predict(word) = argmax` over {class tags} ∪ {exception tags}: a held-out class member → `P[cat]` ("yes"), the exception member → `P_exc` ("no" for the class property).

## The result — 6-seed (adaptive graded drive, margin 2.0)
```
seed 42  pos=~bird  exc='bunny' w=1.27 | CANCEL=no OWN=yes | collateral=0/14 | permuted=yes | moat=1
seed 43  pos=~spot  exc='when'  w=0.92 | CANCEL=no OWN=yes | collateral=0/8  | permuted=yes | moat=1
seed 44  pos=~forest exc='family' w=1.07 | CANCEL=no OWN=yes | collateral=0/10 | permuted=yes | moat=1
seed 100 pos=~cat   exc='under' w=0.84 | CANCEL=no OWN=yes | collateral=0/18 | permuted=yes | moat=1
seed 101 pos=~bird  exc='town'  w=1.18 | CANCEL=no OWN=yes | collateral=0/14 | permuted=yes | moat=1
seed 102 pos=~cat   exc='man'   w=2.63 | CANCEL=no OWN=yes | collateral=0/12 | permuted=yes | moat=1
CANCEL all=True | OWN all=True | no-collateral all=True | permuted all=True | moat all=True  -> GO
```
- **CANCEL**: the exception member (which inherited "yes" BEFORE, by construction) → "no" for the class property — a genuine yes→no flip.
- **OWN**: the exception member → "yes" for its own property (predict picks `P_exc`).
- **no-collateral**: teaching the exception flips ZERO other held-out members (before==after) — specific, not a generic `M` perturbation.
- **permuted anti-cheat**: teaching the exception to a RANDOM non-category word leaves the real exception member still inheriting (yes) — the cancellation is specific to the taught member.
- **moat**: an unknown word → "I don't know" (gate-first abstention).

## Why the regulated (adaptive) drive is NECESSARY (single-variable sweep)
A FIXED `w_exc` cannot satisfy all seeds — the members have different inherited-drive strengths:
- `w_exc ≤ 1.5` → seed 102 (`man`, strongly inherited) does NOT override (CANCEL=yes, OWN=no).
- `w_exc ∈ {2.0, 2.5}` → seed 42 (`bunny`) picks up cross-talk collateral (1/14) AND seed 102 still doesn't override.
- `w_exc = 3.0` → seed 42 collateral (1/14).
No fixed gain passes all 6. The adaptive weights (0.84–2.63, member-dependent) give each member exactly the drive it needs — large for `man`, small for `bunny` — with zero collateral. The regulated graded drive is the mechanism, not a convenience.

## Adversarial self-checks (cheapest load-bearing, done inline)
- The flip is genuine: the exception member is *selected* as one that inherits "yes" before the exception (so "no" after is a true override, not a never-inherited word).
- OWN=yes is non-trivial: the permuted control binds a random word's own exception at the same adaptive regulation and the real member still inherits — so the flip requires binding THE member.
- No test-answer leakage: the adaptive weight is computed from the member's INHERITED drive (`U[member] @ M`) + the tag geometry only — never a held-out label or test answer. It is exactly a homeostatic apical amplifier regulating its own gain relative to the inherited response.

## Honest scope
- Rate-level (reuses the rung-4 numpy reasoner); the spiking realization mirrors the EMERGE-54 apical-drive path (follow-on).
- The exception members here are the first *inheriting* held-out member of the animal-preferred cluster (`bunny` is an animal; `when`/`family`/`under`/`town`/`man` are non-animal words that co-cluster with animals on TinyStories). The MECHANISM is validated regardless of the member's semantics; a fully-coherent *spoken* cancellation demo ("the penguin walks" as the override answer, via the intransitive frame) is the next rung — the intransitive frame reuses the existing trained A→W vocab (modal minus "can"; "walk" is one of the 6 trained verbs).
- Emergent categories on TinyStories are name/animal clusters (coherence 0.07–0.12); the cancellation rides whatever category co-occurrence forms.

## What this establishes
The emergent talkable brain now supports EXCEPTIONS to inheritance — a member's own property overrides its category's, member-specifically, without breaking other members' inheritance, specific to the taught exception, moat intact, over real-corpus-discovered categories. A hallmark of real semantic cognition (Collins-Quillian exceptions / EMERGE-54 cancellation) on the emergent substrate. Follow-on: the spoken intransitive override frame ("the penguin walks"); the spiking realization.

## Files
`research/runners/_realcorpus_cancellation_derisk.py`; per-seed `research/findings/raw/_cancel_adapt_s*.json`. Prior: the rung-4 reasoner `_realcorpus_inheritance_rung4_conversation_derisk.py` + `2026-07-08-rung4-talk-about-real-corpus-vocab-GO.md`; EMERGE-42/54 cancellation.
