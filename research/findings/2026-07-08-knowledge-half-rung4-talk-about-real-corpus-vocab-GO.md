# The KNOWLEDGE half of breadth, rung 4 (GO, 6-seed): TALK to the brain about a broad, REAL-corpus-discovered vocab — discover categories from TinyStories, teach a property in plain terms, and the brain answers a yes/no question about a HELD-OUT real word by inheritance, rejects other-category words, and abstains on unknown words (no-confab moat). The mission payoff of the breadth→knowledge arc. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_inheritance_rung4_conversation_derisk.py` (reuse-by-import: the breadth discovery + rung-1's multi-category argmax inheritance read). numpy-only, offline. NO `sim/` edit.
**Verdict:** GO at K=1024 (broad, 8 categories), 6-seed. The conversational payoff — talk to the brain about a broad real-corpus vocab — demonstrated.

## Why this ran (the mission payoff)
Rung 3 established that mining properties from a shallow corpus yields shallow properties — the design truth (how EMERGE's consoles work) is that the corpus supplies the VOCAB + CATEGORIES (the breadth mechanism) while the FACTS are TAUGHT explicitly. This rung ties them into the conversational payoff: discover a broad vocab from a real corpus → teach a property in plain terms → answer a natural yes/no question about a held-out real word, with the no-confab moat.

## The demonstrated conversation (K=256, the coherent animals category, seed 42)
```
TAUGHT (explicit facts): animals <- [cat, mouse, bird] ; family <- [friend, girl]
Q: does a 'dog'  have the animals property? (animals, HELD-OUT) -> YES   (inherited; dog never taught)
Q: does a 'fish' have the animals property? (animals, HELD-OUT) -> YES   (inherited)
Q: does a 'bear' have the animals property? (animals, HELD-OUT) -> YES   (inherited)
Q: does a 'mom'  have the animals property? (family)             -> NO    (rejected; mom is family)
Q: does a 'dad'  have the animals property? (family)             -> NO    (rejected)
Q: does a 'zzzqqx' have the animals property? (unknown)          -> IDK   [the no-confab moat]
```
The brain discovered the animals category from TinyStories co-occurrence, was told a property about SOME animals, and correctly generalizes it to held-out animals, denies it for family members, and abstains on a word it never discovered.

## The result — 6-seed (42/43/44/100/101/102), TinyStories
| scale | INHERIT-yes (held-out → yes) | REJECT-no (other-cat → no) | MOAT (unknown → IDK) | deranged-yes | verdict |
|---|---|---|---|---|---|
| **K=1024** (8 cats, pos=colors coh 0.152) | **0.958** | **0.972** | **1.000** | 0.292 (collapses) | **GO** |
| K=256 (4 cats, pos=animals coh 0.215) | 1.000 | 0.700 | 1.000 | 0.250 (collapses) | partial |

**GO at K=1024:** every seed passes all gates — held-out inheritance (0.958), other-category rejection (0.972), the moat (1.000 by construction — unknown words have no discovered code to reason over), and the label-derangement control collapses to chance (0.292). The gate-first moat means an unknown word is answered "I don't know" without invoking the reasoner.

## Honest scope (two dependencies mapped)
- **Category count → rejection sharpness.** At K=256 (only 4 categories) INHERIT-yes + MOAT are perfect but REJECT-no is weaker (0.700) — the argmax "does this word belong to the taught category?" discrimination is coarser with few alternatives. At K=1024 (8 categories) rejection is clean (0.972). More discovered categories → sharper rejection.
- **Category coherence → inheritance quality.** Coherent noun-like categories (animals coh 0.215, colors coh 0.152) inherit cleanly. Diffuse verb-like categories (actions/verbs — the largest raw category at K=1024) inherit weakly (~0.25), because verbs share contexts broadly and so their bag-of-words co-occurrence codes do not form a tight category. This is an honest boundary + the next mechanism: verb/relational categories need an argument-structure / syntactic signal, not bag-of-words co-occurrence (EMERGE-72/74's construction-mined argument structure is the project's lever there).
- Rate-level read (reuse rung-1); the spiking realization is rung-2 (GO). The property is a taught tag (the mechanism); the point is the FACT is taught explicitly + the VOCAB/CATEGORY is discovered from the real corpus.

## Two bugs fixed by reading the substance (the discipline)
(1) A single-property associative memory made `p̂` a scalar multiple of the property vector → cosine saturates to +1 → everything answered "yes" (reject 0.000). Fixed to the rung-1 multi-category argmax ("does the word's recalled property argmax to the taught category?"). (2) "Pick the largest category" chose the diffuse verb category at K=1024 → weak inheritance; fixed to "pick the most-coherent category" (the one co-occurrence actually forms), with per-category coherence reported so the dependence is legible.

## What this establishes — the arc landed
The full breadth→knowledge arc, on a real corpus, transformer-free, moat intact: **discover a broad vocab from real experience (structure matches the batch ceiling to 1024) → learn its category structure → teach a fact in plain terms → answer a yes/no question about a held-out real word by inheritance → abstain on the unknown.** Next: wire this to the spiking speaker (EMERGE-59..73 renders the yes/no answer on spikes); the argument-structure signal for verb/relational categories; a factual corpus for richer taught-vs-discovered facts; population-coded read-out to lift the rung-2 absolute spiking accuracy.

## Files
`research/runners/_realcorpus_inheritance_rung4_conversation_derisk.py`; 6-seed `research/findings/raw/_rc_inherit_rung4_K{256,1024}.json`. Prior rungs: 1 (rate) `...-rung1-GO.md`, 2 (spiking) `...-ON-SPIKES-...-rung2-GO.md`, 3 (mined properties) `...-rung3-...-corpus-quality-lever.md`; breadth `...-open-domain-breadth-is-a-data-scale-lever-...md`.
