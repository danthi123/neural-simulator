# PRODUCTIVE regular inflection ON SPIKES (GO): a NOVEL 3sg verb form whose whole-form lexeme was stored NOWHERE is composed NEURALLY as spell(STEM) + spell(AFFIX) — both decoded from `language_output` spikes — replacing the host `emerge_v3` string op. Pinker-Ullman procedural route (Broca + basal ganglia compose a bound-morpheme affix onto a lexically-retrieved stem). NO `sim/` edit.

**Date:** 2026-07-08
**Runners:** `research/runners/_realcorpus_train_affix_pool.py` (the bound-morpheme A→W {s,es,ed,ing,ies}); `_realcorpus_productive_inflection_derisk.py` (the de-risk). numpy. NO `sim/` edit.
**Verdict:** GO — productive regular 3sg inflection composed on spikes, genuinely productive (3sg stored nowhere), controls collapse.

## Why this ran (the honest residual, research-gated)
The relational answer generation ("the dog eats the cat", CYCLE 1023) spelled every word on spikes but the 3sg SURFACE was produced by a HOST function `emerge_v3(base)` (append -s/-es), then the whole surface was spelled by a STORED A→W lexeme. That works for FREQUENT stored 3sg forms (Pinker's words-and-rules: high-frequency regulars are lexically stored) but for a NOVEL verb there is no stored 3sg pool to spell — the productive -s inflection is a host string op. A read-only deep-research gate (`2026-07-08`, Kandel 6e Ch 55 + Pinker-Ullman) reframed it: the PROCEDURAL system (Broca + basal ganglia) composes a productive inflection by concatenating a bound-morpheme AFFIX onto a lexically-retrieved STEM. RANK-1 cheap-first mechanism: the affix is its own spellable A→W pool → a productive 3sg = spell(stem) + spell("-s") on spikes.

## What was built
- **Affix A→W** (`_realcorpus_train_affix_pool.py`): a micro concept-pool bridge that spells the English regular bound morphemes {s, es, ed, ing, ies} as spellable units (drive the affix pool → decode the morpheme from `language_output` firing). Trained on GPU (5 morphemes).
- **The de-risk**: for verbs whose STEM is spellable (BRIDGE-1: run/jump/walk/sleep/play) but whose 3sg surface (runs/jumps/walks/sleeps/plays) is a stored lexeme in NO A→W bridge, the 3sg is produced as spell(stem) + spell("s"), concatenated on the surface (the same class of surface-assembly as the already-on-spikes slot-ordering).

## The result — seed 42 (affix_seed 42)
```
PRODUCE 3sg(run)   -> spell('run')  + spell('s') = "runs"    [exact, stem+affix ON SPIKES, never stored]
PRODUCE 3sg(jump)  -> spell('jump') + spell('s') = "jumps"   [exact]
PRODUCE 3sg(walk)  -> spell('walk') + spell('s') = "walks"   [exact]
PRODUCE 3sg(sleep) -> spell('sleep')+ spell('s') = "sleeps"  [exact]
PRODUCE 3sg(play)  -> spell('play') + spell('s') = "plays"   [exact]
VERDICT: GO (5/5 exact)
```
Anti-cheats all pass:
- **GENUINELY-PRODUCTIVE**: every 3sg form (runs/jumps/walks/sleeps/plays) is a stored lexeme in NO bridge (BRIDGE-1/2/3 + affix) — so producing it genuinely requires neural affixation, not a lookup.
- **WRONG-AFFIX**: none of stem + "-ed" matches the 3sg (the affix identity is load-bearing).
- **AFFIX-ABLATION**: the bare stem ("run") ≠ the 3sg surface ("runs") (the affix slot is load-bearing).

## Multi-seed (the new seed-dependent component) — 3/3 GO
The stem spelling uses the already-validated seed-42 BRIDGE-1; the one NEW seed-dependent component is the affix A→W decode. Validated at affix seeds 42/43/44 (`--affix-seed`): **3/3 GO** — all 5 productive 3sg forms (runs/jumps/walks/sleeps/plays) render exact on spikes at every affix seed, controls collapse. (Expected: the concept-pool A→W arch is extensively multi-seed-validated — v14 16-pool 5-seed etc.) CI guard `tests/test_productive_inflection.py` (1 passed 189s).

## What this establishes
Productive regular inflection is composed ON SPIKES (spell(stem) + spell(affix)), so a novel/unstored 3sg is produced neurally — the host `emerge_v3` string op has a biology-grounded spiking replacement (the Pinker-Ullman procedural route). The stored/lexical route (BRIDGE-3 whole-form 3sg) remains valid for frequent forms; this adds the PRODUCTIVE route for novel ones. Honest residual: the AFFIX CHOICE (allomorphy: -s vs -es vs -ies, the morphophonological rule) is still host-selected here (the de-risk uses the default -s); the phonological-conditioning network is the deeper RANK-3 mechanism. Follow-on: wire neural affixation into the relational producer's VERB slot (spell(stem)+spell(affix) instead of the emerge_v3 surface) for novel verbs.

## Files
`research/runners/_realcorpus_train_affix_pool.py`, `_realcorpus_productive_inflection_derisk.py`. Research gate: the productive-inflection deep-research pass (Kandel 6e Ch 55, Pinker-Ullman declarative/procedural). Prior: the relational generation `2026-07-08-fully-spiking-relational-transitive-answer-generation-GO.md` (where emerge_v3 was the named residual).
