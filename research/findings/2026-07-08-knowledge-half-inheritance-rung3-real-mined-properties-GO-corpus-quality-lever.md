# The KNOWLEDGE half of breadth, rung 3 (GO, 6-seed): the inheritance grounds in REAL corpus-MINED properties (not a synthetic target) — a held-out member inherits its category's mined property, stably across seeds. HONEST caveat: on TinyStories the mined properties are SHALLOW distributional words ('said'/'little'/'big'), not taxonomic facts — property QUALITY is a corpus-content lever. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_inheritance_rung3_mined_properties_derisk.py` (reuse-by-import: the breadth discovery + rung-1's associative-memory inheritance read; a co-occurrence property-miner). numpy-only, offline. NO `sim/` edit.
**Verdict:** GO (the mechanism grounds in real mined properties) with an honest corpus-quality caveat (TinyStories yields shallow properties; a factual corpus is the lever for taxonomic properties).

## Why this ran (closing the "synthetic property" scope note)
Rungs 1–2 taught a SYNTHETIC per-category property target (a random vector). This rung mines the property from the corpus: for each discovered category, the most category-DISCRIMINATIVE co-occurring content word (the word its members co-occur with distinctively vs other categories's members). A held-out member must inherit ITS category's mined-real property; label-derangement must collapse it.

## The result — 6-seed (42/43/44/100/101/102), TinyStories, K=256
- **held-out inherit-acc 0.756** (chance 0.250) | **label-deranged 0.244** (≈ chance, all seeds) → GO: the inheritance grounds in the real mined property, and the genuine discovered grouping is load-bearing (derangement collapses).
- **The mined properties are STABLE across all 6 seeds:** animals→'said', family→'little', actions→'hey', places→'big'.

## The honest caveat (property QUALITY is a corpus lever — the load-bearing read)
The mined properties are SHALLOW: the discriminative scores are tiny (+0.015 to +0.050), and the words ('said', 'little', 'big', 'hey') are children's-story distributional artifacts, NOT taxonomic facts ("animals breathe/eat/move"). TinyStories is a simple, homogeneous children's-story corpus — its categories do not co-occur with clean property predicates, so the "most discriminative word" is a weak surface signal (dialogue words near family/animals, size adjectives near places).
- **What this establishes:** the inheritance MECHANISM is corpus-agnostic — it grounds in whatever category-discriminative signal the corpus provides, stably and with the derangement control collapsing. The property is now a REAL mined fact, not a synthetic target (the rung's goal).
- **What it does NOT establish:** that TinyStories yields real taxonomic property knowledge. It does not — the corpus is too shallow. Real taxonomic property inheritance ("a robin inherits that birds fly") needs a FACTUAL corpus (WikiText / an encyclopedic or knowledge corpus) or explicit property statements. This exactly parallels the breadth conclusion: the mechanism scales/grounds; the DATA (corpus content) is the lever.

## What this establishes + the next lever
Combined with rungs 1–2: the emergent cortex discovers a broad vocab from a real corpus, learns its category structure (matches the batch ceiling to 1024), and reasons (inherits properties) over it — on spikes (rung 2) — grounded in real corpus-mined properties (rung 3, mechanism). The next levers: (a) a FACTUAL corpus (WikiText/knowledge) for real taxonomic property quality (a data fetch, mirroring the breadth corpus fetch); (b) wire the inherited answer into a conversational turn (EMERGE-59..73 speaks it on spikes); (c) population-coded read-out to lift the rung-2 absolute spiking accuracy.

## Files
`research/runners/_realcorpus_inheritance_rung3_mined_properties_derisk.py`; 6-seed `research/findings/raw/_rc_inherit_rung3.json`. Prior: rung 1 `...-rung1-GO.md`, rung 2 (spiking) `...-ON-SPIKES-real-corpus-rung2-GO.md`, breadth `...-open-domain-breadth-is-a-data-scale-lever-...md`.
