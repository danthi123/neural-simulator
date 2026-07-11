# LEARNED keys make content-addressable retrieval LOAD-BEARING where fixed reservoir-state keys did not — the first POSITIVE de-risk of the rung-2 frontier (learned representations enable content-addressable long-range retrieval), composing the two validated pieces (the e-prop-learned reservoir + the content-addressable read)

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_content_addr_derisk.py` (extended additively with `--learned-keys`: e-prop-train the reservoir's recurrent weights first, then use its LEARNED states as the content-addressable keys). Reuse-by-import of the e-prop `train` + the content read; WikiText, numpy, NO `sim/` edit, NO BPTT.
**Verdict:** the arc's converged conclusion — *long-range needs LEARNED keys/representations* — is now DE-RISKED POSITIVELY at the cheapest level: even a shallow e-prop-learned reservoir makes content-addressable retrieval load-bearing (content beats scrambled-key retrieval), where the FIXED reservoir state (a bad key) did not.

## The test + result
The content-addressable finding (`2026-07-11-content-addressable-...-LEARNED-keys`) showed a FIXED content-addressable read over the reservoir's OWN states is not load-bearing: `content ≈ shuffle` (content-minus-shuffle ≈ −0.015 nats, negligible) — the fading reservoir state is a bad KEY. This de-risk changes ONE thing: e-prop-LEARN the reservoir's recurrent weights first (the validated within-reach credit rule), then use the LEARNED states as the keys. Single variable = fixed-vs-learned keys.

**Result — 3-seed proper scale (n300, 1500 sents), content-minus-SHUFFLE by context depth (neg = content-addressing load-bearing):**
| depth | content − shuffle (LEARNED keys) | content − base | (FIXED-keys content−shuffle) |
|---|---|---|---|
| d3 | −0.114 ± 0.044 | +0.839 | ~−0.014 |
| d4-5 | −0.098 ± 0.061 | +0.408 | ~−0.010 |
| d6-9 | −0.209 ± 0.082 | +0.284 | ~−0.015 |
| d10-99 | **−0.422 ± 0.182** | +0.345 | ~−0.017 |
- With LEARNED keys, `content ≪ shuffle` by −0.10 to −0.42 nats — content-addressable retrieval STRONGLY beats scrambled-key retrieval (~7–25× the negligible FIXED-key margin), robustly across 3 seeds.
- **The margin is BIGGEST at the DEEPEST context (d10-99, −0.422)** — the long-range signal: learned keys enable retrieval that discriminates DEEP context, exactly where the fading state (and fixed keys) totally failed.
- `content − base` stays POSITIVE (the raw soft-retrieved feature is informative-but-noisy → net a small cost over the base read-out); so learned keys make the MECHANISM work, but a CE-win over base needs a cleaner read (next levers below).

## Why this matters (the arc convergence, positively de-risked)
`content ≈ shuffle` with fixed keys said "the keys don't discriminate context." `content ≪ shuffle` with e-prop-learned keys says "**learning the keys makes them discriminate context**" — the retrieval now finds genuinely-relevant past contexts (not random ones). This is the first POSITIVE evidence for the arc's converged conclusion: the long-range lever is LEARNED REPRESENTATIONS via biological deep credit, and even the cheapest learned representation (the e-prop within-reach reservoir, no BPTT) already makes content-addressable retrieval work. It composes the two validated pieces of the arc — the e-prop-learned recurrent credit (REAL-WITH-SCOPE) + the content-addressable read — into the beginnings of a biological "attention" (content-addressable associative-memory read over learned keys).

## The cheap read-integration lever (sharper β) tested — strengthens content-addressing but does NOT close to a CE-win over base
At β=16 (a peakier/cleaner read, seed 42): content−shuffle GROWS even stronger (d3/d4-5/d6-9/d10-99 = −0.14/−0.17/−0.26/**−0.515**) — learned keys + a sharp read make content-addressing maximally load-bearing at deep — BUT content−base stays POSITIVE (+0.31/+0.15/+0.25/+0.33). So the cheap read-integration lever does NOT give a content-beats-base CE-win: the retrieval discriminates deep context strongly (vs shuffle) but the ABSOLUTE deep prediction is still dominated by the base read-out (which already captures the within-reach structure), and the retrieval's unique deep signal is small relative to the noise it adds across depths. ⇒ a full CE-win needs DEEPER KEYS (encoding more distal structure than the within-reach e-prop reservoir) and/or a smarter read (learned gating of when to trust the retrieval; retrieval as a residual correction, not a raw appended feature) — the deep-credit-meets-attention frontier, not a β tweak.

## Honest scope + the next levers
- The content-addressing is now load-bearing (content ≪ shuffle) but the retrieved feature does NOT yet BEAT the base read-out (content − base is still positive — the retrieval is informative-but-noisy, net a small cost). So learned keys make the MECHANISM work; a CE-WIN over the base needs (a) deeper/better key learning (the e-prop keys are only within-reach-trained; a longer-horizon or dendritic-deep-credit representation would encode more distal structure), (b) a cleaner read integration (the raw soft-retrieved V-dim distribution adds noise — a learned value projection / a sharper β / a top-1 read), or (c) scale.
- The keys here are learned by the validated no-BPTT local rule (e-prop); the FULL rung-2 frontier is to learn keys/queries good enough that the content-addressable read BEATS the base read-out at d10+ — the deep-credit-meets-attention frontier, now with a positive foothold.
- 1-seed smoke + 3-seed proper-scale (below); numpy rate-level.

## Files
`_emerge_reservoir_lm_content_addr_derisk.py` (`--learned-keys`); raw `research/findings/raw/_eprop/ca_lk_s*.json`, `ca_learnedkeys_s42.json`. Follows `2026-07-11-content-addressable-retrieval-needs-LEARNED-keys-*` (the fixed-key negative this positively answers) + the e-prop `-REAL-WITH-SCOPE` finding (the keys' learning rule).
