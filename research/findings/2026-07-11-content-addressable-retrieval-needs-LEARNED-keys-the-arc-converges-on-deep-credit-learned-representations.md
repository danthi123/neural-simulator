# A fixed content-addressable read over the reservoir's own states does NOT capture long-range structure (content ≈ shuffle ≈ random retrieval) — the fading reservoir state is a BAD KEY. Content-addressable retrieval is the right mechanism CLASS but needs LEARNED keys/queries ⇒ the whole reservoir-LM arc converges, from first principles, on the owner's standing priority: LEARNED REPRESENTATIONS via biological DEEP CREDIT

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_content_addr_derisk.py` (reuse-by-import: the reservoir + corpus loader + context-depth buckets + the validated `train_readout`/`_standardize_fit`). Numpy, WikiText, NO `sim/` edit, NO BPTT. Dispatched by the state-memory research gate's A2 recommendation (a non-fading content-addressable read = modern-Hopfield/attention read, the biological CA3-completion/FHRR-cleanup class).
**Verdict:** the content-addressable long-range lever, in its cheapest FIXED form, is NEGATIVE — and the reason is decisive and points the arc at its fundamental conclusion.

## The de-risk + result
Per token t, append a content-addressable retrieval to the FIXED reservoir read-out: keys = past reservoir states h_τ, query = current state h_t, attn = softmax(β·⟨h_t,h_τ⟩), value = onehot(token that followed the retrieved context). NON-fading (any past τ reachable) + content-addressed. Arms (same reservoir + read-out; only the appended feature differs): `base` (state only), `content` (+ the read), `shuffle` (+ the read but the keys' context-identity is scrambled = the content-addressing anti-cheat).

**Result (n300, 1500 sents WikiText, 6-seed):** the retrieved feature does NOT help and the content-addressing is NOT load-bearing:
| depth | content − base (neg=retrieval helps) | content − SHUFFLE (neg=addressing helps) |
|---|---|---|
| d3 | +1.026 ± 0.15 (HURTS) | **−0.014 ± 0.005** |
| d4-5 | +0.704 ± 0.04 (HURTS) | **−0.010 ± 0.007** |
| d6-9 | +0.492 ± 0.02 (HURTS) | **−0.015 ± 0.005** |
| d10-99 | +0.379 ± 0.06 (HURTS) | **−0.017 ± 0.003** |
- **`content − shuffle` ≈ −0.015 nats at every deep bucket** — content-addressing over the fixed reservoir states is a negligible *whisper* better than SHUFFLED (random) keys, and that whisper is swamped by the +0.4-1.0 nats that the retrieved V-dim feature *hurts* overall (noise the read-out over-parameterizes). ⇒ the content-addressing is NOT load-bearing: the keys (fading reservoir states) do not discriminate contexts.

## Why (the decisive isolation): the fading reservoir state is a BAD KEY
`content ≈ shuffle` is the load-bearing result: if the reservoir states were meaningful keys, retrieving by their similarity would beat retrieving by scrambled keys. They are equal ⇒ **the keys do not discriminate linguistic contexts** — the fading reservoir state is a recency-blurred diffuse vector, so "similar state" ≠ "similar syntactic/semantic context," and content-addressing over it retrieves garbage. (The retrieved feature also *hurts* overall — V extra noisy dims the read-out over-parameterizes — but that is secondary; the load-bearing point is content ≈ shuffle.)

To retrieve a *distal* token by relevance (e.g. the subject 15 tokens back for agreement), the QUERY must encode "I need the subject" and the KEYS "I am a subject" — i.e. **learned syntactic/semantic representations**. That is exactly what a transformer's learned Q/K/V projections are; a fixed reservoir provides none of it. So content-addressable long-range retrieval is the right mechanism CLASS but is **gated on LEARNED keys/queries**, not obtainable as a fixed bolt-on.

## ⇒ THE ARC CONVERGES (the honest, first-principles conclusion)
The reservoir-LM boundary-surpassing chain, run to ground this session, converges on ONE lever:
1. **Fixed reservoir + linear read-out = n-gram-level on real text** (SCALE CAPSTONE) — its bigram edge is a small-data/small-reservoir artifact that vanishes as either axis scales.
2. **Learning the recurrent weights (random-feedback e-prop, no BPTT = rate-analogue of on-bridge BDSP) = GENUINE but MODEST, within-eligibility-horizon credit** (REAL-WITH-SCOPE, adversarially verified) — a first, real step of *learning* representations.
3. **The long-range (d10+) wall is the FADING STATE, not the credit horizon** — extending the credit horizon amplifies within-reach only; extending the state memory (longer τ) FAILS because a leaky state dilutes distal items (STATE-MEMORY FORK).
4. **A non-fading content-addressable store is the right class, but a FIXED read over reservoir states fails (bad keys)** — it needs LEARNED keys (THIS finding).

**⇒ The fundamental lever for long-range language is LEARNED REPRESENTATIONS via biological DEEP CREDIT** — the owner's standing dendritic/deep-credit priority, now independently re-derived from the reservoir-LM investigation. A reservoir (fixed, or shallow-learned-recurrent) is bounded because it cannot LEARN the representations (keys/queries, hierarchical structure) that long-range language needs. The validated e-prop within-reach credit is the FIRST step of that learned-representation direction; the full deep-credit substrate — learning the whole representation, including the keys for a content-addressable/attention-like read — is the genuine frontier (and it is the owner's standing priority, not a wall).

## The two concrete next rungs (both keep one-brain, on-substrate, no-BPTT)
1. **Learned keys for the content-addressable read** (the direct fix): replace the fixed reservoir-state keys with keys LEARNED by the local deep-credit rule (e-prop/BDSP through the retrieval), or compose the read with the project's GO non-fading stores that ALREADY hold specific items (RUNG-2 latch, EMERGE-85/86 theta-gamma buffer, D3 register) under a LEARNED write policy — the "single biological attention head" generalized from the closed grammars to real text. This is the deep-credit-meets-content-addressable frontier.
2. **The spiking realization of the validated within-reach credit** (independent, warranted-tempered): on-bridge BDSP on the reservoir's recurrent synapses (`enable_bdsp` + `bdsp_apical_couples_soma` + population-K; apical = read-out error, no weight transport) — bring the one genuinely-validated piece onto the real spiking substrate.

## Honest scope
1-seed-proper + smoke for the FIXED content read (6-seed confirmation appended); the load-bearing `content ≈ shuffle` signal is consistent across both. This de-risks that a FIXED reservoir-state key fails; it does NOT prove content-addressable retrieval can't help with LEARNED keys — it POINTS there. Numpy rate-level throughout.

## Files
`_emerge_reservoir_lm_content_addr_derisk.py`; raw `research/findings/raw/_eprop/ca6_s*.json`, `caproper_s42.json`, `casmoke_s42.json`. Caps the arc: `2026-07-11-SCALE-CAPSTONE-*`, `-eprop-recurrent-learning-*-REAL-WITH-SCOPE`, `-the-long-range-wall-is-the-fading-STATE-*`.
