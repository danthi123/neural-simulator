# The ALIF adaptation-STATE (the reframe's proposed fix, faithfully implemented + gradient-checked) does NOT surpass the fading-state wall on open text (3-seed: alif d10-99 2.693 > plain e-prop 2.681) — CONFIRMING the fading-state conclusion. + ARC SYNTHESIS: every reservoir-substrate long-range mechanism tested this session is NEGATIVE on open text (a diluting average or a cache/bag), so genuine long-range needs a high-capacity content-selective non-fading LEARNABLE memory whose biological analogue is the deepest unbuilt frontier

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_eprop_recurrent_derisk.py` (`alif`/`alif_readonly` modes added by a controller-verified build subagent; `--grad-check`, `--alif-controls`; additive, default-byte-identical, NO `sim/` edit). Numpy, WikiText, 3-seed. Dispatched by the deep-credit-gated-recurrent research gate (ae00ee22e3b9d784d; Salaj-Bellec 2021 eLife 65459 ALIF adaptive-state, Bellec 2020 e-prop).
**Verdict:** the research gate's decisive reframe — that this session's "longer τ fails" tested the WRONG state variable (a leaky-tanh diluting AVERAGE, not an ALIF per-neuron negative-imprint HOLD) — is tested and **REFUTED on open text**: the faithful ALIF adaptation-state does NOT beat plain e-prop at deep context. The fading-state conclusion STANDS.

## The mechanism + the faithfulness guard (given prior naive-implementation burns)
Rate ALIF (Salaj-Bellec): `a_j(t) = ρ_j·a_j(t−1) + (1−ρ_j)·h_j(t−1)` (a per-unit non-fading slow trace of the unit's own activity, ρ_j log-uniform over ~30–300-token windows), `pre_j = (W_rec h + W_in x + b)_j − β·a_j` (an "activity-silent negative imprint"), read-out feature = `concat([h_t, a_t])` (the adaptation carries distal history forward), credited by the **faithful 2-component Bellec ALIF e-prop eligibility** (coupled `eps_a` → `eps_h`), random feedback, NO BPTT. **Mandatory finite-difference GRADIENT CHECK passed EXACTLY** (eps_h max-rel 2.4e-9, eps_a 2.15e-7; full-grad cos→1.0 as spectral→0, textbook e-prop truncation otherwise) — so this is a correctly-derived ALIF e-prop, NOT the naive dual-eligibility that misled me earlier this session.

## Result — 3-seed absolute d10-99 CE (lower = better; A0 plastic is the bar)
| arm | d10-99 CE (3-seed) |
|---|---|
| **A0 plastic (plain e-prop)** | **2.681 ± 0.079** ← best |
| A2 alif_readonly (state read, not credited) | 2.692 ± 0.076 |
| A1 **alif** (adaptation-state + credited) | 2.693 ± 0.076 |
| fixed (echo-state) | 2.712 ± 0.071 |
- **A1 (ALIF) does NOT beat A0 (plain e-prop)** — it is marginally WORSE (2.693 vs 2.681), and A1 ≈ A2 (crediting the adaptation adds ~0 over merely reading it). The 1-seed smoke's anti-cheats confirmed the mechanism: ADAPTATION-SHUFFLE does not collapse a win (there is none), β=0 ≈ A0, and with sentence-matched windows the adaptation actively HURTS. The scalar-per-neuron adaptation is a diluting average, not a specific-item hold — exactly the fading-state finding's mechanism, whether the fade lives in the membrane leak (hetero) or the adaptation variable (ALIF). (Honest caveat, carried: the harness resets state per sentence over ≤16-token WikiText sentences, so a 30–300-token hold barely accumulates; but the fair 8–40-token window was also negative, and the genuine long-window test = cross-sentence, which is separately NEGATIVE below.)

## ⇒ ARC SYNTHESIS — every reservoir-substrate long-range mechanism tested this session is NEGATIVE on open text
| mechanism | result on open text (WikiText) |
|---|---|
| FIXED reservoir + linear read-out | n-gram-level (SCALE CAPSTONE) |
| e-prop-LEARNED recurrent weights | recovers WITHIN-horizon only (REAL-WITH-SCOPE); no d10+ |
| longer-τ / heterogeneous leaky state | FAILS — a leaky state DILUTES distal items |
| **ALIF adaptation-STATE (this, faithful + gradient-checked)** | **FAILS — marginally worse than plain e-prop; a diluting average, not a hold** |
| content-addressable retrieval, fixed keys | `content ≈ shuffle` (bad keys) |
| content-addressable retrieval, LEARNED keys (append) | content-addressing real but does NOT beat base |
| interpolation read | beats base at deep BUT shuffle-invariant (a within-sentence CACHE, corrected) |
| cross-sentence content-addressable retrieval | NEGATIVE — `content ≈ shuffle`, the uniform BAG is best |

**Across EVERY mechanism, the reservoir substrate's "long-range" benefit is either a diluting average or a cache/bag prior — NEVER a specific-item, content-selective, non-fading HOLD.** The binding limit is CAPACITY + SELECTIVITY: a leaky/adaptive recurrent state cannot hold a specific K-way distal token identity on open prose, and a content-addressable retrieval over such states retrieves a bag, not the relevant item. ⇒ genuine long-range language needs a **high-capacity, content-selective, non-fading, LEARNABLE memory** — which is what a transformer's attention is (a differentiable key-value store with learned Q/K/V), and whose BIOLOGICAL analogue (a learned-write, content-addressed, high-capacity spiking store generalized from the project's closed-grammar RUNG-2/D3/theta-gamma WM buffers to open text) is the **deepest unbuilt frontier** — the owner's standing dendritic/deep-credit priority, now with its precise requirement mapped: not a bigger reservoir, not a better credit horizon, not an adaptive state, but a **high-capacity content-selective non-fading learnable store**.

## Honest scope + what this maps (the scientific deliverable)
This is the project's standing standard: honest negatives that map what the substrate can/can't do. The comprehensive result — the reservoir substrate is within-reach-bounded and NO tested long-range mechanism (state-based or retrieval-based) surpasses it on open text — precisely LOCATES the genuine frontier (a high-capacity content-selective non-fading learnable store) and rules OUT the cheaper paths (reservoir scale, credit-horizon, adaptive state, retrieval bolt-on). All rate-level numpy, gradient-checked where load-bearing, multi-seed, anti-cheated, NO `sim/` edit. The next frontier (the biological high-capacity learnable store) is a major build the project has GO pieces of (RUNG-2/D3/theta-gamma WM on spikes) but only over closed grammars — generalizing a LEARNED-WRITE content-addressed high-capacity store to open text is the genuine remaining work.

## Files
`_emerge_reservoir_lm_eprop_recurrent_derisk.py` (`alif` modes + `--grad-check`); raw `research/findings/raw/_eprop/alif3_s*.json`, `_alif_smoke_*.json`, `_alif_byteid_check.json`. Synthesizes the 2026-07-11 arc: SCALE-CAPSTONE, eprop-REAL-WITH-SCOPE, fading-STATE, cross-sentence-NEGATIVE, learned-keys/interp (corrected), onbridge-BDSP.
