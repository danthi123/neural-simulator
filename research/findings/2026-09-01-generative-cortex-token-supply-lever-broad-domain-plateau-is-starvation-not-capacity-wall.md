---
type: finding
status: positive
date: 2026-09-01
mechanism: wkv-cortex generative-scale — the TOKEN-SUPPLY axis at a FIXED SMALL capacity (single variable)
verdict: >
  GO-TOKEN-LEVER, 6 seeds (42/43/44/100/101/102). At a capacity-MATCHED small WKV cortex (d96/V2000,
  ~0.42M active params) on the BROAD corpus (wikitext103), the deep-context (d10-99) held-out NLL DROPS
  MONOTONICALLY on every seed as training-token-supply rises (delta_nll min->max tokens = 0.646 nats,  <!--derived-->
  6/6), is STILL DESCENDING at the top point (~0.11 nats over the last doubling, 3.0->4.5 tok/param =
  2.7x past the record's ~1.7 operating point, 6/6), the WKV BEATS the fair interpolated trigram at
  every point with the margin GROWING with tokens (6/6 — the OPPOSITE of the record's d512 result where
  the trigram overtook), and the two anti-cheats hold throughout (permute-collapse +0.65->+1.66,
  memoryless-collapse positive at every point). ⇒ the broad-domain fluency PLATEAU the record read as
  "single-layer CAPACITY saturated" was, at a capacity-matched model, TOKEN-STARVATION — a SCALING lever,
  not a capacity/architecture wall. This GO is for THE TOKEN LEVER (the plateau is starvation), NOT for
  "arbitrary prose solved": the top-point NLL 3.932 is still ~0.24 nats above a ~ppl20-40 fluency band,  <!--derived-->
  and the model still needs the corpus/distillation lever to push tok/param into the 20-10000 regime.
lane: generative-cortex-scale
seeds: [42, 43, 44, 100, 101, 102]
external: >
  Grounded in the compute-optimal scaling literature (the same external round that grounded the lane on
  2026-07-21). Hoffmann et al. 2022 "Training Compute-Optimal LLMs" (Chinchilla) arXiv:2203.15556 (~20  <!--derived-->
  tok/param optimal; BELOW it a fixed-size model is token-starved and its held-out loss plateaus — the
  regime the record's d512 measurements all sat in at ~1.7 tok/param). Allal et al. SmolLM2
  arXiv:2502.02737 (135M-param models are fluent at thousands of tok/param). "Beyond Chinchilla-Optimal"  <!--derived-->
  arXiv:2401.00448 (quality keeps improving to ~10000 tok/param). Recorded to  <!--derived-->
  the external-search log under research/queue/ (lane generative-cortex-scale).
instrument: >
  research/runners/_gen_cortex_token_supply_scaling_derisk.py — reuses the VALIDATED WKV LM instrument
  (build_and_train_wkv, eval_perdepth, load_stories, fit_interp_trigram from _emerge_wkv_lm_derisk;
  Vocab/fit_bigram, BUCKETS/_bucket from the reservoir runners). NO sim/ edit. Provenance auto-stamped.
artifacts:
  - research/findings/raw/_gen_cortex_token_supply_scaling.json
  - research/findings/raw/_gen_cortex_token_supply_scaling.json.prov.json
  - research/runners/_gen_cortex_token_supply_scaling_derisk.py
  - research/findings/2026-07-21-gap1-plateau-is-data-starvation-not-capacity-research-gate-plus-probe.md
  - research/findings/2026-07-21-gap1-ceiling-wkv-cortex-beats-fair-bigram-3.35x-scale-progressing.md
  - research/findings/2026-07-20-wkv-cortex-biological-learning-CLOSE-local-rule-readout-retires-BPTT.md
runner: research/runners/_gen_cortex_token_supply_scaling_derisk.py
---

# Generative-cortex scale lever: the broad-domain fluency plateau is TOKEN-STARVATION at a capacity-matched model, NOT a capacity/architecture wall — the fluency critical path is training-token SCALE on the biologizable WKV cortex

**Artifact:** `research/findings/raw/_gen_cortex_token_supply_scaling.json` (6-seed token-supply sweep, provenance-stamped).

## 0. Headline (the decision-relevant result)

The project's deepest gap is brain-native ARBITRARY open prose: the brain's own spiking mouth frames only
structured SVO; arbitrary prose still needs the external Qwen-0.5B scaffold (roadmap Wall #7 / R4). The
substrate-native generator is the **WKV cortex** (an RWKV-style diagonal-SSM linear-attention LM; a FIXED
reservoir + a LOCAL-rule read-out reaches BPTT parity, so the mechanism is biologizable —
`2026-07-20-wkv-cortex-biological-learning-CLOSE-local-rule-readout-retires-BPTT.md`). Its broad-domain
fluency plateau had been read as a **capacity** wall. This de-risk isolates the one axis the record never
cleanly pulled and finds the plateau is **TOKEN-STARVATION**: at a capacity-matched small model, more
training tokens keep lowering the broad-domain held-out NLL, monotonically, 6/6 seeds, still descending 2.7x
past the record's operating point. **The forward path to brain-native arbitrary prose is TRAINING-TOKEN
SCALE (matched-quality token supply) on the biologizable WKV cortex, not a bigger architecture** — and the
"~4 orders of PARAMS" framing of Wall #7 over-states the wall at a capacity-matched operating point.

## 1. What the record already banked — and the confound it left open

`2026-07-21-gap1-plateau-is-data-starvation-not-capacity-research-gate-plus-probe.md` characterized the
broad-domain (wikitext) plateau on all three axes at a BIG d512 model (~9.8M params): **data FLAT**
(150k->850k sentences), **width FLAT** (d512->d1024), **depth MODEST** (~0.06 nats, L1->L4), while a fair
trigram IMPROVED with data — read as "single-layer CAPACITY saturated." But EVERY one of those
measurements sat at wikitext's **~1.7 tok/param** (the len<=16 sentence filter discards most of the corpus),
i.e. deep in the **token-starved regime for a 9.8M-param model** (Chinchilla-optimal ~20 tok/param,
arXiv:2203.15556). "Big model flat + trigram improves" is ALSO the exact signature of a model token-starved  <!--derived-->
FOR ITS SIZE. The record's own next-lever note ("relax the length filter / reach 5-10 tok/param") was
**never run**, so the record never separated (a) capacity-saturation from (b) joint token-starvation that a
capacity-MATCHED model would keep escaping. This de-risk runs exactly that missing single-variable test.

## 2. The single-variable de-risk (method + cleanliness)

Hold a SMALL FIXED capacity and sweep ONLY the training TOKEN SUPPLY, on the BROAD corpus. A small capacity
is the point: the same feasible token budget reaches a MUCH higher tok/param than the record ever hit.

- **Fixed:** WKV cortex d_model=96, vocab=2000 (~0.42M active params), n_layers=1, epochs=6, batch=256,
  recurrence=wkv, corpus=wikitext103, contiguous max_len=40 passages (clean token accounting; sequence
  length held constant — no length-confound).
- **Single variable:** training token supply, via **NESTED prefixes** of ONE fixed per-seed train pool:
  max_train_sents in {4000, 8000, 16000, 32000, 48000} passages = {0.16, 0.32, 0.64, 1.28, 1.92}M tokens =
  **{0.38, 0.76, 1.51, 3.02, 4.54} tok/active-param**. The record's 1.7 sits between points 3 and 4; the
  top point is 2.7x past it.
- **Cleanliness (silent-failure discipline), verified in the artifact:**
  - The EVAL set (idx[cut:][:1500]) and the train POOL (idx[:cut]) are identical across all 5 token points
    within a seed; each point trains on the FIRST k of that fixed pool (nested) — the ONLY thing that
    changes is how many tokens the WKV sees. `eval_ids_sha` is a per-seed constant across points.
  - **Vocab held fixed:** built ONCE per seed from the FULL train pool (not per point), so the V=2000
    output classes are identical across the sweep -> NLL is directly comparable. `vocab_sha`/`V` constant
    across points.
  - **Genuine sequence model, not a count table:** at every point the WKV's deep-context advantage must
    survive PERMUTE (shuffle prefix order) and MEMORYLESS (recurrence-off) collapses — the TRUE anti-cheat,
    kept DISTINCT from the separate quality tell of beating the trigram.
  - **The trigram tell (the record's own decisive control):** the fair interpolated trigram is refit on
    EACH point's train prefix, so we report whether the WKV's margin over it GROWS with tokens (WKV uses
    tokens better than counts) or SHRINKS (counts win, WKV saturates — the record's d512 signature).
  - **Overfit disclosure:** final train NLL recorded per point.

Scope: this is a SMALL-capacity probe of the tok/param RESPONSE DIRECTION on the mechanism's BPTT instrument
(the validated-biologizable WKV, 2026-07-20). The transfer claim is the Chinchilla-universal DIRECTION of
the token response at a capacity-matched operating point, not an absolute fluency run.

## 3. Results (6 seeds, wikitext103, d96/V2000, deep d10-99 held-out NLL)

Per-seed WKV deep NLL vs training tokens (nats; every seed monotone-decreasing):

<!--derived-->
| tok/param | 0.38 | 0.76 | 1.51 | 3.02 | 4.54 | delta(min->max) | top slope (3.02->4.54) |
|---|---|---|---|---|---|---|---|
| seed 42  | 4.570 | 4.401 | 4.232 | 4.038 | 3.923 | 0.647 | +0.115 |
| seed 43  | 4.625 | 4.460 | 4.281 | 4.085 | 3.977 | 0.647 | +0.108 |
| seed 44  | 4.571 | 4.392 | 4.210 | 4.016 | 3.909 | 0.663 | +0.108 |
| seed 100 | 4.558 | 4.396 | 4.239 | 4.028 | 3.926 | 0.632 | +0.102 |
| seed 101 | 4.569 | 4.397 | 4.233 | 4.030 | 3.926 | 0.642 | +0.103 |
| seed 102 | 4.578 | 4.408 | 4.246 | 4.045 | 3.932 | 0.646 | +0.112 |  <!--derived-->
| **mean** | **4.578** | **4.409** | **4.240** | **4.040** | **3.932** | **0.646** | **+0.108** |  <!--derived-->

Gate readout (6/6 on every criterion): `uses_tokens` (delta > 0.10) **6/6**; `still_descending_at_top`
(top-slope > 0.02) **6/6**; `uses_context_at_top` (perm & mless collapse) **6/6**; `beats_trigram_at_top`
**6/6**; `margin_grows_with_tokens` **6/6**. The WKV beats the fair trigram at ALL 30 points, and its
margin over the trigram GROWS with tokens (mean margin +0.240 at 0.38 tok/param -> +0.252 at 4.54
tok/param — it stays wide as data grows, never shrinking toward the trigram) — the OPPOSITE of the record's d512 result, where the trigram
improved with data while the WKV stayed flat and the margin collapsed +0.791 -> +0.296. Anti-cheats
strengthen with tokens (perm-collapse +0.65 -> +1.66; memoryless-collapse positive throughout). Runtime
2288 s / 6 seeds, CPU. Verdict field: **GO-TOKEN-LEVER**.

## 4. The verdict — TOKEN-STARVATION, not a capacity/architecture wall (at a capacity-matched model)

The decisive tell is the CONTRAST with the record at matched deep-context instrument. At the record's BIG
d512 model, more wikitext data did NOT lower the WKV NLL (flat ~4.80) while the trigram improved — the
signature that was read as "capacity saturated." At a capacity-MATCHED small d96 model, the SAME broad
corpus shows the OPPOSITE: the WKV NLL falls monotonically with tokens, stays well ahead of a data-hungry
trigram, and is **still descending ~0.11 nats per doubling at 4.54 tok/param** (2.7x past 1.7). A model that
had truly saturated its capacity could not do this. ⇒ **the broad-domain plateau the record hit was JOINT
TOKEN-STARVATION of a big model on a token-poor corpus, not a capacity/architecture wall.** Data helps only
once the model's capacity is matched to the token budget it is trained on (Chinchilla, arXiv:2203.15556);  <!--derived-->
the record's "flat" simply had the ratio inverted (big model, ~1.7 tok/param).

This refines Wall #7. "The generative cortex is ~4 orders too small" is the honest bound for "fluent about
ANYTHING from scratch at LLM scale", but at a **capacity-matched** operating point the binding constraint is
**matched-quality TOKEN SUPPLY**, and the substrate genuinely SCALES with it. The path to brain-native
arbitrary prose is therefore a token/data-scale arc on the (biologizable) WKV cortex, not a
bigger-architecture prerequisite.

## 5. Generation reality-check (low ppl != fluent, so we sampled it — honestly)

Sampled prose from the top-point (4.54 tok/param) seed-42 model shows real grammatical scaffolding but heavy
`<unk>` — e.g. *"in the th century it &lt;unk&gt; the &lt;unk&gt; were &lt;unk&gt; the book or ..."*,
*"he was the first released in the first &lt;unk&gt; of the ..."*, *"it is a &lt;unk&gt; of a &lt;unk&gt;
and &lt;unk&gt; of ..."*. The `<unk>` density is the **V=2000 lexical cap** (most wikitext content words are
out-of-vocabulary at V=2000 -> rendered `<unk>`), NOT incoherence — function-word order and clause structure
are present. So the NLL drop is a genuine language-model improvement, and lexical fluency is bottlenecked by
capacity+vocab (exactly the residual below). This is consistent with the phi/TinyStories principle
(arXiv:2502.02737): a small model is fluent when the DISTRIBUTION (topic AND vocabulary) is matched to its  <!--derived-->
capacity.

## 6. Honest residuals (not swept under the GO)

1. **Absolute fluency NOT reached.** Mean top-point NLL 3.932 is still ~0.24 nats ABOVE a ~ppl20-40 fluency  <!--derived-->
   band (NLL 3.0-3.69). The GO is for the token LEVER (the plateau is starvation), not for arbitrary prose.
2. **tok/param reached 4.54, not Chinchilla's 20.** The descent is unbroken at 4.54; crossing into the
   fluency band needs pushing tok/param into the 20-10000 regime (arXiv:2401.00448), which wikitext CANNOT  <!--derived-->
   supply (~1.7 tok/param even fully used) — hence the matched-corpus / distillation-as-data lever (S7 below).
3. **Small capacity (d96/V2000).** The DIRECTION (token response at a capacity-matched point) is
   Chinchilla-universal, but the exact tok/param needed to reach fluency, and how it trades against capacity
   and vocabulary breadth, is a scaling-SURFACE the single-model sweep does not map. Next rung S7(b).
4. **BPTT instrument.** The base WKV here is BPTT-trained (a ceiling/scaling instrument). The biologization
   is inherited (fixed reservoir + local-rule read-out at BPTT parity, 2026-07-20); the scaling PROPERTY is
   architecture-level, so it transfers, but the local-rule read-out has not itself been re-run at these
   token scales.

## 7. The forward lever this re-anchors (the fluency critical path)

Because the wall is token supply at a capacity-matched model, the concrete, cheap-FIRST path to brain-native
arbitrary prose is a **token/data-scale arc on the biologizable WKV cortex**, in order:

- **S7(a) — MATCHED-QUALITY TOKEN SUPPLY (the biggest lever, arXiv:2203.15556 / 2401.00448):** the binding  <!--derived-->
  constraint is matched-quality tokens, and wikitext caps at ~1.7 tok/param. The proven small-model recipe
  is **distillation-as-DATA** (the TinyStories/phi recipe — train on broad-TOPIC but simple-STYLE
  teacher-generated text, matched to the small capacity), NOT a train-time soft-target loss (that specific
  method is already banked NEGATIVE, `2026-05-16-generator-increment2-distillation-NEGATIVE.md`). Cheap
  first de-risk: verify the deep-NLL descent CONTINUES past ~4.5 tok/param toward 20+ on a matched
  broad-but-simple corpus at fixed small capacity (single variable = corpus quality/volume).
- **S7(b) — the capacity x tokens x vocab SURFACE:** repeat this token sweep at 2-3 capacities (and a larger
  vocab) to map the iso-tok/param ray and locate the (capacity, tokens, vocab) that first crosses the
  fluency band — this quantifies precisely how far from arbitrary-prose the substrate is, replacing the
  order-of-magnitude "~4 orders" with a measured target.
- **S7(c) — carry it to the biological read-out:** re-run the winning (capacity, tokens) on the fixed
  reservoir + LOCAL-rule read-out (2026-07-20) to confirm the token-scaling benefit survives on the
  brain-native learning rule, then toward the on-substrate realization.

**NO-DEFER note:** this is a verdict on the "bigger-architecture-first" METHOD (banked: not the near-term
lever), and it hands the next method (matched-quality token scale on the biologizable cortex). The
capability — brain-native arbitrary prose — is not deferred; its critical path is re-anchored on the axis
that actually moves the substrate.
