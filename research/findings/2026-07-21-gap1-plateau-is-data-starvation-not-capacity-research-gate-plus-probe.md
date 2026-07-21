# gap#1 broad-domain plateau (ppl ~121) — research gate + probe: it is DATA-STARVATION, not a capacity wall; the data lever was never tested at scale (decisive run in flight)

**2026-07-21.** The gap-close audit flagged gap#1 open-generation as "scale/capacity-bound," and the record's own honest
note: *"whether the d1024 flat is capacity or under-training at 12 epochs is NOT disentangled."* This cycle fired the
research gate (a-1 RAG + external lit) AND ran a probe. The convergent verdict: **the plateau is DATA-STARVATION, and
the data lever was never pulled at a scale that would matter.**

## The research gate (external literature, a-1 RAG)
- **The number IS the story:** the record's own "data-bound at 41M tokens = 0.46 tok/param at 88.6M." **0.46 tok/param
  is ~40× below Chinchilla-optimal (~20 tok/param) and ~300-30,000× below what production small-LMs get.** wikitext
  ppl 121 is a data-starvation signature, not a capacity ceiling — well-trained models in this size class reach
  wikitext ppl ~20-40. (Chinchilla 2203.15556; SmolLM 135M→600B tok ~1,700 tok/param; SmolLM2 135M→2T; "Beyond
  Chinchilla-Optimal" 2401.00448: quality improves to ~10,000 tok/param; single-GPU L20-Edu-135M ≈148 tok/param.)
- **Biggest lever = TOKEN COUNT** (hundreds-to-thousands of tok/param). Then **dedup** (Lee 2022: 10× less memorization
  + fewer steps) + **quality filtering** (FineWeb-Edu/DCLM; SmolLM2 mix 60/40). **Curriculum ORDERING = skip** (BabyLM:
  negligible/negative). **Distillation = use as DATA** (train on teacher-generated text — the TinyStories/phi recipe),
  not a train-time loss.
- **The TinyStories reframe (most relevant):** sub-10M-param models produce fluent coherent prose when the corpus
  DISTRIBUTION is narrowed to the tiny capacity — "broad-domain wikitext at 41M tokens asks a tiny model to cover too
  wide a distribution."
- **ARCHITECTURE finding (from the code read):** the WKV is a **SINGLE recurrent block — NO depth / no layer-stacking**
  (one `emb→LN→{Wk,Wv,Wr,Wo}→one recurrent pass→head`; no `n_layers` arg exists). Diagonal-recurrence / linear-attention
  (SSM `cp_ssm_state = lam·s + (1-lam)·inject`). So depth is a fundamental UNTESTED lever, and a single-layer model may
  saturate on little data regardless (the "capacity" co-limit).
- **OFF the table (already concluded):** attention-over-recurrence (LSTM reaches ~98% of a transformer's long-range
  margin → an O(N²) spiking transformer buys ~nothing); from-scratch spiking BPTT at 25M/50M (overfits, WORSE with
  scale); reservoir/echo-state/e-prop fading-memory gen (loses to a trigram); VSA free-gen (0/16); reservoir SIZE at
  fixed data (overfits). ⇒ NOT more width, NOT attention — the lever is DATA (+ maybe depth).

## The probe (my run this cycle) — within-run OVERFITTING confirms data-starvation (with a confound to disclose)
Ran `_emerge_wkv_lm_derisk.py` d512/wikitext103/v8000, **36 epochs** (vs the record's 12), to test the epoch lever.
Added a per-epoch train-loss print. Train loss fell steeply **5.85 (ep1) → 4.02 (ep12) → 3.10 (ep36)** — clearly still
falling at ep12 (the record's measurement point). BUT the **held-out DEEP (d10-99) NLL = 6.296, vs-trigram −0.347 →
no-go** — WORSE than the recorded 12ep 4.796 (+0.791 GO), and now LOSES to the trigram. Train ↓↓ while held-out ↑ =
**textbook OVERFITTING** = the data-starvation signature (more gradient passes over too-little data memorize the train
set).
- **CONFOUND (disclosed, silent-failure discipline):** the run trained on **n_tr=60000, NOT 150000** — `--max-train-sents`
  **defaults to 60000** (`:329`) and caps the train set (`:385`), so my `--n-sentences 150000` was silently capped. So
  this is 60k-data/36ep, NOT a clean epoch-lever test vs the 150k/12ep baseline (which set the cap higher: its log shows
  `n_tr=150000`). The WITHIN-RUN train-vs-held-out divergence (3.10 vs 6.296) is still a clean overfitting signal on 60k;
  the cross-run comparison to 4.796 is confounded (less data + more epochs).
- The record's **data lever "150k→400k = 4.796→4.798 FLAT" is REAL** (both logs confirm n_tr=150k/400k, not capped) —
  but 150k-400k sentences of wikitext is **~<1 tok/param**, still deep in the starved regime; per the gate, the plateau
  should only break at 20-200+ tok/param (millions of sentences), which the record NEVER TESTED (it stopped at 400k).

## The decisive experiment (IN FLIGHT: `_gap1_datalever_d512_1M_12ep.log`, run bi27mss18)
d512/wikitext103/v8000, **--n-sentences 1000000 --max-train-sents 1000000**, 12 epochs (matched to the baseline epoch
count, so ONLY data changes; ~16× the 60k probe, ~2.5-6× the 150-400k baselines). Compare held-out deep NLL to 4.796.
- **If deep NLL DROPS meaningfully → the data lever is real** (the gate's #1 lever confirmed; the 150k→400k "flat" was
  just both-too-small) → the gap#1 fluency path is MORE DATA (+ dedup/quality), not bigger model / more epochs.
- **If FLAT → the SINGLE-LAYER architecture is the co-limit** → depth (multi-layer, currently absent) is the next lever
  (a code change), with the gate's caution that attention buys little.

## The DATA CEILING (measured this cycle) — reframes the whole path
Counted wikitext103's usable sentences at the runner's len-3-16 filter: **~1.65M sentences ≈ 17.1M tokens = ~1.7
tok/param** for the 9.8M d512 model. ⇒ **even training on the ENTIRE corpus stays ~12-120× below the 20-200 tok/param
the gate says broad-domain fluency needs** — wikitext103 alone CANNOT reach "fluent about anything" at this model size,
regardless of the 1M run's outcome (the 1M run tests whether *more within-corpus data* helps at all — a direction
check, not a fluency run). Two compounding facts: (a) the corpus is small; (b) the **len≤16 filter discards most tokens**
(typical wikitext sentences exceed 16 words → dropped/split), so the effective corpus is a fraction of the 540MB.
**⇒ the local levers, sharpened:** (1) **relax the length filter** (max_len 16→48) + **add corpora** (FineWeb-Edu/DCLM
per the gate) to reach ~5-10 tok/param (helps, still short of fluency); (2) **NARROW the distribution to the model's
capacity** — the gate's TinyStories insight, and the project's OWN in-domain result (d512 TinyStories → ppl ~24,
coherent prose) IS the small model being FLUENT on a matched distribution; "fluent about ANYTHING" is the one that
needs big-model+big-data (the field's scale wall, managed via the 21M scaffold — C1 GO). ⇒ the honest gap#1 map: the
substrate-native LM is FLUENT on a domain matched to its size (TinyStories), data-STARVED on broad-domain (wikitext
1.7 tok/param), and broad-domain fluency is a compute/data-scale arc, NOT a mechanism wall.

## ⇒ DECISIVE RESULT (the data-lever run, 2026-07-21): the plateau is SINGLE-LAYER CAPACITY, NOT data — the WKV SATURATES
`d512/wikitext103/v8000, n_tr=850000, 12ep` (the data lever pulled 5.6× the baseline). Deep (d10-99) held-out NLL:
| n_tr | WKV deep NLL | fair trigram | WKV−trigram margin |
|---|---|---|---|
| 150k (baseline) | 4.796 | 5.587 | **+0.791** |
| 400k | 4.798 | 5.259 | +0.461 |
| **850k (this run)** | **4.811** | **5.106** | **+0.296** |
- **The WKV deep NLL is FLAT (~4.80) across 150k→850k — 5.6× more data does NOT lower it.** Combined with the record's
  WIDTH-flat (d512→d1024 = 4.798→4.813), the plateau is ROBUST to BOTH data and width.
- **The decisive tell: the fair trigram IMPROVES with data (5.587→5.106) while the WKV stays flat → its vs-trigram
  margin SHRINKS (+0.791→+0.296).** A simple count model USES the extra data; the single-layer WKV does NOT. ⇒ the
  WKV's (single-layer) CAPACITY has SATURATED on broad-domain wikitext — it is CAPACITY/ARCHITECTURE-bound, **NOT
  data-starved-that-more-data-fixes** (at this corpus scale) and **NOT under-training** (the 36ep probe overfit).
- **This RESOLVES the record's undisentangled "capacity vs under-training at 12 epochs" question: CAPACITY** (data +
  width both flat; the model saturates while a trigram improves). The reframe corrects this finding's own earlier
  "data-starvation is the whole story" (the gate's literature framing) at THIS operating point: the small SINGLE-LAYER
  model cannot use more data — so the prerequisite lever is **DEPTH** (the architecture has NO layer-stacking; `n_layers`
  never existed), then a fundamentally bigger model + corpus (the field's scale wall). Data helps only once the model
  has the capacity/depth to use it. Both the ceiling (wikitext 1.7 tok/param) AND the saturation (WKV flat while trigram
  improves) point the same way: broad-domain "fluent about anything" is an ARCHITECTURE (depth) + scale arc, not a
  more-wikitext-data lever. The next cheap de-risk = **add depth (multi-layer WKV)** at fixed 150-850k data → does the
  WKV deep NLL drop below ~4.80 (depth breaks the single-layer plateau)?

## ⇒ DEPTH DE-RISK RESULT (2026-07-21) — depth is a REAL but MODEST + SATURATING lever; it does NOT reach fluency
Added multi-layer stacking to the WKV (pre-norm residual blocks, `--n-layers`; the `n_layers=1 → 4.793 ≈ 4.796`
validity gate PASSED, so the refactor is sound and the comparison is clean). Same config (d512/wikitext103/v8000/150k/12ep):
| n_layers | deep (d10-99) NLL | vs-trigram | train loss | overfit? |
|---|---|---|---|---|
| 1 | 4.793 (gate ≈ 4.796) | +0.79 | — | — |
| **2** | **4.738** | **+0.850** | 4.719 | no (train ≈ held-out) |
| 4 | 4.735 | +0.852 | 4.721 | no |
- **Depth 1→2 HELPS ~0.05 — 2-SEED ROBUST** (seed 42: 4.793→4.738 = −0.055; seed 43: 4.869→4.820 = −0.049; real
  generalization — train ≈ held-out, NOT overfitting). **2→4 is DIMINISHING** (seed 42: 4.738→4.735 = −0.003 flat;
  seed 43: 4.820→4.797 = −0.023 small) — so ~0.06-0.07 TOTAL L1→L4, a diminishing-returns lever (big step 1→2, small
  step 2→4), NOT a clean saturation but clearly bounded.
- ⇒ the single-layer plateau is NOT the absolute limit (depth lowers it ~0.06-0.07), but depth is a MODEST,
  DIMINISHING-returns lever that does NOT break the plateau toward fluency (~ppl 20-40 target; L4 is still ppl ~112-114).
  Combined: **data flat (150k→850k) + width flat (d512→d1024) + depth modest-diminishing (~0.06 total, 2-seed)** — the
  plateau is a fundamental SMALL-MODEL + LIMITED-DATA limit. Broad-domain "fluent about ANYTHING" needs the field's
  big-model + big-data SCALE (managed via the 21M scaffold, C1 GO), NOT more wikitext, more width, or a few more layers.
- Honest scope: 2-seed (42, 43); the L1→L2 ~0.05 benefit is robust across both, the L2→L4 diminishing benefit is
  seed-variable (flat/0.023) but small either way; the qualitative conclusion (depth gives a bounded improvement, does
  NOT reach fluency) is robust. The `--n-layers` WKV refactor (gate-verified) is a reusable lever for future scale-up.
  ⇒ **gap#1 investigation COMPLETE: the broad-domain plateau is a scale (model+data) arc, characterized on all three
  axes (data/width/depth); the substrate-native mechanism is GO and fluent on a matched domain (TinyStories ppl 24) —
  "fluent about anything" is the field's scale wall, not a mechanism/architecture gap.**

## Honest status
gap#1's substrate-native mechanism WORKS + generalizes (beats fair count baselines at depth, in- and broad-domain);
open-fluent generation "about anything" is bottlenecked, and this cycle DISENTANGLES the bottleneck: **data-starvation
(literature-decisive) + a single-layer architecture co-limit, NOT model width or training budget on fixed data.** The
1M-sentence run resolves which dominates. This is a scale/data + architecture arc, per THE LAW a characterized frontier
with the next lever named — not a wall. Runner unchanged except the additive per-epoch train-loss print.
