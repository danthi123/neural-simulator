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

## Honest status
gap#1's substrate-native mechanism WORKS + generalizes (beats fair count baselines at depth, in- and broad-domain);
open-fluent generation "about anything" is bottlenecked, and this cycle DISENTANGLES the bottleneck: **data-starvation
(literature-decisive) + a single-layer architecture co-limit, NOT model width or training budget on fixed data.** The
1M-sentence run resolves which dominates. This is a scale/data + architecture arc, per THE LAW a characterized frontier
with the next lever named — not a wall. Runner unchanged except the additive per-epoch train-loss print.
