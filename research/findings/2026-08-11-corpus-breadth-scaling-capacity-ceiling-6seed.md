---
type: finding
status: go
date: 2026-08-11
mechanism: corpus-mined SVO breadth swept K=40..320 through the RF-phasor VSA store; the no-confab moat's capacity ceiling located on three synthetic axes
lane: H-memory (conversation integration / grounded-knowledge breadth capacity)
seeds: [42, 43, 44, 100, 101, 102]
follows: research/findings/2026-08-10-INTEGRATION-6-corpus-learned-facts-into-live-chat-6seed.md
artifacts:
  - research/findings/raw/lanes/stageA/corpus_breadth_scaling_capacity_ceiling_6seed.json
---

# Corpus-breadth SCALING → the VSA no-confab moat CAPACITY CEILING (6-seed GO): breadth scales to the FULL corpus, the moat never leaks; the live ceiling is a provisioning cap, not the moat

**2026-08-11.** INTEGRATION #6 gave the live chat corpus-LEARNED breadth (2→9 grounded subjects at K=40, moat 0)
and named the CAPACITY question: how far does breadth scale before the VSA moat margin degrades? This de-risk answers
it. Runner:
[`research/runners/_corpus_breadth_scaling_capacity_ceiling_derisk.py`](../runners/_corpus_breadth_scaling_capacity_ceiling_derisk.py).
Reuse-by-import of #6 (mining, moat battery, provenance, empty-kb control, live chat, byte-identity); NO `sim/` edit;
SIM_BACKEND=numpy; cfg.seed-controlled; query_patient memoised per measurement (a deterministic pure read of the
frozen store → answer-identical, the same rationale #6's chat memo documents).

## Headline (6/6 seeds 42/43/44/100/101/102)

**Breadth scales to the FULL corpus with the no-confab moat intact.** Sweeping K (the top-K mined TinyStories SVO
triples, vocab growing with K), the STANDALONE RF-phasor store's grounded-subject BREADTH rises **9 → 19 → 31 → 38**
across K=40/80/160/320, with **recall 1.000** on every stored (a,v) cue and **0 moat false-accepts** at every K.
This is well above #6's K=40 breadth of 9 → **GO**.

| K | n_facts | \|V\| | breadth | recall | moat_FA | perm_overlap | emptyKB_new | row |
|---|---|---|---|---|---|---|---|---|
| 40 | 46 | 42 | 9 | 1.000 | 0 | 0.07 | 0 | GO |
| 80 | 86 | 54 | 19 | 1.000 | 0 | 0.05 | 0 | GO |
| 160 | 166 | 65 | 31 | 1.000 | 0 | 0.08 | 0 | GO |
| 320 | 253 | 68 | 38 | 1.000 | 0 | 0.09 | 0 | GO |

(Values are the min breadth / min recall / summed moat-FA / max provenance-overlap / summed empty-kb-new over the 6
seeds; all 6 seeds returned identical breadth/recall/moat at each K.) Attribution (`tools.lab.attributable_to`): the
breadth is attributed to the stored FACTS by subtracting the empty-kb SAME-VOCAB control (breadth 2, new-subject
answers 0 at every K) — the vocab growth alone grounds nothing.

## Scoping reality (measured, honest): the corpus is exhausted before any moat leak

Under the shipped noun/verb inventory (`_ANIMALS|NOUNS_EXTRA` = 45 nouns, `VERBS` = 16), the 3.95M-token TinyStories
corpus contains only **247 distinct clean SVO triples over 68 distinct concepts**. So K≥320 **caps** at 247 mined
facts (+6 curated = 253) / \|V\|=68. The corpus cannot reach the "~320-concept single-bridge" figure the
2026-06-04/G.20 mapping flagged — and that figure is anyway a DIFFERENT substrate (a sparse-distributed 5-bridge
ensemble; single-bridge ≈ 64 there). The RF-phasor composer here is a distinct substrate, so its capacity had to be
measured directly (block B).

## The capacity-ceiling instrument (block B) — where the RF-phasor moat WOULD leak (synthetic, seeds 42/43/44)

The store is a **list of INDEPENDENT per-fact composites** (`self.kb`), NOT one superposed memory, so scaling the
NUMBER of facts adds **no inter-fact crosstalk** — a query unbinds ONE 3-bind composite and cleans up against the \|V\|
codebook. Three genuine capacity axes, each swept to a leak or a bound (cleanup accuracy = min over seeds):

- **b1 CODEBOOK axis D × \|V\| {68..8192}.** At the operating **D=128**, per-role cleanup accuracy is **1.0** with
  **0 moat false-accepts** through **\|V\|=8192 concepts** — ~120× the corpus's 68 — and the cleanup margin (true cos −
  best-competitor cos) falls only from **min 0.2287** (\|V\|=68) to **0.1176** (\|V\|=8192), staying far above the 0
  leak threshold (a ~√(ln\|V\|)/√D decay). To give the metric discriminating power (a metric pinned at ceiling is
  uninterpretable), the SAME sweep at a **stress D=32** shows cleanup accuracy FALLING **0.9531 → 0.6667** as \|V\|
  grows 68→8192, with a genuine moat false-accept appearing at \|V\|=2048 — proving the instrument CAN detect a leak,
  so the D=128 hold is a real ceiling, not an always-pass.
- **b2 SUPERPOSITION axis L {2..6}** (role-fillers bundled into ONE composite): cleanup accuracy **1.0 to L=4**, then
  0.9958 (L=5), 0.9826 (L=6) — the within-fact superposition ceiling is **L≈5** (the store()'s own "±1 scheme K=5
  boundary" question; FHRR sits right at it). SVO facts are **L=3**, two binds of headroom.
- **b3 DIMENSION axis D {8..128}** (fixed 3-bind, \|V\|=256): cleanup accuracy climbs **0.2604 (D=8) → 0.5521 → 0.9167
  → 0.9948 → 1.0 (D=128)** — the first row with perfect cleanup + 0 moat false-accepts is **D=128** (min ok D=128,
  ~2× the D=32 knee where cleanup is 0.9167). **moat_FA stays 0 even at D=8** — the moat is CONSERVATIVE: recall/cleanup
  degrades first; a mis-decoded cue still does not match the SPECIFIC untaught queried pair, so false-accepts never
  precede recall loss.

**Located ceiling.** The moat did NOT leak anywhere within the corpus: recall≥0.95 & moat==0 hold to the full corpus
(253 facts / 38 grounded subjects at K=320). The RF-phasor codebook holds **≥8192 concepts** at D=128; the corpus (68)
sits ~120× below it. **The binding wall is not a moat leak — it is query LATENCY O(K·D)** (a resonate over 2·K·D
neurons; ~0.08 s at 40 facts → ~0.7 s at 320), the faithful-but-slow substrate cost the sharded/multi-bridge store
addresses.

## The LIVE-pipeline ceiling is a provisioning cap (onebrain_k_max=32), NOT the moat — and it is the real bottleneck

Tier-1 ran the live mouth-free chat at the scaled K=320 (seed 42, +byte-identity): grounded replies rise **4 → 9**,
confabulated **0**, OOD turns abstain, `_gm_posthoc_verify` drops **100%** of unsupported props — the live loop still
holds, moat intact. But the live grounded count **plateaus at 9** while the standalone store's breadth is 38. Diagnosed
to root cause (NOT a moat leak): the live chat uses the **CoResidentOneBrainComposer** on the merged bridge, whose
substrate slice is provisioned for **`onebrain_k_max=32`** fact-blocks. `OneBrainComposer.store`
(`research/runners/one_brain_composer.py:605`) raises `RuntimeError` on fact #33, and `_store_facts` catches+skips it,
so the co-resident store holds **exactly 32 facts (7 distinct subjects → 9 grounded turns; verified directly:
co_resident_kb_len=32)**. Extra corpus facts are simply not stored → the moat abstains on them (0 confab — the
conservative direction). So there are **two ceilings**: the VSA moat-margin ceiling (≥8192 concepts, far above reach)
and the **live-pipeline provisioning cap `k_max=32`** — and the latter, not the moat, is what bounds live breadth
today. Raising `k_max` re-provisions the rf slice and lifts it, but the enlarged single-bridge store makes each query
markedly slower (the O(K·D) latency wall again) — which is why the honest successor is a **sharded / multi-bridge
store**, not merely a bigger `k_max`.

## Anti-cheats (#6's, preserved; all pass at every K)

1. **Permuted-corpus provenance** — mined-set overlap **0.05–0.09** across seeds/K (all < 0.5): the breadth is
   corpus-ORDER-derived, not a hand list.
2. **Expanded moat battery** — untaught in-vocab cues → **0 false-accepts** at every K, every seed.
3. **Empty-kb same-vocab control + attribution** — with the EXPANDED vocab but 0 corpus facts, breadth stays **2** and
   new-subject answers = **0** at every K; `attributable_to` attributes the breadth to the FACTS, not the vocab.
4. **Capacity sweep** — this de-risk IS #6 anti-cheat 4 generalised to K=320 + the block-B ceiling instrument.
5. **Surface-confab scan** — `_detect_ungrounded` = 0 on every live reply.
6. **Byte-identity** — `build_one_brain(seed)` vs `build_one_brain(seed, vocab=DEFAULT_VOCAB)` bit-identical on the
   substrate threshold hash, num_neurons (25531), composer concept codes, and the full mouth-free transcript (seed 42).

## What is brain-based vs a declared scaffold (per THE LAW + docs/TERMS.md)

**Genuinely brain-based.** Recall and the no-confab moat are RF-VSA brain reads (`query_patient` = a spiking VSA unbind
+ cleanup); the located ceilings are properties of that spiking substrate. **Declared scaffolds** (identical to #6):
host SVO mining (the linguistic-environment boundary), `comp.store` (the composer-as-idealization host VSA write),
`_gm_fact_to_english` (host text interface). Block B's synthetic concepts are a **labelled instrument** to locate the
substrate ceiling — NOT a corpus-breadth claim (provenance does not apply to it).

## Burn-down successors (named, per THE LAW)

The latency wall O(K·D) and the `onebrain_k_max` provisioning cap share one successor: a **sharded / multi-bridge VSA
store** (query cost per shard, breadth across shards) — the same multi-bridge direction the G.20 320-concept ensemble
took. The host mine+store scaffold's successor is the synaptic co-occurrence cortex
(`research/findings/raw/_foundational_curriculum_scaling_scoping.md`). Neither is a moat weakness — the moat holds by
construction across every axis measured here.

## Reproduce

```bash
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m \
    research.runners._corpus_breadth_scaling_capacity_ceiling_derisk \
    --seeds 42,43,44,100,101,102 --Ks 40,80,160,320 --live-K 320 --live-seeds 42 --cap-seeds 42,43,44 \
    --out research/findings/raw/lanes/stageA/corpus_breadth_scaling_capacity_ceiling_6seed.json
```
(Tier-0 breadth/recall/moat is 6-seed — the scaling GO; the ~11-min/seed live chat at K=320 is an at-scale
confirmation on seed 42, on top of #6's 6/6 live result at K=40. The `onebrain_k_max` diagnosis was confirmed by a
direct co-resident build: k_max=32 → 32 facts stored / 9 grounded turns.)
