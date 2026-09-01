---
type: finding
status: draft
date: 2026-09-01
mechanism: wkv-cortex generative-scale — S7(a) matched-quality-corpus token-supply reaching the fluency band (TinyStories)
verdict: >
  RESULTS PENDING. This is a METHOD/DESIGN doc + a single-seed pipeline smoke, NOT a scored result — the
  scored 6-seed pool sweep is queued (research/queue/pool.queue) and has not returned. Do not cite a
  GO/NO-GO/PASS/FAIL verdict from this document; it carries none. Update this doc (or supersede it with a
  results doc that cites it) once the pool artifact lands.
lane: generative-cortex-scale
seeds: [42, 43, 44, 100, 101, 102]
seeds_status: queued, not yet executed (single-seed 42 pipeline SMOKE only has run so far, at toy scale)
external: >
  Same external round the parent finding grounded (2026-09-01, c71cc7c9): Hoffmann et al. 2022 "Training
  Compute-Optimal LLMs" (Chinchilla) arXiv:2203.15556 (~20 tok/param optimal; below it a fixed-size model is
  token-starved). Allal et al. SmolLM2 arXiv:2502.02737 and the TinyStories/phi small-model-fluency recipe
  it documents (a small model is fluent when the DISTRIBUTION — topic AND vocabulary — is matched to its
  capacity) is why TinyStories specifically is the matched-quality corpus tried here. "Beyond
  Chinchilla-Optimal" arXiv:2401.00448 (quality keeps improving to ~10000 tok/param — the token points here
  stop at ~23 tok/param, well short of that ceiling, by corpus/wallclock budget, not by a claimed limit).
instrument: >
  research/runners/_gen_cortex_token_supply_scaling_derisk.py — the SAME validated WKV-cortex token-supply
  instrument as the parent finding (build_and_train_wkv / eval_perdepth / load_stories / fit_interp_trigram
  from _emerge_wkv_lm_derisk; unchanged anti-cheats). Extended this task only with (a) a warning instead of
  a silent skip when a --token-points value exceeds the seed's train pool, and (b) a documented TinyStories
  invocation in the module docstring — --corpus and --token-points were already plain CLI args, so no
  structural runner change was needed for the corpus or token-count axes themselves.
artifacts:
  - research/findings/raw/_gen_cortex_s7a_tinystories_smoke.json
  - research/findings/raw/_gen_cortex_s7a_tinystories_smoke.json.prov.json
  - research/runners/_gen_cortex_token_supply_scaling_derisk.py
  - research/findings/raw/_gen_cortex_token_supply_scaling.json
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
runner: research/runners/_gen_cortex_token_supply_scaling_derisk.py
---

# S7(a): does the token-supply descent reach the fluency band on a matched-quality corpus (TinyStories) at fixed small capacity? — DRAFT, results pending

**Status banner: DRAFT.** The 6-seed scored sweep is queued to the mini-PC pool, not yet returned. Everything
below is (1) the restated question, (2) the design and why it should isolate the right variable, and (3) a
single-seed toy-scale SMOKE that proves the pipeline runs end-to-end on the new corpus — not a scored answer.
The main session harvests the pool result and finalizes this doc (or files a results doc that supersedes it).

## 0. The question this answers

`2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md`
(c71cc7c9, just landed) showed that at a capacity-matched small WKV cortex (d96/V2000, ~0.42M active params)
the broad-domain (wikitext103) deep-context held-out NLL keeps falling with training tokens, 6/6 seeds, still
descending at the corpus's ceiling of **4.5 tok/param** — but that ceiling sits **~0.24 nats above** a
~ppl20-40 fluency band (NLL 3.0-3.69). It could not say whether the descent, given enough tokens, actually
**reaches** that band, because wikitext103 (broad-domain, complex) exhausts the eval-comparable token supply
at that point. It named the untried next rung explicitly (S7(a) in its own §7): repeat the identical sweep on
a **matched-quality, simple-STYLE, broad-topic** corpus — the TinyStories/phi recipe (arXiv:2502.02737) — at
the SAME fixed small capacity, since that corpus has enough raw text to push tok/param past 20 (Chinchilla-
optimal).

**Two possible outcomes, both decisive:**
- **Reaches the band** → the token-supply lever provably reaches fluency on matched-simple text at this tiny
  capacity; the residual gap to open, arbitrary prose is corpus BREADTH (matched-quality broad text), not
  architecture. This would sharpen Wall #7's "~4 orders of params" framing considerably.
- **Plateaus above the band** → a genuine residual survives even with abundant matched-style tokens at
  d96/V2000; the next rung (S7(b), already named in the parent) — a second capacity point — becomes load-
  bearing to separate capacity from tokens, rather than optional.

## 1. What changed, what stayed fixed (one-variable discipline preserved)

Everything the parent finding held fixed stays fixed: **d_model=96, vocab_cap=2000 (~423,248 active params),
n_layers=1, recurrence=wkv, epochs=6, max_len=40, batch=256**, the same eval-set-fixed / vocab-fixed-per-seed
protocol, and the same two anti-cheats (permute-collapse, memoryless-collapse) kept distinct from the
trigram-beats quality tell. The **only** variable this run changes relative to the parent is the **corpus**
(wikitext103 → TinyStories, `data/corpus/tinystories_train.txt`) and, because that corpus supports it, the
**--token-points** ceiling (up to 240,000 train passages vs. the parent's 48,000).

TinyStories (`data/corpus/tinystories_train.txt`, 119.8MB, one continuous text stream with `<|endoftext|>`
story separators) regex-tokenizes to **~23.66M word tokens** total (checked directly, not derived from an
artifact — a plain `re.findall(r"[a-z']+", ...)` count over the full file). `load_stories` (unchanged, reused
from `_emerge_wkv_lm_derisk`) already treats any corpus as one contiguous token stream cut into fixed
max_len passages, so it needed no TinyStories-specific parsing — the `<|endoftext|>` markers are simply
stripped by the tokenizer regex like any other non-alpha punctuation, and passages span story boundaries the
same way the parent's wikitext passages span sentence boundaries (by design, per the runner's R4 open-prose
cross-sentence rationale).

## 2. The queued design: token points and tok/active-param

<!--derived-->
| max_train_sents (k) | tokens (k×40) | tok/active-param |
|---|---|---|
| 4,000   | 160,000   | 0.378 |
| 8,000   | 320,000   | 0.756 |
| 16,000  | 640,000   | 1.512 |
| 32,000  | 1,280,000 | 3.024 |
| 64,000  | 2,560,000 | 6.048 |
| 128,000 | 5,120,000 | 12.097 |
| 200,000 | 8,000,000 | 18.901 |
| 240,000 | 9,600,000 | 22.682 |

The first five points (0.38 → 6.05 tok/param) overlap the parent's wikitext range for a direct like-for-like
comparison at matched tok/param; the last three (12.1 / 18.9 / 22.7) are the new territory, bracketing
Chinchilla's ~20 tok/param optimum from both sides. `--n-sentences 300000` gives each seed's 85% train pool
(~255,000 passages) enough headroom above the top point (240,000) that no point should hit the newly-added
skip-warning; if it does fire in the returned log, that is itself a signal the pool run's split landed short
and the top point(s) should be treated as absent, not silently averaged in.

## 3. Pipeline smoke (single-seed, toy scale — NOT the scored answer)

Before queuing 6 seeds × 8 points at real capacity to the pool, a single-seed smoke (`--smoke`: seed 42 only,
d_model=48, epochs=3, n_sentences=12,000, token_points=[2,000, 6,000], ~48s wall on CPU) confirmed the runner
loads and trains on TinyStories with no code path unique to wikitext silently misbehaving. Artifact:
`research/findings/raw/_gen_cortex_s7a_tinystories_smoke.json` (provenance-stamped;
`research/findings/raw/_gen_cortex_s7a_tinystories_smoke.json.prov.json` — re-stamped after an unrelated
concurrent process's exit hook raced and mis-attributed the first sidecar, see the commit that fixed it).

At this toy scale (n=1, tiny d=48, only 2 points to 1.416 tok/param) the smoke's own numbers are **not
informative about S7(a)'s real question** — they exist only to prove the pipeline is sound: deep NLL fell
4.570 → 4.8535 is wrong, correct value 5.8992 → 4.8535 across the two points (delta_nll_min_to_max_tokens =
1.0457), `uses_context_at_top` true (perm_collapse +1.313, mless_collapse +0.222 at the top point — the WKV
genuinely uses order and memory even at this toy scale), and `still_descending_at_top` true. The smoke does
**not** yet beat the fair trigram (`beats_trigram_at_top` false, margin -0.7966 nats) — expected at 1.4
tok/param and d=48, consistent with the parent's own low-token-point behavior on wikitext, not a regression.
The smoke's own `verdict` field reads `GO-TOKEN-LEVER` (the runner's generic n=1 aggregate label) — this is a
pipeline-health signal only, explicitly not a claim about S7(a) at real scale or seed count.

## 4. What's queued, and where the real result lands

The full 6-seed run at real capacity is queued to the mini-PC pool (`tools/pool_queue.sh add`, validated
against local + all 3 nodes' argparse before staging, corpus file rsynced to all 3 nodes, code provisioned to
commit `64bc485e0` / `d85090c2`). It is a single pool job running all 6 seeds sequentially inside one process
(matching the parent's own invocation shape), so the output is one aggregate JSON with the runner's built-in
6-seed verdict — `uses_tokens` / `still_descending_at_top` / `uses_context_at_top` / `beats_trigram_at_top` /
`margin_grows_with_tokens` per seed, plus the mean-top-point-NLL-vs-fluency-band residual, exactly the schema
`research/findings/raw/_gen_cortex_token_supply_scaling.json` (the parent's artifact) already uses.

The `--json` output target is `research/findings/raw/` + `_gen_cortex_s7a_tinystories_token_supply.json`
(not yet on disk — this is the pool job's declared output path, recorded verbatim in
`research/queue/pool.queue`). Estimated runtime: the parent's wikitext sweep (5 points, top 4.5 tok/param,
epochs=6) took 2287.9s for all 6 seeds sequentially on CPU; this sweep's per-seed token-epoch volume is
roughly 6.4x larger (8 points reaching 22.7 tok/param vs. 5 points reaching 4.5), so a multi-hour pool job is
expected — appropriate for background non-Claude compute, not an inline run.

## 5. Honest scope notes (carried forward from the parent, still true here)

1. This remains a **small-capacity** (d96/V2000) probe. Reaching the fluency band here would say the lever
   works AT THIS capacity on matched-style text — it would not by itself map the (capacity, tokens, vocab)
   surface (that is S7(b), separately named in the parent).
2. TinyStories is intentionally **simple-style**: a positive result here answers "can token supply alone
   reach fluency on text matched to a tiny model's capacity", not "can this substrate write arbitrary broad
   prose" — that residual (corpus breadth beyond simple style) is the honest gap either outcome leaves open.
3. The base WKV instrument remains BPTT-trained (a scaling/ceiling instrument, biologization inherited per
   the 2026-07-20 local-rule-readout finding, not itself re-run at these token scales).
4. The gate quirk the task brief flagged (citing a `.jsonl` artifact silently truncated to `.json` in
   `PATH_RE`/`ARTIFACT_RE`, causing a false MISSING) was **fixed on main same-day** (commit `7e2edc08e`,
   landed just before this branch was cut) — noted here for the record; this doc only cites `.json`
   artifacts regardless, so it was never exposed to it.
