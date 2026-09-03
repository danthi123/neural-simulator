---
type: finding
status: contributing
date: 2026-09-03
mechanism: ordered content-addressable attention (assoc_t, hippocampal time-cell "when" signal) + an exhaustive investigation of the ~-0.12 margin_vs_trigram fluency bound
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: assoc_t NO-GO at the bound; the bound is a SHARED data/tokenizer-regime limit (with an exactness-gap caveat) — next levers = read-integration gate + predictive objective, NOT more attention capacity
artifacts:
  - research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json
---

# Ordered attention lands AT the ~-0.12 fluency bound — three families converge, and an exhaustive investigation says the lever is read-integration/objective, not capacity

**Status:** assoc_t (ordered content-addressable attention) is a NO-GO at the bound, BUT the order fix worked exactly as the diagnosis predicted, and a 10-agent bound-investigation (record + external literature) turned the negative into a ranked, evidence-grounded next-lever ladder. Owner steer: pursue open fluency FULLY.

## Result — ordered attention fixes the bag underfit, then lands at the shared bound

<!--derived-->
From `research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json`, deepest bucket (10-99), per-seed margin_vs_trigram: −0.169, −0.054, −0.202, −0.057, −0.196, −0.205 → **mean −0.147** (min −0.205, max −0.054; high variance). Anti-cheats healthy (mless-collapse ~+1.4, perm-collapse ~+3.6 — uses memory + order).

<!--derived-->
| deployable-mouth family (depth-2, contiguous, 6-seed) | mean margin_vs_trigram |
|---|---|
| bag content-addressable attention (no order) | −0.347 |
| **ordered attention (assoc_t, +time-cell "when")** | **−0.147** |
| spiking SSM dual-nonneg (recurrence) | −0.125 |
| HiPPO structured SSM (seed 42) | −0.126 |
| exact-math wkv, sentence-mode | −0.125 |
| exact-math wkv, CONTIGUOUS (1-seed) | **+0.02 (crossed)** |

The temporal/order signal did exactly what the diagnosis predicted: it resolved the bag underfit (−0.347 → −0.147, +0.20). But ordered attention lands AT the ~−0.12/−0.15 zone the linear recurrence and HiPPO already occupy — **not a surpass**. This is the July content-addressable arc's predicted "content+order is NECESSARY-BUT-NOT-SUFFICIENT" pattern (`2026-07-11-LEARNED-keys-make-content-addressable-retrieval-load-bearing`).

## The bound-investigation verdict (10-agent workflow: record + external literature)

<!--derived-->
**Primary verdict: the ~−0.12 is a SHARED data/tokenizer-regime bound, not an architecture-specific wall** — four unrelated families pile on one number; the bound is SOFT (2× contiguous tokens moved exact wkv −0.125 → +0.02); BPE amplifies it (word-level tokenization beats a fair trigram by +0.3 to +0.8 nats while BPE V=8001 loses — greedy sub-word merges make the trigram an unusually strong baseline).

**Completeness-critic caveats (load-bearing — do NOT over-read the "shared bound"):**
1. In the MATCHED contiguous regime the families do NOT all sit at −0.12: exact wkv is at **+0.02 (crossed)** while spiking/HiPPO/assoc_t cluster below — evidence FOR an **exactness/architecture gap** (exact-math read pulling ahead of approximate/spiking reads), not purely a shared bound. The "convergence" partly compared sentence-mode wkv to contiguous-everything-else.
2. The decisive wkv −0.125 → +0.02 jump is **1-seed** AND confounds token-count with **context-length** (sentence-mode resets state every 3-16 words, so the deep bucket is measured on a sparse tail). So "more tokens crosses" is fragile on both counts.
3. External literature's strongest SAME-budget datapoint points at **OBJECTIVE, not memory or data**: at 10M words a causal+masked hybrid hits BLiMP 0.794 vs a tuned n-gram's 0.633 and a plain causal LSTM's 0.661 (recurrence-alone barely ties the n-gram = this arc's failure mode).
4. Honesty flag: 3 of 8 investigators returned placeholder content — the synthesis rests on 5 substantive digests.

## The pivotal diagnostic + the ranked next-lever ladder (all brain-based, being fired)

**PIVOTAL DIAGNOSTIC (running now on GPU):** a 6-seed contiguous **wkv** — settles whether +0.02 replicates (exactness gap: exact attention crosses, spiking lags → the deliverable is the SPIKING PORT) or was a 1-seed fluke (genuine shared bound → objective/capacity is the lever).

Ranked mechanisms (fire order gated on the diagnostic; the exact-math families crossing is NOT the deliverable — only the DEPLOYABLE SPIKING mouth crossing counts):
1. **Complete the content-addressable read** — the learned trust GATE + residual-correction (BUILT: `--assoc-gate`, 6-seed QUEUED after the diagnostic), then deeper keys + a learned write policy. Matches the symptom; the July learned-key deltas (up to −0.42 nats at d10-99) exceed the residual gap.
2. **A predictive/predictive-coding OBJECTIVE** (multi-step / future-token auxiliary loss; being built `--pred-aux-weight`) — the critic's under-ranked-but-strongest same-budget external lever; cheap, composes with everything.
3. **Capacity** (depth then width) to escape the ~6.4 tok/param reversal band.
4. **Matched-quality token doubling + a ULM/char tokenizer variant** — a DIAGNOSTIC (the token story rests on 1 seed), NOT the primary bet; wikitext103 (on disk) is domain-mismatched, so not raw-volume.

## Reproduce

```bash
# assoc_t 6-seed (300W cap standing):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence assoc_t --n-layers 2 --d-model 192 --batch 128 --tokenizer bpe \
    --corpus data/corpus/simplewiki.txt --contiguous --max-len 40 \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 \
    --json research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json
```
