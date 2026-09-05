---
type: finding
status: locked-not-executed
claim_check: measured
date: 2026-09-05
mechanism: additive `--eval-corpus` instrument on `_emerge_wkv_lm_derisk.py` — hold the held-out EVAL fixed while scaling the TRAINING-token supply, so the broad-domain token/data-bound wall can be tested WITHOUT a download (local wt103 + Simple English Wikipedia combined training, wt103 held-out eval)
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [43]
runner: research/runners/_emerge_wkv_lm_derisk.py
external: >
  The token-supply / data-scale axis this instrument tests is the externally-grounded lever from the lane's own
  6-seed GO (2026-09-01-...-plateau-is-STARVATION-not-capacity-wall): Hoffmann et al. 2022 "Training
  Compute-Optimal LLMs" (Chinchilla) arXiv:2203.15556 (below ~20 tok/param a fixed model is token-starved and  <!--derived-->
  its held-out loss plateaus). This is a DATA-axis continuation of that GO, not another architecture lever.
artifacts:
  - research/runners/_emerge_wkv_lm_derisk.py
builds_on:
  - research/findings/2026-09-05-mouth-objective-lever-flat-on-broad-domain-architecture-is-wrong-axis-NO-GO.md
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
verdict: >
  INSTRUMENT BUILT + VERIFIED; the decisive single-variable test is QUEUED (locked-not-executed), result
  pending the controller's GPU harvest. Three architecture levers (objective FLAT, delta-rule sub-bar,
  content-addressing exhausted) came back small on the broad-domain wt103 mouth, while the lane's OWN 6-seed GO
  says the plateau is TOKEN/DATA-starvation. The cheapest way to move the DATA axis WITHOUT a download is to add
  the already-local Simple English Wikipedia (data/corpus/simplewiki.txt, ~142M, SAME Wikipedia domain, simpler
  English) on top of the ~fully-used wt103 as broad-domain training tokens. This required an instrument that
  scales TRAINING tokens while holding the EVAL fixed and comparable to the wt103 baseline. Added an ADDITIVE,
  default-OFF `--eval-corpus` flag: when set, the held-out eval is drawn from that corpus with the SAME
  rng/permutation/cut/truncate arithmetic used for `--corpus`, so at a shared seed the eval set is byte-identical
  to a standalone run on the eval-corpus; training stays on `--corpus`, and any train passage whose word tokens
  exactly match an eval passage is DROPPED (content-based decontamination) so the extra-token training never sees
  the held-out eval. Proven byte-identical when OFF (identical per_seed sha256 between the unedited and edited
  runner; no output-key leak) and correct when ON (eval-bucket counts match a standalone run exactly; the
  decontamination drops the eval-overlap passages the concatenation introduces). The decisive run — linattn
  trained on combined wt103+simplewiki, evaluated on the wt103 held-out (same d192/2-layer/bpe/max-len-40/4-epoch
  s43 config as the baseline) — is queued on gpu_queue.sh. PRE-REGISTERED: direction-positive if the deep bucket
  (10-99) margin_vs_trigram lifts >=+0.03 off the baseline's -0.286 WITH the anti-cheats (permute/memoryless
  collapse) holding; that would justify a larger same-domain corpus (a download the controller handles
  separately). No sim/ edit; no production change; the mouth default remains linattn.
---

# `--eval-corpus` instrument: hold the eval fixed, scale the training-token supply — the token/data-bound mouth wall tested WITHOUT a download

## Why (the settled context this builds on)
The broad-domain (wt103) own-voice-mouth wall is TOKEN/DATA-bound, not architecture. Three architecture levers
came back small on this exact test — the predictive-coding OBJECTIVE essentially FLAT at depth
(`2026-09-05-mouth-objective-lever-flat-on-broad-domain-...`), the delta-rule a sub-bar lift, content-addressing
exhausted — all far from the ~+0.3-0.57 needed to cross the trigram at depth. Read against the lane's OWN 6-seed
GO (`2026-09-01-...-plateau-is-starvation-not-capacity-wall`: more unique tokens -> monotonic deep-NLL drop,
margin GROWS with tokens, beats the trigram), the axis that moves the substrate is TRAINING-TOKEN SUPPLY. wt103
is ~fully used locally, but `data/corpus/simplewiki.txt` (Simple English Wikipedia, ~142M) is the SAME Wikipedia
domain in simpler English — so wt103 + simplewiki is a clean ~1.25x broad-domain token increase with a
compatible distribution, testable NOW with no download.

## What was built (additive, default-OFF, no sim/ edit)
`_emerge_wkv_lm_derisk.py` gains one flag, `--eval-corpus`:
- When UNSET (default None): every path is byte-identical to today.
- When SET: the held-out EVAL set is drawn from `--eval-corpus` using the SAME rng/permutation(0.85 cut)/
  truncate arithmetic that `--corpus` uses, so at a shared seed the eval set is byte-identical to a standalone
  run on that corpus. TRAINING stays on `--corpus`. Any train passage whose word tokens exactly match an eval
  passage is dropped from the training pool (content-based decontamination), so the extra-token training never
  sees the held-out eval (silent-failure discipline: contamination would spuriously inflate the lift).

This makes the single variable clean: only the TRAINING-token supply changes; the eval is held fixed and
directly comparable to the wt103-train/wt103-eval baseline.

## Derived — verification (exact measurements, small deterministic smokes)
<!--derived: all figures below are direct reads of throwaway smoke runs described inline; the decisive-run figures do not exist yet (test queued) -->
- BYTE-IDENTICAL WHEN OFF: a small BPE contiguous smoke run on the UNEDITED runner and on the EDITED runner
  (with `--eval-corpus` unset) produced an IDENTICAL `per_seed` sha256; the `eval_corpus` output key is absent
  when off; top-level keys identical. The flag adds only guarded, skipped code when unset.
- CORRECT WHEN ON: a smoke with `--corpus combined --eval-corpus wt` (where `combined` = wt-slice ++ tinystories
  so the training pool CONTAINS the eval passages) matched a standalone `--corpus wt` run's eval-bucket counts
  in every depth bucket (1/2/3/4-5/6-9/10-99), confirming the eval set is byte-identical; and the decontamination
  correctly dropped the eval-overlap passages the concatenation introduced (train count fell below the standalone
  cap by exactly the overlap), confirming the extra-token training never sees the held-out eval.

## The decisive test (queued; single-variable = training-token supply, eval fixed)
Combined training corpus = wt103 ++ simplewiki (a data-prep concatenation with a boundary seam; a
matched-quality broad-domain corpus is host-legit curriculum per the mission, EXEMPT — not a cheat). Config
matches the baseline exactly (recurrence linattn, uniform-decay, bpe, contiguous, max-len 40, d_model 192,
n_layers 2, epochs 4, seed 43, max-eval-sents 4000), evaluated on the wt103 held-out. Baseline for comparison:
`research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json` (train wt103 / eval wt103, deep-bucket
10-99 margin_vs_trigram -0.286, WKV 4.749, trigram 4.463).

PRE-REGISTERED decision: direction-positive if the deep (10-99) bucket lifts meaningfully (margin_vs_trigram
>= +0.03 off -0.286, and/or the absolute WKV deep NLL drops below 4.749) WITH the anti-cheats holding (permute
and memoryless collapse). A positive direction justifies a larger same-domain corpus (a download the controller
handles separately); a flat/negative result banks "~1.25x local broad-domain tokens is not enough to move the
deep margin" and points to the volume/quality axis.

## Honest scope
Single-seed (s43) direction-test, matched to the single-seed baseline — labeled direction-test, not a 6-seed
claim. Additive; no production change; the mouth default remains linattn. The extra training tokens are
simplewiki passages added on top of a same-size random wt103 subset; the trigram control is refit on the same
(combined) training data, so `margin_vs_trigram` remains the fair within-run control, and the absolute WKV deep
NLL is reported alongside it to guard against a margin that lifts only because the trigram shifted.
