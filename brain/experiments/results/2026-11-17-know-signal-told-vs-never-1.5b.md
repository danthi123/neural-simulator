# Result: "do I know this" signals at 1.5B, told vs never-told (re-analysis, no new run)

Data: `results/2026-11-13-confidence-gate-1.5b/cgate_s*.json` (AWS, six seeds 42/43/44/100/101/102, DEV fact bank, slept
1.5B cortex, hippocampus off; 12 told questions in the telling's words, 12 reworded, 12 never-told per seed). The
11-13 analysis used told right-vs-wrong only; this is the told-vs-never AUC the record listed as never measured.

| signal | told (same wording) vs never | reworded vs never | told + reworded vs never (pooled) |
|---|---|---|---|
| cue (cortical cue response) | 0.84 | 0.61 | 0.73 |
| mean_logp (answer) | 0.88 | 0.52 | 0.70 |
| min_logp | 0.80 | 0.55 | 0.67 |
| first_logp | 0.68 | 0.42 | 0.55 |
| shift (adapters on - off) | 0.57 | 0.62 | 0.60 |
| mean_logp_off | 0.51 | 0.38 | 0.44 |

Per seed, pooled cue 0.56-0.86, mean_logp 0.57-0.83. Reading: at 1.5B the signals separate a question in the telling's
own words from a never-told one, but not a reworded told question from a never-told one: they track wording overlap
with what was replayed, not knowledge, the same failure as at 0.1B (`PREREQUISITES.md` Metacognition). Never-told
replies put told names in new slots ("Your horse is called Pixel.", "Your aunt lives in Oslo.").

Open conversation (re-scored 11-17 with the new `open_conv.invented` mark): the 1.5B brain invents an answer to 96%
of never-told questions (intact 76/81, hippocampus off 79/81, AWS + pool, 11-15/16 runs) and never says it does not
know. The pretrained cortex alone (no memory, no night) invents 79-86%; with the night off 7/13, with the night 15/15
(small n): the cortex already confabulates, and the night makes it total.
