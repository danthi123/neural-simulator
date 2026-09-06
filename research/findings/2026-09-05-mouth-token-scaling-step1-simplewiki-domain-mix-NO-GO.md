---
type: finding
status: negative
claim_check: measured
date: 2026-09-05
mechanism: mouth broad-domain token-scaling STEP-1 (no-download proxy) — train the deployable linattn mouth on
  wt103+simplewiki (~+8% same-language but SIMPLER-domain tokens), eval on the wt103 held-out deep-context buckets
  vs the wt103-only baseline. --eval-corpus decontaminates 2052 eval-overlap passages.
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [43]
runner: research/runners/_emerge_wkv_lm_derisk.py
artifacts:
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103plus_simplewiki_evalwt103_s43.json
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
builds_on:
  - research/findings/2026-09-05-mouth-objective-lever-flat-on-broad-domain-architecture-is-wrong-axis-NO-GO.md
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
verdict: >
  NO-GO on the deep margin, but the test is a WEAK/confounded proxy that does NOT actually exercise the
  same-domain token-supply lever. Adding ~8% SIMPLER-domain data (simplewiki) to wt103 training and evaluating on
  the wt103 held-out deep buckets left the deep-context (10-99) margin_vs_trigram slightly WORSE, not better
  (figures in the marked body; single-seed s43 A/B vs the wt103-only baseline). Anti-cheats are HEALTHY (large
  permutation- and memoryless-collapses), so this is a real, clean measurement, not a broken instrument: the model
  learned genuine order/context structure and still lost a little ground at depth — consistent with the added
  simpler-domain tokens pulling the model marginally toward simpler text and away from wt103's harder long-range
  distribution. CRUCIAL SCOPE: this is a DOMAIN-MIX test, NOT a same-domain scale-up. The 2026-09-01 6-seed GO
  (token-supply is the lever) was CAPACITY-MATCHED SAME-DOMAIN scaling; wt103 is now ~fully used (~2.1M sentences),
  so a clean same-domain scale test needs MORE HARD same-domain tokens, which requires a corpus DOWNLOAD.
  Per THE LAW this is a method verdict on the cheap no-download proxy, not a capability wall: naive
  simpler-domain padding is banked as unhelpful (mildly harmful); the real same-domain token-supply lever is
  UNTESTED here and remains the standing hypothesis. The next method — a large HARD-same-domain corpus (10-100x
  tokens toward Chinchilla-optimal) — is a DOWNLOAD (owner-permission-gated) + major-compute (hardware-linked)
  fork, SURFACED to the owner, not unilaterally triggered on a negative proxy. Mouth default remains linattn; Qwen
  scaffold not retired.
---

# Mouth token-scaling STEP-1: simplewiki domain-mix is NO-GO — and it doesn't test the real lever

## What ran
`_emerge_wkv_lm_derisk.py --recurrence linattn` trained on `wt103_plus_simplewiki.txt` (2,273,197 passages after
`--eval-corpus` decontaminated 2052 wt103-overlap passages), evaluated on the wt103 held-out deep-context buckets,
s43, 4 epochs, d_model=192, 2 layers — byte-identical config to the wt103-only baseline except the training corpus.
The cheap, no-download STEP-1 of the token-scaling fork (owner delegated the mouth fork 2026-09-05).

## Derived — deep-context margins vs trigram (s43; direct reads of the two cited artifacts)
<!--derived: this-run from research/findings/raw/_emerge_wkv_lm_linattn_wt103plus_simplewiki_evalwt103_s43.json; baseline from research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json; lift is their difference -->
| depth | 1 | 2 | 3 | 4-5 | 6-9 | 10-99 |
|---|---|---|---|---|---|---|
| wt103+simplewiki (this run) | 0.964 | -0.627 | -0.524 | -0.459 | -0.397 | -0.312 |
| wt103-only baseline | 0.989 | -0.570 | -0.454 | -0.402 | -0.356 | -0.286 |
| lift | -0.025 | -0.057 | -0.070 | -0.057 | -0.041 | -0.026 |

Deep-bucket (10-99) lift = **-0.026** (WORSE; the pre-set direction-positive bar was >=+0.03). Every bucket is
flat-to-slightly-worse. Anti-cheats on this run (deep bucket): wkv 4.765 vs wkv_perm 8.641 (permutation-collapse
+3.876) and wkv_memoryless 5.878 (memoryless-collapse +1.113) — both large, so the model genuinely uses order and
context; the measurement is clean.

## Reading it (no-defer)
The deep margin did not lift — it slipped slightly. But the honest scope is decisive: **this did not test the
same-domain token-supply lever.** simplewiki is a *different, simpler* domain; adding ~8% of it to wt103 training
while evaluating on wt103 is a DOMAIN-MIX experiment, and the result says naive simpler-domain padding mildly hurts
the harder eval (the model drifts toward simpler text). The 2026-09-01 finding that established token-supply as the
lever was CAPACITY-MATCHED SAME-DOMAIN scaling (deep-NLL drops monotonically, margin grows with same-domain
tokens). wt103 is now ~fully used, so the same-domain scale-up cannot be tested without new HARD same-domain data.

⇒ Two things are now banked: (1) architecture levers = wrong axis (objective/delta-rule/content-addressing, prior
findings); (2) naive simpler-domain data-mixing = unhelpful/mildly-harmful (this run). The same-domain
token-supply lever remains the standing hypothesis and is UNTESTED at scale here. Testing it cleanly needs a large
HARD-same-domain corpus (10-100x tokens toward Chinchilla-optimal), which is a DOWNLOAD (owner-permission-gated per
the download safety rule) + major GPU-compute (connects to the 2nd-GPU/hardware plan). That fork is SURFACED to the
owner — not unilaterally triggered on a negative cheap proxy.

## Honest scope
Single-seed (s43) A/B, direction-test not a 6-seed claim. Additive; no production change; mouth default remains
linattn and the Qwen articulation scaffold is not retired. The elapsed run was 15418.8s (~4.3h) on the 3090.
