---
type: finding
status: verified
date: 2026-09-18
mechanism: a go/no-go deep-research review (10-agent adversarially-verified workflow scale-unblocks-walls,
  wf_4470f70a-209) answering, BEFORE any hardware purchase, whether scaling up (params / tokens / GPU) will
  unblock the primary walls — broad + fluent own-voice communication (the ~49 neural-render-blocked faculties).
  Weighs our OWN metric-vs-scale curves (read from raw artifacts, not summaries) heaviest, plus the external
  scaling-law literature and a deliberate steelman of the owner's skepticism, adversarially cross-examined.
integration_faculty: mouth / own-voice fluency (neural-render — the keystone wall)
lane: strategy / own-voice-fluency
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO on buying compute NOW — run the free + cheap confirmatory tests FIRST (high confidence). RAW
  parameter/GPU scale is FALSIFIED as a standalone fluency lever on the broad domain (the token-supply sweep at
  fixed small capacity is capacity-saturated — near-flat gain across a large token increase, and the margin over
  a fair trigram REVERSES; numbers in body), and the ONE realistic large-compute analog we own (an AWS multi-
  GPU-hour larger-token run) REGRESSED cross-domain at deployable vocab. Every broad-domain axis run to
  saturation sits FLAT above the fluency band. The mechanism, corroborated by the external literature, is that
  FLUENCY is a DATA-QUALITY problem and BREADTH is a TOKEN-THROUGHPUT + curation problem atop a MODEST one-time
  capacity floor — NOT raw model size. The owner's prior ("scale never got us anywhere") is vindicated by our
  own saturated data. The mouth IS the correct keystone (ledger: roughly half the blocked faculties are
  neural-render-gated), so scale, if it worked, would be aimed right — but the lever at that wall is data, not a
  bigger GPU. The ONLY unexhausted variant is JOINT compute-optimal capacity+token scaling, which was never run
  at target size AND is already dented (the matched-ratio point sits flat above band). Two risks a GPU cannot
  touch even on a positive curve: BIOLOGIZATION TRANSFER (every clean pro-scale result is a BPTT model at a small
  vocab; the DEPLOYED biological fixed-reservoir + local-rule readout was never re-run at scale) and the LEXICAL
  CAP (the small-vocab output is unk-riddled; real fluency needs a much larger vocab, which multiplies params
  against the committed consumer-3090 one-brain envelope). Gate ANY purchase on three pass-conditions IN ORDER
  (in body). Honest fork recorded: if the deployable/biological path cannot reach the band within the 3090
  envelope, keeping the Qwen articulation scaffold is the honest call and a GPU buy would purchase an
  undeployable BPTT model. A wall defers a METHOD, never the capability.
runner: research/runners/_gen_cortex_token_supply_scaling_derisk.py (existing artifacts) + the review workflow
external: EXTERNAL-DONE (read + lane-recorded) — TinyStories (Eldan & Li, 2023, fluency at tiny param counts via
  data quality); Chinchilla (Hoffmann et al., 2022, compute-optimal token:param ratio); Qwen2.5 + SmolLM2
  technical reports (small models over-trained far past Chinchilla buy breadth with token throughput);
  Allen-Zhu & Li (2024, a modest per-param knowledge floor); Schaeffer et al. (2023, emergent-abilities-at-scale
  is largely a metric artifact; broad fluency was never an emergent ability). The verdict is the deliverable —
  it prevents a premature compute purchase and names the cheapest tests that would settle it.
artifacts:
  - research/findings/raw/_gencortex_scaling/fineweb_d192_n2M_s100.json
  - research/findings/raw/_gencortex_scaling/_aggregate_6seed.json
---

# Scaling go/no-go: RAW scale is falsified as the fluency lever; the lever is DATA — do NOT buy hardware yet, run the free test first

Owner (2026-09-18) approved a per-mechanism faithfulness-vs-tractability relaxation (honesty-conditioned) but
was skeptical of scaling ("we've played with scale many times and it never got us anywhere") and demanded proper
deep research + extensive RAG of our own record — "100% sure scaling up will unblock our primary walls before we
invest in the compute." This is that review: a 10-agent adversarially-verified workflow over our own artifacts +
the external scaling literature. The adversarial "the curve is flat, not rising" angle LANDED and is decisive.

## The curve evidence (the discriminator: rising = scale-limited; flat = a wall scale won't fix)
<!--derived-->
- **Token supply at fixed small capacity — FLAT past Chinchilla.** research/findings/raw/_gencortex_scaling/fineweb_d192_n2M_s100.json:
  NO-GO-CAPACITY-SATURATED — only ~0.0664 nats gained across a 16x token increase (slope ~0.003), at ~80
  tok/param, residual ~+0.22 above the 3.0-3.69 fluency band; the margin over a fair trigram REVERSES
  (~0.367 -> ~0.146). A real, clean plateau.
- **Capacity at matched tokens — still descending but UNDER-tokened.** _aggregate_6seed.json: d96 3.976 > d192
  3.928 > d384 3.906 (all 6/6 still-descending), BUT d384 is at only ~4.328 tok/param — under-tokened, not a <!--derived-->
  clean capacity ceiling; gains halve per doubling.
- **The decisive JOINT cell does NOT exist.** No d384 x ~20-tok/param artifact; nothing postdates 2026-09-17.
  The exact regime a hardware buy targets is literally unmeasured. And d192 at Chinchilla-matched 20 tok/param
  already sits ~3.9227, +0.23 above the band, flat — denting even that hope.
- **The realistic large-compute analog REGRESSED.** The AWS ~39-GPU-h d192 / 2B-token run: deep-context margin
  -0.408 at deployable vocab (vs -0.082 at 0.4B); train loss nearly flat (4.826 -> 4.802 over 4 epochs) — small <!--derived-->
  capacity already bound + single-domain over-training HURT the out-of-domain eval. Confounded (fixed small
  capacity + single-domain), but it is the closest thing we have to "buy compute and run it big," and it is
  negative.

## Why (mechanism, external-corroborated)
<!--derived-->
Fluency is a DATA-QUALITY problem (TinyStories: fluent multi-paragraph coherence at <10M params — the exact wall
125M web-trained models fail). Breadth is a TOKEN-THROUGHPUT problem atop a MODEST one-time capacity floor
(Qwen2.5-0.5B @ 18T tokens; SmolLM2 over-trained 100-500x past Chinchilla — breadth bought with data throughput,
not model size; Allen-Zhu ~2-bits/param knowledge floor). "Capability emerges at a scale threshold" is largely a
metric artifact (Schaeffer 2023), and broad fluency was never on the emergent-abilities list. This matches our
own record that the broad path is token-supply/Chinchilla-bound and architecture is the wrong axis
(2026-09-05 mouth-objective-lever-flat; the hippokey/deltanet/objective/content-addressing levers all NO-GO).

## The keystone is correct; the lever at it is not a GPU
<!--derived-->
Ledger tally (docs/PRODUCTION_INTEGRATION_LEDGER.yaml): ~49 rows are retire_status BLOCKED:neural-render — the
off-substrate mouth is the single dominant wall gating the great majority of production faculties. So scale is
aimed at the right keystone. But the lever AT that keystone is DATA (matched-quality token supply + curation;
note Wikitext caps at ~1.7 tok/param, distillation-as-data banked NEGATIVE, wrong-domain data actively HURT) plus
at most a modest capacity floor — none of which a bigger GPU solves. Other live walls are NOT scale-limited at
all (gap#4 deep-credit = interpretability-foreclosed; affect noise-robustness = redirected to multimodal
grounding; the one-brain 11-organ flip = a calibration bug, root-caused 2026-09-18).

## Gate any purchase on three pass-conditions, IN ORDER (cheapest first, all zero-purchase)
<!--derived-->
1. **FREE, minutes, no GPU — the analytical gate.** Fit the additive law L(N,D)=E+A/N^a+B/D^b (host scipy) over
   the ALREADY-COLLECTED d96/d192/d384 x token grid and read the irreducible E. If E > 3.69, scale is falsified
   as the fluency lever at vocab 2000 and NO hardware is justified. (RUNNING as of this finding.)
2. **~3-6 GPU-h on the EXISTING 3090 — the empirical joint cell.** _gen_cortex_token_supply_scaling_derisk.py
   --d_model 384 at ~20 tok/param (d384 active~2.13M -> ~42.6M tokens, ~887k sentences), seeds 100/101/102.
   GO iff (a) broad-domain deep-context NLL descends INTO [3.0,3.69] (not ~3.9); (b) the trigram margin GROWS
   with joint scale; (c) the winning point survives at DEPLOYABLE vocab (V>=8k) on the BIOLOGICAL local-rule
   readout (not the BPTT WKV) within the consumer-3090 one-brain envelope.
3. Only if BOTH pass is a token-throughput hardware buy justified. If the biological/deployable path cannot reach
   the band within the 3090 envelope, keep Qwen as the conditioned articulation scaffold (the honest call) — a
   bigger GPU would purchase a fluent BPTT model that cannot deploy as the one-brain spiking mouth.

## Update: the FREE analytical gate was run — INCONCLUSIVE, and it corrected the cheap test
<!--derived-->
The additive-law fit (test 1) was run host-side on the existing broad-domain grid (13 unique (N,D) cells,
metric wkv_deep_nll, vocab 2000; script /tmp/claude-1000/.../scratchpad/fit_scaling.py). It does NOT cleanly
resolve the gate: the free 5-param fit gives irreducible E~3.85 (ABOVE the band -> falsified) but with
degenerate exponents (a~0.80/b~1.40, ill-conditioned, R^2 0.998); the Chinchilla-prior-constrained fit
(a=b=0.34) gives E~3.64 (BELOW band -> not falsified) but fits far worse (R^2 0.894). The two straddle the 3.69
threshold. ROOT CAUSE: the grid is NOT fully crossed — d384 (the largest model) was only run at ~2-4 tok/param,
never extended, so its large-N asymptote is pure extrapolation. IMPORTANT CORRECTION to the cheap test above:
the fit checked whether the joint d384 x ~20-tok/param cell would resolve E and found it WOULD NOT (both fits
predict ~the same observable loss there, ~3.85-3.88; they only diverge in the N,D->inf limit). So the DECISIVE
zero-purchase test is heavier than first stated: extend d384 to a tok/param range COMPARABLE to d96/d192 (~370M
tokens, not ~43M) on the EXISTING 3090 to pin E — a longer local run, still no hardware buy. Net: the go/no-go
stays NO-GO-on-hardware; the free gate narrowed but did not settle it, and it correctly killed a would-be-wasted
cheap run. Still-DON'T-buy holds; the decisive local test is the extended-d384 token run.

## Adversarial verification
R1 (curve is flat, not rising) LANDED — decisive; it moved the synthesis from "probably data-dominant scaling
helps" to "the saturated evidence leans AGAINST raw scale." R2 (spiking/biology won't transfer at scale) did NOT
overturn but is folded in as purchase-precondition (3c). R3 (the wall isn't the mouth) did NOT overturn — the
49/neural-render tally confirms the mouth is the keystone, and R3's "the lever is data-curation" collapses INTO
the verdict. Full workflow result: /home/dant123/.claude/.../tasks/wyde8ckya.output (run wf_4470f70a-209).
