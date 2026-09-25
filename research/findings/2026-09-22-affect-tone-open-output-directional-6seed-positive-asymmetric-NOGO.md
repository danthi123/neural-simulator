---
type: finding
status: verified
claim_check: measured
date: 2026-09-22
lane: language (own-voice mouth / affect grounding) — roadmap §8 near-term-speech
seeds: [42, 43, 44, 100, 101, 102]
mechanism: the live affect->tone decode coupling `webapp/wkv_mouth_generator.py::_apply_affect_bias` (a saturating,
  margin-to-top1-aware, habituating additive logit bias over the WARRINER-gated affect lexicon), driven by the real
  spiking affect organ's signed differential through `webapp/open_ended_chat.py::answer_turn`
  (`valence_from_affect` -> `_WKV.generate(valence=,arousal=)`), lesioned by `BRAIN_AFFECT_LESION=1` (clamps the
  organ differential -> valence 0.0 -> `_apply_affect_bias` early-returns, an EXACT decode no-op). Measured LIVE
  through `webapp.server.brain_chat` on the EXACT deployed config (BRAIN_OPEN_ENDED=1, linattn WKV mouth, broad
  scope, real onebrain composer + real spiking affect organ), CPU/numpy.
verdict: NO-GO on the directional sub-condition of the strict-disjoint independent-lexicon OPEN-output ruler, at
  this operating point — a method verdict on the RULER + operating point, NOT a falsification of the affect->tone
  capability (which is characterized here as real-but-POSITIVE-ASYMMETRIC). The instrument is trustworthy on every
  validity axis (clean null, clean control, disjoint lexicon, per-seed ckpt, form-not-content preserved, fluent),
  so the NO-GO is a real negative, not UNDEFINED. HARD RULE 2: banked, not tuned to pass.
artifact: research/findings/raw/_affect_tone_open_output/affect_tone_open_output_verdict.json
runner: research/runners/_lbf_affect_tone_open_output_derisk.py (6-seed, fresh-subprocess-per-arm)
supersedes-scope-of: research/findings/2026-09-04-linattn-affect-coupling-sharpness-aware-GO.md (that single-seed
  GO's instrument was BINARY "raw text differs byte-for-byte lesion0 vs lesion1"; this finding shows the byte-diff
  is real but is NOT a directional tonal effect on an independent ruler — the right ruler reveals the asymmetry).
---

# Affect→tone over the OPEN mouth reply is real but POSITIVE-ASYMMETRIC and sub-band — directional NO-GO (6-seed)

## What was built (roadmap §8 near-term-speech, first rung)

The affect→tone coupling was GO only single-seed and only on a BINARY "raw output differs byte-for-byte" instrument
(`phase4_linattn_flip_confirmation_rerun.py`), scored with a tone proxy that reused the SAME WARRINER lexicon the
bias boosts (the circular-proxy trap the 2026-09-04 flip-confirmation finding named). This rung upgrades the
verification to the RIGHT ruler: a 6-seed, DIRECTIONAL, INDEPENDENT-lexicon, DISTRIBUTIONAL lesion probe on the
freely-generated reply, plus a default-OFF `measure_affect_tone_open_output` (`LB_AFFECT_TONE_OPEN_PROBE`) in
`load_bearing_fraction.py` so the #1 metric can score affect→tone by the OPEN-output ruler once it earns a GO
(mirrors `measure_open_ended_distributional`). Additive + flag-gated: OFF ⇒ byte-identical.

Per seed ∈ {42,43,44,100,101,102}, six FRESH-SUBPROCESS arms (the _RNG private-timeline confound forbids one
process for multiple arms): pos / neg (affect active) · lesion / lesion_rep (the exact-no-op neutral baseline +
determinism) · ctrl_pos / ctrl_neg (valence MONKEYPATCHED to a mood-DECOUPLED random-sign sequence — the
attribution control). Each arm ran 10 free-talk TONE prompts (unknown ⇒ free-gen, where tone lives + the
moat-holds check) + 1 KNOWN topic (the content-identity guard). Reply = `open_ended.raw`. Tone scored by an
INDEPENDENT 356-word signed lexicon with ZERO overlap with the 180-word WARRINER boosted set (enforced + counted at
load), VADER-style `s/sqrt(s^2+15)` + negation. δ preregistered from the lesion arm's own per-prompt tone-noise
band. NLTK VADER's own `vader_lexicon.txt` is absent on disk and no-download discipline forbids fetching it — the
lexicon is the plan's sanctioned curated-disjoint fallback (residual: swap in true VADER when its data ships).

## Result: a clean, trustworthy NO-GO on the directional gate

Artifact: `research/findings/raw/_affect_tone_open_output/affect_tone_open_output_verdict.json` (top-level
`preconditions` block + per-seed values + per-arm JSONs in the same directory).

Caveat on seed independence (honest): different-seed mouth CHECKPOINTS (verified distinct sha256, resolved
per-seed with no seed42 fallback) produced NEARLY-IDENTICAL free-gen text on these prompts — e.g. seed43 and
seed102 differ on only 1 of 10 tone prompts — so their aggregate tone metrics coincide. The linattn mouth's
free-gen is largely seed-invariant on this prompt set; the 6 seeds therefore provide limited independent
variation of the FORM, though each arm genuinely ran its own per-seed mouth. This does not change the verdict
(the directional effect is sub-band on every seed), but it means the 6-seed replication is of a near-deterministic
mouth, not of 6 independent draws.

<!--derived--> (all numeric values below are rounded presentations of the cited verdict artifact + its per-arm JSONs in the same directory)

Instrument validity (all preconditions clean, so the reading is trustworthy): determinism byte-identical 6/6;
attribution control clean 6/6; content-identity (facts + known byte-identical intact vs lesion) 6/6; moat holds
(every unknown free-talk prompt stayed known=False under the active bias) 6/6; fluency max salad-fraction 0.155 ≤  <!--derived-->
0.16; lexicon overlap with WARRINER = 0; every seed's mouth checkpoint resolved to its OWN per-seed file (no
seed42 fallback confound).

Directional, under the PREREGISTERED band δ=0.1346 (pop-std of the neutral per-prompt tone): **NO-GO** — all 12  <!--derived-->
per-direction states (6 pos + 6 neg) are `null` (below the neutral tone-noise band).

| seed | tone_pos | tone_neg | tone_lesion | gap_pos | gap_neg | pos | neg |
|------|----------|----------|-------------|---------|---------|-----|-----|
| 42   | +0.1209  | +0.0959  | +0.0250     | +0.0959 | +0.0709 | null | null |  <!--derived-->
| 43   | +0.1209  | +0.0250  | +0.0250     | +0.0959 | +0.0000 | null | null |  <!--derived-->
| 44   | +0.0750  | +0.0709  | +0.0250     | +0.0500 | +0.0459 | null | null |  <!--derived-->
| 100  | +0.0750  | +0.0250  | +0.0250     | +0.0500 | +0.0000 | null | null |  <!--derived-->
| 101  | +0.1209  | +0.0459  | +0.0250     | +0.0959 | +0.0209 | null | null |  <!--derived-->
| 102  | +0.1209  | +0.0250  | +0.0250     | +0.0959 | +0.0000 | null | null |  <!--derived-->

The neutral baseline `tone_lesion` = +0.0250 on ALL 6 seeds (a stable, reproducible neutral tone floor). `gap_pos`  <!--derived-->
is POSITIVE on every seed; `gap_neg` is ≥ 0 on every seed (never the required negative).

## Why NO-GO: a POSITIVE-ASYMMETRIC, lexically-localized effect (the scientific content)

Two secondary reads (reported, NOT the gate) explain the NO-GO and are consistent across all 6 seeds:

- Under a SEM band δ=0.0426 (δ/√10, the natural noise scale for a mean — reported to show the effect is not  <!--derived-->
  merely a band artifact): the POSITIVE direction is `correct` on ALL 6 seeds (positive mood reliably raises the
  free reply's tone, visible even to the disjoint independent lexicon); the NEGATIVE direction is `wrong` on 2
  seeds and `null` on 4 — negative mood never yields a negative tonal shift on any seed under any band.
- Boosted-word diagnostic (net signed WARRINER-word count — CIRCULAR by design, mechanism check only): pos-minus-neg
  sign is +1 on all 6 seeds; `gap_pos−lesion` in WARRINER words is consistently positive (+0.0045..+0.0109) while  <!--derived-->
  `gap_neg−lesion` ≈ 0 (±0.0006). So even at the mechanism level the negative mood surfaces essentially NO extra  <!--derived-->
  negative words.

<!--derived-->
The affect organ produced correctly-signed differentials (pos +0.038, neg −0.038, lesion clamped 0.0), so the
mechanism is doing the right thing; the ceiling is the DECODE ROUTE: the additive logit bias can only surface words
that are already near the top-1 margin, and in the linattn mouth's wiki-descriptive free-gen distribution positive
evaluative words (nice/good/great…) sit near the margin while negative words do not — so positive mood can nudge
tone up but negative mood has nothing near-margin to pull down. The affect→tone coupling over open output is
therefore real and directional for POSITIVE mood, absent for NEGATIVE mood, and lexically localized to the boosted
lexicon (weak even there) — it does NOT generalize to a broad, symmetric, independent-lexicon tonal shift. The
phase4 single-seed byte-diff GO was real but did not constitute a directional tonal effect; the directional /
independent ruler (the right ruler) is what exposes the asymmetry.

## Banked methods (the wall carries its biology-first surpass — HARD RULE 2)

The strict-disjoint independent-lexicon directional open-output ruler at this operating point (live valence ~0.16,
additive decode bias, wiki-style free-talk prompts) is BANKED. Next methods, biology-first, to make negative-mood
tone load-bearing over open output: (1) the brain-based `BRAIN_WKV_MOUTH_AFFECT_NEURAL` neuromodulator-driven
coupling (the tracked shortcut burn-down) which acts on the read mechanism rather than post-hoc logits and could
lower negative-word thresholds; (2) a coupling that shifts the decode DISTRIBUTION (e.g. temperature / threshold
modulation) rather than an additive top-margin bias, so negative words can enter; (3) affect-laden prompt domains
where negative words are near-margin (not neutral wiki description); (4) the eventual real-Qwen mouth where affect
steers via the prompt (a separate, deferred rung).

## Honesty boundary + shortcut status

The expressed tone read-out is a FUNCTIONAL measure — the free reply's word choice tracks (for positive mood) the
spiking affect signal; never a felt/phenomenal claim. `_apply_affect_bias` remains HOST decode-time arithmetic over
an already-neural valence (a tracked shortcut); the brain-based target is `BRAIN_WKV_MOUTH_AFFECT_NEURAL`
(out of scope here).

## Reproduce

```
CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy python -m research.runners._lbf_affect_tone_open_output_derisk \
    --controller --parallel 2 --memcap-gb 10           # 36 CPU arms, ~2h at 2-parallel
CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy python -m research.runners._lbf_affect_tone_open_output_derisk --score-only
CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy python -m research.runners._lbf_affect_tone_open_output_derisk --selftest
```

Caveat (honest): the attribution control decoupled VALENCE but not AROUSAL, so ctrl_pos/ctrl_neg raw replies were
not byte-identical; the control's directional TONE gap was nonetheless < δ on all 6 seeds, so attribution holds.
The directional NO-GO is independent of the control.
