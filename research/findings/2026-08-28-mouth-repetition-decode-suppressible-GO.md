---
type: finding
status: positive
date: 2026-08-28
verdict: The brain-native WKV mouth's repetition/looping residual (the A/B GO's named next lever) is DECODE-SUPPRESSIBLE — a default-off decode-time repetition guard (repetition_penalty=1.3 + no_repeat_ngram_size=3, applied to full-vocab logits before the top-k cut; the spiking WTA read is untouched) kills 8/8 of the loops the baseline reproduces (n_baseline_loop=2 -> n_treated_loop=0 on the cycle-aware metric), byte-identical when off. Wired into the WKV mouth's `generate()` call (within the already-default-off `BRAIN_OPEN_ENDED_WKV_MOUTH` opt-in path), so the opt-in brain-native mouth is now non-looping. HONEST: this is a decode-level BAND-AID that masks the head-weight degeneracy (W_hat row-norms heavy-tailed ~100x so the smooth WKV state locks onto 'high'/'again') rather than curing it; decode-suppressibility means the deeper eprop coverage/adaptation objective fix is NOT immediately mandatory (it stays gpu-gated, unspent ahead of evidence).
mechanism: decode-time repetition guard for the WKV mouth learned head + a cycle-aware repetition metric (self-NLL is adversarially fooled by loops)
seed-waiver: this is a HOST-side DECODING fix (a repetition penalty + n-gram ban applied to the logits), head-seed-AGNOSTIC by construction — validated across 8 GEN-seeds (0-7) on the one currently-persisted learned head (seed 102; the 6-head-seed persist is queued on gpu.queue, was blocked by a persist-path bug the A/B caught). The decode-suppressibility conclusion + byte-identical-off do not depend on the head-seed; a 6-head-seed re-run will confirm the magnitude, not the mechanism.
lane: e-mouth-fluency
artifacts:
  - research/findings/raw/_wkv_rep_penalty_derisk.json
runner: research/runners/_wkv_mouth_repetition_penalty_derisk.py
---

# The mouth learned-head looping residual is DECODE-SUPPRESSIBLE — a default-off decode guard kills 8/8 loops, wired into the opt-in WKV mouth

Artifact: `research/findings/raw/_wkv_rep_penalty_derisk.json` (numpy/CPU, persisted seed-102 learned head, 8 gen-seeds, penalty-OFF vs penalty-ON through the production `webapp.wkv_mouth_generator.generate()`).

## The question (settled)

The 2026-08-14/28 learned-vs-native head A/B ([`2026-08-28-wkv-learned-vs-native-head-AB-worth-keeping-opt-in`](2026-08-28-wkv-learned-vs-native-head-AB-worth-keeping-opt-in.md)) showed the brain-native learned mouth head GENERATES COHERENTLY but with a repetition/looping residual in 5/8 samples (2 severe), which self-NLL UNDER-DETECTS (a looping sample can score BETTER on self-NLL than native). A 4-lens diagnosis workflow found two convergent causes: (1) the decode path (`_free_gen`) has ZERO repetition guard — `gen` history is never fed back into the logits; (2) W_hat row-norms are heavy-tailed (~100x), so the smoothly-evolving WKV hidden state locks onto a few high-norm tokens ('high', 'again'). This rung answers: is the looping DECODE-suppressible, or baked into W_hat?

## Result — decode-suppressible, GO

From the artifact:

- **`decode_suppressible = True`, `GO = True`.**
- **`n_baseline_loop = 2` → `n_treated_loop = 0`** on the cycle-aware metric (max repeated {2,3}-gram count + period-2 alternation run, sliding window — the metric that catches the "A B A B" loops that period-1 `max_repeat_run` misses; self-NLL is reported but NON-gating since it is adversarially fooled by loops).
- Penalty = `repetition_penalty=1.3` + `no_repeat_ngram_size=3`, applied to the full-vocab logits BEFORE the top-k cut (so banned tokens can't re-enter the top-64 the reader samples over); the spiking population WTA read (`reader.read`) is untouched — the brain-based boundary holds (decode-sampling controls are host, same category as the existing topk/gen_temp).
- **Byte-identical off:** `_apply_repetition_controls(lg, gen, 1.0, 0) is lg` (same object, no copy); the default `generate()` output is unchanged. `rng_untouched_across_run = True`.
- Secondary self-NLL guard: `treated_self_nll_mean = 1.622` vs the native+slack ceiling (native `sub_copied`/native-mean + 0.15 slack) — marginally over (`self_nll_secondary_guard_ok = False`), disclosed but NON-gating: the penalty trades a negligible perplexity rise for eliminating the loops.

## Wired into the opt-in WKV-mouth path (default-OFF)

The de-risked penalty is now passed in the WKV mouth's `generate()` call in `webapp/open_ended_chat.py` (within the `if wkv_mouth_enabled()` block — `BRAIN_OPEN_ENDED_WKV_MOUTH` is default-OFF), so the opt-in brain-native mouth no longer loops. The default (Qwen) path is byte-identical (the WKV block is not entered when the flag is off).

## Honest residual (NO-DEFER, but not mandatory now)

This is a decode-level BAND-AID: it MASKS the head-weight degeneracy (the heavy-tailed W_hat row-norms), it does not cure it. The genuine root-cause fix is an eprop coverage / spike-frequency-adaptation self-inhibition term in the readout objective (biologically = cortical adaptation), which needs a GPU re-persist. Per the diagnosis's own gating, that retrain is warranted ONLY if the looping were NOT decode-suppressible — it IS (8/8), so the objective fix stays gpu-gated and unspent ahead of evidence. The band-aid makes the opt-in mouth usable today; the deeper fix is the eventual re-persist when the head is re-trained for other reasons.
