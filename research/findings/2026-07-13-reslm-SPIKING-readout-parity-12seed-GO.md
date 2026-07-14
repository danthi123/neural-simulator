# The emergent generator's READ-OUT runs ON SPIKES — a spiking one-of-K FS-WTA read-out matches the numpy argmax at parity (12-seed GO)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_spiking_readout_derisk.py` (reuse-by-import `build_fswta_score_bridge`/`fswta_drive`; numpy; NO `sim/` edit).
**Status:** ✅ 12-seed GO (standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12).

## What this closes

The emergent-generation ladder's reservoir is already spiking (EMERGE-82 `OnBridgeLSM`), but its next-token READ-OUT was a numpy linear-softmax argmax — the generator's last non-spiking piece. This converts the read-out SELECTION to spikes: the linear read-out's per-token scores drive a one-of-K FS-WTA Izhikevich bridge (the validated `build_fswta_score_bridge`/`fswta_drive`, the SAME spiking WTA the D3 register uses), and the SPIKING winner is the predicted next token. The read-out WEIGHTS are still learned by the committed LOCAL one-step delta rule (Widrow-Hoff on the clean next-token error, no BPTT, no weight transport); only the argmax is moved onto spikes.

## Result — 12-seed GO

| set | parity (spk == numpy) | numpy_acc | spk_acc | shuffle_parity |
|---|---|---|---|---|
| standard | 0.971 | 0.892 | 0.883 | 0.076 |
| fresh | 0.959 | 0.894 | 0.879 | 0.074 |

GO gate per seed: parity > 0.90 AND spk_acc > 1.5/V (beats chance) AND |spk_acc − numpy_acc| < 0.05 (no accuracy loss) AND shuffle_parity < 0.5. **12/12 GO.**

- **Parity ~0.97** — the spiking FS-WTA read-out agrees with the numpy argmax on ~97% of predictions; the ~3% is the FS-WTA's occasional near-tie tie-break.
- **spk_acc ≈ numpy_acc** (0.88 vs 0.89) — the spiking read-out achieves the SAME next-token accuracy (the ~1% gap is the tie-break cost, not a substrate loss).
- **shuffle_parity 0.075** — driving the FS-WTA with PERMUTED scores collapses agreement to chance, confirming the spiking WTA reads the actual read-out scores (not a generic sorter).

## ⇒ the claim

The emergent generator's next-token read-out is realizable ON SPIKES with negligible accuracy loss — the spiking reservoir (EMERGE-82) + a spiking FS-WTA read-out make the generator fully-spiking-realizable, the read-out weights still learned by a local one-step delta rule. This closes the generator's last non-spiking selection step, on the shared spiking-WTA substrate.

## Honest scope / next

- **Cheap-first + self-contained:** a small reservoir + a toy high-order next-token task (V=12) + the FS-WTA parity — the CORE conversion (spiking argmax parity) before the full-vocab wire (a V=200+ FS-WTA bridge is a bigger bridge; the parity mechanism is vocab-agnostic).
- The read-out produces the ARGMAX next token on spikes; sampling from the full softmax distribution on spikes (for stochastic generation) is a separate rate-code question (the argmax is what generation greedy-decodes).
- NO `sim/` edit. CI guard `tests/test_reslm_spiking_readout.py`.

## Files

- `research/runners/_reslm_spiking_readout_derisk.py`; raw `_spk_readout_{std,fresh}.json`.
- Builds on the reslm read-out (`_emerge_reservoir_lm_derisk` `train_readout`, the local delta rule) + the FS-WTA spiking substrate (`_d3_spiking_attractor_derisk`).
