# WKV cortex PHYSICALLY MERGED onto ONE bridge — forward-identical to the two-bridge faculty (GO)

**Date:** 2026-07-20 · **Status:** GO (seeds 42/43/100, byte-exact) — the WKV cortex's TWO internal bridges (the
`cp_ssm_state` read-out + the RF spike-encoder) are physically consolidated onto ONE `SimulationBridge` with two
regions, and the merged faculty's forward is BYTE-IDENTICAL to the stock two-bridge `OnBridgeWKVFaculty` (accumulated
state `0.000e+00`, next-token logits `0.000e+00`, greedy generation identical). NO `sim/` edit.

## What this builds (owner end-goal)

"Fully closing all gaps INHERENTLY means fully-spiking, one brain, single shared substrate." The WKV cortex itself
ran on TWO bridges (`self.b` = the ssm-state read-out + `self._rfb` = the RF spike-encoder). This merges them onto
ONE bridge — the first PHYSICAL consolidation step (the prior de-risks proved the co-residence risks byte-clean;
this realizes the merge on the real faculty forward).

## The structure

ONE `SimulationBridge`, two brain-regions:
- **`chan`** (region 0, `2*D` neurons) — holds `cp_ssm_state`, the WKV leaky read-out state. `read_idx` maps into it
  (region 0 ⇒ same indices as a chan-only bridge, so nothing downstream changes).
- **`encoder`** (region 1, `2*D` RF oscillators) — the spiking input encoder, driven by the masked
  `rf_kick(neuron_mask=encoder) + rf_resonate_steps` loop (NOT `_run_one_simulation_step`, which would run the ssm
  block and corrupt the WKV state — the reason the merge needs the masked resonate path, proven byte-identical in the
  encoder-equivalence de-risk).

Per token: encode the input on the `encoder` region (masked resonate → read phases → decode) → write `cp_ssm_inject`
at `read_idx` → `_run_one_simulation_step` (the ssm block advances `cp_ssm_state`; the Izhikevich step touches the
encoder region's `v`/`u`, harmless — re-kicked next token) → read `cp_ssm_state`.

## Result (`_gap_wkv_onebridge_merged_derisk.py`, seeds 42/43/100)

Merged (chan=512 + encoder=512 on ONE bridge) vs stock two-bridge faculty, token stream "the penguin can not fly the
owl can fly the":
- **accumulated state max|err| = `0.000e+00` (byte-clean) — all seeds.**
- **next-token logits max|err| = `0.000e+00` (identical) — all seeds.**
- **greedy generation merged == two-bridge: True** — both emit `you help me get the cheese and milk` (D=256 grounded-ft
  ckpt), all seeds.

Byte-exact (not merely close) because the WKV state reads `cp_ssm_state` (`lam*s + (1-lam)*inject`, NO firing → does
not depend on the neuron thresholds the extra region perturbs) and the encoder reads independent-oscillator RF phases
(no synapses, no thresholds). CI: `tests/test_wkv_onebridge_merged.py` (GPU + ckpt, else skip).

## Read-out — honest scope

- **⇒ the WKV cortex (ssm read-out + RF spike-encoder) runs on ONE `SimulationBridge`, forward-identical to the
  two-bridge faculty.** The faculty's own bridge-count drops 2 → 1; the two regions co-reside byte-clean.
- **This is the WKV-internal merge, not yet the FULL single substrate.** The full end-goal build adds the composer
  (RF phasor — already proven co-resident with the ssm read-out in the crux de-risk) + the on-bridge learning (the
  delta rule over `cp_ssm_state` — already on this same bridge) so a whole grounded turn (comprehend → reason →
  spiking render) runs on ONE bridge. Every co-residence piece is now byte-clean; the remaining work is wiring the
  composer's fact-store region onto this same bridge object + running the De-risk-5 turn end-to-end on it.
- **Next:** (1) add the composer as a third region on this bridge + run a grounded turn end-to-end on the single
  bridge; (2) the on-bridge fluency THROUGHPUT lever (batched stepping) to reach the off-bridge ppl ~40 live.

Runner: `_gap_wkv_onebridge_merged_derisk.py` (`--seed`, `--n-tokens`, `--ckpt`).
