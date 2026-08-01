---
type: finding
status: contributing
date: 2026-08-01
mechanism: recombinative-replay
lane: H-gap5
---

# gap#5 RANK 3 (imagination) — the gamma-WTA timing method, applied to recombination via the EXTRACTED mean transition matrix, sits at the geometric chance level: a characterized boundary, cause located

**One-line:** The RANK 3 research gate (`2026-07-22-gap5-RANK3-imagination-recombinative-replay-research-gate.md`) named
**theta/gamma phase-organized replay** as the next method for novel recombination at a shared branch node, after both
direct-composition spiking methods hit a co-ignition boundary. This applies that method in its cheapest form — the
**gamma-WTA + post-fire silence** primitive that is 3/3-seed GO for RANK 2 forward order
(`2026-07-22-gap5-gamma-WTA-timing-fixes-replay-order-cheap-GO.md`) — over the **extracted between-assembly transition
matrix** of the shared-node topology. **Verdict: NEGATIVE, and the cause is located.** The method reads at its
**2/3 geometric chance** because the extracted mean transition weight at the shared hub is **undifferentiated** (learned
successors ≈ unlearned out-edges, ~20 to everything) and **INERT to 4× more chain encode** (`chain_fwd` 24→96 left it at
~18–21). Per THE LAW this is a **method verdict on the extracted-matrix proxy, not the capability** — it re-confirms and
extends the gate's boundary, and sharpens the named next method to a **full phase-gated SPIKING replay** (postsynaptic
thresholding on the actual potentiated-synapse subset, which the diluted mean-weight proxy discards).

## What was built (NO `sim/` edit; reuse-by-import; numpy-deterministic default)
`research/runners/_gap5_gamma_recombination_derisk.py` — composes the two banked primitives:
- the shared-node topology (`_prepare_sequence` + additive `chain_edges`: `A→B→C` + `X→B→Y`, B=assembly 1 shared) from
  `_gap5_recombination_derisk.py`; and
- the `_extract_W` between-assembly transition matrix + gamma-organized replay (per gamma cycle: `drive[j] = W[cur][j] +
  noise`; the winner fires; **post-fire silence** removes it) from `_gap5_gamma_wta_replay_derisk.py`.

Starting from a predecessor (A or X), the walk reaches the shared hub B and its next winner is the **B-exit**: to the
stored successor (`A→B→C` / `X→B→Y`) or the **recombined** one (`A→B→Y` / `X→B→C`, never stored as a whole). Metrics:
`learned_exit_frac` (B exits to a learned successor C/Y at all), `recomb_frac` (of learned exits, the recombined
fraction), with the gate's mandated anti-cheats: **NO-SHARED** (`X→D→Y`, B≠D), **NO-NOISE**, **NO-ENCODE**, **SCRAMBLE**.

## Result (GPU cupy, CUBLAS deterministic reductions; seed 42 shown, 6-seed below)
The metric floor is a **geometric chance of 2/3**: after `A→B` silences {A,B}, three candidates remain {C, X, Y} and
**two of them (C, Y) are learned successors**, so a signal-free argmax lands on a "learned successor" 2/3 of the time.

| arm | learned_exit | recomb_frac | note |
|-----|-------------:|------------:|------|
| MAIN (chain_fwd=24) | 0.695 | 0.571 | ≈ chance 0.667 |
| NO-ENCODE | 0.651 | — | ≈ chance (no learned edges) |
| SCRAMBLE | 0.712 | — | ≈ chance (structure destroyed) — but so is MAIN |
| NO-SHARED | — | 0.481 | should be ~0; leaks to chance 0.5 |
| MAIN (chain_fwd=96, 4× encode) | 0.637 | 0.483 | still chance; **W inert** |

- **The learned successor weights are weak and inert:** `W[B→C], W[B→Y]` = 20.8, 23.2 at `chain_fwd=24` and 17.7, 21.0 at
  `chain_fwd=96` — **4× more chain encode did not strengthen them** (BTSP coincidence encode saturates at the hub).
- **MAIN ≈ NO-ENCODE ≈ SCRAMBLE ≈ 0.667**, and **NO-SHARED recomb ≈ 0.5** (not ~0): every arm is at chance. The
  gamma-WTA has **no learned-successor signal to ride**.

## Cause (the DIAGNOSTIC read — an instrument check, not a tuning lever)
The added `_diff_stats` compares the mean **learned** out-edges of B (`B→C`, `B→Y`) against the **unlearned** ones
(`B→X`, `B→A`). Across seeds the ratio is **≈ 1** — `learned_succ / unlearned_out` = **1.17, 1.06, 0.74** (seeds
42/43/44; seed 44's learned edges are even *weaker* than its unlearned ones). The two are indistinguishable. When
learned ≈ unlearned, the extracted mean transition matrix carries **no successor signal**, so the argmax can only read
the geometric chance. The shared **hub dilutes/saturates** the
per-successor transition weight — unlike RANK 2's linear chain, where the same `_extract_W` + gamma-WTA method WORKED
because the adjacent-forward weight was strong (the RANK 2 GO's adjacent chain ≫ skip). **This is an instrument-valid
negative** (the metric is fine; there is genuinely nothing to read), not a metric artifact.

## 6-SEED CONFIRMATION (default config chain_fwd=24, seeds {42 43 44 100 101 102}, GPU cupy, CUBLAS deterministic)
**0/6 GO — every arm at chance, robust across seeds.** Means (range):
- MAIN `learned_exit` **0.631** (0.569–0.676) ≈ chance 0.667; MAIN `recomb_frac` **0.497** (0.446–0.533) ≈ 0.5.
- **NO-ENCODE `learned_exit` 0.671** (0.654–0.695) and **SCRAMBLE `learned_exit` 0.674** (0.532–0.825) — both ≈ MAIN,
  i.e. removing the learned edges or scrambling them **does not lower** the read: MAIN was never above them.
- **NO-SHARED `recomb` 0.499** (0.459–0.546) — should be ~0 for a real branch; it sits at chance 0.5 (the control is
  not clean because the "successor" weights are undifferentiated).
- **DIFF ratio (learned_succ / unlearned_out) 1.14** (0.74–1.44) — learned and unlearned B-out-edges are
  indistinguishable at every seed (seed 44 has learned < unlearned). This is the instrument check: there is genuinely no
  successor signal in the extracted matrix, so the argmax reads chance. A robust, correctly-measured negative.

## Verdict — per THE LAW, a METHOD verdict; the capability is not abandoned
- **Banked failing method:** gamma-WTA + post-fire silence over the **extracted mean between-assembly transition matrix**
  on the shared-hub recombination topology → reads the 2/3 geometric chance; the hub's mean transition weight is
  undifferentiated and inert to 4× encode. Two levers on the one defect (chain_fwd 24, 96) → parameter search stopped
  (research gate).
- **Why the mean-weight proxy is the wrong instrument for the hub:** it averages over all coincidence-masked ca3→ca3
  synapses, diluting the strong potentiated **subset** that a real spiking read would threshold on. RANK 2's linear chain
  had a strong-enough MEAN so the proxy sufficed; the shared hub does not.
- **Named next method (sharpened from the RANK 3 gate's "theta/gamma next-arc"):** a full **phase-gated SPIKING replay**
  where the gamma reset operates on the actual CA3 spiking dynamics and postsynaptic thresholding rides the potentiated
  synapse subset — vehicle `_gap5_spiking_gamma_replay_derisk.py`. A larger bounded build; the mean-weight argmax proxy
  is retired for the hub.
- **gap#5 core unaffected:** completion CLOSED (2026-07-18); replay-boundary SURPASSED 6-seed GO; RANK 1 reactivation
  6-seed GO; RANK 2 forward order gamma-WTA 3/3. RANK 3 imagination remains the open rung — now a boundary characterized
  from a SECOND angle (the timing method fails on the extracted-matrix proxy, cause = the hub's undifferentiated mean
  transition weight), with the spiking method named.

Artifacts: `research/findings/raw/gap5_r4/gamma_recomb_s*.json` + `_gamma_recomb_6seed_gpu.sh`. Runner:
`research/runners/_gap5_gamma_recombination_derisk.py` (no `sim/` edit).
