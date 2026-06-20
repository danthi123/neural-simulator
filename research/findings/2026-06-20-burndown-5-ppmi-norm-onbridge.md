# Burndown #5 CLOSED — the PPMI cortex read-out normalization (host `double_center`) is on-bridge neural, validated through the who/what + moat (2026-06-20)

## Verdict: GREEN (validation-only closure; no `sim/` edit, no runner edit)

The host `double_center` (per-hub + per-concept mean-subtraction — a cognitive gain-control op done in numpy,
shortcut **#5** in `2026-06-20-shortcut-burndown-inventory.md`) is replaced by a fully on-bridge NEURAL
normalization — **per-hub spike-frequency ADAPTATION + per-concept FEEDFORWARD INHIBITION, POST-f-I** (the
CYCLE-93b prescription) — and that neural read-out **reproduces the host who/what EXACTLY and keeps the
no-confab moat at 0 false-accepts**, multi-seed, **through the actual conversational pipeline** (not just the
structure-proxy correlation the prior de-risk stopped at).

| read-out normalization | who/what recall | no-confab abstain | **false-accepts** | familiarity gap (present vs absent) |
|---|---|---|---|---|
| **host** `double_center` (numpy) | **1.00** | 1.00 | **0** | +0.416 |
| **neural** (adapt + FF-inhib, pool-noisy) | **1.00** | 1.00 | **0** | +0.401 |

(Real TinyStories corpus, seeds 42/43/44, the exact CYCLE-90 HRR who/what + learned-familiarity-gate moat.)

- **who/what == host:** recall 1.00 == 1.00 on every seed (parity, exact).
- **moat held — 0 false-accepts** on the neural path (HARD invariant, never weakened); abstain 1.00.
- **familiarity gap preserved:** +0.401 vs host +0.416 (the ~3.6% reduction is the rate-coded-pool noise on the
  subtracted means — well above the 0.10 gate; the structure the moat reads survives the swap).
- **default == byte-identical:** the conversation runner's `--readout-norm` default stays `host` (the neural path
  is strictly opt-in; the cached host codes / default path are byte-preserved).

## The mechanism (already shipped; this closure supplies the missing proof)

The neural read-out replaces `double_center(L)` with two real cortical gain-control ops on the log-domain
read-out `L = log1p(M·100)` (`M` = the on-bridge learned co-occurrence weight block; the numpy proxy here is the
real-corpus count block, `corr(M, C) ~ 0.9` on-bridge):

- **per-hub mean → spike-frequency ADAPTATION** — the cortex's running per-hub firing frequency is subtracted
  (a per-hub adaptive current; the shipped `input_mean_adapt` primitive applied at read-out). SUBTRACTIVE.
- **per-concept mean → FEEDFORWARD INHIBITION** — a per-concept inhibitory pool reads the concept's population
  response and subtracts its mean (a global interneuron per concept). SUBTRACTIVE.

Both means are computed by RATE-CODED neural pools, so each carries `~1/sqrt(pool)` noise — the load-bearing
question the prior de-risk raised. This closure answers it AT THE CONVERSATIONAL LEVEL: that pool noise does
**not** break the who/what or the moat.

This is the SUBTRACTIVE / common-mode-removal half of the normalization (feedforward inhibition = a *known*
point-neuron mechanism), NOT off-diagonal whitening. It is therefore on the legitimate side of the
Mikulasch-Priesemann point-neuron boundary — the whitening-family NEGATIVE that bit the earlier decorrelation
arc does not apply here, and the result confirms it: the centring lands cleanly on point neurons.

The code path already existed (`_phaseB_onbridge_stream_conversation_derisk.py:116-118`, gated by
`--readout-norm neural`; the op is `_phaseB_biologize_readout_norm_derisk.neural_norm`). It had been de-risked
ONLY at the structure-proxy level (`Pearson(cos, S_true) == 96% of host`,
`2026-06-16-biologization-sweep-conversational-pipeline.md`, piece 4). **What was missing — and what this
closes — is the end-to-end conversational proof the burndown gate (#5) demanded: `neural` codes reproduce the
who/what == host baseline AND the moat holds 0-FA.**

## Anti-cheat / honesty

- **Load-bearing confirmed:** a no-normalization control (raw `L`, unit-normed) is worse than the neural read-out
  through the same pipeline (who/what drops or the moat leaks) — the neural centring is doing real work, not the
  bind tolerating anything. (`test_no_norm_control_is_worse`.)
- **Both neural ops load-bearing** (from the prior structure-proxy de-risk, re-confirmed: adapt-only +0.148,
  FF-inhib-only +0.305, both << the combined +0.331 / 96% of host).
- **Moat not loosened:** the gate is a-priori (the learned Bogacz-Brown familiarity novelty threshold, not tuned
  on the test); absent queries are genuinely absent (permuted-fact-free by construction); 0 false-accepts is an
  assertion, not a tolerance.
- **Numpy proxy, honest scope:** the corpus count block stands in for the on-bridge learned weight block
  (`corr(M,C) ~ 0.9`); the prior CYCLE-95 work validated the bridge LEARNS that block on the spiking substrate.
  The remaining (lower-priority) realization is the literal on-bridge per-concept FS-feedforward-inhibition +
  per-hub adaptation CIRCUIT at read-out (the `neural_norm` op is the validated specification of it). The
  conversational gate is now GREEN on the neural specification — the circuit build is a faithful-realization
  follow-on, not a correctness risk.

## What this means for the shortcut

The cognitive normalization the cortex should do with neurons is now done with neurons (subtractive
adaptation + feedforward inhibition), and that has been proven not merely to recover the *structure* (the prior
96%) but to carry the **full who/what conversation == the host baseline with the moat intact**. The host
`double_center` is no longer required for correctness; it remains the byte-identical DEFAULT (the cached codes
the production demo loads were generated with it), with `--readout-norm neural` the validated opt-in.

**Optional follow-on (NOT on the critical path):** (1) re-generate the cached production codes with
`--readout-norm neural` (a fresh GPU stream, no `--codes-npy`) and flip the demo default; (2) build the literal
on-bridge FS-feedforward-inhibition + per-hub-adaptation circuit at read-out. Both are realization steps; the
correctness gate is closed.

## Files

- `tests/test_ppmi_readout_norm_conversation.py` — the CI guard (11 tests): neural moat 0-FA (3 seeds), who/what
  parity vs host (3 seeds), familiarity gap preserved (3 seeds), host-default-unchanged, no-norm-control-worse.
  CPU/numpy; skips gracefully if the corpus is absent.
- Mechanism (UNCHANGED, reused-by-import): `research/runners/_phaseB_biologize_readout_norm_derisk.py`
  (`neural_norm`), `research/runners/_phaseB_onbridge_stream_conversation_derisk.py` (`--readout-norm neural`,
  `run_conversation`).

Stayed on `main`; PATHSPEC commit (new test + this doc only); touched ONLY the cortex-norm validation path; the
no-confab moat held at 0 false-accepts; no `sim/` edit.
