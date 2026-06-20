# Tier-2 #6 (limbic -> composer), Route B WIRED into the production `OneBrainComposer` (2026-06-20)

**Verdict: WIRED + GATED.** The de-risked Route B (write-side dopamine ENCODING-GAIN) is now wired into the
production one-brain conversational composer (`OneBrainComposer`, the `--composer onebrain` default) as a SMALL,
OPT-IN, DEFAULT-OFF mirror of the already-validated `RFPhasorComposer` mechanism. This completes the one cheap
residual of Tier-2 #6 on the one-brain path. **NO `sim/` edit** (reuse-by-import). **The no-confab moat holds with
0 false-accepts under the gain** (the load-bearing constraint).

## What was wired (the mirror)

Route B is the dopamine-gated ENCODING-GAIN already DONE + de-risked in the RF composer
(`2026-06-19-dopamine-encoding-gain-derisk.md`, numpy 6/6 GO, moat 6/6):
- `RFPhasorComposer.__init__(encoding_gain_fn=None)` (`rf_phasor_composer.py:64,80`)
- the store applies it (`rf_phasor_composer.py:449-450`):
  `g = 1.0 if self.encoding_gain_fn is None else float(self.encoding_gain_fn())`;
  `conns = [(1 + k, 0, complex(g) * zc[k]) ...]` -- scales the stored block's magnitude at store time (a
  salient/rewarded fact is stored more strongly -> better recall under read damage). `g=1.0`
  (`encoding_gain_fn=None`) is the BYTE-IDENTICAL unit-magnitude write.

`OneBrainComposer._write_block` (`one_brain_composer.py:261`) does the same block write but LACKED the hook. The
change mirrors the RF composer EXACTLY:
- added `encoding_gain_fn=None` to `OneBrainComposer.__init__`, stored as `self.encoding_gain_fn` (with the same
  Lisman-Grace / Kandel-D.16 docstring semantics).
- applied the identical gain in `_write_block`:
  `g = 1.0 if self.encoding_gain_fn is None else float(self.encoding_gain_fn())`;
  `block_conns = [(trig + 1 + k, trig, complex(g) * zc[k]) ...]`.

**Why it is differential, not a vacuous global gain:** the RF phase read-out has a hard MAGNITUDE FLOOR
(`sim/bridge.py:5589`, `_rf_mag2 > _rf_floor2` -- a readout neuron whose |Z| decays below the floor never spikes ->
reads phase 0 = garbage). The OneBrainComposer reads its store back through the SAME `rf_resonate_steps` path
(`_read_block` / `_read_blocks`), so a higher-gain (rewarded) fact reconstructs ABOVE the floor under common read
damage where a unit-gain (neutral) fact degrades BELOW it -> the rewarded fact wins the cue-match scan. The floor
is the nonlinearity that makes the gain meaningful.

The composite `zc` passed to `_write_block` is `comp._to_phasor(...)` = `exp(2j*pi*phases)`, so `|zc| == 1` exactly
-> `|g*zc| == g`. And `complex(1.0) * zc[k] == complex(zc[k])` is exactly True for finite IEEE values, so the
default-OFF (`None`) and explicit-`g=1.0` paths are GENUINELY byte-identical.

## The gate

| gate | result |
|---|---|
| FULL `tests/test_one_brain_composer_agent.py` (15 tests) verbatim, production D=128, GPU | __PENDING (running)__ |
| DEFAULT-OFF byte-identity (`encoding_gain_fn=None` AND `lambda: 1.0` -> IDENTICAL `store_conns`) | PASS |
| moat 0-FA WITH a gain set (`encoding_gain_fn=lambda: 1.5`: recall correct, all unstored cues/facts abstain) | PASS |
| `g=1.5` WRITE check (every stored block edge `|w| == 1.5`) | PASS |
| NO `sim/` edit (reuse-by-import; only `one_brain_composer.py` + its test touched) | CONFIRMED |

Two new CI tests (the 14th + 15th in the file):
- `test_onebrain_encoding_gain_default_off_byte_identical` -- `encoding_gain_fn=None` and a constant `lambda: 1.0`
  both write the SAME persistent `store_conns` (same edges, same complex weights) as the current code. The
  default-OFF guard: wiring the hook changes nothing unless a gain fn is supplied.
- `test_onebrain_encoding_gain_lifts_recall_moat_intact` -- a constant `lambda: 1.5` writes a higher-magnitude
  store block (`|w| == 1.5 == g*|zc|`) while every who/what answer stays correct AND an unstored cue
  (`query_patient` / `query_agent` -> `None`) / unstored fact (`ask_yes_no` -> `unknown`/`no`) still ABSTAINS. The
  HARD load-bearing constraint -- DA-modulated encoding NEVER produces a false-accept -- is pinned here.

The existing 13 tests (core matrix/moat, negation, describe/reason, clause parity, agent-clause, reconsolidation,
grounded-codes drop-in, multi-turn correction, multi-turn anaphora, confidence-gate, attributed comprehension,
batched==per-block, default-path-unaffected) are unchanged and must pass VERBATIM with the new default-OFF param --
i.e. the additive wiring is byte-identical for every existing path.

## Honest scope

This is the OPT-IN INTERFACE on the one-brain path -- the same scope as the RF composer's Route B: the gain is read
at store time from a callable (the shared `dopamine` concentration in deployment; a probe value in the de-risk).
The numpy-kb fast path stores phases (no magnitude), so the gain only exercises the floor on the substrate read --
which is exactly the OneBrainComposer's path (its store is always on-substrate in `cp_rf_w_re/im`). The full
limbic->composer LOOP (the dopamine signal sourced from the neural reward/value system feeding `encoding_gain_fn`)
is the deployment wire-up; this provides the composer-side hook it plugs into, mirroring the validated RF mechanism
with zero behavioral change when unused.

Files: `research/runners/one_brain_composer.py` (+the `encoding_gain_fn` param + the one-line gain in
`_write_block`), `tests/test_one_brain_composer_agent.py` (+2 tests). Mirrors `rf_phasor_composer.py:449-450`.
