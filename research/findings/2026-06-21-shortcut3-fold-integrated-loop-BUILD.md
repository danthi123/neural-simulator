# Shortcut #3 BUILD — fold the spiking K-way sequencer into the production composer, retire the host first-match `_scan` (2026-06-21)

**The build.** Per the committed plan (`docs/plans/2026-06-21-shortcut3-fold-host-scan-to-spiking-sequencer-plan.md`,
commit 94a5e237), the production `OneBrainComposer`'s **host first-match cue-match loop** — the Python `for / if /
return` over stored fact blocks that decides which stored fact answers a who/what query (and answer-vs-abstain) — is
retired in favour of the **validated spiking K-way sequencer** (a gated-disinhibition match cascade + a BG first-match
priority WTA), behind an opt-in flag. This closes a brain-based-only host shortcut: the cue-match comparison + the
answer/abstain routing are now neurons firing, not host bookkeeping.

**Reuse-by-import, NO `sim/` edit** (the sequencer reuses already-shipped bridge primitives). The per-block spiking
reconstruction (`_read_blocks`) was already on-bridge; the residual host op was only the first-match loop.

---

## The flag + the helper

**Flag** (`OneBrainComposer.__init__`, default-OFF = byte-identical = the host-`_scan` oracle, mirroring
`enable_spiking_cleanup`):

```python
OneBrainComposer(..., integrated_loop=False, sequencer_match_thresh=0.06,
                 sequencer_gain=0.11, sequencer_sigma=1.0, sequencer_input_gain=1.0)
```

The op-point is the validated K=32 surpass op-point (`2026-06-21-shortcut3-K32-capability-surpass.md`):
`match_thresh=0.06`, `gain=0.11`, `sigma=1.0`, `input_gain=1.0`, `retreat=divnorm`.

**Helper** `_seq_block(agent, action)` — returns the SELECTED block index (or `None` = abstain). OFF → the host
first-match loop (byte-identical). ON → the spiking decision: lazily build + cache the K-way sequencer fabric
(`build_sequencerK_bridge`, S0) + the divnorm score bridge (`build_divnorm_score_bridge`, S5) per K; derive the
per-block decoded-line drives from the composer's OWN on-bridge cleanup scores (`block_cleanup_scores` +
`make_block_drives`, S2) — rebuilt only on a dirtied/grown store (`_seq_dirty`); run the spiking decision
(`run_sequencerK_with_drive` → `decision_to_block`). An absent cue WORD abstains before the sequencer (the moat).
Imports are lazy (`_seq_imports`) so a flag-OFF composer never imports the de-risk runners.

## The wire (build-1 scope)

The (agent, action) **hot-path** sites delegate to `_seq_block` when `integrated_loop=True`:

| site | method | routed |
|---|---|---|
| `:609` | `query_patient(agent, action)` | ✅ via `_seq_block` (then reads the kb patient TYPE + decodes the selected block) |
| `:623` | `ask_yes_no(agent, action, patient)` | ✅ via `_seq_block` (then the patient-equality + polarity body read → yes/no/unknown) |
| `:698` | `_find_cued_block(agent, action)` | ✅ via `_seq_block` → reconsolidation (`update_on_mismatch`) + `query_chain`/`reason_chain` inherit the spiking decision |

**Documented build-1 bounded follow-ons (kept on the host read, still abstaining via the oracle):**
- `query_agent(action, patient)` (`:618`) — a **swapped-cue** cascade (the sequencer is built for the (agent, action)
  cue). Named follow-on; widening to (action, patient) is a clean second cascade.
- `render_fact(agent)` / `describe` (`:635`) — an **agent-only 1-role** cascade. Named follow-on.
- `count_facts` stays host (it counts, not routes — not the routing control flow).

This is an honest partial conversion, not a moat hole: the production who/what + yes-no + reason hot paths go fully
spiking; the residual ops still abstain correctly (the host read is the test oracle).

---

## The HARD GATE — every claim gated, all green

### (1) Answer-identity + (2) moat + anti-cheats — composer-API de-risk, OVERALL GO

`research/runners/_phaseB_onebrain_integrated_loop_fold_derisk.py`: two composers per seed (host oracle
`integrated_loop=False` vs `integrated_loop=True`) on the K=32 production fact set (V=72), the full who/what + moat
battery (`query_patient` / `ask_yes_no` / `update_on_mismatch` abstain) + the LESION / permuted / NO-DIVNORM-raw
anti-cheats. numpy-CPU, V=72, D=128, seeds 42/43/44, `match_thresh=0.06`:

| K | ==host (answer-identity) | moat (FA_total) | recon-abstain | lesion-safe | permuted | raw-fails | verdict |
|---|---|---|---|---|---|---|---|
| 2 | 3/3 | 3/3 (**0**) | 3/3 | 3/3 | 3/3 | 3/3 | **GO** |
| 4 | 3/3 | 3/3 (**0**) | 3/3 | 3/3 | 3/3 | 3/3 | **GO** |
| 8 | 3/3 | 3/3 (**0**) | 3/3 | 3/3 | 3/3 | 3/3 | **GO** |

`OVERALL: GO`. The `integrated_loop=True` composer is **answer-identical** to the host-`_scan` oracle on the full
who/what + abstention matrix (every `is None` / `"unknown"`); the **moat holds 0-false-accept** at the validated
`match_thresh=0.06` (never traded); the anti-cheats behaved (LESION→abstain, permuted→cyclic-shift, NO-DIVNORM
raw→fails so the divnorm is load-bearing). Finding: `2026-06-21-shortcut3-fold-integrated-loop-derisk.md`.

### (3) 320-scale (V=320, K=32, GPU)

_See the "320-scale" section below (the production-tier confirmation)._

### (4) Default-OFF byte-identical — the shipped suites pass verbatim

- `tests/test_one_brain_composer_agent.py` (11 tests, GPU): _see below_.
- `tests/test_consolidated_320_conversation.py` (CPU, rf path, `integrated_loop` default-OFF): **2 passed**
  (byte-unaffected — the rf production path does not route through the sequencer).

### CI guard

`tests/test_onebrain_integrated_loop_fold.py` extends the suite: `integrated_loop` defaults False + the OFF path is
byte-identical (Task 1); `_seq_block` selects the same block index as the host first-match on present cues + abstains on
absent/cross (Task 2); `query_patient` / `reason_chain` / `update_on_mismatch` route through `_seq_block` (spied) ==
host with the moat intact (Task 3); `ask_yes_no` affirmative→yes / negated→no / wrong-patient→unknown / unstored→unknown
== host (Task 4). Multi-seed (42/43), numpy-CPU, the de-risk's V=72 vocab at D=128 (the validated op-point + margin).
**All green** (12 tests, 165–202 s).

---

## NO `sim/` edit — confirmed

`git diff --stat -- sim/` is empty across the build (every commit staged with a narrow pathspec; verified
`git diff --cached --name-only -- sim/` empty at each commit). Every primitive the sequencer uses
(`couple_gate_to_pool`, `_apply_gate_couplings`, `set_transmission_gate`, `cp_external_input_current`,
`enable_vectorized_gate_couplings`, `input_divisive_norm`) was already shipped.

## Commits (on `main`, both remotes)

- Task 1 — flag + lazy plumbing (default-OFF byte-identical)
- Task 2 — `_ensure_sequencer` + the spiking `_seq_block` branch
- Tasks 3+4 — route `query_patient` / `_find_cued_block` / `ask_yes_no` through `_seq_block`
- Task 5 — composer-API answer-identity de-risk = OVERALL GO (K∈{2,4,8}, 3 seeds)
- Task 6 — 320-scale GPU confirmation _(below)_
- Task 7 — CI guard + the combined suite
- Task 8 — opt-in `--integrated-loop` through the demo + `BrainConversationalAgent` (default-OFF)
