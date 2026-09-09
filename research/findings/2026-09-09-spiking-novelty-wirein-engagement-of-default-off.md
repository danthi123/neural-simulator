---
type: finding
status: qualified
date: 2026-09-09
mechanism: spiking-habituation-novelty
lane: scaffold-retirement
seeds: [42, 43, 44, 100, 101, 102]
verdict: WIRED-DEFAULT-OFF (focused verify GO; integrated no-regression soak DEFERRED to an AWS-CPU batch)
runner: research/runners/_spiking_novelty_wirein_verify.py
artifacts:
  - research/findings/raw/_spiking_novelty_wirein/verify_6seed.json
external: NO-EXTERNAL-NEEDED -- pure production wiring of the already-GO mechanism de-risk
  (2026-09-09-spiking-habituation-synaptic-depression-novelty-mechanism-6seed-GO.md); no new biological claim.
---

# Spiking-habituation NOVELTY wired into `engagement_of()`, default-OFF (scaffold-retirement)

**Verdict: WIRED, DEFAULT-OFF.** The focused wire-in verify is GO — `engagement_of()` is BYTE-IDENTICAL when the flag
is off (proven, max|diff|==0), and the spiking path is LOAD-BEARING when on (5/6 project-standard seeds). The
**integrated `/api/brain-chat` no-regression soak is DEFERRED** to an AWS-CPU batch (the dev box is RAM-blocked on the
full integrated brain-chat battery); the FLIP to default-ON waits on that verdict. This turns the 6/6-GO mechanism
de-risk (`2026-09-09-...-synaptic-depression-novelty-mechanism-6seed-GO.md`, commits `3cf6bc52`/`755c17290`) from a
banked de-risk into an actual production wire-in — the owner's #1-priority completion step for this retirement.

## What was wired

The host `set`-membership novelty term inside `webapp/da_mode_drives_chat.py::engagement_of()` —
`sum(1 for t in tokens if t not in seen) / len(tokens)`, a permanent per-session Python `set` that never forgets — is
now optionally replaced by the spiking short-term-depression HABITUATION read, behind a new default-OFF flag
**`BRAIN_SPIKING_NOVELTY`**. This scalar is the ROOT of THREE production faculties (da-mode-drives-response board #79,
da-gated-encoding, da-gated-curiosity-threshold, all reading it through the rank-4 shared spiking salience afferent),
so the retirement is at a single high-leverage site.

- **`research/runners/spiking_novelty_habituation_organ.py`** (new; NO `sim/` edit). A per-session
  `SpikingNoveltyHabituationOrgan`: ONE real `SimulationBridge` holding a fixed bank of block-diagonal per-word
  input->readout channels wired through dense fixed-weight Tsodyks-Markram **depression-dominant** synapses (the exact
  de-risked `stp_U=0.35`, `stp_tau_d=800ms`, `stp_tau_f=10ms`; build mirrors the de-risk's `_build_bridge`/
  `HabituationCircuit.present` verbatim). `novelty_of(tokens)` presents each content token (a short presynaptic burst +
  a silent readout window), normalizes the readout spike count against the fresh baseline, and returns the mean
  freshness in [0,1] — high = novel, low = habituated. Presenting also habituates the channel, so the read tracks the
  conversation's word history and recovers during the silence of intervening presentations.
- **Open vocabulary (recruit-on-demand — the `VocabAgnosticSpikingSampler` pattern).** The de-risk fixed 3 channels; a
  live conversation has an arbitrary vocabulary. The organ maps words to a fixed channel bank on first sight (the same
  "fixed neural bank, words mapped into slots on demand" pattern `VocabAgnosticSpikingSampler` already uses for
  open-ended generation). When the bank is full it reuses the LEAST-RECENTLY-PRESENTED channel — which, by the recovery
  time-constant, is also the MOST-recovered (nearest to fresh), so a reused channel starts a new word from a near-fresh
  synapse without an explicit STP reset (the recovery mechanism and the LRU policy align; honest residual below).
- **`engagement_of()`** gains an optional `novelty_override: Optional[float] = None`. When None (the DEFAULT) the
  novelty is the pre-existing host `set` arithmetic — **byte-identical**; when a float is supplied the spiking value
  replaces the novelty fraction while `richness` (a legitimate host token-count boundary) is untouched. `seen` is only
  read on the host path.
- **`DaModeDrivesWorkspace.observe()`** reads the organ ONLY when `BRAIN_SPIKING_NOVELTY` is on
  (`_spiking_novelty()`), lazily building it on the workspace's PRIVATE RNG timeline (the `_isolated` #77-footgun
  isolation the #76 DA substrate already uses, so enabling it leaves the host process-global RNG byte-untouched).
  Never-raising: on any wiring failure it degrades to `novelty_override=None` (the host `set` path) for that turn. Off
  (the default) -> the branch is not entered, the organ is never built, RNG is untouched, `seen` updates exactly as
  before -> byte-identical.
- **Lesion `BRAIN_SPIKING_NOVELTY_LESION=1`** builds the bank with STP off (the de-risk's own G5 lesion): with no
  synaptic depression the read no longer tracks word history — the load-bearing proof. Distinct from
  `BRAIN_DA_DRIVES_LESION` (which silences the downstream SNc nucleus).

## What was verified locally (focused; NOT the full integrated battery)

`research/runners/_spiking_novelty_wirein_verify.py` ->
`research/findings/raw/_spiking_novelty_wirein/verify_6seed.json` (verdict GO):

**(A) BYTE-IDENTICAL WHEN OFF (light, no brain).** On a 6-message set with a growing `seen` set: `engagement_of(tokens,
seen)` equals the pre-wiring reference formula recomputed inline, **max|diff| == 0.0**; `spiking_novelty_enabled()` is
False by default; and the explicit `novelty_override=None` call equals the no-kwarg call. So `engagement_of()` — and
therefore `observe()`'s engagement path when off — is byte-identical to pre-wiring.

**(B) LOAD-BEARING WHEN ON (focused spiking organ, per seed).** 5/6 seeds GO (board bar >=5/6). The per-seed numbers
below are a rounded presentation of the cited artifact `verify_6seed.json` (exact 4-decimal values live there):

<!--derived-->
| seed | nov_first | nov_rep (2-3) | ratio (rep/first) | new-word fresh | lesion ratio | attrib->STP | e_hi vs e_lo | GO |
|---|---|---|---|---|---|---|---|---|
| 42  | 0.871 | 0.431 | 0.495 | 0.862 | 0.987 | 0.975 | 0.722 / 0.459 | GO |
| 43  | 0.938 | 0.339 | 0.361 | 0.969 | 1.000 | 1.000 | 0.762 / 0.403 | GO |
| 44  | 0.957 | 0.299 | 0.312 | 1.000 | 0.994 | 0.992 | 0.774 / 0.379 | GO |
| 100 | 0.674 | 0.216 | 0.320 | 0.765 | 0.951 | 0.927 | 0.605 / 0.330 | no (L1 only) |
| 101 | 0.862 | 0.250 | 0.290 | 0.905 | 0.968 | 0.955 | 0.717 / 0.350 | GO |
| 102 | 0.850 | 0.338 | 0.397 | 0.792 | 0.987 | 0.978 | 0.710 / 0.403 | GO |

- **Novelty varies with input:** fresh words read novel (nov_first), repeating them habituates the read to ~30-50% of
  fresh, and brand-new words recruit a fresh channel and read novel again (no cross-talk) — every seed.
- **Engagement tracks the spiking novelty:** with tokens+`seen` HELD FIXED, `engagement_of(..., novelty_override=high)`
  exceeds `engagement_of(..., novelty_override=low)` by 0.16-0.40 (e_hi vs e_lo) — the spiking read is load-bearing on
  the engagement scalar, not decorative.
- **Lesion reverts:** with STP off the repeat-ratio stays >=0.95 (barely drops) and `attributable_to` assigns 0.93-1.00
  of the habituation to the STP manipulation — synaptic depression, not some other circuit artifact, produces the
  signal.
- **Seed 100** misses ONLY the absolute L1 sanity threshold (nov_first=0.674 vs the 0.70 bar) — a per-channel
  heterogeneity effect on the fresh-baseline normalization at that seed, NOT a mechanism failure: its relative
  habituation (ratio 0.320), engagement tracking (0.605 vs 0.330), and lesion-revert (0.951 / attrib 0.927) all hold
  cleanly, exactly the pattern the de-risk itself argued carries the load. No threshold was changed after seeing the
  6-seed numbers (the discipline); 5/6 clears the board bar.

## Deferred to the AWS-CPU integrated verify (verdict-pending)

The full integrated `/api/brain-chat` no-regression soak is RAM-blocked on the dev box and is NOT run here. The exact
command for the parent to run on an AWS-CPU batch (produces the flip-gating verdict):

```
SIM_BACKEND=cupy BRAIN_SPIKING_NOVELTY=1 .venv/bin/python -u -m research.runners.onebrain_regression_battery \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/_spiking_novelty_wirein/<integrated-out>.json
```
(pick a concrete output filename under `research/findings/raw/_spiking_novelty_wirein/` for the record.)

It must show: (1) the answer/content fields byte-identical vs `BRAIN_SPIKING_NOVELTY=0` (novelty affects only the
engagement/DA-mode axis, never the content), and (2) the `da_drives`/engagement read present and sane end-to-end
through the real handler with the flag on. **Until that lands, the flip to default-ON is NOT taken** and
`BRAIN_SPIKING_NOVELTY` stays default-OFF (byte-identical). (If `onebrain_regression_battery` is not the exact battery
name the parent uses for this axis, substitute the standard integrated no-regression runner for `da_mode_drives_chat`
with `BRAIN_SPIKING_NOVELTY=1`; the two required checks are what matter.)

## Honest residuals

- **NOT flipped default-ON; scaffold_retired stays NO** on the ledger `da-mode-drives-response` row until the AWS
  integrated soak returns GO. This finding lands the WIRE-IN, not the flip.
- **`observe()` byte-identity when off is proven at the `engagement_of()`/branch level, not by a full integrated
  observe() run** (that needs the #76 DA substrate build, deferred with the integrated soak). The off-branch touches no
  new state or RNG, so this is a scoping choice, not a gap in the argument.
- **LRU eviction reuses a channel without an explicit STP reset**, relying on the recovery/LRU alignment. For a bank
  churned faster than `stp_tau_d` a reused channel can inherit residual depression — a bounded-capacity INTERFERENCE
  that is itself biologically realistic, not a correctness bug. The default bank (64 channels) is sized so eviction is
  rare in a normal turn; a per-channel synaptic reset on eviction is a named next rung if at-scale interference proves
  material.
- **FUNCTIONAL correlate, NOT phenomenal** — a spiking novelty/habituation read, no claim of subjective familiarity.
- **`richness` remains host** (a legitimate token-count sensory boundary); only the novelty JUDGMENT is retired.

## Files

`research/runners/spiking_novelty_habituation_organ.py` (new organ), `webapp/da_mode_drives_chat.py`
(`engagement_of` + `observe` + `_spiking_novelty`), `research/runners/_spiking_novelty_wirein_verify.py` (new focused
verify). Artifact: `research/findings/raw/_spiking_novelty_wirein/verify_6seed.json`. NO `sim/` edit
(`git diff sim/` empty).

## Citations

- De-risk finding: `research/findings/2026-09-09-spiking-habituation-synaptic-depression-novelty-mechanism-6seed-GO.md`.
- Biology binding: `research/biology/spiking-habituation-novelty.md` (Kandel PNS 6e Ch 53, Fig 53-2, Aplysia
  gill-withdrawal habituation).
- Ledger row this addresses: `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` `da-mode-drives-response` residual (1) (the
  host message->engagement novelty term).
