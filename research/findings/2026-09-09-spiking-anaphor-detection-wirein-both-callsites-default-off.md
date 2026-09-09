---
type: finding
status: qualified
date: 2026-09-09
mechanism: spiking-closed-class-pattern-completion
lane: scaffold-retirement
seeds: [42, 43, 44, 100, 101, 102]
verdict: WIRED-DEFAULT-OFF (focused verify GO 6/6; integrated no-regression soak DEFERRED to an AWS-CPU batch)
runner: research/runners/_spiking_anaphor_wirein_verify.py
artifacts:
  - research/findings/raw/_spiking_anaphor_wirein/verify_6seed.json
external: NO-EXTERNAL-NEEDED -- pure production wiring of the already-GO mechanism de-risk
  (2026-09-09-spiking-anaphor-detection-CA3-pattern-completion-6seed-GO.md, status live); no new biological claim.
---

# Spiking CA3 pattern-completion anaphor DETECTION wired into both host `set`-membership call-sites, default-OFF

**Verdict: WIRED, DEFAULT-OFF.** The focused wire-in verify is GO (6/6) -- BOTH call-site detection helpers are
BYTE-IDENTICAL when the flag is off (proven, 0 mismatch over 21 tokens across both sites), and the spiking detection is
LOAD-BEARING when on (6/6 project-standard seeds: every anaphor detected, zero false positives, corrupted-cue recovery,
lesion reverts). The **integrated `/api/brain-chat` no-regression soak is DEFERRED** to an AWS-CPU batch (the dev box is
RAM-blocked on the full integrated brain-chat battery); the FLIP to default-ON waits on that verdict. This turns the
6/6-GO mechanism de-risk (`2026-09-09-spiking-anaphor-detection-CA3-pattern-completion-6seed-GO.md`, commit `1a8152a8a`)
from a banked de-risk into an actual production wire-in -- the owner's #1-priority scaffold-retirement completion step.

## What was wired

Two call sites gate the ALREADY-SPIKING referent resolution (`held_referent()` / the WTA biased-competition read) on a
bare host Python `set`-membership DETECTION step -- "is the current token one of my known closed-class pronouns at all?".
That DETECTION is now optionally moved onto the spiking substrate as CA3-style autoassociative pattern completion,
behind ONE new default-OFF flag **`BRAIN_SPIKING_ANAPHOR`**:

- **`brain_chat_tui.py::ChatBrain._resolve_anaphora`** -- `anaphors = {"it","that","they","them","this"}; ... if tl in
  anaphors:` -> `if self._is_anaphor_token(tl, anaphors):`.
- **`multi_turn_agent.py::MultiTurnAgent._resolve`** -- `if not (isinstance(word, str) and word.lower() in _ANAPHORS):`
  -> `if not (isinstance(word, str) and self._anaphor_is(word)):`.

- **`research/runners/spiking_anaphor_detection_organ.py`** (new; NO `sim/` edit, `git diff sim/` empty). A per-session
  `SpikingAnaphorDetectorOrgan` that reuses-by-import the de-risk's OWN circuit + constants + helpers
  (`SpikingLoopContextBuffer`, `ANAPHORS`, `PATTERN_SIZE`, `ATTRACTOR_WEIGHT`, `BUF_KW`, `_unused_pool`,
  `_cortex_local_index`, `_fresh_probe`, `decide_pronoun`) so the wired mechanism is byte-for-byte the de-risked one: a
  cortico-PFC NMDA-bistable loop with one Hebbian outer-product attractor per closed-class word. ONE scratch buffer per
  session reads off the deterministic pattern allocation; every CLASSIFICATION builds its OWN FRESH quiescent buffer --
  the de-risk's OWN load-bearing lesson (the NMDA-bistable assemblies LATCH permanently, so a reused buffer reads every
  later probe as already-ignited; a fresh buffer per decision is the correct unit).
- **The DECISION is on the substrate; the string->cue encoding is a DECLARED host shortcut** (exactly as the de-risk +
  biology binding already declare -- "the string-to-neuron-index encoding a live deployment would need"). `is_anaphor`
  maps a token to a cue over cortex_ctx neurons and returns whether it COMPLETES to an ignited stored assembly above
  threshold (`decide_pronoun` on `cp_firing_states`). The encoding is EXACT-token (a known anaphor -> its FULL assembly;
  any other token -> a cue drawn from the unused pool, which de-risk G3 proved cannot ignite, reported not-an-anaphor
  without a build). Deliberately EXACT-token, NOT a host string fuzzy-match, because a single-character edit tolerance
  over a 2-4-letter closed-class vocabulary collides with common real words ("what"~"that", "the"/"then"~"them"/"they")
  -- a specificity claim this wire-in does not attempt and the de-risk never validated. So on CLEAN typed text the ON
  path recognises exactly the host set's tokens, but via the substrate's ignition (lesion-reverts); the DELIVERABLE is
  that the decision now lives on the neurons.
- **The de-risked SURPASS is exercised through the organ, not overclaimed on the string path.** `probe_corrupted_cue`
  presents a NOISY/PARTIAL cue (only 20% of a stored assembly's neurons, the de-risk's G2) and the substrate completes
  it to the correct anaphor -- a token an exact `x in {...}` cannot recognise at ANY corruption level. This is the
  proven capability that justifies the substrate detector and is available for a future noisy-perception path.
- **Both call sites are never-raising** -- on ANY organ error each falls back to its exact pre-existing host `set` test,
  so a wiring failure never changes a turn's contract or crashes it. **RNG-isolated** (`_isolated`, the #77 footgun): the
  substrate build + read run on the organ's own private RNG timeline, leaving the host process-global RNG byte-untouched.
- **Lesion `BRAIN_SPIKING_ANAPHOR_LESION=1`** builds every buffer with `attractor_weight=0.0` (the de-risk's own G4
  untrained-network lesion): with no recurrent CA3 completion a driven cue cannot self-sustain, nothing ignites, real
  anaphors read as not-an-anaphor, and the downstream resolution reverts to pass-through -- the load-bearing proof.

## What was verified locally (focused; NOT the full integrated battery)

`research/runners/_spiking_anaphor_wirein_verify.py` -> `research/findings/raw/_spiking_anaphor_wirein/verify_6seed.json`
(verdict GO):

**(A) BYTE-IDENTICAL WHEN OFF (light, no brain).** With `BRAIN_SPIKING_ANAPHOR` unset, over 21 representative tokens
(the 5 anaphors + near-neighbours like "the"/"what"/"then"/"than"/"his" + content words + mixed-case/punctuated forms):
`ChatBrain._is_anaphor_token(stub, tl, anaphors)` == `tl in anaphors` for EVERY token (**mismatch == 0**), and
`MultiTurnAgent._anaphor_is(stub, w)` == `w.lower() in _ANAPHORS` for EVERY token (**mismatch == 0**), and
`spiking_anaphor_enabled()` is False by default. With the flag off both helpers short-circuit before touching `self` or
the substrate, so the only change to `_resolve_anaphora`/`_resolve` is a call returning the identical boolean -> both
call sites are byte-identical to pre-wiring.

**(B) LOAD-BEARING WHEN ON (focused spiking organ, per seed).** 6/6 seeds GO (board bar >=5/6). The per-seed numbers
below are a rounded presentation of the cited artifact `verify_6seed.json` (exact 4-decimal values live there):

<!--derived-->
| seed | clean detect | false-positive rate | corrupted-cue recovery (keep=0.20) | lesion clean | lesion corrupt | attrib->attractor | GO |
|---|---|---|---|---|---|---|---|
| 42  | 1.000 | 0.000 | 0.933 | 0.000 | 0.000 | 1.000 | GO |
| 43  | 1.000 | 0.000 | 0.867 | 0.000 | 0.000 | 1.000 | GO |
| 44  | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | GO |
| 100 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | GO |
| 101 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | GO |
| 102 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | GO |

- **L1 clean detection:** every one of the 5 known anaphors is DETECTED on the substrate on a clean cue -- every seed.
- **L2 specificity:** content words ("cat","dog","fish","hello","banana","house","running","water") are NOT falsely
  detected (false-positive rate 0.000) -- their unused-pool cue cannot ignite (de-risk G3), every seed.
- **L3 corrupted-cue SURPASS:** a cue containing only 20% of a stored assembly's neurons still completes to the correct
  anaphor 0.867-1.000 of the time (>=0.85 bar) -- the capability an exact `x in {...}` structurally cannot have.
- **L4 lesion reverts:** with the attractor weights removed, BOTH clean detection AND corrupted-cue completion collapse
  to 0.000, and `attributable_to` assigns 1.000 of the effect to the recurrent attractor -- the CA3 completion, not the
  host encoding, is doing the recognition.

## Deferred to the AWS-CPU integrated verify (verdict-pending)

The full integrated `/api/brain-chat` no-regression soak is RAM-blocked on the dev box and is NOT run here. The exact
command for the parent to run on an AWS-CPU batch (produces the flip-gating verdict):

```
SIM_BACKEND=cupy BRAIN_SPIKING_ANAPHOR=1 .venv/bin/python -u -m research.runners.onebrain_regression_battery \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/_spiking_anaphor_wirein/<integrated-out>.json
```
(pick a concrete output filename under `research/findings/raw/_spiking_anaphor_wirein/` for the record.)

It must show: (1) the anaphora-resolved turn content byte-identical vs `BRAIN_SPIKING_ANAPHOR=0` on clean typed text
(the exact-token encoding recognises exactly the host set's tokens, so the resolved question -- and therefore the answer
-- is unchanged), and (2) the spiking detection present + sane end-to-end through the real handler with the flag on.
**Until that lands, the flip to default-ON is NOT taken** and `BRAIN_SPIKING_ANAPHOR` stays default-OFF (byte-identical).
(If `onebrain_regression_battery` is not the exact battery name the parent uses for the anaphora path, substitute the
standard integrated no-regression runner that exercises `_resolve_anaphora`/`_resolve` with `BRAIN_SPIKING_ANAPHOR=1`;
the two required checks are what matter.)

## Honest residuals

- **NOT flipped default-ON.** This finding lands the WIRE-IN; the ledger `anaphora-wm` row stays not-retired until the
  AWS integrated soak returns GO. `status: qualified` (the de-risk finding remains the mechanism's one `status: live`
  answer).
- **No string-typo surpass is claimed at the live text boundary.** The exact-token encoding means the ON path on clean
  text detects the same tokens as the host set (via the substrate). The pattern-completion surpass over exact-match is
  the substrate's proven CUE-level capability (L3 / `probe_corrupted_cue`), latent until a noisy-perception input path
  supplies a degraded cue -- a deliberate scope choice, not a gap: an edit-distance string fuzzy-match over a 2-4-letter
  vocabulary was assessed and rejected for its lexical false positives (it would collide with "what"/"the"/"then").
- **Call-site byte-identity when off is proven at the helper level, not by a full integrated turn** (that needs the
  MultiTurnAgent/ChatBrain brain build, deferred with the integrated soak). The off path touches no new state or RNG, so
  this is a scoping choice, not a gap in the argument.
- **Per-token substrate builds** (a fresh buffer per closed-class candidate) make the ON path slower than a `set` test;
  content words short-circuit without a build (unused-pool cue cannot ignite), so the cost is bounded to actual
  closed-class candidates. Speed is secondary per the mission; the AWS soak measures the integrated latency.
- **FUNCTIONAL correlate, NOT phenomenal** -- a spiking recognition read, no claim of subjective familiarity.

## Files

`research/runners/spiking_anaphor_detection_organ.py` (new organ), `research/runners/brain_chat_tui.py`
(`_is_anaphor_token` + `_resolve_anaphora` + `_anaphor_organ`), `research/runners/multi_turn_agent.py` (`_anaphor_is` +
`_resolve` + `_anaphor_organ`), `research/runners/_spiking_anaphor_wirein_verify.py` (new focused verify). Artifact:
`research/findings/raw/_spiking_anaphor_wirein/verify_6seed.json`. NO `sim/` edit (`git diff sim/` empty).

## Citations

- De-risk finding: `research/findings/2026-09-09-spiking-anaphor-detection-CA3-pattern-completion-6seed-GO.md`.
- Biology binding: `research/biology/spiking-closed-class-pattern-completion.md` (Kandel PNS 6e, CA3 pattern completion;
  Marr's recurrent-collateral proposal).
- Ledger row this addresses: `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` `anaphora-wm` residual ("host pronoun
  detect/substitute around it").
