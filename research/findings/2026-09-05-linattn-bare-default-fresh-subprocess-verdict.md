---
type: finding
status: go
date: 2026-09-05
mechanism: fresh-subprocess-per-arm re-verification of the linattn mouth's bare (unset `BRAIN_WKV_MOUTH_*`)
  production default through the real `webapp.server.brain_chat` pipeline -- each arm (prime turn + topic
  turn) runs in its OWN freshly-launched Python interpreter
  (`research/findings/raw/_linattn_flip_verify/_check_d_run_one_arm.py`), so none of check_b/phase6's
  same-process residual state (`webapp/wkv_mouth_generator._RngIsolation`'s continuing per-seed RNG
  timeline, `_get_readout`/`_affect_bias_ids` per-seed caches, session-level mood-EMA/habituation) can leak
  between arms. Methodology ported from `research/runners/_wkv_mouth_affect_neural_verify.py::_run_arm`
  (the SAME discipline `research/findings/2026-09-04-affect-coupling-neural-not-host-PARTIAL.md` used for
  its own 6-seed x 3-prompt load-bearing table), applied here at the `webapp.server.brain_chat` boundary
  rather than the lower `generate()` call because bare-default RESOLUTION is itself under test.
seed-waiver: single production seed (42). Not a reduction from the project's 6-seed non-negotiable battery
  so much as an observation that the battery's premise does not hold for this specific test:
  `webapp/server.py` hardcodes `seed=42` at every organ construction call site
  (`_WARM_QWEN_RENDERER = QwenRenderer(seed=42)`, roughly ten `get_organ(seed=42)` call sites,
  `SelfInitiationOrgan(seed=42)`, etc. -- grep-confirmed, no call site threads a request/session seed
  through), and only the seed42 linattn checkpoint is committed
  (`bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed42.npz`; seeds 43/44/100/101/102 are
  `.gitignore`d). A 6-seed sweep of `brain_chat` itself would therefore call the IDENTICAL pipeline six
  times, not six different configurations. Diversity is instead added on the PROMPT axis for the moat gate
  (2 independently-fabricated unknown topics, vs check_b/phase6's 1); the known-topic prompt matches
  established precedent for direct comparability with check_b/phase4/phase6's own numbers.
lane: language (own-voice mouth / production flip gate) -- record-integrity + instrument correction
seeds: [42]
verdict: GO. `FRESH_SUBPROCESS_BARE_DEFAULT_CLEAN_GO: true` -- ALL FIVE gates pass under genuine
  fresh-subprocess isolation: resolution (bare defaults resolve to linattn/bpe/broad, ckpt exists),
  determinism (fresh-subprocess A raw == C raw, byte-identical), affect load-bearing (fresh-subprocess A
  raw != B raw), moat (both independently-fabricated unknown topics read `known: false` in both lesion
  arms), fluency (salad-fraction well under the 0.3 heuristic threshold -- see the body table for the exact
  figure). This demonstrates that
  check_b's own dropped NO-GO (`research/findings/2026-09-05-linattn-check-b-dropped-nogo-recovered-and-
  reproduced.md`) was the same-process session/RNG-timeline confound `phase6_linattn_clean_isolation.py`'s
  own docstring already names for the original `phase4` script -- NOT a genuine defect in the flip's affect
  coupling or its bare-default resolution. The flip remains production-INERT (`BRAIN_OPEN_ENDED` default-
  OFF, confirmed byte-identical by `check_a_off_byte_identical.py`); this finding gates a FUTURE
  `BRAIN_OPEN_ENDED=1` flip, not current production.
artifacts:
  - research/findings/raw/_linattn_flip_verify/check_d_bare_default_fresh_subprocess_clean_verify.py
  - research/findings/raw/_linattn_flip_verify/_check_d_run_one_arm.py
  - research/findings/raw/_linattn_flip_verify/check_d_bare_default_fresh_subprocess_clean_verify.json
  - research/findings/2026-09-05-linattn-check-b-dropped-nogo-recovered-and-reproduced.md
  - research/runners/_wkv_mouth_affect_neural_verify.py
---

# The linattn flip's bare default IS clean under fresh-subprocess isolation -- GO (check_b's NO-GO was the same-process confound)

## Why re-testing, not just recording, was necessary

The companion finding (`2026-09-05-linattn-check-b-dropped-nogo-recovered-and-reproduced.md`) recovered and
independently reproduced `check_b_bare_default_linattn_and_affect_go.py`'s dropped NO-GO
(`BARE_DEFAULT_FLIP_CONFIRM_GO: false`) and showed its own "smoke turn" self-diagnosis was false -- removing
the smoke turn changed nothing, byte-for-byte. That still left the real question open: is the bare-default
determinism failure a genuine defect in the flip, or an artifact of check_b's own same-process, four-
sequential-turn shared-session design -- the SAME confound `phase6_linattn_clean_isolation.py`'s docstring
diagnoses for the original `phase4` script ("the mood EMA... AND the affect-fix's habituation state evolve
ACROSS turns... CONFOUNDS the lesion0-vs-lesion1 attribution")? check_b inherited that design unmodified,
never adopting even phase6's fresh-SESSION fix, let alone this project's own stronger fresh-SUBPROCESS
precedent. Fresh SESSIONS in one process are not sufficient either: `webapp/wkv_mouth_generator._RngIsolation`
keeps one CONTINUING, per-seed RNG timeline across every `generate()` call in a process regardless of
session id (by the class's own docstring), so only a brand-new process actually starts every stateful cache
at zero -- exactly the isolation `research/runners/_wkv_mouth_affect_neural_verify.py::_run_arm` already
uses for this project's own 6-seed affect-coupling load-bearing table.

## Results

(`research/findings/raw/_linattn_flip_verify/check_d_bare_default_fresh_subprocess_clean_verify.json`,
top-level `verdict`)

| gate | result |
|---|---|
| resolution (bare defaults resolve to linattn/bpe/broad, ckpt exists) | **True** |
| determinism (fresh-subprocess A raw == C raw) | **True** |
| affect load-bearing (fresh-subprocess A raw != B raw) | **True** |
| moat (2 unknown topics x 2 lesion arms, all `known: false`) | **True** |
| fluency (salad-fraction of arm A < 0.3) | **True** (0.0411 <!--derived-->) |
| **FRESH_SUBPROCESS_BARE_DEFAULT_CLEAN_GO** | **True** |

Every arm in this run is a completely independent process: its own interpreter, its own fresh
`_RngIsolation` state, its own fresh per-seed readout/bias-id caches, its own fresh session. With that
isolation, arm A (lesion=0) and arm C (lesion=0, repeated in a wholly separate process ~510s later) produce
**byte-identical** raw text (`rows.Q1_known_topic.A_lesion0.raw == rows.Q1_known_topic.C_lesion0_repeat.raw`,
exact string compare in the artifact), and arm B (lesion=1) diverges from A starting mid-sentence: "...his
**HOME** in new york city to the united states secretary of state..." (A/C) vs "...his **FATHER**
abbreviated to her mother she studied law school..." (B) -- the SAME divergence point check_b's own
(confounded) run showed for this exact prompt, now obtained without the same-process artifact. This
directly demonstrates that check_b's NO-GO was the same-process session/RNG-timeline confound, not a
genuine defect in the flip's affect coupling or its bare-default resolution.

The moat gate held on both fabricated topics despite the mouth continuing to generate fluent, specific-
sounding (and entirely invented) detail for each -- e.g. the "zltrinqua dynasty" topic's continuation
invents an author's bibliography, the "glorbaxian empire" topic's invents a US congressional biography --
`known: false` correctly gates BOTH regardless of how plausible the free-generated prose reads, matching
the SAME `known`-flag semantics check_b/phase6 already established for this exact moat check.

## Verdict

**GO.** All five gates pass under fresh-subprocess isolation. check_b's dropped bare-default NO-GO is
explained: it was a same-process instrument confound (four sequential turns sharing one session's mood-
EMA/habituation state and one process's continuing `_RngIsolation` timeline), not a real defect in what the
bare production default resolves to or how it behaves. `docs/TERMS.md`'s condition for "GO" ("the gate's
OWN verdict is positive") is met here on the gate's own combined boolean, not a metric lifted from a
partial pass.

## Scope and honest limits

1. **Single production seed (42)** -- see the seed-waiver above; this is what the deployed pipeline actually
   runs on every request, not a scope-reduction from a 6-seed claim. There is no seed axis in `brain_chat`
   to sweep.
2. **One known-topic prompt** (matching check_b/phase4/phase6 precedent, for direct numeric comparability)
   **and two unknown-topic prompts** (one more than precedent). A wider prompt sweep was not run here: each
   fresh-subprocess arm pays a full brain-build + Qwen-renderer-load cost (~210-274s measured across the 7
   arms this run), so a broader sweep is a real compute-time tradeoff, not a free addition.
3. **This gates a FUTURE `BRAIN_OPEN_ENDED=1` production flip, not current production.** `BRAIN_OPEN_ENDED`
   remains default-OFF; `check_a_off_byte_identical.py` already confirms `webapp.wkv_mouth_generator` is not
   even imported on that path. Nothing in this finding changes anything live -- no rush, but the record
   needed to be honest before that future flip is ever considered.
4. **The fluency gate is the project's own established salad-fraction heuristic** (most-common-token share
   in the generated text), not a calibrated fluency metric or a human read -- the same acknowledged limit as
   check_b/phase5/phase6's identical heuristic (`docs/TERMS.md`; honest-residual #6 of
   `2026-09-04-linattn-affect-coupling-sharpness-aware-GO.md`). The salad-fraction value measured here
   (0.0410958904109589) is identical to check_b's own value for the same prompt/lesion combination, since
   the underlying generated text is byte-identical -- an independent cross-check that check_d's arm A
   genuinely reproduces the same generation, not a coincidence of the metric.
5. **The moat check verifies the `known` flag, not the absence of invented-sounding detail in free
   generation.** As in check_b/phase6, the free-gen continuation on a fabricated topic can still read as
   fluent, specific prose (an invented bibliography, an invented congressional career) -- the honesty
   boundary under test is whether the system PRESENTS that as grounded factual knowledge (`known: true`),
   which it correctly does not, on every arm tested.

## Reproduce

```bash
# from the repo root; forces CPU/numpy in every fresh child subprocess (CUDA_VISIBLE_DEVICES="" set per-arm)
.venv/bin/python research/findings/raw/_linattn_flip_verify/check_d_bare_default_fresh_subprocess_clean_verify.py
```

Needs the committed seed42 linattn checkpoint (`bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed42.npz`)
and `data/corpus/tinystories.txt`, which is `.gitignore`d and must be copied in from a checkout that has it
-- the same provenance gap every prior script in this verify family already notes
(`research/findings/2026-09-04-linattn-affect-coupling-sharpness-aware-GO.md`, Honest Residual 5). Total
wall time this run: 1713.5s (~28.5 minutes) for 7 fresh-subprocess arms plus one cheap resolution check.

## Provenance

Read this session: `check_b_bare_default_linattn_and_affect_go.py` and its recovered dropped JSON in full,
`phase6_linattn_clean_isolation.py`, `research/runners/_wkv_mouth_affect_neural_verify.py` in full (the
fresh-subprocess pattern this script's own child process, `_check_d_run_one_arm.py`, mirrors),
`webapp/wkv_mouth_generator.py`'s `_RngIsolation` class and its docstring, and `webapp/server.py`'s organ
construction call sites (grepped for `seed=` to confirm the seed-42-hardcoded claim in the seed-waiver
above). No `sim/` edit; no code changes to the flip itself. Both new scripts
(`check_d_bare_default_fresh_subprocess_clean_verify.py`, `_check_d_run_one_arm.py`) are hand-authored
orchestrators (not `research.runners` modules), matching check_a/b/c's own convention in the same directory.
