---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: A10 (midnight plan S15c) -- the SNc reward/context afferent that `da-mode-drives-response` (webapp/da_mode_drives_chat.py) folds into its engagement EMA is, when BRAIN_REWARD_VALUE_AFFERENT=1, driven by an EXISTING spiking read (the surprise-organ confirm/violate rate off `research/runners/surprise_production_organ.py`, falling back to the affect-appraisal valence ladder off `webapp/affect_drives_chat.py` when no expectation-bearing assertion is present) instead of the host `engagement_of()` novelty+richness scalar named as the residual at webapp/da_mode_drives_chat.py:89. New module `webapp/reward_value_afferent_chat.py`; a 4-line additive hook inside `da_mode_drives_chat.observe_turn`. Default OFF (BRAIN_REWARD_VALUE_AFFERENT, BRAIN_REWARD_VALUE_LESION).
seeds: [7]
verdict: PRE-REGISTRATION only (filed before any evaluation run governed by it). Seed 7 is a DEV/CALIBRATION seed, not a gate seed. No result is claimed here.
runner: research/runners/_reward_value_afferent_derisk.py
artifacts:
  - research/findings/raw/_surprise_organ_homeostat/summary.json
---

# A10 reward/value afferent driven by a spiking read: PRE-REGISTRATION (filed before any measured run)

Filed 2026-09-24 on branch `research/reward-value-afferent`, cut from origin/main `d7b2a2bb5`, in its own commit
before any evaluation artifact this file governs exists. Seed 7 only (a dev/calibration seed per the midnight
plan's own rule -- seeds 42/43/44/100/101/102 are reserved for a real 6-seed gate at the frozen SHA `F`, not run
here).

## Prior evidence this design relies on (already on disk; not re-run here)

`research/findings/raw/_surprise_organ_homeostat/summary.json` (the 6/6-seed homeostat GO the surprise organ's
own `ensure_built()` runs by default) reports `mean_confirm_after_max=0.34722222222222215` Hz, well below
`mean_confirm_before_max=2.7295524691358026` Hz, and `all_surprise_specificity_ok=true`
(`mean_novel_registers_frac=1.0`, `mean_contradict_registers_frac=1.0`) -- i.e. the calibrated organ's
confirm/contradict separation this module's `normalized = clip(hz / (2*threshold), 0, 1)` map depends on is
ALREADY established and lesion-verified there; this prereg reuses it rather than re-deriving it.

## The residual this closes (partially)

`docs/plans/.../GAP_CLOSURE_MISSION.md`'s declared-crutches register (item "OPEN -- reward/value afferent") and
`webapp/da_mode_drives_chat.py`'s own docstring (HONEST RESIDUALS #1, line 89) both name the same gap: the
message -> SNc reward/context afferent that sets the self-produced tonic-DA mode (rest/focus/arousal, hence the
engagement suffix on `/api/brain-chat` replies) is a HOST `engagement_of()` computation (novelty fraction +
content-word richness), not a spiking read. `da-mode-drives-response`'s own lesion (`BRAIN_DA_DRIVES_LESION`)
proves the DOWNSTREAM SNc->DA transduction is neural; it says nothing about the UPSTREAM afferent's origin.

Separately, `webapp/da_mode_drives_chat.py:89` also flags that `da-mode-drives-response`'s afferent scalar is
"still environmental" per the board-#76 finding -- i.e. this same residual is already tracked as the "next rung."

## What is built (default OFF, additive, byte-identical off)

- `webapp/reward_value_afferent_chat.py` (new module):
  - `reward_value_enabled()` / `reward_value_lesioned()` read `BRAIN_REWARD_VALUE_AFFERENT` /
    `BRAIN_REWARD_VALUE_LESION` (default unset -> both False).
  - `spiking_reward_value(chat, message, seed)` returns `None` (no usable spiking source this turn -> caller
    falls through to the pre-existing host `engagement_of()` path, UNCHANGED) unless one of two EXISTING spiking
    reads applies:
    1. **surprise-organ path (primary).** Reuse-by-import `research.runners.surprise_production_organ`
       (`extract_assertion`, `get_organ`, `surprise_enabled`) -- the SAME 6/6-GO D2 predictive-coding mismatch
       circuit `webapp/server.py`'s own surprise block reads, and the SAME process-shared organ instance (no
       duplicate build). When this turn's message is an expectation-bearing assertion
       (`extract_assertion(message)` resolves AND `chat.inner.what_does(agent, action)` recalls a stored
       patient), the organ's `judge()` firing rate is normalized against its OWN calibrated confirm/contradict
       threshold (`normalized = clip(hz / (2*threshold), 0, 1)`) and mapped to the SNc afferent range
       (`pa = normalized * da_mode_drives_chat._MAX_AFFERENT_PA`, the SAME 0..1400 pA calibration
       `da-mode-drives-response` already uses). A CONFIRM turn (asserted == stored) reads near-zero afferent; a
       CONTRADICT turn reads near-max -- prediction error as the salience/reward-context signal (Schultz 1998
       RPE; the same novelty-as-salience logic `engagement_of()` already used, now off a genuinely spiking
       mismatch unit instead of a host `set`-membership/token-count formula).
    2. **affect-valence fallback (secondary, declared weaker).** When no expectation-bearing assertion exists
       this turn but this session's `AffectDrivesWorkspace` (board #81, `webapp/affect_drives_chat.py`) has
       already run this turn (population V+/V- ladder read off `cp_firing_states`), the magnitude of its current
       `ema_valence` is used as the afferent's normalized magnitude. This is read-only (no extra appraisal /
       substrate build); it never fires before `affect_drives_chat.observe_turn` has populated
       `chat._affect_drives_workspace` for this turn.
  - The lesion (`BRAIN_REWARD_VALUE_LESION=1`) cuts the READ this module relies on:
    - on the surprise path, it reuses the SAME per-call lesioned twin the organ already exposes
      (`sorg.judge(..., lesion=True)` -- the prediction-edges-zeroed bridge `research/runners/
      _spiking_expectation_rpe_derisk.py` trains), so a CONFIRM and a CONTRADICT turn read the SAME (elevated)
      rate -- the differentiation this module's afferent depends on COLLAPSES. This is a genuine neural-lesion
      (cuts synaptic prediction edges), not a host override, and it does NOT touch `BRAIN_SURPRISE_LESION` (a
      separate, independent per-call argument on the SAME shared organ) or the production `surprise_info`/
      `surprise_prefix` block computed later in the same turn.
    - on the affect-valence fallback, the lesion forces the read magnitude to a fixed 0.0 (a declared,
      WEAKER simplification -- not a synaptic cut of the #81 ladder, which has its OWN independent lesion
      `BRAIN_AFFECT_DRIVES_LESION`). The load-bearing GO claim below is scoped to the surprise path only; the
      affect-valence fallback is reported but excluded from the GO criteria.
- `webapp/da_mode_drives_chat.py::observe_turn` gains a 4-line additive branch: when `afferent_override` is not
  already set by the pre-existing `BRAIN_DA_DRIVES_INDUCE` escape AND `reward_value_enabled()`, call
  `spiking_reward_value()` and use its `pa` as the SNc afferent (attaching the trace as response key
  `da_drives.reward_value`). `BRAIN_REWARD_VALUE_AFFERENT` unset -> the branch's own guard is False -> the
  function is never called -> `afferent_override` stays exactly what it was before this change -> BYTE-IDENTICAL.
  No `webapp/server.py` edit: both of `da_mode_drives_chat.observe_turn`'s existing call sites pick this up
  unchanged.
- `research/runners/_reward_value_afferent_derisk.py`: the capability-gate runner (mirrors
  `_da_encoding_wired_verify.py`'s A/B/C template, `tools.verdict.Verdict` + `tools.lab.attributable_to`), fresh
  sessions through the REAL `webapp.server.brain_chat` handler, numpy backend:
  - (A) OFF byte-identity: `BRAIN_REWARD_VALUE_AFFERENT` unset -> no `reward_value` key, `da_drives.afferent_pA`
    computed the OLD way (host `engagement_of()`), identical to HEAD on the `confirm`/`contra` turns.
  - (B) ON, load-bearing: on a FRESH session, `"the dog chase the cat"` (CONFIRM -- the built-in dog-chase-cat
    fact) vs `"the dog chase the fish"` (CONTRADICT) differ in `da_drives.reward_value.normalized` /
    `.pa` / `.mode`, attributable to the surprise-organ read (`tools.lab.attributable_to` against the lesioned
    control).
  - (C) LESION: the SAME confirm-vs-contradict pair under `BRAIN_REWARD_VALUE_LESION=1` -- the differential
    VANISHES (both read the lesioned organ's elevated, undifferentiated rate).
- `research/runners/lbf_rows/reward_value_afferent.py`: `EXTRA_LESIONS["reward-value-spiking-afferent"]`
  (`flag=BRAIN_REWARD_VALUE_LESION, value=1, kind=neural-lesion`) and `EXTRA_PROBES` entry keyed on the existing
  `contra` turn (`session "surp2"`, already in `onebrain_regression_battery.PROBE_TURNS`), fields
  `["da_drives.reward_value.source", "da_drives.reward_value.normalized", "da_drives.mode"]`, per the
  `research/runners/lbf_rows/<module>.py` interface (this file does NOT edit `FACULTY_LESIONS`/`FACULTY_PROBES`
  directly; it is picked up once the AG-REG import hook lands). Because `BRAIN_REWARD_VALUE_AFFERENT` and
  `BRAIN_DA_DRIVES` must BOTH be set for this row to read anything, it runs as an OPT-IN capability row
  (`--extra-env BRAIN_DA_DRIVES=1 BRAIN_REWARD_VALUE_AFFERENT=1`), exactly like the other S15 conditional flags --
  never inside the default `adequate`/`thin` probe sets.

## Declared residuals (named, not claimed closed)

1. **The read -> pA transduction is a host scaffold.** `normalized = clip(hz / (2*threshold), 0, 1)` and
   `pa = normalized * _MAX_AFFERENT_PA` are a fixed linear host map, exactly the same class of residual
   `da-mode-drives-response` already declares for its OWN afferent map (`e -> pA`). Closing this fully needs a
   spiking transduction stage (e.g. the mismatch rate driving the SNc population directly via a synapse, not a
   Python `float` multiply) -- named as the next rung, not attempted tonight.
2. **`extract_assertion` + `chat.inner.what_does`** are the SAME host language-comprehension boundary the
   surprise organ's own production wiring already uses and already declares (agent/action/patient token
   extraction; the RECALLED expected patient is the brain's own spiking recall, not a host lookup) -- reused
   verbatim, not a new shortcut.
3. **The affect-valence fallback's lesion is a forced-zero override**, not a synaptic cut -- declared weaker
   than the surprise path's genuine neural lesion and EXCLUDED from tonight's GO claim.
4. **`BRAIN_REWARD_VALUE_AFFERENT` gates on `surprise_enabled()`** -- if `BRAIN_SURPRISE=0` (the surprise organ
   fully disabled project-wide), this module's primary path is silently unavailable for that request and it
   falls through to the affect-valence fallback or, absent that, to the pre-existing host engagement afferent.
   This coupling choice (never run a source whose parent faculty is off) is declared, not hidden.
5. **No independent opus-skeptic pass has run.** S15(c)'s own success criterion asks for "an opus skeptic
   traces the path from user text to SNc current and fails it on any keyword or regex classifier." This
   PREREGISTRATION records a self-trace (below) done by the same agent that wrote the code -- NOT an independent
   review. It is named here as an open item, not silently assumed passed.

### Self-trace (not a substitute for an independent opus pass)

Text -> `extract_assertion` (regex tokenization + a fixed function-word/WH-word set; no keyword-to-reward
mapping -- it only decides IF a clause is a 3-content-token SVO, not what the reward value is) -> the AGENT and
ACTION tokens address existing trained circuit BLOCKS (`_block_for`, a topographic assignment learned at build
time, not a per-run classifier) -> `chat.inner.what_does` (the brain's own spiking recall, not a lookup table)
-> `sorg.judge()` drives the SAME predictive-coding mismatch circuit through `cp_firing_states[surprise]` -> the
returned Hz is a population spike-rate read, not a string/keyword compare -> the pA map is a fixed linear
rescale of that ONE scalar. No branch classifies message CONTENT into a reward category; the only text-dependent
step is which trained block index cue/patient tokens hash to, and that hash is dictionary-order/round-robin
(`_block_for`), not sentiment/keyword-based.

## GO criteria (scored by the runner, seed 7 only -- a de-risk, not a gate verdict)

- (A) OFF byte-identity: no `reward_value` key on either arm; `da_drives.afferent_pA`/`.mode` on the `confirm`/
  `contra` turns identical to a HEAD build with the same env.
- (B) ON: `reward_value.source == "surprise"` on both turns; `contra`'s `normalized`/`pa` strictly greater than
  `confirm`'s; `attributable_to(...)` against the lesioned control >= 0.9.
- (C) LESION: `confirm` and `contra` `reward_value.normalized` differ by less than 1e-6 (both read the
  prediction-removed twin).

GO on seed 7 is a DE-RISK, not a 6-seed verdict. Per S15's own rule, the flag's 6-seed capability gate
(seeds 42/43/44/100/101/102) runs AT the frozen SHA `F` in B2b (S28) -- never on seed 7 alone -- and the
default-ON decision additionally needs the SOUND opus review named above.
