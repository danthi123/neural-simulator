---
type: finding
status: live
date: 2026-09-05
mechanism: curiosity-graded-topic-novelty
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
runner: research/runners/_curiosity_graded_novelty_derisk.py
artifacts:
  - research/findings/raw/_curiosity_graded_novelty_derisk.json
external: NO-EXTERNAL-NEEDED -- reuses two ALREADY in-repo, already-validated spiking/learned mechanisms
  (the DR-1 curiosity ASK-pool novelty transduction, on-bridge 6-seed GO; and the Bogacz-Brown anti-Hebbian
  familiarity projector, catalog D.04, 6-seed-GO'd at V=320 and already reused by INTEGRATION #7's spiking
  familiarity gate). The biology (perirhinal repetition-suppression familiarity discrimination scaling
  continuously with cue fidelity) is the SAME citation already grounded for those two mechanisms.
---

# Curiosity's novelty signal was a binary host constant (`NOVEL_SIGNAL=0.95` on every abstain) -- retired with a graded Bogacz-Brown familiarity/mismatch read of the SPECIFIC topic. 6/6-seed GO (reproduced across 3 independent runs), additive, default-OFF, byte-identical off, lesion load-bearing.

**Verdict: GO (de-risk, default-OFF -- NOT flipped on).** `research/coordination/scaffold_retirement_backlog.md` rank-10 named "curiosity novelty binary host constant (HIGH, de-risked start)". Verified against CURRENT code: the de-risked start is DR-1 (`2026-07-23-DR1-curiosity-inversion-6seed-GO.md` / `-ONBRIDGE-spiking.md`), which built the genuinely-spiking `current_novelty_signal -> from_novelty -> excitability_drive -> cp_firing_states[ask]` pathway and validated it against a real Bogacz-Brown familiarity gate -- but only inside an isolated multi-concept ask-and-learn demo. The production call site (`webapp/server.py::_curiosity_followup`, reached on every no-confab-moat ABSTAIN) never used a graded read at all:

```python
j = _get_curiosity_organ().judge(novelty=_CU.NOVEL_SIGNAL, lesion=_CU.curiosity_lesioned())
```

`curiosity_production_organ.py`'s own docstring already named this residual: *"NOVELTY = the ABSTAIN (a binary epistemic gap) ... a graded familiarity-gate novelty (Bogacz-Brown) is the next rung."*
This session builds that rung: `TopicNoveltyGate` (new, additive, in `curiosity_production_organ.py`) reads a CONTINUOUS [0,1] novelty for the SPECIFIC topic word off the SAME Bogacz-Brown anti-Hebbian projector (`AntiHebbianFamiliarity`, catalog D.04) the v320 gate (`2026-06-11-familiarity-gate-v320-GO.md`) and INTEGRATION #7's burn-down #2 (`2026-08-10-INTEGRATION-7-burndown2-spiking-familiarity-gate-moat-fully-spiking-6seed.md`) already use.
It binds the cue via the SAME genuine time-stepped resonate-and-fire spike-phasor neuron (`phase_sum_neuron`, Orchard Algorithm 1) that gate's spiking realization already uses. Neither is reinvented; both are reused by import.

## What was built

- **`research/runners/curiosity_production_organ.py`** (additive; no existing code touched):
  - `TopicNoveltyGate` -- imprints words the brain already holds (`imprint`/`imprint_vocab`) into an `AntiHebbianFamiliarity` projector, rendering each word's cue via a FIXED per-word phase code (seeded by a stable hash of `(seed, word)`, mirroring `SpikingConjunctiveFamiliarityGate.act_phase`'s fixed-per-concept pattern) bound with a fixed "TOPIC" role phase via `phase_sum_neuron`. `novelty(word, noise=0.0)` reads the graded [0,1] mismatch energy; `noise>0` renders a jittered/partial draw of the SAME word (the project's established `env.draw()`-vs-`env.proto()` pattern). `lesion()` fully clears the learned pool (fresh `AntiHebbianFamiliarity`), matching `SpikingConjunctiveFamiliarityGate.lesion()`'s own discipline.
  - `graded_novelty_enabled()` / `graded_novelty_lesioned()` -- `BRAIN_CURIOSITY_GRADED_NOVELTY` / `BRAIN_CURIOSITY_GRADED_NOVELTY_LESION`, both default OFF.
  - `get_topic_gate(seed, lesion)` / `topic_novelty(topic, known_vocab, seed, lesion)` -- the process-shared singleton (mirrors `CuriosityProductionOrgan`'s own bridge/les split: the lesion arm is a SEPARATE instance that is NEVER imprinted, permanently reading the ceiling). `topic_novelty` degrades to the `NOVEL_SIGNAL` constant on a falsy topic or any internal error -- it can never crash a turn or corrupt the moat.
- **`webapp/server.py::_curiosity_followup`** (one additive block, ~13 new lines): when `graded_novelty_enabled()`, `novelty_val = _CU.topic_novelty(topic, _brain_vocab(chat), lesion=_CU.graded_novelty_lesioned())` replaces the constant fed to `judge()`, and a `graded_novelty` trace key (`on`, `value`, `lesioned`) is attached. Default-OFF: `novelty_val` stays `_CU.NOVEL_SIGNAL` and no new key is attached -- `git diff` shows only new code inside a `if _gn_on:` branch; the pre-existing line is otherwise unchanged.
- **`research/runners/_curiosity_graded_novelty_derisk.py`** (new de-risk/verify runner) -- the 6-seed gate below.
- **`tests/test_curiosity_graded_novelty.py`** (new, 19 tests) -- unit-level pins of `TopicNoveltyGate`'s gradation/lesion/determinism/fallback contract (fast, no on-bridge substrate build).
- **`tests/test_webapp_server.py`** (+2 tests) -- `test_brain_chat_curiosity_graded_novelty_default_off_is_byte_identical` and `..._on_attaches_trace_and_lesion_flips_it`, following the SAME established pattern as the sibling `test_brain_chat_xedge_curiosity_d6_*` tests (the "what does the wombat eat" abstain probe).

No `sim/` edit (`git diff sim/` is empty) -- every neuron/synapse/pathway already existed in the DR-1 curiosity bridge; this session only changes what NOVELTY VALUE is fed into it.

## The 6-seed gate (42/43/44/100/101/102, numpy-CPU; artifact `_curiosity_graded_novelty_derisk.json`; reproduced across 3 independent runs, all 6/6 GO each time)

Each seed builds its OWN fresh `TopicNoveltyGate` AND its OWN fresh `CuriosityProductionOrgan` substrate (NOT the module-level `get_organ()` singleton, which is process-shared and ignores `seed` after its first build -- production always runs it at a fixed seed=42, so a genuine 6-seed check of the downstream spiking coupling needs 6 independent builds).
Four groups on 12 synthetic words each: `known` (imprinted, clean cue), `noisy_lo`/`noisy_hi` (the SAME imprinted words, a mild/strong phase-jitter draw), `novel` (never imprinted, unrelated words).
Each `want_hz` figure is the mean of `WANT_REPS=12` independent `judge()` calls (the ASK pool's OU-process trial noise is well documented in this codebase to swamp a single read at a small novelty gap -- `test_brain_chat_xedge_curiosity_d6_no_regression_on_ordinary_turns`'s own docstring reports 129.17 vs 126.39 Hz calling `judge()` twice on one build with zero env change).

Representative (seed 42; all 6 seeds show the same pattern, see the artifact):

| quantity | known | noisy_lo | noisy_hi | novel |
|---|---|---|---|---|
| graded novelty (deterministic) | 0.0000 | 0.0350 | 0.7647 | 0.9787 |
| want_hz (mean of 12 reads) | 4.70 | 5.73 | 91.16 | 131.71 |
| curious (threshold 65.89 Hz) | False | False | True | True |

**Old constant, same organ, same seed:** `novelty=0.95 -> want_hz=126.26 -> curious=True` -- i.e. the pre-existing behavior reads `curious=True` on literally every abstain, regardless of topic.

**Lesioned twin (never imprinted, the production `lesion=True` semantics) -- ALL 4 groups collapse to the SAME ceiling:** novelty 1.0000 for known/noisy_lo/noisy_hi/novel alike (spread 2.2e-16, pure float noise), driving `want_hz=134.87`, `curious=True` -- i.e. the lesion makes the mechanism REVERT to the old constant's undifferentiated always-curious behavior, exactly as it should (severing the LEARNED weights removes the brain's ability to tell topics apart, not its ability to be curious at all).

**Permuted control (imprint a DISJOINT decoy vocabulary, then query the REAL "known" words against it):** the "known" words, never actually imprinted in this arm, read 0.9803 -- indistinguishable from `novel` -- while the words ACTUALLY imprinted in this arm (the decoys) read 0.0000. The low reading in the real arm is caused by the SPECIFIC imprint<->query correspondence, not incidental word-string shape.

### GO gates (all 6/6 across 3 independent runs)
1. **Graded order**: `known < noisy_lo < noisy_hi < novel`, strictly, on the deterministic novelty value.
2. **Want tracks it**: the SAME strict ordering on the denoised `want_hz` read off the REAL on-bridge ASK pool.
3. **Full-range margins**: `novel - known >= 0.3` novelty units and `>= 20` Hz on `want_hz` (observed: ~0.97-0.98 and ~120-145 Hz across seeds -- comfortably clear).
4. **Discriminates the old constant**: the constant reads `curious=True` on every seed (by calibration, `want_novel_hz` is always above threshold at construction); the graded read reads `curious=False` for `known` and `curious=True` for `novel` on every seed -- the crave decision now depends on the topic.
5. **Lesion collapses**: the lesioned spread is `< 1e-3` on every seed (observed ~2e-16, pure float noise) vs a real spread `>= 0.3` (observed ~0.97-0.98).
6. **Lesion reverts to the old behavior**: the lesioned `want_hz` drives `curious=True` on every seed (matching the old constant's always-curious signature, not a "never curious" failure mode).
7. **Permuted control collapses**: a "known" word queried against a disjoint decoy-imprinted gate reads within `0.3` of the `novel` group's mean, and more than `0.3` above the decoy words actually imprinted in that arm.

## Anti-cheats
- **Fresh substrate per seed for the downstream claim** (not the process-shared singleton) -- see above; the first version of this runner silently re-served seed 42's build for all 6 "seeds" (identical `threshold_hz=65.89` printed for every seed line), a real bug caught by noticing the calibration was seed-invariant when it should not have been, fixed before the GO was recorded.
- **Denoised want_hz** (12-rep mean, not a single call) -- a first pass at 1 rep/novelty-value scored 4/6 GO, failing ONLY the finest step (known vs. a 0.035-novelty-unit noisy draw) on 2 seeds; this is the SAME pre-documented OU-noise-floor residual `test_brain_chat_xedge_curiosity_d6_no_regression_on_ordinary_turns` already names for this organ, not a mechanism failure -- averaging resolves it (a legitimate measurement fix scoped entirely to this de-risk's own test harness, touching neither the shared organ's calibration nor its production behavior).
- **Lesion verified to still hold at measurement time** (`docs/TERMS.md`'s "lesion" condition): the lesioned twin is NEVER imprinted at any point in its lifetime (no plasticity pathway runs against it), so there is nothing that could regrow it between construction and read.
- **Never crashes a turn**: `topic_novelty()` degrades to `NOVEL_SIGNAL` on a falsy topic or ANY exception (a malformed vocab, a backend hiccup) -- pinned by `tests/test_curiosity_graded_novelty.py::test_topic_novelty_never_raises_on_bad_vocab`.
- **Byte-identical off**, asserted in the data (not inferred from reading the code): `tests/test_webapp_server.py::test_brain_chat_curiosity_graded_novelty_default_off_is_byte_identical` posts the established "what does the wombat eat" abstain probe with the flag unset and asserts `curiosity.novelty == NOVEL_SIGNAL` exactly AND `"graded_novelty" not in curiosity` -- a real HTTP round trip through `/api/brain-chat`, not a unit-level read of the flag function alone.

## Honest scope, terms, and open questions (per `docs/TERMS.md`)
- **NOT "closed" / NOT "on-by-default" / NOT "scaffold-retired".** This is a de-risked, WIRED (reachable from `/api/brain-chat` -- `webapp/server.py::_curiosity_followup` calls `_CU.topic_novelty` on the live turn once the flag is set), default-OFF mechanism. The host constant `NOVEL_SIGNAL` remains the production default until `BRAIN_CURIOSITY_GRADED_NOVELTY` is flipped on; retirement is enabled, not yet executed by default. A default-ON flip needs its own no-regression soak against the live production default (mirroring how `da-gated-curiosity`/`da-gated-encoding` were flipped only after a dedicated soak) -- this de-risk does not attempt that.
- **The gradation axis validated is CUE FIDELITY** (a clean vs. a noisy/partial draw of the SAME word), not between-different-word lexical-semantic relatedness -- the word->phase code is a FIXED per-word draw carrying no semantic structure of its own (a declared host boundary, exactly like the v320 gate's percept->phase projection and the curiosity organ's own wh-frame language scaffold).
  Two genuinely different, semantically-close words (e.g. "dog"/"puppy") are not claimed to read closer to each other than to an unrelated word under THIS scheme; only "the SAME concept, viewed cleanly vs. noisily" is validated as graded, which is nonetheless sufficient to retire the CONSTANT (a topic the brain has actually imprinted, even imperfectly, now reads measurably less novel than one it has never seen at all).
- **Capacity-bounded**: the anti-Hebbian basis holds at most `2*D` orthogonal directions (`D=256 -> 512`). This validates the MECHANISM at battery scale (24-36 words per seed), not production-vocabulary scale (which can run to thousands of words) -- the v320 gate's own "does this hold at V=320" question is a named next rung for `TopicNoveltyGate` specifically, not re-litigated by this de-risk.
- **The production wiring imprints an EVER-GROWING vocabulary** (`_brain_vocab(chat)`, re-applied every turn, idempotent per-word) into a process-shared singleton with no eviction policy -- a long-running production process would eventually approach the capacity bound above. Named, not solved; the mechanism stays default-OFF, so this is not currently a live operational risk.
- **FUNCTIONAL correlate, NOT phenomenal** -- this reads + reports a novelty/familiarity CORRELATE feeding an existing curiosity-drive correlate; it claims no subjective wanting or felt novelty.
- **CO-RESIDENT.** `TopicNoveltyGate` is a small, separate host-rate-form projector (not yet a `cp_firing_states` population of its own); it rides on the SAME "composer-as-idealization" declared boundary the v320 gate and INTEGRATION #7 already carry (the CUE is a genuine spike-phasor bind; the novelty READOUT is the Bogacz-Brown rate-form energy on that bind's I/Q render) -- consistent with, not a regression from, the project's existing precedent for this exact family of gates.

## Files
`research/runners/curiosity_production_organ.py` (+`TopicNoveltyGate`, `graded_novelty_enabled`, `graded_novelty_lesioned`, `get_topic_gate`, `topic_novelty`, `_word_phase`, all additive), `webapp/server.py` (+1 additive block in `_curiosity_followup`), `research/runners/_curiosity_graded_novelty_derisk.py` (new), `tests/test_curiosity_graded_novelty.py` (new, 19 tests), `tests/test_webapp_server.py` (+2 tests). Artifact: `research/findings/raw/_curiosity_graded_novelty_derisk.json`.

## Citations
- Scaffold-retirement map (this de-risk's mandate): `research/coordination/scaffold_retirement_backlog.md` rank-10.
- The de-risked start (reused verbatim): `research/findings/2026-07-23-DR1-curiosity-inversion-6seed-GO.md`, `research/findings/2026-07-23-DR1-curiosity-inversion-ONBRIDGE-spiking.md`, runner `research/runners/_curiosity_seek_learn_onbridge_derisk.py`.
- The production curiosity organ this wires into (reused, unmodified elsewhere): `research/runners/curiosity_production_organ.py`, `research/findings/2026-08-12` (Gate-B D3 wire-in).
- The Bogacz-Brown anti-Hebbian familiarity projector + the v320 spiking realization (reused verbatim): `research/findings/2026-06-11-familiarity-gate-v320-GO.md`, `research/findings/2026-08-10-INTEGRATION-7-burndown2-spiking-familiarity-gate-moat-fully-spiking-6seed.md`, `research/runners/cortex_learned_cleanup_derisk.py::AntiHebbianFamiliarity`, `research/runners/_spiking_conjunctive_familiarity_gate.py::SpikingConjunctiveFamiliarityGate`.
- The genuine spike-phasor bind (reused verbatim): `research/runners/spiking_phasor_fhrr.py::phase_sum_neuron` (Orchard & Jarvis 2023, "Hyperdimensional Computing with Spiking-Phasor Neurons," Algorithm 1).
- The pre-documented OU-noise-floor residual on this organ's `want_hz` read (context for the averaging fix): `tests/test_webapp_server.py::test_brain_chat_xedge_curiosity_d6_no_regression_on_ordinary_turns`'s own docstring.
- Attribution / lesion discipline: `docs/TERMS.md`, `tools/lab.py`.
