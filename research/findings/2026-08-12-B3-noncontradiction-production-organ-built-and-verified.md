---
type: finding
status: qualified
date: 2026-08-12
mechanism: onebrain-noncontradiction-assertion-gate-organ
lane: MOAT
artifacts:
  - research/findings/raw/_b3_noncontradiction_organ_verify.json
  - research/findings/raw/_burndown_B3_onebrain_negation_moat.json
---

# B3 non-contradiction assertion-gate — PRODUCTION ORGAN built + standalone-verified (numpy-CPU), reuse-by-import of the 6-seed-GO de-risk. WIREABLE.

**Verdict: WIREABLE.** The B3 de-risk (`_burndown_B3_onebrain_negation_moat_derisk.py`) is a genuine 6-seed GO whose load-bearing element is a genuinely-SPIKING polarity recall on the production one-brain composer, verified-first this session (see §1). A co-resident-shape production organ (`b3_noncontradiction_production_organ.py`) now wraps that de-risked gate for the USER-ASSERTION path, with a standalone verify harness proving intact-fires / lesion-collapses / flag-off-byte-identical / moat-preserved on natural inflected input. `status: qualified` (organ built + verified; NOT yet wired into `/api/brain-chat` — that is the integrate agent's step, spec in §4).

Raw artifacts: organ verify `research/findings/raw/_b3_noncontradiction_organ_verify.json` (ALL_OK 6/6, D=128, numpy-CPU, preconditions carried via `tools.verdict.Verdict`) and the de-risk reproduce `research/findings/raw/_burndown_B3_onebrain_negation_moat.json` (status GO, 6/6 seeds, D=128).

## 1. VERIFY-FIRST (the E1 lesson: a claimed GO can be a mis-read / host-formula / seed-fragile)

- **Genuinely spiking, not a host formula.** `ask_yes_no` -> `_read_blocks`/`_decode_batched_mem` -> `_select`. The matched-filter polarity scores are read from the resonator cleanup membrane `cp_membrane_potential_v` (a real substrate read); with `enable_spiking_cleanup=True` (the OneBrainComposer constructor default, which the de-risk uses) the WINNER is `_spiking_select`, a WTA over `cp_firing_states` of a cached Izhikevich concept bank. The gate proper is the ONE host boolean `stored != asserted` — the exact thin comparison the project already accepts as the moat (`_contradicts == (ask_yes_no == "no")`).
- **Reproduced numpy-CPU.** Re-ran the de-risk. Seed 100 @ D=256 reproduced the finding EXACTLY: INTACT recall neg/aff 1.0/1.0, moat FA=0, rejections 6/6, over_block=0, canon "dog!eat grass"="no"; LESION FA=3 + canon->"yes"; NOSTORE rejections=0; SHUFFLE recall=1.0 tracks=1.0 moved=1 => GO=True. (D=128 multi-seed reproduce: see raw.)
- **Lesion is load-bearing, not a bug.** In the storage lesion (store all AFFIRM) the canonical negation genuinely reads "yes" on the substrate — the negation is really gone, so the 0->N false-accepts are the moat correctly going inert.

## 2. What the de-risk left as the residual, and what the organ adds

The de-risk closed the MECHANISM (negations store, recall, and drive an assertion-path gate). Its residuals were: (a) NOT wired to the production endpoint; (b) negation DETECTION is host, upstream; (c) the production acquire path stores ZERO negations (`polarity="AFFIRM"` hard-coded in `_maybe_acquire`). The organ closes (a) at the code level (a stateless, reuse-by-import gate the endpoint can call) and EXPOSES `detect_polarity` + `extract_polar_assertion` so the wiring can fix (c). (b) remains a declared host residual (learned/spiking polarity classifier = next rung).

**One additional real gap the verify-first caught:** the organ operates on SURFACE content tokens (like the sibling surprise organ), but a user types INFLECTED forms ("the dog eats grass") while the substrate stored the LEMMA ("dog eat grass"). A first pass tested with lemma-matched toy input and hid this; on natural input the gate was INERT (recall returned "unknown" -> accept). Fixed with a minimal, DECLARED surface-first/lemma-fallback (single trailing-"s"/"-ies"->"y" strip on the action, used ONLY when the surface form does not recall, so a genuine surface hit is never corrupted). Irregulars ("goes"/"does") need the shared D4 lemmatizer — named, not smuggled.

## 3. The organ + verify harness (this session's deliverable)

- `research/runners/b3_noncontradiction_production_organ.py` — `NonContradictionProductionOrgan.check(recall, text, lesion)` returns None (out of scope) or a decision dict `{svo, asserted_polarity, recalled_yn, stored_polarity, reject}`. Imports `_assert_gate` + `FLIP` from the de-risk (NO gate reimplementation). Default-ON (`BRAIN_NONCONTRADICTION_GATE`), lesion flag (`BRAIN_NONCONTRADICTION_LESION`). Stateless — reads the REAL production recall (`chat.inner.is_it_true` == `composer.ask_yes_no`); adds NO co-resident bridge (strictly simpler than affect/surprise/comprehension).
- `research/runners/_b3_noncontradiction_organ_verify.py` — standalone, numpy-CPU. Every content test uses natural inflected text end-to-end through a real `OneBrainComposer`.

**Verify results (ALL_OK, 2-seed + 6-seed D=128):**
- INTACT: rejects a contradiction of a NEGATE fact ("the dog eats grass") AND of an AFFIRM fact ("the cat doesn't eat fish"); accepts the consistent restatement, a novel assertion, and a different-patient assertion; returns None for a question; lemma-fallback maps "eats"->"eat".
- LESION (recall bypass): the SAME contradiction is ACCEPTED -> gate inert. STORAGE lesion (store all AFFIRM): canon reads "yes", contradiction slips through.
- FLAG-OFF: the organ is read-only; canon recall identical before/after the whole battery (byte-identical turn when the caller skips `check`).
- MOAT: unknown SVO -> accept, never a fabricated rejection.

## 4. Wiring spec (for the integrate agent) — see the structured result

Hook in `webapp/server.py::brain_chat` AFTER D2-surprise and BEFORE the rich/gate store; on `reject` return an honest functional NOTICE and DO NOT reach `chat.gate()` (so the held belief is never overwritten). Also route the acquire path (`brain_chat_tui._maybe_acquire`) through `detect_polarity` + `extract_polar_assertion` so heard negations store as NEGATE with the SAME normalization (recall + store agree). Default-ON; escape `BRAIN_NONCONTRADICTION_GATE=0`; lesion `BRAIN_NONCONTRADICTION_LESION=1`. Composes with the existing default-on organs with zero overlap (B3 = same-SVO/opposite-polarity; surprise = different-patient).

## 5. Honest residuals

- Negation DETECTION + verb MORPHOLOGY are host, upstream (declared; learned/spiking versions are the next rung; the composer already RECALLS polarity on the substrate).
- STORE-SIDE wiring (negations must store as NEGATE) is part of the integrate step, not yet landed.
- Scale demonstrated D=128/256, 6 facts, 2-word polarity codebook (decode robust regardless of main-vocab size); production-scale (V~1454) polarity-margin confirm is a cheap follow-on.

## 6. Reproduce

```bash
SIM_BACKEND=numpy NEURAL_SIM_DISABLE_LLM=1 PYTHONPATH=. .venv/bin/python \
    -m research.runners._b3_noncontradiction_organ_verify --seeds 42 43 44 100 101 102 --D 128
SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python \
    -m research.runners._burndown_B3_onebrain_negation_moat_derisk --seeds 42 43 44 100 101 102 --D 256
```
