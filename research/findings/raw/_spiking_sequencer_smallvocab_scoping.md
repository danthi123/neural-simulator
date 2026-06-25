# Purity #4 — the spiking cue-match sequencer reverts at SMALL vocab: read-only scoping (2026-06-25)

**Type:** READ-ONLY deep-research scoping at a CONFIRMED boundary (the standing research-gate first move). NO edits, NO
runs, NO webapp. One scoping document + the final message. Stayed on `main`.

**The purity item (#4, class-b #1 — the LARGEST live default-on conversational host residual).** The production
`OneBrainComposer` / `RFPhasorComposer` answer who/what (and yes-no / describe / reason-chain) by a host **cue-match
scan** — a Python `for`-over-stored-fact-blocks + `if cue-matches then return-the-answer-role else continue/return-None`
— and the abstention decision (no block matches → `None`/`"unknown"`) IS that loop's else-branch (the no-confab moat).
The validated spiking replacement (the **K-way sequencer**: a gated-disinhibition match cascade + a BG first-match
priority/abstain WTA) is `integrated_loop=True`. It is **default-ON at the 320-concept flagship**
(`consolidated_320_conversation_demo.py:242 set_defaults(integrated_loop=True)`) and **default-OFF at the small-vocab
library agent** (`one_brain_composer.py:116` / `brain_conversational_agent.py:154` `integrated_loop=False`). This doc
scopes WHY it reverts at small vocab, ranks the cheapest fix, and gives the verdict.

**The 4-move deliverable (standing SURPASS form):** (1) DIAGNOSIS — what the host scan does + ISOLATE the genuine
residual + the PRECISE small-vocab failure; (2) REFRAME via biology; (3) RANK cheap-first options; (4) the anti-cheats +
de-risk + GO bars + VERDICT.

---

## 0. TL;DR (the load-bearing answers)

- **The residual is THREE host ops, of decreasing size:** (i) the **cue-match COMPARISON** (`got==want`) + the
  **first-match ROUTING** (which stored block answers) — the bulk; (ii) the **abstention DECISION** (no match → `None`)
  = the moat's else-branch — same op, the safe tail; (iii) `query_agent` (the swapped (action,patient) cue) +
  `render_fact`/`describe` (the 1-role agent-only cue) still on the host `_scan` even when `integrated_loop=True` (named
  build-1 follow-ons, `one_brain_composer.py:1008/1050`). The (agent,action) hot paths (`query_patient`, `ask_yes_no`,
  `_find_cued_block`→reconsolidation/`reason_chain`) ARE converted on the ON path.

- **The small-vocab revert is NOT a sequencer/WTA/threshold failure. It is an UPSTREAM cleanup-margin (code-fidelity)
  failure.** At the library test config (V=15, K=4, **fresh random codes**, D=128), the per-block matched-filter
  **cleanup of the AGENT role produces ZERO firing** on ≥half the blocks — `win_rate 0.000` (the correct block's
  spiking match pool does not fire at all) — so `present_ok` stays 0/4 (seed 42) or 2/4 (seeds 43/44) at **EVERY**
  `match_thresh` ∈ {0.06, 0.02, 0.005, 0.001} (and at 0.0+ε). The action-role decode is clean; the AGENT role is the
  one that washes out. The moat held throughout (`worst_off_rate 0.000`, `host_absent_all_none True`) — the failure
  direction is OVER-ABSTENTION (the SAFE direction). Evidence: `_burndown_1A_c2_smallvocab_derisk.json` +
  `_burndown_1A_c2_smallvocab_derisk.py:104-109` (`win = r[tgt]`, the correct block's `m{tgt}` rate read at
  `match_thresh=0.0`).

- **The decisive contrast (why 320 is GO but small-vocab reverts):** at 320 the demo feeds **stream-learned cortex
  codes** (`_phaseB_stream_codes_320_neural_seed42.npy`; high cleanup margin) — every present-cue target match-pool rate
  is **0.116–0.196**, the worst no-match leak is 0.000, so `match_thresh=0.06` sits cleanly in the open gap → GO 4/4,
  moat 0-FA (`2026-06-21-shortcut3-K32-capability-surpass.md` + the fold BUILD's V=320/K=32 GO). At small vocab the
  fresh-random-code agent cleanup is `0.000` — there is **no signal of any height** to admit. The boundary is the
  **upstream code fidelity / cleanup margin**, which is HIGH for the stream-learned 320 codes and LOW for fresh random
  codes at small V. **The sequencer is not the limiter** (it GO'd at the harder 1-of-K=32 selection; the cleanup WTA
  GO'd at the harder 1-of-V=320 selection).

- **Why a bigger vocab makes it BETTER, not worse (the counter-intuitive part):** "small vocab" is a proxy for "low
  cleanup margin." A who/what agent at V=15 uses fresh random codes whose pairwise cosine spread + the FHRR
  bind/unbind/cleanup chain leave the *agent role* (the role unbound LAST in the cleanup matvec, after the
  superposition-of-K-bindings cross-terms accumulate) below the RF magnitude floor on some blocks → it reads phase-0
  garbage → the matched filter lands flat → zero match-pool firing. The 320 demo's stream-learned codes have a wide
  familiarity/cleanup margin (the PPMI local-normalization gives them structure), so the same chain clears the floor.
  More precisely: the lever is CODE MARGIN, and stream-learning RAISES it.

- **The verdict (honest):** **PARTIALLY closeable — and the cheapest close does NOT touch the sequencer or the
  threshold.** Two cheap-first paths, in order: **(R-A, recommended first)** make the LIBRARY agent default carry the
  same high-margin codes the production path uses (or, equivalently, make the small-vocab CI config use stream-learned /
  PPMI-structured codes), so the cleanup margin that gates the sequencer is the PRODUCTION margin, not the fresh-random
  worst case → then flip `integrated_loop` default-ON for the onebrain library path. **(R-B)** lift the upstream cleanup
  margin itself (population redundancy / per-role re-kick / a confidence-gate-driven graceful abstain) so even
  fresh-random small-V codes clear the floor. The genuinely-irreducible residual — IF both fail — is a characterized
  **fresh-random-code-at-small-V cleanup-margin boundary**: the host scan stays the default at the
  numpy-CPU/test-oracle/fresh-random-code path, the spiking sequencer is the default at the stream-learned/production
  path (where it already is). **This is NOT a sequencer boundary and the gap CANNOT and MUST NOT be closed by lowering
  the match threshold or loosening the moat** (the moat is 0-FA at every threshold at BOTH scales; the failure is a
  present-cue MISS, not a confab).

- **The standing-practice framing:** the "C-2 reverts at small vocab" headline is a SLIGHT MIS-LOCATION of the residual.
  The reverted op (the sequencer) is downstream of the actual limiter (the cleanup of low-margin codes). The honest
  surpass move is therefore NOT "re-tune the sequencer for small K" (already proven not to help — `win_rate 0.000` at
  every threshold) but "raise the upstream cleanup margin at the library default," which the production path already
  does for free via stream-learned codes.

---

## 1. DIAGNOSIS — what the host scan does, the isolated residual, and the PRECISE failure

### 1.1 What the host cue-match scan does (verified in code)

The host control flow is the SAME op in two syntactic forms (`one_brain_composer.py`, and the rf sibling
`rf_phasor_composer.py`):

```python
# OneBrainComposer._scan (one_brain_composer.py:917-921) -- and the inlined twins at the call sites:
def _scan(self, cue, answer_role):
    for got in self._read_blocks():                              # (a) iterate the stored fact blocks  [CONTROL]
        if all(got.get(role) == want for role, want in cue.items()):  # (b) the MATCH compare           [CONTROL]
            return got.get(answer_role)                          # (c) ANSWER (emit the answer role)     [CONTROL]
    return None                                                  # (d) ABSTAIN (the no-confab moat)      [CONTROL]
```

`_read_blocks()` (`:714`) is ALREADY on-substrate: it reconstructs+unbinds+cleans each stored block on the co-resident
RF bridge (the batched `_read_all_blocks`, `:626`) and returns `{role: word}` dicts. The cleanup SELECTION (the
per-role argmax→word) is ALSO on-substrate when `enable_spiking_cleanup=True` (`_spiking_select`, `:507`, the NEF
Izhikevich WTA, default-ON for onebrain via the 1A sentinel). **So the residual host op is ONLY (b)+(c)+(d): the
cue-match COMPARISON, the first-match ROUTING, and the abstain decision.** That is exactly what the sequencer replaces.

The rf sibling's host scan is `_scan_first_match` (`rf_phasor_composer.py:435-444`): a batched matmul cleanup → a numpy
`w == val` mask → `idx[0] if len else None`. Same residual (comparison + first-match + abstain), via numpy.

### 1.2 ISOLATE the genuine residual (the SURPASS move-1 — pin the exact bytes)

| residual op | where (host) | converted on the ON path? | the genuine residual |
|---|---|---|---|
| **(b) cue-match COMPARE** (`got==want`) | `_scan:919`; inline `query_patient:892-894`, `ask_yes_no:1035`, `_find_cued_block` | **YES** — the spiking gated-disinhibition match cascade (`m{b}` fires iff block b's decoded cue-roles == the cue), computed for ALL blocks in PARALLEL (no Python loop). `2026-06-19-onebrain-sequencer-derisk.md` GO 6/6. | **the upstream signal it reads** (the cleanup margin), not the compare |
| **(c) first-match ROUTING** (return the FIRST matching block) | `_scan` loop order; `query_patient:987` | **YES** — the K-way first-match priority WTA (`ans{b}→inh{b}→{ans{j>b}}∪abstain`, the BG A.04 disinhibition). GO to K=16 (S0, 6 seeds) + K=32 (the fold, `match_thresh=0.06`). | the WTA is in proven range (1-of-32 < the cleanup's 1-of-320) |
| **(d) ABSTAIN decision** (no match → `None`) | `_scan:921`; `query_patient:988`, `ask_yes_no:1033` | **YES** — the abstain channel (tonic default suppressed by ANY match); on the ON path the absent-cue WORD is caught before the fabric (`_seq_block:900-901`, the moat). 0-FA at every K, every seed, BOTH scales. | the moat is structurally safe — NOT the residual |
| **(e) `query_agent` (action,patient) + `render_fact`/`describe` (agent-only)** | `query_agent:1008` → `_scan`; `render_fact:1050` host loop | **NO** — STILL host even with `integrated_loop=True` (the sequencer is built for the (agent,action) cue) | a named build-1 follow-on — a swapped-cue + a 1-role cascade (clean second/third sequencers) |

**The genuine, isolated residual is therefore TWO things, NOT one:**
1. **The thing that actually blocks the small-vocab default-on flip = the UPSTREAM cleanup margin** (§1.3). The
   sequencer's match COMPARE reads the cleanup-driven decoded line; at low code fidelity that line never lights for the
   agent role, so the compare has nothing to fire on. *This is the residual the headline mis-attributes to the
   sequencer.*
2. **A genuine remaining host-cognition residual = `query_agent` + `render_fact`/`describe`** (the swapped/1-role
   cues, still on `_scan` on the ON path). This is a clean, bounded conversion (more match cascades), independent of the
   small-vocab issue, and is the part of #4 that is unambiguously "host doing cognition" once the (agent,action) path is
   ON.

### 1.3 The PRECISE small-vocab failure (the SURPASS move-1 — measure, don't hand-wave)

The committed de-risk `_burndown_1A_c2_smallvocab_derisk.json` (V=15, K=4, fresh random codes, D=128, seeds 42/43/44,
the EXACT `test_one_brain_composer_agent` config) reads the per-cue match-pool rate at `match_thresh=0.0` then
re-thresholds. The numbers:

```
seed 42:  dog,go win 0.000 | cat,come win 0.000 | bird,look win 0.000 | river,stop win 0.000   (worst_off 0.000 all)
          present_ok 0/4 at thresh {0.06, 0.02, 0.005, 0.001};  false_accept 0;  moat_0fa True
seed 43:  dog,go win 0.241 | cat,come win 0.000 | bird,look win 0.241 | river,stop win 0.000
          present_ok 2/4 at EVERY thresh;  false_accept 0;  moat_0fa True
seed 44:  dog,go win 0.222 | cat,come win 0.000 | bird,look win 0.000 | river,stop win 0.229
          present_ok 2/4 at EVERY thresh;  false_accept 0;  moat_0fa True
```

**The signature is unambiguous:**
- **The correct block's match pool fires at rate `0.000`** on the failing cues. The decode rule is `rates[j] > th`
  (`_burndown_1A_c2_smallvocab_derisk.py:115`), so a rate of exactly 0.000 is BELOW every positive threshold AND below
  0.0+ε. **No threshold re-cal can recover a 0.000 rate** — there is nothing to detect. This is why a gain/sigma sweep
  (`gain {0.11→0.001} × sigma {1,100}`, per the migration-PARTIAL update) found NO operating point that cleanly isolates
  the agent winner: `gain≥0.11 → empty` (the divnorm starves the already-tiny drive), `gain≤0.03 → the winner lights
  WITH runner-ups` (`clean_exact 0/4`).
- **It is the AGENT role specifically** that washes out (the migration-PARTIAL update: "the divnorm-WTA **agent-role**
  cleanup decode produces ZERO firing on ≥half the blocks — the action-role decode is clean"). Mechanistically, in the
  per-block read the roles are unbound from the SAME stored composite (a superposition of K role-filler bindings); the
  agent-role unbind's cross-terms (the other K−1 bindings' residue) are largest, and at low code margin the recovered
  agent phasor's |Z| dips below the RF magnitude floor (`sim/bridge.py:5589`, `_rf_mag2 > _rf_floor2` — a readout neuron
  whose |Z| decays below the floor never spikes → reads phase 0 = flat cleanup → zero match firing).
- **The moat held at EVERY threshold, EVERY seed** (`false_accept 0`, `worst_off_rate 0.000`, `host_absent_all_none
  True`). The failure is a PRESENT-cue MISS (over-abstention), the SAFE direction. The moat is NOT at risk and was NOT
  the failing axis.

### 1.4 The decisive 320-vs-small contrast (why one is GO and the other reverts)

| config | codes | target match-pool rate (present cues) | worst no-match leak | `match_thresh=0.06` outcome | source |
|---|---|---|---|---|---|
| **production 320 demo** | **stream-learned cortex** (`_phaseB_stream_codes_320_neural`) | **0.116–0.196** | 0.000 | **GO 4/4, moat 0-FA** | `2026-06-21-shortcut3-K32-capability-surpass.md`; fold BUILD V=320/K=32 GO |
| **K=32 stress (V=72)** | fresh random, but 32 facts → wide margin | min 0.116 | ≤0.014 | GO 3/3, moat 0-FA | `_phaseB_onebrain_sequencerK_k32_margin` + the surpass |
| **library small-vocab** | **fresh random, V=15, K=4** | **0.000** on ≥half | 0.000 | present_ok 0/4–2/4 at EVERY thresh | `_burndown_1A_c2_smallvocab_derisk.json` |

The K=32 case (the one the 2026-06-20 scoping FEARED was the boundary) turned out to be a wrong-threshold artifact —
GO at 0.06. The small-vocab case is a genuinely different failure: **the cleanup margin (code fidelity), not the
sequencer**. The lever that separates GO from revert is the **code margin**, and **stream-learning raises it**. (Note
the K=32 stress table also uses fresh random codes yet GO's — because with 32 facts the cue space is denser and the
specific agent-role cross-term collisions that zero a fresh-random V=15 agent cleanup don't recur; the failure is
small-V-specific, an SNR-vs-redundancy interaction, not "fresh-random always fails.")

**⇒ The genuine residual that blocks the default-on flip is the upstream cleanup margin of fresh-random codes at small
V. The sequencer is correctly converted and is not the limiter.** This is the SURPASS move-1 finding: most of the
"blocker" is already solved (the compare/route/abstain are spiking + GO at production), and the genuine residual is a
TINY, precisely-located thing — the agent-role cleanup firing of a handful of low-margin codes.

---

## 2. REFRAME via biology — how does the brain scan memory for a cue-match + decide "no match"?

The host scan conflates two biological operations the project already has catalog grounding for; the reframe sharpens
WHICH one the small-vocab failure sits in.

### 2.1 The cue-match scan = CA3 pattern completion, NOT a serial `for`-loop (catalog D.05 / D.13)

The brain does NOT iterate stored items with an equality test. A partial cue drives a **CA3 recurrent autoassociator**
(D.05, Kandel 6e Ch 54 pp 1357, 1360–1361; Marr 1971; O&N 1978 "missing-stimulus" single-unit data) that **converges
to the full stored pattern in PARALLEL** (D.13). The trade-off the catalog names is exactly the one in play:
*"too much completion → confused episodes; too little → no generalization"* (D.13). The sequencer's gated-disinhibition
match-cascade + first-match WTA is a faithful-enough functional stand-in for "all stored blocks complete in parallel,
the best match wins" (the catalog A.04 BG WTA at GPi/SNr is the selection; the cleanup matched-filter is the
completion). **The reframe's payload: the small-vocab failure is the "too LITTLE completion" horn** — a low-margin
agent code fails to complete (the recovered phasor falls below the magnitude floor → flat cleanup → no match), which is
exactly D.13's "too little → no retrieval." The fix biology points at is *better-separated, higher-fidelity stored
patterns* (DG pattern separation D.12 feeding crisp CA3 attractors) — i.e. the PPMI/stream-learned code structure the
production path already supplies — NOT a different scan mechanism.

### 2.2 The "no match" decision = a familiarity/novelty signal, NOT the scan's absence (catalog D.23)

The abstention (the no-confab moat) is biologically a **familiarity/novelty signal** — O&N's hippocampal "misplace"
system (D.23, O&N Ch 2.3 pp 89–101, Ch 4.7.2 pp 195–209) fires when the current cue MISMATCHES the stored map; Kandel
attaches the same match/mismatch comparator to perirhinal/EC-III (D.04). The project ALREADY has a validated neural
realization of this: the **learned Bogacz-Brown familiarity gate** (`2026-06-11-familiarity-gate-v320-GO.md`,
agreement 168/168, zero moat-breaches) and, on the composer, the **`confidence_gate`** (`one_brain_composer.py:256-262`,
`_margin`:500-505) — a cue-role cleanup margin that BLANKS a noise-dominated (unfamiliar) block so every consumer
abstains. **The reframe's payload: at small vocab the system is ALREADY in the "low familiarity / flat margin" regime
for the agent role — the cleanup IS flat (`win_rate 0.000`)** — so the *correct biological outcome* is abstention. The
moat is firing correctly; the problem is that a PRESENT fact is being read as unfamiliar because its code didn't
complete. This re-confirms §2.1: the fix is upstream fidelity, not the gate (and the gate must NOT be loosened — that
would trade the moat).

### 2.3 The rate-code wall (catalog E.07) — the floor that bites

The recovered-agent-phasor-below-the-RF-floor mechanism is the documented **rate-code / point-neuron readout floor**
(the project's recurring family: the Mikulasch-Priesemann point-neuron limit; the cleanup's "off-target emits zero
spikes" Stewart-Tang-Eliasmith threshold). At low code margin the *signal* (the correct phasor magnitude) is small and
the floor clips it to zero. The catalog-grounded lift the project has used before is **population redundancy** (the
"population code lifts the single-neuron read-out from 47% → 100-108%" result,
`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO`) — average many noisy readouts to clear the floor.
This is R-B below.

**⇒ Biology reframe verdict:** the host scan stands in for CA3 completion + a familiarity gate, both already validated
neurally. The small-vocab revert is the "too-little-completion / below-floor" horn, whose biological fix is
better-separated higher-fidelity stored codes (DG→CA3; PPMI/stream cortex) and/or population redundancy — NOT a
different sequencer and NOT a looser gate.

---

## 3. RANK cheap-first options to close the small-vocab gap (the SURPASS move-3)

Ordered cheapest-first. Each ends with a `==host` + moat-0-FA gate. The first that GO's makes the spiking sequencer the
library default.

### R-A — give the library default the PRODUCTION cleanup margin (CHEAPEST; recommended FIRST; likely sufficient)

**The insight:** the production path is GO **because of its codes, not its sequencer**. The library/test default reverts
**because of its codes** (fresh random, V=15). So the cheapest close is to remove the code-margin difference between the
two defaults — not to re-engineer the sequencer.

Two sub-forms, in order:
- **R-A1 (the cleanest):** make the small-vocab CI/library config that the default-on flip must pass use
  **stream-learned / PPMI-structured codes** (the same kind the 320 demo uses), at a small vocab. The stream cortex is
  already validated at 64 concepts (`2026-06-15-...GO`); a small (V≈15–32) stream-learned or PPMI-normalized codebook
  gives the agent-role cleanup a non-zero margin (the structure that lifts `win_rate` above the floor). Then flip
  `BrainConversationalAgent(composer_kind="onebrain")` `integrated_loop` default to the None-sentinel pattern already
  used for `enable_spiking_cleanup`/`enable_learned_assoc` (`brain_conversational_agent.py`, the 1A flips):
  **auto-resolve ON for the onebrain production path, OFF for the rf/rate test-oracle + numpy-CPU path.** This is exactly
  the precedent the close-out audit set for C-3/C-5.
- **R-A2 (a smaller config knob, if R-A1 is too heavy for CI):** raise the small-vocab agent's code DIMENSION D (the
  fresh-random margin grows with D — more independent components, lower cross-term collision probability) and/or use the
  composer's existing `grounded_codes=` path with a precomputed higher-margin small codebook, until the agent-role
  `win_rate` clears the floor on all seeds. This keeps fresh-random codes but moves them out of the worst-case margin.

**Effort:** ~0.5–1 day (a config/sentinel change + the de-risk re-run; R-A1 reuses the stream cortex; NO `sim/` edit).
**GO bar:** the agent-level onebrain suite (`test_one_brain_composer_agent.py`, the 6 tests that reverted) passes with
`integrated_loop` auto-ON, `==host`, moat 0-FA; the rf/numpy-CPU path byte-unregressed. **Risk:** low — it directly
removes the measured cause (the code margin), and the production path already proves the sequencer works at production
margin.

### R-B — lift the UPSTREAM cleanup margin so even fresh-random small-V codes clear the floor (if R-A is rejected)

If the owner wants the spiking sequencer default-ON even at the fresh-random worst case (a stricter purity bar), raise
the cleanup margin itself. Cheapest-first, all catalog-grounded:
- **R-B1 — population redundancy on the agent-role readout** (the documented rate-code-wall lift,
  `2026-06-15-...GO`): replicate the agent-role cleanup neurons N-fold and average their firing before the match — lifts
  the sub-floor agent signal above threshold. The `_spiking_select` NEF bank already has an `n_per` noise-averaging knob
  (the cleanup's `NEF_CLEANUP_OP`); the same idea applied to the match-pool drive.
- **R-B2 — per-role re-kick before the agent-role cleanup** (reuse `persistent_loop`'s `_dev_rekick_into`,
  `one_brain_composer.py:488-496`): re-kick the unbound agent Q-register as a CLEAN UNIT PHASOR before its cleanup matvec
  (it is currently carried live across the unbind→cleanup handoff). The re-kick normalizes the register to unit
  magnitude → the recovered agent phasor is lifted off the floor regardless of the cross-term residue. This is
  byte-identical to a host round-trip and already default-ON for the clause path; extending it per-role on the flat read
  is a clean composer-layer change.
- **R-B3 — confidence-gate-driven graceful read** (`confidence_gate>0`, already built `:256-262`): NOT a fix for recall
  (it makes the flat-margin block abstain, which is what already happens), but it CONVERTS the failure into a clean,
  intended abstention rather than a silent miss — useful as a documented graceful-degradation default if R-A/R-B1/R-B2
  don't fully recover small-V recall. (This is the lossy-memory-OK direction the owner sanctioned —
  `feedback_moat_not_hard_lossy_memory_ok` — but it is recall-degrading, so it is the LAST resort, not a close.)

**Effort:** R-B1/R-B2 ~1–2 days each (composer-layer reuse-by-import; R-B2 reuses the shipped `_dev_rekick_into`; likely
NO `sim/` edit). **GO bar:** the fresh-random small-V agent-role `win_rate` clears the threshold on all seeds → the
agent suite passes with `integrated_loop` ON on fresh-random codes, `==host`, moat 0-FA. **Risk:** medium — it is a
genuine SNR lift on the documented rate-code-wall, but it must clear the floor on the WORST seed (seed 42, where ALL
four agent cleanups read 0.000) without admitting any moat leak.

### R-C — the swapped/1-role follow-on sequencers (the OTHER residual; independent of small-vocab)

Separately from the small-vocab issue, convert the remaining host scans on the ON path: **`query_agent`** (an
(action,patient) cue → a second match cascade) and **`render_fact`/`describe`** (an agent-only 1-role cue → a 1-role
cascade). These are the build-1 named follow-ons (`one_brain_composer.py:1008/1050`). They are clean reuse of the same
gated-disinhibition primitive (the loop is already role-agnostic — `_phaseC_task2_wholeturn_loop.py`
`block_role_scores`), they have the SAME code-margin dependency as `query_patient` (so they should be done AFTER R-A so
they inherit the high-margin codes), and they retire the last default-on host-cognition control flow on the
conversational path. **Effort:** ~1–2 days. **GO bar:** `query_agent`/`render_fact` `==host` at the validated K + the
moat. **Risk:** low (proven primitive).

### Ranking summary

| rank | option | what it fixes | effort | risk | recommend |
|---|---|---|---|---|---|
| 1 | **R-A1** (None-sentinel + stream/PPMI small codes) | the small-vocab default-on flip (removes the code-margin gap) | 0.5–1 d | low | **DO FIRST** |
| 1b | R-A2 (raise D / grounded small codebook) | same, if R-A1 too heavy for CI | 0.5 d | low | fallback for R-A1 |
| 2 | R-B2 (per-role re-kick) | lift fresh-random small-V margin off the floor (strict purity) | 1–2 d | med | if owner wants ON at fresh-random worst case |
| 2b | R-B1 (population redundancy) | same | 1–2 d | med | alt to R-B2 |
| 3 | R-C (swapped/1-role sequencers) | the OTHER host residual (query_agent/render_fact) | 1–2 d | low | after R-A (inherits margin) |
| last | R-B3 (confidence-gate graceful abstain) | converts the miss to a clean abstain (NOT recall recovery) | 0.5 d | low | only if R-A/R-B fail recall |

---

## 4. THE ANTI-CHEATS + DE-RISK + GO BARS + VERDICT

### 4.1 The anti-cheats — the moat is SACROSANCT, 0-FA at small AND large vocab (the HARD gate)

Per the prompt's explicit constraint and `feedback_moat_not_hard_lossy_memory_ok` (kept where free — and here it is
free, structurally protected by the gated match), every stage's GO REQUIRES:
1. **Moat 0-false-accepts at BOTH scales.** Every absent/cross cue → `None`/`"unknown"` at small vocab (V=15/K=4) AND at
   production (V=320/K=32). A single false-accept at any seed, any K, any vocab is a FAIL. **The small-vocab gap CANNOT
   be closed by lowering `match_thresh`** — the de-risk already shows `win_rate 0.000` is below every threshold, so
   threshold-lowering would either still miss (safe) or, past 0.0, start admitting noise (a moat breach). The fix MUST
   be upstream margin (R-A/R-B), NEVER a looser gate.
2. **Answer-identity `==host`** across the FULL who/what + abstention matrix (present cues answer the right block; the
   patient/agent label `==` the host `_scan` on the same store), at the relevant K, multi-seed.
3. **Sequencer-LESION fails SAFE** (sever the result→op drive → abstain, never a wrong block) — `lesion_fails_safe`,
   already in the de-risks.
4. **Permuted-rule INVERTS** (cyclic-shift `m{b}→ans{(b+1)%K}` → the decision follows the rule, not a fixed scan order).
5. **NO-DIVNORM raw control FAILS** (the on-bridge normalization is load-bearing).
6. **OFF == byte-identical** (the production regression guard — the rf/numpy-CPU path unchanged).
7. **Multi-seed:** the standing rule — 6 seeds for any noise-sensitive cleanup/match sweep; 3 seeds for the production
   320 confirm (the demo precedent).

**The crucial anti-cheat for THIS scoping:** any R-A/R-B GO must show the agent-role `win_rate` cleared the floor
**because the code margin rose** (R-A: the stream/PPMI codes; R-B: redundancy/re-kick), demonstrated by the same
`_burndown_1A_c2_smallvocab_derisk.py` margin read (`win_rate` per cue) going non-zero — NOT because the threshold
moved. The de-risk runner already reports `win_rate` at `match_thresh=0.0`, so this is directly checkable.

### 4.2 The cheap-first de-risk (smallest test that settles it)

- **R-A de-risk (the one real lever, run FIRST):** re-run `_burndown_1A_c2_smallvocab_derisk.py` (GPU, the EXACT agent
  config) but with the composer constructed on a **small stream-learned / PPMI codebook** (R-A1) — or higher-D
  fresh-random (R-A2) — instead of the default fresh-random V=15. **GO = the agent-role `win_rate` is non-zero on all 4
  cues, all 3 seeds, AND `present_ok 4/4` at `match_thresh=0.06`, AND `moat_0fa True`.** This isolates "is the cause the
  code margin?" in one run. If GO, R-A1 is the close: add the None-sentinel `integrated_loop` default + re-run
  `test_one_brain_composer_agent.py` (the 6 reverted tests) with onebrain auto-ON.
- **R-B de-risk (only if R-A is rejected):** the same runner with R-B2 (per-role `_dev_rekick_into` before the agent
  cleanup) on the UNCHANGED fresh-random V=15 codes. GO = the same `win_rate`-non-zero + `present_ok 4/4` + moat-0-FA on
  fresh-random codes (the strict purity case).

### 4.3 GO bars (when #4 is CLOSED)

#4's small-vocab default-on is CLOSED when, on the library `BrainConversationalAgent(composer_kind="onebrain")` (the
default object), `integrated_loop` is ON-by-default and:
1. `test_one_brain_composer_agent.py` (the 6 tests that reverted: `what_does`/`yes-no`/`reason_chain`/clause/multiturn)
   pass with the spiking sequencer, `==host`, moat 0-FA;
2. the rf/numpy-CPU/test-oracle path is byte-unregressed (the host `_scan` retained as the explicit oracle);
3. (R-C, for the OTHER residual) `query_agent` + `render_fact` route through spikes `==host` + moat 0-FA.

### 4.4 VERDICT (the SURPASS move-4 — closeable-and-how-cheaply vs irreducible-and-why)

**CLOSEABLE — and the cheapest close does NOT touch the sequencer or the threshold (it removes the code-margin gap
between the library default and the production path).** The "C-2 reverts at small vocab" headline mis-locates the
residual: the reverted op (the K-way sequencer) is CORRECTLY CONVERTED and GO at production (1-of-K=32 selection) and at
the harder 1-of-V=320 cleanup; what actually reverts is the **upstream matched-filter cleanup of the AGENT role at low
code fidelity** — `win_rate 0.000` at fresh-random V=15, which no threshold can recover and which the moat (correctly)
turns into abstention. The PRODUCTION path is GO precisely because its stream-learned codes have a wide cleanup margin
(target rates 0.116–0.196).

- **Cheapest close (R-A1):** make the library default carry production-margin codes (stream/PPMI small codebook) via the
  already-established None-sentinel pattern, then flip `integrated_loop` default-ON for the onebrain path. ~0.5–1 day, NO
  `sim/` edit, low risk. This is the SURPASS-consistent close: the genuine residual is TINY (a handful of low-margin
  agent codes) and the production path already solves it for free.
- **Stricter close (R-B):** if the owner wants the sequencer default-ON even on fresh-random worst-case codes, lift the
  upstream margin (per-role re-kick / population redundancy) — the documented rate-code-wall lift. ~1–2 days, medium
  risk.
- **The OTHER genuine host residual (R-C):** `query_agent` + `render_fact`/`describe` are STILL host on the ON path — a
  clean, bounded, independent conversion (more match cascades), to be done after R-A so they inherit the margin.

**The genuinely-irreducible part (IF R-A AND R-B both fail to clear the floor on fresh-random small-V codes without a
moat breach — no current evidence suggests this):** a characterized **fresh-random-code-at-small-V cleanup-margin
boundary** (the rate-code/point-neuron floor clipping a sub-floor recovered agent phasor) — in which case the host scan
stays the default at the numpy-CPU/test-oracle/fresh-random path and the spiking sequencer is the default at the
stream-learned/production path (where it already is). **This residual is a CODE-FIDELITY boundary, NOT a sequencer
boundary, and is NEVER closeable by loosening the match threshold or the moat** (the moat is 0-FA at every threshold at
both scales; the failure is a present-cue MISS, the safe direction). But the strong expectation — given the production
path's GO on margin-rich codes and the catalog-grounded lifts available — is **R-A GO**, i.e. the gap closes by giving
the library the codes the production path already uses, and the spiking sequencer becomes the onebrain default.

---

## 5. SOURCES (file:line / finding verified)

- **The host scan (the residual):** `research/runners/one_brain_composer.py` (`_scan`:917-921; `query_patient`:976-1003;
  `query_agent`:1005-1016 [still host]; `ask_yes_no`:1018-1041; `render_fact`:1043-1058 [still host];
  `query_chain`:1060-1069; `_seq_block`:884-915 [the host branch :889-895 + the spiking branch]; `_select`:539-545 +
  `_spiking_select`:507-537; `_read_blocks`:714-719; `confidence_gate`:256-262, `_margin`:500-505; `integrated_loop`
  default :116 + the docstring :131-143). `research/runners/rf_phasor_composer.py` (`_scan_first_match`:435-444 — the
  numpy `w==val` host loop + abstain header :12).
- **The small-vocab revert (the PRECISE failure):** `research/findings/raw/_burndown_1A_c2_smallvocab_derisk.json`
  (V=15/K=4, fresh random, D=128, 3 seeds: `win_rate 0.000` on ≥half the cues, `present_ok 0/4–2/4` at EVERY thresh,
  `false_accept 0`, `host_absent_all_none True`); `research/runners/_burndown_1A_c2_smallvocab_derisk.py:104-115`
  (`win = r[tgt]` = the correct block's `m{tgt}` rate at `match_thresh=0.0`; `decode` = `rates[j] > th`);
  `research/findings/2026-06-23-spiking-default-migration-PARTIAL.md` (the 6 agent tests reverted; the 2026-06-24
  UPDATE: NOT a `match_thresh` re-cal — the agent-role divnorm-WTA cleanup is ZERO firing; gain/sigma sweep found no
  operating point; root cause = LOW cleanup margin of fresh random codes at small V).
- **The 320-GO + the threshold:** `research/findings/2026-06-21-shortcut3-fold-integrated-loop-BUILD.md` (the 4-gate
  fold COMPLETE; V=320/K=32 GO seed 42, moat 0-FA; the 837K-neuron fabric; the memory-safe gate-3 unblock);
  `research/findings/2026-06-21-shortcut3-fold-integrated-loop-derisk.md` (K∈{2,4,8} 3-seed GO at `match_thresh=0.06`);
  `research/findings/2026-06-21-shortcut3-K32-capability-surpass.md` (the 0.15 NEGATIVE = wrong-threshold artifact;
  target rates 0.116–0.196, no-match floor 0.000; GO at 0.06).
- **The deployment + the default-on decision:** `research/findings/2026-06-24-closeout-audit-default-on.md` (C-2 STAYS
  default-OFF at the library = CHARACTERIZED divnorm code-margin BOUNDARY at small vocab; ON at the 320 demo);
  `research/runners/consolidated_320_conversation_demo.py:242` (`set_defaults(integrated_loop=True)`) + :105/132
  (feeds `_phaseB_stream_codes_320_neural` stream-learned codes via `grounded_codes`).
- **The sequencer (proven, K-generalized):** `research/findings/2026-06-19-onebrain-sequencer-derisk.md` (the K=2 kernel
  GO 6/6 — the gated-disinhibition match + BG production rule; the coincidence-AND walls vs gated-disinhibition robust);
  `research/findings/2026-06-20-burndown-3-S0-kway-sequencer.md` (K∈{2,4,8,16} GO 6 seeds);
  `research/runners/_phaseB_onebrain_sequencerK_derisk.py` (`build_sequencerK_bridge`/`run_sequencerK`).
- **The biology reframe (catalog, verified):** `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` —
  A.04 (BG WTA at GPi/SNr, :128-138 — the selection primitive, "implemented"), D.05 (CA3 recurrent autoassociator —
  pattern completion substrate, :1137-1148), D.13 (pattern completion — partial cue → full pattern; "too little → no
  retrieval", :1235-1246), D.23 (misplace/novelty detection — the familiarity/no-match signal, :1059-1066), E.07
  (ganglion rate code, :1415 — the rate-code-wall family). The Bogacz-Brown familiarity gate
  (`2026-06-11-familiarity-gate-v320-GO.md`) + the population-code rate-wall lift
  (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`) are the project's own neural realizations.
- **The reusable lifts:** `_dev_rekick_into` / `_loop_rekick` (`one_brain_composer.py:488-496`, the per-op clean-phasor
  re-kick, already default-on for the clause path); the None-sentinel default pattern (`brain_conversational_agent.py`,
  the 1A C-3/C-5 flips); the RF magnitude floor (`sim/bridge.py:5589`, the clip that bites at low margin).

_Read-only scoping deliverable. NO code written, NO experiments run, NO webapp. Every cited file:line + finding verified
against the source; the small-vocab failure mode (`win_rate 0.000`, threshold-independent) and the 320-vs-small code-
margin contrast were extracted from the committed JSON + the de-risk runner + the surpass finding. The no-confab moat is
the HARD gate in every proposed stage and is NEVER weakened by any option here — the small-vocab gap is an upstream
code-fidelity boundary, not a moat or threshold knob._
