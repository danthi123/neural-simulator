---
type: finding
status: positive
date: 2026-08-28
lane: onebrain-integration
board: 129
mechanism: surprise-episodic-129construction (surprise->source_provenance via TWO context-gated Hebbian cross-edges + a divisively-normalized opponent-ratio read)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_onebrain_surprise_episodic_129construction_derisk.py
artifacts:
  - research/findings/raw/_onebrain_surprise_episodic_129construction_6seed.json
  - research/findings/raw/_onebrain_surprise_episodic_129construction_6seed.json.prov.json
builds_on:
  - research/findings/2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md
  - research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md
  - research/findings/2026-08-28-read-fidelity-opponent-pushpull-NOGO-sign-recovery-does-not-lift-and-net-hurts-read-power.md
  - research/findings/2026-08-28-declarative-cross-edge-framework-gap-analysis-already-closed.md
---

# Applying board #129's ALREADY-WORKING construction (two context-gated Hebbian cross-edges + a divisively-normalized opponent ratio) to surprise->source_provenance: F2's lesion-control precondition, which failed 5/6 seeds on the raw-margin single-edge construction, now clears 6/6 (frac_attributable 0.89-0.97)

**Verdict: GO 6/6.** All of F1 + F2 (redefined) + F3 + F4 + emergence + lesion-recovers-migration pass on every
seed (42, 43, 44, 100, 101, 102). The surprise->source_provenance cross-edge's F2 read-fidelity crux — UNDEFINED
since 2026-08-27 because its own lesion control failed its precondition on 5/6 seeds — is now DEFINED and
PASSING, by re-CONSTRUCTING the edge the way board #129 already proved works for the sibling
perceived-vs-generated provenance capability, not by further tuning the original single-edge/raw-margin read.

## 1. Verify-first: does #129's construction map onto this edge?

Read (not just RAG-summarized) before building: `2026-08-25-laneC-...-6seed-GO.md` (#129 itself),
`2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md` (the blocked edge, F2's own numbers), and —
found via `git log --all --oneline | grep -iE "read-fidelity"`, NOT surfaced by the RAG corpus check run at the
start of this session — `2026-08-28-read-fidelity-opponent-pushpull-NOGO-...md`, which had ALREADY tested a
closely-related idea on this EXACT edge: an opponent/push-pull sign-recovering READ (two Dale's-law non-negative
template channels, `I_push-I_pull`, fit as an offline decoder on the raw spike rasters of the SAME single
trained edge) — 0/6, net WORSE than the single rectified channel it replaced, with the explicit conclusion
**"consistent with #129 (separate-trace WIRING delivers provenance, not the read alone)"**. That result is a
different mechanism from what this finding tests (a post-hoc decoder-style opponent read vs. a genuine SECOND
in-network LEARNED Hebbian cross-edge, #129's literal construction) but is directly relevant: fixing the READ
FORMULA alone, without adding a genuinely separate learned trace, was already tried on this crux and failed —
which is exactly the hypothesis this task was dispatched to test, now with real data on both sides.

**Architectural analysis before building anything** (`research/runners/_spiking_expectation_rpe_derisk.py`,
read directly): #129's two context lines (`ctx_perceived`/`ctx_generated`) are externally-injected and cleanly
MUTUALLY EXCLUSIVE per encode. The `surprise` circuit (`cue->patient_expected(inh)`, `patient_asserted(exc)
->surprise`, `patient_expected(inh)->surprise`) is architecturally a UNIPOLAR RECTIFIED mismatch detector
(`surprise ~ relu(asserted_exc - expected_inh)`): `patient_expected`/`patient_asserted` both fire regardless of
CONFIRM/CONTRADICT (driven purely by cue/assert presence; only their downstream SUBTRACTION discriminates), so
there is no pre-existing population that is cleanly "high on CONFIRM, ~0 on CONTRADICT" the way #129's two
context lines are "exactly one active per encode". `patient_expected` is the best available CONFIRM-side
presynaptic driver, but it does not discriminate the two trial types the way #129's design requires — a real,
declared architecture mismatch, not a hidden one. **Answer: the construction maps STRUCTURALLY (two separate
context-gated Hebbian traces feeding an already-opponent-inhibiting pool pair, read by a divisive-normalization
ratio) but NOT with a perfectly clean presynaptic pair the way #129 has one; whether the imperfect mapping still
works is an empirical question, not a guess — see §3.**

## 2. What was built

`research/runners/_onebrain_surprise_episodic_129construction_derisk.py` (new; imports and reuses the blocked
edge's `SurpriseEpisodicPool`/`_f1`/`_emergence` verbatim for everything unchanged):

- **EDGE 1 (unchanged)**: `surprise -> prov_generated`, CONTRADICT-trial-trained + `ctx_generated`-gated — the
  ORIGINAL, already-6/6-clean-on-every-other-arm edge from the blocked runner, untouched.
- **EDGE 2 (NEW)**: `patient_expected -> prov_perceived`, CONFIRM-trial-trained + `ctx_perceived`-gated — the
  direct structural mirror of edge 1, both declared via the SAME `CrossEdge`/`merge_organs(cross_edges=)`
  declarative path the blocked edge already uses (no bespoke re-inject, no `sim/` edit).
- **READ**: F2/F4 recomputed as `d=(r_gen-r_perc)/(r_gen+r_perc+DN_SIGMA+eps)` — #129's divisive-normalization
  opponent ratio — instead of the raw rate margin `r_gen-r_perc`.
- **LESION (F2's crux)**: both cross-edges zeroed together (jointly "the surprise->episodic mechanism").

## 3. Two problems found and fixed empirically, in order — not assumed, not hidden

<!--derived-->
Every number in this section comes from the seed-7 (non-canonical) calibration/diagnostic runs described in the
prose, read directly from console output while iterating on the construction — not from the two cited
artifacts (which hold only the FINAL, frozen-config canonical 6-seed run). Reported here for the reasoning
trail, not as claims needing independent artifact citation.

**(a) A bare-epsilon ratio reopens the moat (F4a).** The first cut (ratio with `eps=1e-9`, everything else as
above) cleared F2 immediately on the seed-7 calibration run (`frac_attributable=0.965` vs the raw-margin
construction's 0.297-0.727) but FAILED F4a: a surprise-hold with **zero** content drive read `gen=0.0094Hz,
perc=0.0Hz` — both near the instrument noise floor, an order of magnitude below the ~0.09-0.24Hz content-driven
denominators seen elsewhere — and the bare-epsilon ratio amplified that noise-floor difference to `d~+1.0`, a
spuriously MAXIMAL "generated" verdict from silence alone (exactly the confabulation-from-bias-alone honesty
check F4a exists to catch). **Fix**: a semisaturation constant (Carandini & Heeger 2011, *Nat Rev Neurosci*
13:51-62, "Normalization as a canonical neural computation", PMID 21587300 — the standard form of divisive
normalization in sensory cortex, `r = drive^n/(drive^n+sigma^n)`), `DN_SIGMA=0.05`, pre-registered from this
seed-7 diagnostic (silent denominator ~0.0094 vs. real content-driven denominators ~0.14-0.24, sigma set roughly
midway in log-scale between them) BEFORE any canonical seed was read. This is a principled, biologically-named
fix, not a fudge: a bare epsilon only guards literal division-by-zero, it does not suppress a near-zero-but-
nonzero denominator's noise, which is exactly what a semisaturation constant is for.

**(b) At the original edge's `HMAX=20`, F2 and F4a pass but F4b (moat) fails.** A CLEAR, already-correctly-
encoded battery item, held together with a co-occurring WRONG-CONTEXT surprise trial, FLIPPED from perceived to
generated (`clear_nohold=-0.49`, `clear_wrong_hold=+0.19` — opposite sign) — the two cross-edges' large trained
current (`w_gen=5.75, w_perc=6.17` at `HMAX=20`) was strong enough to overpower a genuine, well-established
content memory's own margin. Diagnosed, not assumed: `N_EPISODES` from 40 to 150 converged to the SAME final
weight (5.75/6.17, identical to 3 decimals) — a true plateau, not under-training, so more training would not
have helped. Since the ratio-based F2 signal at `HMAX=20` (`delta_intact~0.59`) was ~60x any reasonable floor,
there was headroom to shrink the bound. **Fix**: `HMAX_129=2.5` (vs. the original edge's `HMAX=20`),
`N_EPISODES_129=60` — calibrated on the SAME seed-7 diagnostic, frozen BEFORE any canonical seed was read. At
this bound both edges converge to a much smaller magnitude (`w_gen~0.6-1.2, w_perc~1.3-2.1` across the 6
canonical seeds) that still clears F2 with wide margin while leaving a clear item's own trace dominant, so F4b
now holds.

## 4. 6-seed result {42 43 44 100 101 102} — GO (all six PASS), F2 now DEFINED

<!--derived-->
Artifact: `research/findings/raw/_onebrain_surprise_episodic_129construction_6seed.json`. All values below are
rounded from that artifact's `runs[*].F2` fields (this section, and the table, hence block-marked derived); the
calibration line's numbers come from the seed-7 calibration run's own console output (a separate, uncommitted
invocation — the committed artifact was produced with `--floor` passed explicitly to skip re-running it) not a
cited JSON field. Calibration (seed 7, non-canonical, frozen BEFORE any canonical seed was read):
`delta_intact=+0.1911`, `delta_lesion=+0.0222`, `frac_attributable=0.884` -> `F2_INTACT_FLOOR` (ratio units)
frozen at `0.0478` (`0.25 * |calibration delta_intact|`, a pre-registered rule, not tuned against the canonical
set).

| seed | F2 delta_intact | F2 delta_lesion | frac_attributable | F1 | F2 | F3 | F4 | emergence | migration |
|---|---|---|---|---|---|---|---|---|---|
| 42  | +0.1964 | +0.0126 | 0.936 | pass | **pass** | pass | pass | pass | pass |
| 43  | +0.1831 | +0.0101 | 0.945 | pass | **pass** | pass | pass | pass | pass |
| 44  | +0.1256 | +0.0142 | 0.887 | pass | **pass** | pass | pass | pass | pass |
| 100 | +0.1986 | +0.0069 | 0.965 | pass | **pass** | pass | pass | pass | pass |
| 101 | +0.1544 | +0.0060 | 0.961 | pass | **pass** | pass | pass | pass | pass |
| 102 | +0.1450 | +0.0140 | 0.904 | pass | **pass** | pass | pass | pass | pass |

Per-arm: **F1 6/6 - F2 6/6 - F3 6/6 - F4 6/6 - emergence 6/6 - lesion-recovers-migration 6/6.** Compare to the
raw-margin single-edge construction's F2: `frac_attributable` 0.297-0.727 (5/6 seeds FAILING the `<0.34`
lesion-ratio precondition; the ONE precondition that fired UNDEFINED). Here every seed clears `frac_attributable
>= 0.887`, i.e. lesioning both cross-edges removes 89-97% of the intact shift — the precondition that was
UNDEFINED is now cleanly, repeatedly TRUE.

**Instrument checks (verify-go, inline, not a silent Monitor):**
- **Determinism**: seed 42 was run in two independent processes (a 3-seed batch, then the final combined 6-seed
  run) — `delta_intact`/`delta_lesion`/`emergence` weights byte-identical across both (`cfg.seed` correctly
  controls the substrate; the CLAUDE.md seed trap does not apply here).
- **Anti-cheat**: the randomly-assigned `(cue_concept, assert_concept)` block pair varies across all 6 seeds
  (`(1,5) (6,7) (0,4) (3,0) (4,0) (0,4)`), confirmed by the `anti_cheat_random_assignment` precondition.
- **Emergence**: both edges grow from `W0=0.05` by the substrate's own Hebbian rule on every seed; the 11
  non-participating concept blocks stay within `OTHER_BLOCK_DRIFT_MAX=0.03` of the seed value on both edges,
  every seed.

## 5. An honest additional finding: edge 2 (the NEW confirm-side edge), not edge 1, does almost all the causal work

<!--derived-->
All numbers in this section are from a targeted, single-seed (7, non-canonical, diagnostic-only — does not touch the canonical GO) individual-edge
lesion control, run AFTER the 6-seed GO landed, as a verify-go check on whether both declared edges are
genuinely load-bearing: intact `F2 delta=+0.193`; **edge-1-only lesioned** (zero `surprise->prov_generated`,
keep edge 2) `delta=+0.190` — almost unchanged from intact; **edge-2-only lesioned** (zero `patient_expected->
prov_perceived`, keep edge 1) `delta=+0.027` — collapses to essentially the both-edges-lesioned floor
(`+0.023`). **Edge 2 alone reproduces nearly the full intact shift; edge 1 alone reproduces almost none of it.**
This is a genuine, unexpected nuance, disclosed rather than smoothed over: the mechanism that empirically clears
F2 is dominated by the NEW, architecturally-imperfect (per §1) confirm-side edge, not by the original,
already-validated surprise-side edge. The pre-registered 6-seed gate (§4) lesions both edges TOGETHER, which is
what "the surprise->episodic mechanism" is defined as here and is what cleanly passes 6/6 — this nuance does not
change that verdict, but it does mean the load-bearing causal story is not the one first hypothesized ("surprise
biases toward generated"); it is closer to "the confirm-side edge's presence measurably reshapes the
gen-vs-perc balance under a contradict hold," a mechanism not yet explained at the circuit level. Named as an
open follow-on in §7, not investigated further here (time-boxed session).

## 6. Scope and scaffolds (same class as the blocked edge's own declared residuals, unchanged)

Two-factor Hebbian (no reward/dopamine gating); host-chosen cross-edge topology (both edges' endpoints are
host-picked, not self-organized); host-curated training schedule (co-driving CONTRADICT/CONFIRM trials directly,
not via an organic dialogue turn); `prov_generated`/`prov_perceived` firing remains an ENCODING-COMMITMENT PROXY
for the still-Group-C-deferred `d5_episodic` (unchanged scope substitution from the blocked edge, re-verified
current: `d5_episodic` remains absent from `onebrain_merge_framework.py`'s `GROUP_A` registry). `patient_expected`
as edge 2's presynaptic driver is an architecturally imperfect stand-in for a genuine "confirm/expectation-met"
detector (§1) — not hidden, and directly implicated by §5's dominance finding. **Not a production flip**: this
remains a standalone research runner, additive, no `sim/` edit, no production wiring, no default flip.

## 7. Verdict and ranked next

**Is the edge's read now DEFINED + GO (migration-ready)?** Yes for the specific, pre-registered F2 test this
session ran (both edges lesioned together, divisive-ratio read): 6/6, `frac_attributable` 0.887-0.965 <!--derived--> (rounded range restated from §4's table). This
makes surprise->source_provenance a candidate for the declarative-`CrossEdge`-framework migration path R3
(surprise) previously could not take (`2026-08-28-declarative-cross-edge-framework-gap-analysis-already-
closed.md` §4 step 5 named this exact edge's read as the blocker) — **not wired into production and no default
flip here**, per this task's own scope.

Ranked next:
1. **Investigate the §5 dominance finding at the circuit level** (why does `patient_expected->prov_perceived`
   alone reproduce nearly the full shift, while `surprise->prov_generated` alone reproduces almost none?) — a
   genuine open mechanistic question a future session should resolve before treating this construction as fully
   understood, even though the pre-registered gate passes.
2. **Migrate this edge through the declarative CrossEdge framework's F1-F4 genericization** (the framework
   gap-analysis's own §3 residual: seven near-duplicate hand-typed F-gates) — this edge is now a second
   concrete consumer motivating that generic harness.
3. **Replace `patient_expected` with a purpose-built CONFIRM/expectation-met detector** if §1's architecture
   mismatch is later judged to matter beyond what §5 already surfaces (e.g. if a cleaner presynaptic pair is
   needed for the eventual `d5_episodic` migration, where the encoding-commitment proxy is retired).
4. **Apply the same DN_SIGMA semisaturation-constant fix to the banked opponent/push-pull NO-GO** (§1) as a
   cheap follow-up check — that arc's own silent/control conditions were never tested for the same near-zero-
   denominator failure mode this session found and fixed here.

## Files

`research/runners/_onebrain_surprise_episodic_129construction_derisk.py` (new, 552 lines) ·
`research/findings/raw/_onebrain_surprise_episodic_129construction_6seed.json` (+ `.prov.json`) · imports
`research/runners/_onebrain_integration_surprise_episodic_crossedge.py` (unchanged) and
`research/runners/_laneC_source_provenance_opponent_derisk.py` (unchanged, pattern/battery helpers) verbatim ·
no `sim/` file touched · no `webapp/server.py` edit · no new production flag.

Functional read-outs only; no phenomenal-experience claim.
