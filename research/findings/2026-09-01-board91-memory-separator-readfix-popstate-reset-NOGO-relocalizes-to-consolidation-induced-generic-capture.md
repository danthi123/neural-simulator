---
type: finding
status: no-go
date: 2026-09-01
runner: research/runners/_replay_dg_pattern_separation_readfix_popreset.py
artifacts:
  - research/findings/raw/sep_readfix/popreset_6seed.json
---

# Board #91 scope + cheap de-risk: resetting the population set-point's host-side controller state at the read boundary does NOT fix the memory-separator READ — and the collapse turns out NOT to be specific to the subordinate cue at all

**Board #91** — lane-H memory-separator, the READ-side frontier the #90 finding named
after banking the dg->answer WRITE family (per-granule output transform, BCM
selectivity) as exhausted. This finding tests the most concrete, cheaply-testable
form of the #90 finding's own candidate (2), "characterize + clear the
post-consolidation reactivation persistence," and reports a decisive 6/6-seed NO-GO,
plus a new re-localizing measurement: the collapse the #90 finding characterized is
NOT specific to the subordinate memory's own cue. A genuinely novel, untaught,
independently-drawn input pattern collapses onto the dominant engram just as strongly
(mean Jaccard 0.90 post-consolidation vs 0.44 pre-consolidation, 6/6 seeds), which
neither the #90 finding's own diagnostic nor #73/#78 measured.

## Verdict

**NO-GO** for "reset `bridge._pop_state` at every read/probe boundary" as the fix for
#91. All three preconditions hold (the reset verifiably lands on every seed; the OFF
arm reproduces the #90/#78 residual exactly, matching its committed DG-Jaccard values
to 2 decimals on every seed; the dominant memory's own recall does not regress), so
this is a real negative, not an instrument failure. The control shows **exactly zero
separation** between the ON and OFF arms (`m1_reactivates_own` stays 0/6 either way) —
the manipulation demonstrably executes and changes nothing about which engram the read
recruits.

## What was built (the named mechanism, no `sim/` edit)

`bridge._pop_state` (`ever`/`integ`/`drive`/`silent`) is a plain Python dict the #78
population-set-point controller (`_install_pop_controller`,
`_replay_dg_pattern_separation_popsetpoint.py:125`) attaches to the bridge instance
and updates from inside a monkey-patched `bridge._run_one_simulation_step`. It is host
bookkeeping, not a `cp_*` numeric array, so the #90 finding's "snapshot every
per-neuron cp_* array" check could not see it by construction. `_drain()`
(`_reset_dynamics`, `_replay_dg_pattern_separation_gate.py:80`) zeros
membrane/conductance/synapse-timer/Hebbian-coactivity arrays **directly** — it never
calls `bridge._run_one_simulation_step` — so the controller's own documented
"2 consecutive silent DG steps -> reset ever/integ/drive" event-boundary reset never
gets a chance to fire during `_drain`. The #90 finding's own diagnostic only reset
`pop_state` inside `_isolated_reactivation_rate` (the BCM twin-selectivity signal, run
on a fresh, never-consolidated twin bridge where the reset is close to redundant); the
actual behavioral read (`_probe`) and the read-time reactivation diagnostic
(`_read_reactivation`) never did. The #90 finding's "not restored by clearing the #78
pop-controller integrator" cleared `integ` only — with `ever` left populated,
`err = nfired - k` recomputes to roughly the full excess on the very next step, and the
**proportional** term alone (`kp * err`, kp=45) reproduces most of the saturated
basket drive before `integ` could matter, which predicts exactly the reported
non-fix. This file clears all four fields at the top of every read/probe call
(`_read_reactivation`), completing candidate (2) from the #90 finding in the form that
had not actually been tested.

Biology motivating the lever (still a valid mechanism in general, just not the locus
here): real dentate/CA3 feedforward inhibition is fast and input-locked (Pouille &
Scanziani 2001, *Science* 293:1159-1163 — feedforward inhibition enforces a narrow
temporal window so a cell's read-out tracks the current input volley, not residual
ongoing activity), and CA3 pattern completion is understood to retrieve from the
presented cue, not from whichever ensemble was active before it (Neunuebel & Knierim
2014, *J Neurosci* 34:3999-4009). A slow controller whose state survives across cue
boundaries is exactly the kind of persistent process this predicts should NOT decide a
clean read — which is why the lever was worth the cheap test, even though it turned
out not to be the operative locus.

## Results — 6 seeds (42/43/44/100/101/102), pop_state reset ON vs OFF (`popreset_6seed.json`)

<!--derived-->
_Numbers from the cited artifact. `own` = the cue reactivates its own engram
(Jaccard to own > Jaccard to the other); PRE = pre-consolidation baseline
(fresh, never-consolidated bridge, same seed)._

| seed | dgJ(e0,e1) | OFF m1→(e0,e1) | ON m1→(e0,e1) | m1 own? | m0 own? (ON) | novel→e0 (ON) | novel→e0 (PRE) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 42  | 0.56 | (1.00, 0.56) | (1.00, 0.56) | no | yes | 0.95 | 0.52 |
| 43  | 0.47 | (1.00, 0.47) | (1.00, 0.47) | no | yes | 0.85 | 0.45 |
| 44  | 0.53 | (1.00, 0.53) | (0.93, 0.59) | no | yes | 0.86 | 0.34 |
| 100 | 0.56 | (1.00, 0.56) | (0.98, 0.58) | no | yes | 0.96 | 0.48 |
| 101 | 0.48 | (1.00, 0.48) | (1.00, 0.48) | no | yes | 0.88 | 0.40 |
| 102 | 0.60 | (1.00, 0.60) | (1.00, 0.60) | no | yes | 0.90 | 0.43 |

**Pooled:** `m1_reactivates_own` ON **0/6**, OFF **0/6** (zero separation — the
manipulation lands and changes nothing); `m0_reactivates_own` ON **6/6** (no
regression); `novel_no_spurious_completion` (Jaccard to both engrams < 0.5) ON **0/6**
— a genuinely novel, untaught cue completes to the dominant engram on every seed, ON
or OFF, reset or not. Mean novel→eng0 Jaccard: **0.437 pre-consolidation** vs **0.900
post-consolidation** (`build_bridge_popsetpoint` fresh vs after `base._consolidate`,
same seed, same wiring) — the collapse is consolidation-induced, not a static wiring
degeneracy, but it captures ANY sufficiently strong drive, not specifically the
subordinate memory's cue.

## The re-localization (narrows, but does not resolve, the #90 "characterized, locus
not fully isolated" mystery)

Two dissociations, both new relative to #90:

1. **Not the population-controller's host state.** The #90 finding's remaining
   candidate — an "uncaptured global/instrument state the down-state reset does not
   clear" — had one clean, named suspect (`bridge._pop_state`) once the code is read
   closely. Clearing it completely, at the correct call sites, on every seed, changes
   nothing. Combined with #90's own per-neuron `cp_*` snapshot check (also clean), the
   uncaptured state is neither a `cp_*` array nor the pop-controller's host dict. What
   remains unaccounted for: any OTHER host-side bookkeeping the popsetpoint/bcm
   runners attach to the bridge instance (none found by inspection of
   `_replay_dg_pattern_separation_{bridge,popsetpoint,bcm,readout}.py`), or a
   per-synapse quantity outside the fixed (non-plastic) DG-competition weights that
   this file did not separately snapshot.
2. **Not specific to the subordinate cue.** The #90 finding characterized the defect
   as "the subordinate memory's own input reactivates the dominant engram." This file
   shows the capture is broader: an independently-drawn, never-taught pattern (the
   `_input_patterns(seed, cfg, "dissimilar")["m1"]` draw — a different RNG stream than
   either taught memory) reactivates the SAME dominant engram just as strongly
   (mean 0.90 vs the taught subordinate cue's 0.94 mean to eng0 in this run's OFF arm).
   This reframes #91: the mechanism to build is not "restore the subordinate memory's
   privileged access to its own engram" specifically — it is "make the post-
   consolidation DG read CUE-LOCKED again" for essentially any input, which is
   candidate (3) from the #90 finding (a read-time attractor/novelty gate) rather than
   an instrument-level clear. Candidate (1) (a CA3-style autoassociator completing
   onto the cue's own private core) remains untried and is now better-motivated: the
   generic capture is consistent with the DG competition's winner set becoming
   INPUT-INDEPENDENT after consolidation (something makes a fixed subset of granules
   win almost regardless of which input drives them), which is exactly the failure
   mode a genuine per-cue completion/attractor stage would need to correct rather than
   an instrument reset.

## Where the next lever should go (banked, not built here)

Given (1) above, the productive next move is almost certainly to look for what makes
the DG competition's winner set input-independent post-consolidation, rather than
another host-state-reset variant. Two concrete candidates, in order of cheapness:

- **Per-cue z-scored competition (candidate 3, made concrete).** Normalize each DG
  granule's effective threshold/gain by its OWN drive under the CURRENT cue relative
  to its drive under a broad reference set, so competition selects for
  cue-RELATIVE excitation rather than absolute excitation. This is the DG "detonator
  gain must be relative, not absolute" framing already used successfully elsewhere in
  this codebase (`2026-08-10-gap5-leverA-DG-detonator-gain-...`) for a different arc's
  completion, and is a runner-side transform on the read step, no `sim/` edit.
- **A genuine CA3-style completion stage (candidate 1).** Add a small recurrent
  auto-associative layer downstream of DG that completes toward the memory's own
  PRIVATE granules specifically (not toward whichever DG cells the raw feedforward
  competition favors) — the heavier build the #90 finding flagged; only worth it if
  the cheaper candidate above is also tried and insufficient.

**GO-gate for either lever** (unchanged from #90's bar, now sharpened with the novel-cue
anti-cheat this file adds): `m1_reactivates_own` (Jaccard to own engram > Jaccard to
the other) on >=5/6 seeds, WITH the OFF arm reproducing the #90/#78 residual
(dissociation), WITH no regression on `m0_reactivates_own`, AND WITH a genuinely novel
untaught cue NOT completing to either taught engram (Jaccard < 0.5 to both) —
this file's own checks already implement that gate; only the manipulation function
(`_read_reactivation`'s per-cue competition transform, or a new completion stage)
needs to change.

    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readfix_popreset \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/sep_readfix/popreset_6seed.json

(Runs in well under a minute on CPU/numpy; `n_dg=200`, tiny substrate — cheap to
iterate on either candidate above using the same harness, swapping only
`_read_reactivation`'s body.)

## Tracked scaffolds (host, not brain)

Inherited from the #71/#78/#90 runners: host-defined input patterns + answer
assemblies; host reinstatement of each memory's input during replay; scheduled
down-states; the WRITE/READ transmission-gate phase; a rate-window Hebbian coactivity
write; the population set-point PI controller (host, unmodified here). NEW this file:
the read-boundary `pop_state` reset is a runner-side host-dict clear (not a `sim/`
edit, a no-op on any substrate without the popsetpoint controller); the
pre-consolidation baseline measurement (a third, never-consolidated bridge) is new
instrumentation, not a mechanism.

## Sources

EXTERNAL-SEARCH-RAN: CA3/DG pattern separation vs completion; feedforward vs
population-level inhibition timing; engram dominance/capture (logged to the
corpus-check record, 2026-09-01, via `tools/before_you_build.sh`).

- Pouille, F., Scanziani, M. (2001). Enforcement of temporal fidelity in pyramidal
  cells by somatic feed-forward inhibition. *Science* 293:1159-1163. — fast,
  input-locked feedforward inhibition as the biological contrast to the slow,
  history-carrying population controller tested (and refuted as the locus) here.
- Neunuebel, J.P., Knierim, J.J. (2014). CA3 retrieves coherent representations from
  degraded input: direct evidence for CA1 pattern completion and dentate gyrus
  pattern separation. *J Neurosci* 34:3999-4009. — completion is understood to
  operate on the PRESENTED cue; motivates why a cue-independent winner set (this
  finding's re-localization) is the wrong regime, not merely an unlucky history.
- Leutgeb, J.K., Leutgeb, S., Moser, M.-B., Moser, E.I. (2007). Pattern separation in
  the dentate gyrus and CA3 of the hippocampus. *Science* 315:961-966. — already cited
  by the #90/#78 lineage; DG orthogonalizes, CA3 completes.

Internal: reads and does NOT re-derive
`2026-08-19-memory-separator-BCM-selectivity-write-writes-private-granule-but-NOGO-relocalizes-to-read-reactivation.md`
(#90 — the WRITE-family exhaustion + the read-reactivation-collapse characterization
this file tests one candidate fix for) and
`2026-08-19-replay-dg-pattern-separation-popsetpoint` / board #78 (the population
set-point consolidation this file's OFF arm reproduces exactly). Board ledger:
`GAP_CLOSURE_MISSION.md` line naming #91 as "READ-TIME (CA3 completion onto the
private core / clear reactivation persistence)"; this finding banks the "clear
reactivation persistence via the pop-controller" sub-candidate as tried and
insufficient, and sharpens the remaining two candidates with the novel-cue evidence.
