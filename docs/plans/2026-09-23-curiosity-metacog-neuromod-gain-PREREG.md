# Pre-registration (v2, superseding v1): a spiking locus-coeruleus-analog population diffusely projecting onto curiosity's ASK pool, driven by metacog's own comparator through real synapses

**AMENDMENT HISTORY.** v1 (2026-09-23, commit `7ba619532`) pre-registered a mechanism built as a HOST rate relay:
a `GainRecorder` read the comparator's firing fraction each step in Python and wrote it to
`core_config.current_novelty_signal`, which the neuromodulator subsystem's own concentration ODE then turned into
current on `ask`. Adversarial review (key `v2:fcc2f777`) correctly rejected this: `GainRecorder._step` computing
`sum(firing)/len(firing)` is host code standing between two neural populations, not a synapse, and CLAUDE.md
lists neuromodulators among the things that must be neurons/synapses. **This v2 document REPLACES the mechanism
under test** (§1 below), written and committed BEFORE any new run of the rebuilt mechanism — the v1 seed-42
artifact (`_curiosity_metacog_neuromod_gain_smoke_s42.json`) is VOID under this rebuild; a fresh seed-42
calibration run follows in a SEPARATE, later commit (this is the `gates/prereg_before_run` discipline: a
prereg commit carries no `research/findings/raw/**` artifact). **CORRECTION (fix round 2, per adversarial
re-review):** the v1 artifact was NOT removed in the same commit as this document. The repository log shows
it was actually deleted in commit `57fad1d0c` (the fix round's merge-origin/main commit, whose message names
only the FAILURE_LOG/matrix cleanup and does not mention this deletion), one commit BEFORE this v2 document's
own commit `72b744c4c`. The artifact was void either way (it measured the superseded host-relay mechanism)
and no run ever cited it as evidence for v2, but the prose above misstated which commit did the removal;
this note corrects the record rather than rewriting already-published history.

**The two pool lines staged from `f9dbafc9c`** (the v1 mechanism's isolated revision, referenced in that build
round's `honest_residuals`) **are SUPERSEDED by this document** — they governed a mechanism (a host relay) that
no longer exists in this file. They are not removed here (`research/queue/*` is live state this fix round does
not edit), but any result they eventually produce is NOT a verdict on the mechanism this document describes and
must not be read as one.

## 0. Why this rung, and what wall it answers (unchanged from v1)

`docs/plans/2026-09-23-curiosity-metacog-conflict-xedge-PREREG.md` built ONE declared point-to-point CrossEdge
(metacog's margin comparator `meta_schema` -> curiosity's `ask` pool, fixed weight 4.0). Its 6-seed verdict
(`research/findings/2026-09-23-cpu-lane-harvest-curiosity-metacog-conflict-xedge-6seed-no-go.md`) was **NO-GO**:
4/6 seeds pass (3/5 held-out), and — the part this rung targets — **even where it passes, the synaptic drive
alone sits 3x-14x BELOW production's own 19-24 Hz curious threshold** (S1, secondary in that PREREG). The review
correction on that lane named the next lever explicitly: *"the missing companion drive / neuromodulatory gain
that the real system runs"* — not another retune of the same fixed-weight edge.

**The wall question first (CLAUDE.md's standing rule): what does the real system run alongside a single
excitatory synapse that the conflict_xedge rung replaced with a constant?** Answer, grounded externally
(recorded `research/queue/.external_searches.jsonl`, lane=`curiosity`, 2026-09-23): **Aston-Jones & Cohen (2005),
Annu Rev Neurosci 28:403-450, "An integrative theory of locus coeruleus-norepinephrine function: adaptive gain
and optimal performance."** The LC-NE system's tonic mode, driven by cortical utility/conflict monitoring
(ACC/OFC), broadcasts a global population excitability gain — not a point-to-point glutamatergic increment.

## 1. Claim under test (v2 — the STRUCTURAL claim is now primary; v1's §1 stated the operating-point claim as
primary and left the S1 amendment (§6) contradicting it — this is the fix)

**CORRECTED (fix round 2, per adversarial re-review):** the claim below originally read "driven through TWO
co-existing, independently-lesionable, ALL-SPIKING pathways" and, in §4, "TWO co-existing, INDEPENDENTLY
lesion-attributable, ALL-SPIKING pathways". Both overstate what the seed-42 data show. The `edge_lesion` arm
(point-edge off, `lc_ne` pathway intact) is 0.0 Hz at every one of the 11 evidence levels, on every seed measured
so far — i.e. `lc_ne` alone drives NOTHING; `attrib_edge` (§below data) reads 1.0. `lc_ne` is therefore NOT a
second, independent driver of `ask`; it is a SUB-THRESHOLD MODULATOR of the edge-driven response — its own
contribution only shows up as a change in the COMBINED arm's dynamic range on top of the edge's drive (G3), never
as a response it can produce by itself. This is closer in kind to a gain than to a parallel pathway, and is said
so explicitly here rather than left implicit in the data. The corrected claim:

Curiosity's ASK pool firing is a monotone, class-symmetric, mechanism-specific function of metacognition's own
spiking margin computation, driven through TWO co-existing, ALL-SPIKING pathways — ONE that drives `ask` on its
own (the point-edge) and ONE that only MODULATES the first one's response and is silent alone (`lc_ne`):

1. **The frozen point-to-point edge** (unchanged from the conflict_xedge rung): `x_metacog_meta_to_curiosity_ask`,
   `meta_schema -> ask`, fixed weight 4.0.
2. **NEW (v2): a dedicated spiking relay population, `lc_ne`** (a locus-coeruleus analog, `LC_N` neurons,
   excitatory-typed, no internal recurrence), wired through TWO fixed-weight, dense, uniform CrossEdges:
   - `x_metacog_meta_to_lc_ne`: `meta_schema -> lc_ne` (the comparator's excitatory principal cells — NOT its
     inhibitory relay `meta_margin_fs` — drive `lc_ne`; this is also the more anatomically apt choice, since
     LC's cortical afferents are glutamatergic projection neurons, not local interneurons).
   - `x_lc_ne_to_curiosity_ask`: `lc_ne -> ask`, DIFFUSE (every `lc_ne` neuron synapses onto every `ask` neuron).

**No host scalar of any kind reaches `ask` or `lc_ne`.** Host code only (a) drives metacog's input evidence,
exactly as production does, and (b) reads out the resulting spike rasters for logging/statistics AFTER a step
has already run — it never writes into a step before it runs. There is no modulator ODE, no `current_novelty_
signal`, and no `enable_neuromodulator_subsystem` in this mechanism (v1 had all three; v2 has none).

### v1-vs-v2, and why the rebuild (kept for the record, not part of the claim under test)

v1's `GainRecorder` counted BOTH `meta_schema` and `meta_margin_fs` spikes toward the host relay's rate
(sign-blind: a spike from either region counted the same). v2 cannot do this with real synapses without
confronting Dale's-law sign: `meta_margin_fs` is typed inhibitory (`exc_fraction=0.0`, it is the comparator's own
lateral-inhibition relay), so a real synapse from it onto `lc_ne` would deliver INHIBITORY current, not add to a
sign-blind rate the way host summation did. Rather than build a mixed excitatory/inhibitory afferent onto `lc_ne`
(which changes the read from "population activity, unsigned" to "net synaptic drive, signed" — arguably MORE
faithful, since real neurons integrate signed current, not raw spike counts, but a different constant regime),
v2 drives `lc_ne` from `meta_schema` alone. `meta_schema`'s own firing rate already reflects the net effect of
the relay's inhibition (the relay acts ON `meta_schema`, so `meta_schema`'s rate already carries that
computation) — this is a genuine simplification, declared, not hidden, and it is why v2's calibration constants
(§2) do not reuse any v1 numeric value.

## 2. Operating point (declared; calibration seed = 42 only)

Inherited unchanged from the conflict_xedge PREREG: `CMP_EXC=(1.2,2.4)`, `CMP_REL=2.5`, `CMP_INH=(6.0,14.0)`,
relay 2x30, `XEDGE_W=4.0`.

NEW for v2, chosen from calibration probes on **seed 42 only**, 2026-09-23, before the calibration run this
document governs commits: `LC_N` (population size), `CMP_TO_LC_W` (`meta_schema -> lc_ne` uniform dense weight),
`LC_TO_ASK_W` (`lc_ne -> ask` uniform dense weight). All three are hand-set, seed-42-only, and declared as
residuals on `LC_ORGAN.scaffold_residuals`. The calibration criterion (identical in kind to v1's, re-run against
the new mechanism): the LARGEST `LC_TO_ASK_W` that (a) keeps G1 (monotone, rho<=-0.8) and G7 (class-symmetric)
intact and (b) clears G3's floor (the gain pathway independently carries >= `G3_GAIN_ATTRIB_MIN` of the combined
ASK dynamic range) is frozen; if no value clears both simultaneously, the AMENDMENT LOG (§6) records exactly
which constraint failed and at what values, per `docs/BUILD_LANE_CHECKLIST.md`'s "do not stage a run whose
pre-registration already predicts failure" rule.

- Seeds 43/44/100/101/102 are HELD OUT; the verdict reports them separately, exactly as the prior PREREG did.
- `LC_N`/`CMP_TO_LC_W`/`LC_TO_ASK_W` are fit ONLY on seed 42's response under the evidence grid — never on any
  held-out seed's ASK response, and never on whether a held-out seed passes a gate (that would be circular).

**v3 amendment (fix round 2): the calibration criterion above is EXTENDED, not replaced, by G10 (§3).** `CMP_TO_
LC_W`/`LC_TO_ASK_W` must now ALSO leave `lc_ne`'s own per-level firing evidence-graded (G10), not merely leave
`ask`'s COMBINED response monotone (G1) — a constraint the original v2 calibration could not check because
`lc_ne`'s own rate was not recorded. **Measured: `CMP_TO_LC_W=30.0` DOES saturate `lc_ne`** (rho(evidence, lc_ne
Hz)=+0.93, 40-44 Hz at every one of the 11 levels — G10 FAILS). `CMP_TO_LC_W` is RECALIBRATED to **5.0** (seed 42
ONLY; a coarse scan over {5, 5.5, 6, 8, 10} found the graded regime narrow — 5.0 gives rho_lc=-0.963, 5.5 gives
-0.766, and >=8.0 already flips positive/saturated), re-checked against G1/G3/G7/G8/G10 TOGETHER at the new value
(all pass: G1 rho=-0.991, G3 attrib_gain=0.246, G7 rho_swap=-0.991, G8 rho_relay=0.982, G10 rho_lc=-0.963 range=
2.78 Hz), and disclosed here rather than silently overwritten. `LC_TO_ASK_W=1.0` is UNCHANGED (it governs `lc_ne`'s
OUTPUT onto `ask`, not its own input drive) and was re-verified, not re-tuned, at the new operating point.

## 3. Gates (per seed; GO requires every REQUIRED gate on 6/6 seeds 42/43/44/100/101/102)

**RECLASSIFICATION (v2, per review v2:fcc2f777): G4 and G5 are INTEGRITY checks, not required evidence.** The
review found both pass BY CONSTRUCTION on v1's own data: this pool has NO path from `ask`/`lc_ne` back into
metacog (so a metacog-exact check under an ask-side lesion cannot fail by the pool's own topology — G5), and
v1's edge-lesion arm was already flat (0 Hz at every level), so a joint (both-pathways) lesion was trivially
also flat, making G4 pass on the `rho_both is None` branch regardless of whether the new pathway does anything
(G4). v2's rebuild does not remove the structural fact behind either: there is still no feedback path for G5, and
whether `edge_lesion` happens to be flat on v2's own calibrated weights is exactly the load-bearing-magnitude
question G3's floor already measures directly. Both are still SCORED and reported per seed; neither gates GO.

| id | required | measures | pass condition |
|---|---|---|---|
| G1 | yes | monotone coupling, COMBINED (edge+gain) arm | Spearman rho(evidence, level-mean ASK Hz) over 11 levels <= -0.8, AND intact ASK range >= 1.0 Hz (else UNDEFINED = fail) |
| G3 | yes | the `lc_ne` pathway is independently load-bearing | closing `lc_ne -> ask`'s transmission_gate (edge intact) removes >= `G3_GAIN_ATTRIB_MIN` of the COMBINED arm's ASK dynamic range (`tools.lab.attributable_to`); a floor, not the full effect, because the edge alone already carries some of the range |
| G6 | yes | determinism | the combined-intact digest (sha256 over per-rep ASK rates, balances, raster hashes) is identical in a FRESH subprocess |
| G7 | yes | class-symmetry anti-cheat | evidence driven into the OTHER assembly: rho <= -0.8 on the combined arm |
| G8 | yes | mechanism specificity | lesioning metacog's comparator relay (relay->meta inhibition), edge+gain intact: rho > -0.5, or UNDEFINED — the coupling needs the margin computation, not just metacog activity or raw comparator noise |
| G4 (INTEGRITY, not gating) | no | joint necessity | with BOTH the edge lesioned AND `lc_ne -> ask` closed, the coupling FAILS G1's own bar (rho > -0.8, or UNDEFINED from a flat arm) |
| G5 (INTEGRITY, not gating) | no | metacog unchanged, EXACT, under every lesion arm | metacog balance, threshold, confident flags (==) and workspace/workspace_fs + comparator spike-raster sha256 identical to the combined-intact arm |
| S1 (secondary, reported not gating) | no | reaches the PRODUCTION operating point | on each seed: `max(ask_hz at an uncertain/not-confident level) >= threshold_hz` (the seed's OWN `CuriosityProductionOrgan` calibration) AND `max(ask_hz at a confident level) < threshold_hz` |
| G9 (secondary, reported not gating) | no | permutation null | Spearman rho over the 88 per-rep observations vs 10,000 permutations, one-sided p<=0.01 — same pseudo-replication caveat as the conflict_xedge review raised |
| G10 (v3, REQUIRED) | yes | `lc_ne`'s own firing is evidence-GRADED, not a saturated tonic bias | on the COMBINED arm: Spearman rho(evidence, `lc_ne` level-mean Hz) <= `LC_GRADED_RHO_MAX` (-0.3) AND `lc_ne`'s own Hz range across the 11 levels >= `LC_GRADED_MIN_RANGE_HZ` (0.3 Hz); a None rho (zero-variance/saturated) or a sub-floor range both FAIL, never UNDEFINED-as-pass |

**G8's None-never-passes handling** (the bug the review flagged on v1's PRIOR rung, the conflict_xedge one) is
kept: a genuinely flat relay-lesion arm reports UNDEFINED and never silently passes. **v2 also moves BOTH G4's
and G8's predicates into module-level functions (`gate_g4_joint_lesion`, `gate_g8_relay_lesion`) that `run_seed`
and `--selftest` both call** — the review's other flagged gap (v1's selftest asserted its OWN local copies of the
gate logic, with hard-coded thresholds, so a real regression in `run_seed`'s scoring would not have been caught
by `--selftest` passing). **v3 (fix round 2) adds `gate_g10_lc_evidence_graded` to the same shared module-level
set, tested by `--selftest` identically.**

**WHY G10 EXISTS (fix round 2, per adversarial re-review).** `lc_ne`'s own firing was never recorded in v2 — the
artifact only carried `ask`'s response, never `lc_ne`'s. `CMP_TO_LC_W=30` was chosen ~10x above `lc_ne`'s firing
ONSET (measured 0 spikes at weight 3.0, ~940 at weight 30.0), but only at evidence=1.0, a single point. Without a
per-level reading, a near-ceiling (saturated) `lc_ne` could look, from the ASK-side data alone, exactly like "the
gain pathway carries the comparator's margin signal" (G3 passing) while actually delivering a roughly CONSTANT
excitatory bias whose apparent evidence-dependence in the COMBINED arm comes entirely from `ask`'s own threshold
nonlinearity interacting with the edge's graded drive — not from anything graded in `lc_ne` itself. G10 measures
`lc_ne`'s own per-level Hz (now recorded by `RecorderWithLC`/`coupled_sweep_lc`, `research/runners/
_curiosity_metacog_neuromod_gain_derisk.py`) directly, so this distinction is data, not inference from `ask` alone.

Integrity smokes (reported, NOT counted as evidence — pass by construction if the code is right; v2 IMPLEMENTS
the byte-off check v1 only promised):
- **byte-off (v2: actually run, not a hard-coded `True`).** The combined pool's base connectivity, with the
  point-edge and BOTH `lc_ne` CrossEdges excluded (which excludes every `lc_ne` synapse, since it has no internal
  connectivity of its own), is compared via `onebrain_crossedge_gate.verify_byte_off` against the conflict_xedge
  rung's own `coupled=False` pool, built by literally calling that rung's own `build_pool(seed, coupled=False)`
  — an exact (`==`) dict-of-(row,col)->weight compare, not `allclose`.
- restore-exact: re-reading after every lesion arm restores the combined-intact digest.
- `neuromodulator_subsystem_enabled` / `no_host_novelty_signal`: read directly off `pool.bridge.core_config`
  after building the pool (a real read, not a hard-coded claim — v1's runner had
  `"gain_pathway_off_by_default_on_base_organ": True` as a literal, never measured; v2 has no neuromodulator
  installation anywhere in this mechanism, so both read as their off/zero defaults by construction of the code
  path, and the artifact shows the actual measured value either way).

## 4. What a GO would and would not mean

- **GO means:** on this 2-organ (+comparator, +`lc_ne`) merged pool, curiosity's ASK pool firing is a monotone,
  class-symmetric, mechanism-specific function of metacognition's own spiking margin computation, driven through
  TWO co-existing, ALL-SPIKING pathways: the frozen point-to-point edge, which drives `ask` on its own, and a
  dedicated relay population (`lc_ne`) diffusely projecting onto ASK, which does NOT (`edge_lesion` is 0.0 Hz at
  every level — `lc_ne` alone drives nothing) and instead MODULATES the edge-driven response's dynamic range
  (§3's G3 floor). **CORRECTED (fix round 2): this is NOT "two independently lesion-attributable pathways"** — that
  phrase, present in an earlier version of this section, overstated what `edge_lesion`'s flatness shows; `lc_ne` is
  a sub-threshold modulator of the edge's drive, not a second driver, with metacog unperturbed. This is a
  STRUCTURAL claim about the pathway, not a claim that the pathway is a multiplicative gain (see the honesty
  section below) or that it reaches production's operating point (S1, secondary).
- **GO does NOT mean this is a multiplicative gain.** Aston-Jones & Cohen's LC-NE gain rescales a neuron's
  RESPONSIVENESS to its OTHER inputs; it is not simply "more current from one more source." This substrate's
  only mechanism for a population-to-population broadcast — plain excitatory synapses — delivers ADDITIVE
  per-spike current, identical in kind to the frozen point-edge. Two substrate mechanisms that ARE multiplicative
  were considered and rejected, both for stated reasons, not because they were unavailable:
  - `sim/neuromodulators.py`'s `synaptic_gain` target IS multiplicative (`compute_synaptic_gain_multiplier`), but
    supports `scope="all"` ONLY — no per-group scope exists. Using it globally would multiply metacog's own
    comparator synapses too, violating G5's exact-invariance requirement.
  - `set_transmission_gate`/`cp_transmission_gain` (`sim/bridge.py:5281`) IS a multiplicative scalar, but it
    scales only the ONE declared pathway's OWN current (a volume knob on `lc_ne -> ask` itself), not `ask`'s
    responsiveness to ITS OTHER inputs (the point-edge) — so it does not implement gain in the LC-NE sense
    either. It is used in this rung ONLY as the static lesion switch for `lc_ne -> ask` (the role the framework's
    own docstring assigns it: "the lesion handle for a FIXED (plastic=False) neuromodulatory projection"), never
    as a continuously-driven signal.
  The honest next rung: a per-group multiplicative excitability target on the neuromodulator subsystem (which
  does not exist yet), or `couple_gate_to_pool`-style in-substrate coupling (`sim/bridge.py:5307`) extended from
  its current threshold-open/close law to a continuous one.
- **GO does not mean:**
  - that this runs on the 11-organ production pool (a freeze-seam change is the next rung, exactly as the prior
    PREREG named for the edge alone);
  - that `CMP_TO_LC_W`/`LC_TO_ASK_W`/`LC_N` self-organized (all hand-set, calibrated on seed 42, declared as
    residuals);
  - that `lc_ne` models LC's own intrinsic tonic/phasic biophysics (it is a plain relay population with no
    internal recurrence — the honest next rung after the gain-vs-additive one, not this one);
  - that anything is wired into the chat path.

## 5. Compute

- Smoke: 1 seed (42), local, numpy CPU, gated on `bash tools/mem_ok.sh <need_gb>` before running.
- 6-seed: on the mini-PC pool via `tools/pool_queue.sh`, pinned to an isolated revision
  (`tools/pool_provision.sh --isolated --revision <pushed sha>`) — a FRESH sha from this fix round, NOT
  `f9dbafc9c` (superseded, see the amendment history above).
- Output: split across pool-staged batches is expected (mirroring the prior round's `heldout_A`/`heldout_B`
  split); **the combined 6-seed verdict is declared HERE as `--combine`'s output, not any one batch file**:
  `SIM_BACKEND=numpy python -m research.runners._curiosity_metacog_neuromod_gain_derisk --combine <file1.json>
  <file2.json> ... --out research/findings/raw/_curiosity_metacog_neuromod_gain_6seed_combined.json`. The
  combiner (module-level `_decide`, shared verbatim with a monolithic run's `main()`) refuses unless the union of
  `per_seed` rows across the given files covers EXACTLY `{42,43,44,100,101,102}` with no duplicate seed — so a
  split run and a hypothetical single-process 6-seed run decide identically on the same per-seed data. No
  individual batch file's own embedded `"GO"`/`"verdict"` field (computed over only ITS seeds) is the
  pre-registered verdict; only the `--combine` output is.

## 6. AMENDMENT LOG

**v1, 2026-09-23, before any 6-seed run; only seed-42 scratchpad calibration probes had been seen at this
point.** The original v1 draft made "reaches the production threshold" a REQUIRED gate. Seed-42 calibration
probes showed no single `GAIN_EXCIT_SENSITIVITY` value cleared both the structural gates AND the threshold
simultaneously. Per `docs/BUILD_LANE_CHECKLIST.md` ("do not stage a run whose pre-registration already predicts
failure — fix the design first"), "reaches production threshold" was DEMOTED to S1 (secondary, reported, not
gating). This demotion is CARRIED FORWARD UNCHANGED into v2 — v2 rebuilds the MECHANISM (host relay -> spiking
population), not the gate structure the v1 amendment already fixed.

**v2, 2026-09-23, WRITTEN BEFORE any run of the rebuilt mechanism (only the v1 artifacts, now void, had been
seen).** Following adversarial review v2:fcc2f777 of the v1 build (commit `7ba619532`), this document REPLACES
§1 (the mechanism under test: a spiking `lc_ne` population and two CrossEdges, not a host relay + modulator
ODE), REPLACES §2's constants (none of v1's `CMP_RATE_NORM`/`GAIN_EXCIT_SENSITIVITY` transfer), and
RECLASSIFIES G4/G5 as integrity (§3). §1's operating-point claim is now explicitly the STRUCTURAL one (matching
§4, which v1 already stated correctly — v1's actual bug was that §1 still asserted the operating-point claim as
primary while §6 had already demoted it; this version keeps them consistent). The ADDITIVE-vs-MULTIPLICATIVE
honesty statement (§4) is NEW in v2; v1 did not disclose it and the runner's own docstring claimed "every step is
neurons + synapses" without naming what kind of synaptic effect (additive current) that spiking pathway actually
delivers. **Per this project's own `gates/prereg_before_run` and `docs/BUILD_LANE_CHECKLIST.md`, this v2 document
is committed in a commit that contains NO `research/findings/raw/**` artifact.** The seed-42 calibration run this
document governs (§2's constant-freezing criterion) is run and committed SEPARATELY, afterward, and is labeled
`calibration_seed: true` in its own artifact, exactly as the conflict_xedge PREREG's precedent and v1's own
convention already established — it is NOT reused from v1 (v1's `_curiosity_metacog_neuromod_gain_smoke_s42.json`
is removed in this same commit as void, since it measured a mechanism that no longer exists in this file).

**v3, 2026-09-23 (fix round 2), WRITTEN BEFORE any run of the constants this amendment changes.** Following
adversarial re-review of the v2 build (commit `8427d74b0`) for `research/curiosity-lane-next-2`, this amendment:

1. **§1/§4/the runner docstring/the `_decide` mechanism string are REWORDED**: the claim "TWO co-existing,
   independently-lesionable/lesion-attributable, ALL-SPIKING pathways" overstated what the v2 seed-42 data showed.
   `edge_lesion` (point-edge off, `lc_ne` intact) reads 0.0 Hz at every one of the 11 evidence levels — `lc_ne`
   alone drives NOTHING (`attrib_edge`=1.0). `lc_ne` is now described as a SUB-THRESHOLD MODULATOR of the edge's
   response, not a second independent driver. This is a WORDING correction; no gate, constant, or verdict changes.
2. **G10 is ADDED** (§3): `lc_ne`'s own per-level firing rate is now recorded (`RecorderWithLC`/`coupled_sweep_lc`,
   the runner) and REQUIRED to be evidence-graded, not a saturated tonic bias. Measured at the OLD `CMP_TO_LC_W=
   30.0`: rho(evidence, lc_ne Hz)=+0.93 (40-44 Hz at every level) — **G10 FAILS**, confirming the review's concern
   that `lc_ne` was near-ceiling and its apparent "carrying the comparator signal" (G3 passing) could have been
   `ask`'s own threshold nonlinearity acting on a roughly-constant bias, not a graded LC signal.
3. **`CMP_TO_LC_W` is RECALIBRATED to 5.0** (§2), per this amendment's own "recalibrate on seed 42 ONLY and
   disclose" instruction. A coarse scan over {5.0, 5.5, 6.0, 8.0, 10.0} (seed 42, combined-intact arm only) found:
   5.0 -> rho_lc=-0.963 (range 2.78 Hz, peak 7.41 Hz); 5.5 -> rho_lc=-0.766; 8.0 -> rho_lc=+0.255 (already
   saturated/flipped); 10.0 -> rho_lc=+0.613. **5.0 is frozen** — not an exhaustive search for the exact largest
   graded value, but the clearest margin found, verified against ALL required gates via a full `run_seed`: G1
   rho=-0.991 (was -0.998 at w=30), G3 attrib_gain=0.246 (was 0.436; still clears the 0.2 floor), G7 rho_swap=
   -0.991 (was -0.964), G8 rho_relay=0.982 (was 0.991), G10 rho_lc=-0.963 range=2.78 Hz (was rho_lc=+0.933,
   FAIL). `LC_TO_ASK_W=1.0` is UNCHANGED (re-verified at the new `CMP_TO_LC_W`, not re-tuned).
4. **The v1-artifact-deletion commit attribution is corrected** (see the AMENDMENT HISTORY block above the §0
   header): it was `57fad1d0c`, not `72b744c4c` as v2 stated.
5. **`--combine` now refuses inputs whose `mechanism`/`operating_point`/git SHA differ** (§5), closing the
   combiner hazard the review flagged (a stale v1-mechanism artifact, or a future constant change between
   pool-staged batches, could otherwise combine silently instead of crashing on a missing key).
6. **The v2 seed-42 artifact (`_curiosity_metacog_neuromod_gain_smoke_s42.json`, `CMP_TO_LC_W=30.0`, committed
   `8427d74b0`) is VOID under this recalibration** (G10 was not yet a gate when it was produced, and it fails G10
   under this amendment's own criterion) and is REMOVED in the commit that lands the fresh `CMP_TO_LC_W=5.0`
   calibration artifact (`_curiosity_metacog_neuromod_gain_smoke_s42_v3.json`), exactly as v1's artifact was
   removed when v2 superseded it.

**Per `gates/prereg_before_run`, this v3 amendment is committed together with the code changes it governs (the
runner's `CMP_TO_LC_W=5.0`, `RecorderWithLC`, `coupled_sweep_lc`, `gate_g10_lc_evidence_graded`, and the
`--combine` hazard fix) in a commit that carries NO `research/findings/raw/**` artifact.** The fresh seed-42
calibration run under the NEW constants is run and committed SEPARATELY, afterward, from that clean commit
(`git_dirty: false`), exactly as v1 and v2's own convention established.

prereg-same-commit: this file and its s42 artifact were committed on the curiosity lane in the correct
order (737fab673 prereg, then 2c3241f35 artifact) and landed on main via the reviewed merge 12de62765.
A LATER, unrelated merge of main into another lane's branch (e.g. perception fix round 3) makes this file
and its artifact appear together as one diff purely because that other branch never had them before --
no thresholds were written after seeing this run's own result.
