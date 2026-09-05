---
type: plan
status: prep
lane: memory-learn-through-use, continuous-life/generation
date: 2026-09-04
---

# Build-ahead prep: the SWR/CA3 learn-through-use unblock (board #107) and the CA3 generative-attractor port (board #104)

Scope, not build: this session prepares a runner + GO-gate + anti-cheats for #107's genuinely-next mechanism,
smoke-tests it (tiny, structural, numpy/CPU, not the decisive de-risk), and separately writes up an already-run
but never-recorded #104 result. Both are queued/named for later compute, per this pass's own mandate
("prepare... don't fully build... run only a tiny smoke").

## Part 1 -- board #107: correcting the framing, then the genuinely-next mechanism

### The literal title is a dead end; the record has already moved seven mechanisms past it

Board #107 is titled "SWR-envelope op-point is the convergent unblock for learn-through-use". Taken literally
that would mean: tune the SWR E/I-transient envelope's operating point on the CA3 store to produce discrete
forward replay. That specific mechanism is CLOSED, decisively, and has been since 2026-08-27:
[[2026-08-27-swr-envelope-learn-through-use-NOGO]] swept drive/inhibition op-points 25x and found no
segmentation-producing point exists on that store -- the bistable-completion architecture itself never rests
(silence <=0.09, co-fire 0.97-0.98), so there is no "envelope tuning" left to try there. Building a fresh
SWR-envelope-op-point runner, as the board's literal title would suggest, would re-derive a already-closed result
using one of several runners that already exist for it
(`research/runners/_gap5_swr_envelope_replay_derisk.py`).

The record has moved substantially past that framing, each step banked as its own finding:
[[2026-08-27-ecker-adex-store-learn-through-use-NOGO]] (the Ecker AdEx store DOES segment into discrete forward
replay -- the architecture wall is surpassed -- but plain replay-driven STDP symmetrizes the band instead of
strengthening it forward), [[2026-08-27-conduction-delay-directional-replay-learn-through-use-PARTIAL]] and
[[2026-08-27-btsp-directional-write-learn-through-use-PARTIAL]] (a forward-edge conduction delay + BTSP-eligibility
gating makes the write net-directional 6/6), [[2026-08-27-graded-recall-instrument-learn-through-use-NOGO]] (a
proven-graded recall-depth instrument shows weak-cue recall actually DECREASING after that write on 5/6 seeds --
root cause: pure-potentiation BTSP leaves the reverse band deepening at ~84% of forward), and
[[2026-08-27-reverse-edge-heterosynaptic-depression-learn-through-use-NOGO]] (suppressing the reverse band
eliminates that deepening and flips the MEAN recall change from decreasing to increasing, but only 2/6 seeds
individually clear the gain bar -- driving the suppression 20-30x further does not move the other 4 at all).
[[2026-08-27-decorrelation-read-shared-fidelity-wall-PARTIAL]] separately tested and closed the hypothesis that
this residual is the SAME wall as the mouth's read-SNR issue (it is not -- the mouth's was a stale-weights
measurement artifact, LTU's is "a separate genuine effect", and a retina/LGN-style decorrelation read for it comes
back decisively negative across all six seeds tested). Board #107's own text already records this whole progression in its append log;
this plan's contribution is picking the correct NEXT step from it, not re-deriving the history.

### The two named-but-untested candidates, and which one this session builds

The reverse-edge-depression finding names two candidates verbatim: "characterize the READ-side noise floor
directly (repeated graded-instrument reads of the SAME frozen weights ...) and/or investigate whether forward-band
ABSOLUTE magnitude at encode time (not just its ratio to reverse) is the actual limiting factor ... a homeostatic
forward-band STRENGTHENING process (rather than further reverse suppression) is the next candidate lever." Neither
has been attempted for learn-through-use as of this session (the decorrelation attempt was a read-side fix of a
DIFFERENT kind -- common-mode removal across assemblies -- not a magnitude/noise-floor characterization, and it
never touched the write side at all).

**This session builds candidate 2** (forward-band absolute-magnitude homeostasis) as a complete, ready-to-queue
runner. **Candidate 1** (the read-noise-floor diagnostic) is named but not built here, because a clean version of
it needs a small additive change to a shared function five other findings already depend on being unchanged --
see "Candidate 1, scoped but not built" below.

### What was built: `research/runners/_gap5_forward_band_homeostatic_scaling_ltu_derisk.py`

The mechanism: Turrigiano-style multiplicative synaptic scaling (Turrigiano GG, Nelson SB, "Homeostatic plasticity
in the developing nervous system", Nat Rev Neurosci 2004;5(2):97-107, PMID 14735113) applied ONLY to the CA3
forward between-assembly band, ONCE, between `encode` and `consolidate` -- rescale every forward-band synapse by
the SAME factor toward `mult * this_seed's_own_post-encode_forward_band` (a per-seed-relative target, since
baselines vary ~2x seed to seed per the reverse-edge finding's own numbers). The identical mechanism CLASS
(multiplicative rescale of a population's own synapses toward a set point computed from the population's own
measured state) already ships and is validated elsewhere in this repo --
`webapp/da_encoding_drives_chat.py::apply_substrate_homeostasis` -> `OneBrainComposer.apply_homeostatic_scaling`,
6/6 GO -- targeting a per-engram unit set-point on the D5 store. This applies the same FORM to the Ecker CA3
forward band's absolute magnitude on the standalone research substrate instead; it is a new application of an
already-accepted mechanism class, not a re-derivation of that GO and not a host-invented shortcut (the scale
factor is read from the substrate's own STDP-grown weight state, never from an environment-side oracle).

Because the rescale is a one-time discrete step on already-encoded weights rather than a per-step update-rule
change, this runner needed no new cupy step loop: it reuses `build_store` / `encode` / `rest_and_replay` /
`measure_band` / `measure_band_from` / `_load_weights` (from `_gap5_ecker_adex_ca3_stdp_band_derisk` and
`_gap5_ecker_replay_learn_through_use_derisk`) and the established, UNMODIFIED
`consolidate_by_btsp_replay_delayed` write, plus the proven-graded `_read_graded` / `verify_instrument` instrument
(from `_gap5_graded_recall_learn_through_use_derisk`). No `sim/` edit. All non-lever hyperparameters are held
IDENTICAL to the established write's own decisive config, so a result is attributable to the one new variable.

**The GO gate** (per seed): directional (dw_fwd > dw_rev + dw_min, inherited for free since the write rule is
untouched) AND the rescale actually reaches its own target (adj_fwd rises by >= `--fwd-raise-min` when mult > 1)
AND weak-cue recall gains (depth_frac or tau rises by >= 0.05, the SAME bar the whole lane has used) AND headroom
held (weak-cue depth_frac before is not already at ceiling) AND lesion-controlled (the NO-SEED arm's forward
weight change is near zero) AND -- the one genuinely new anti-cheat this mechanism needs -- **USE-DEPENDENT**: the
SEEDED arm's recall gain must exceed the NO-SEED arm's gain by more than a margin. Aggregate bar: >=5/6 seeds (the
lane's established 6-seed bar).

**Why the use-dependence control is new and necessary.** Unlike the BTSP-directional-write and reverse-edge-
depression mechanisms (both gated by replay ignition, so their own NO-SEED arm nulls the mechanism itself), this
rescale happens BEFORE consolidation regardless of whether replay ever ignites afterward. So the standard
NO-SEED lesion here does double duty: it isolates "does a statically stronger band read better regardless of
replay" (a bigness effect with nothing to do with learn-through-USE) from "does replay-driven consolidation on
TOP of a homeostatically-corrected band durably strengthen recall" (genuine use-dependence) -- mirroring the
"two legs" pattern the D5 organ's own learn-through-use GO used (clamp isolates plateau-vs-cue-current; disjoint-
held completion makes it retrieval-driven). A positive result here cannot be "scaling alone was already enough".

**Byte-identical-off**, asserted in the data, not read from the code: at `--fwd-scale-mult 1.0` (the default) the
rescale function returns the input array untouched (no arithmetic executed at all, so the property does not rely
on `x*1.0==x` as a floating-point coincidence). Verified this session at tiny scale (n_mem=3, asm_size=12, seed
42): `EXACT_HASH_MATCH=True max_abs_diff=0.000e+00` against the established write's own output.

**Structural smoke** (this pass's mandate -- confirm it imports/parses/starts a step, not a decisive result):
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk --smoke
--seeds 42` ran to completion at tiny scale (3 assemblies x 12 neurons, mult forced to 1.5 to exercise the new code
path) and wrote `research/findings/raw/gap5_ecker_adex/forward_band_homeostatic_scaling_ltu_smoke.json` with
`smoke_ok: true`. This is explicitly NOT a decisive result -- the network is far too small/short for the recall
metric to mean anything; it only proves the pipeline runs end to end without crashing on the new code path.

### Candidate 1, scoped but not built: the read-side noise-floor diagnostic

The reverse-edge-depression finding's other named candidate is a repeatability probe: read the SAME frozen
post-consolidation weights multiple times and see how much the graded recall-depth metric varies, to separate
genuine per-seed substrate variance (what the forward-band mechanism above targets) from instrument/read noise
(a different fix entirely -- e.g. averaging over repeated reads, or a different statistic). It was not built this
session because a MEANINGFUL version of it is not free: `rest_and_replay` derives its cue-cell-subsample and
env/period-choice RNGs FROM the passed substrate `seed` (see `_gap5_ecker_adex_ca3_stdp_band_derisk.rest_and_replay`),
and this repo's own seeding discipline makes a fresh build at a fixed seed fully deterministic
(`tests/test_determinism.py::TestSubstrateActuallySeeded`) -- so repeating a read at the SAME seed on frozen
weights would trivially report zero noise, not a real diagnostic. A genuine probe needs a READ-TRIAL seed
independent of the substrate-build seed threaded through those two RNGs -- a small, additive parameter on a
function five other findings in this lane already depend on being byte-identical unless explicitly asked to
change. Named here as the exact next addition (add `read_trial_seed=None` to `rest_and_replay`, defaulting to
`seed` when absent so every existing call site is byte-identical-off by construction), not attempted under this
pass's "don't fully build" mandate.

### Ready to queue

```
SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk \
  --fwd-scale-mult 1.5 --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/gap5_ecker_adex/forward_band_homeostatic_scaling_ltu_6seed_mult1.5.json
```
`--fwd-scale-mult 1.5` is a reasonable starting point (a 50% strengthening), not a fitted value -- this pass did
not run the decisive test, so no mult has been chosen by outcome. A single-seed scan is cheaper and should run
first to pick a value before committing GPU time to the 6-seed decisive run:
```
SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk \
  --scan-mult --seeds 42 --scan-mults 1.0 1.25 1.5 2.0 3.0
```
Queue command (gpu.queue, once a mult is chosen from the scan): `bash tools/gpu_queue.sh add 'SIM_BACKEND=cupy
.venv/bin/python -u -m research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk --fwd-scale-mult <chosen>
--seeds 42 43 44 100 101 102 --out research/findings/raw/gap5_ecker_adex/forward_band_homeostatic_scaling_ltu_6seed.json'`

## Part 2 -- board #104: an already-run result, written up

While RAG-checking the record for #107 this session also found that #104's own named next step -- the
production-scale (n_ca3=2000, emergent) 6-seed cupy verify -- actually ran on 2026-08-28 (`gpu_queue.log` lines
519214-521033, rc=0) and was never written up or synced to the board. That gap is now closed:
[[2026-09-04-generative-wander-production-scale-6seed-PARTIAL-blend-balance-collapse]]. Summary for this plan's
purposes: 1/6 seeds clear the runner's own strict per-seed bar; the dominant failure is a blend-balance collapse
(the released blend state fails to preserve both driven memories) on 4/6 seeds, with a genuine open question about
its cause (the most readily-available covariate, emergent assembly-size disparity, does not explain the worst-
performing seed, which has the most EQUAL assembly sizes of all six). No mechanism runner is built for #104 in
this pass -- a root cause is not yet identified, and naming one without evidence would repeat the exact
premature-conclusion pattern this project's workflow exists to catch. The finding names the concrete next DIAGNOSTIC (per-seed
blend-settle firing-rate instrumentation against a wider covariate set) rather than a fix.

## Honest residual (both parts)

Neither #107's forward-band-homeostasis mechanism nor #104's blend-balance question has been decided by this
pass -- both are prepared/named, not resolved, exactly as instructed. Board #107's Vikunja text and board #104's
Vikunja text both still read as of session start in a way that undersells or misses what the record now says (see
this session's Vikunja updates to both tasks); a full `sync-documentation` pass over `GAP_CLOSURE_MISSION.md` /
the master roadmap's wall-ledger for both boards is not attempted in this pass and is flagged as the next
mechanical step for whoever picks this up. The forward-band-homeostasis runner's own decisive verdict is unknown
until it is actually queued and run at the established scale; this document does not pre-judge it.
