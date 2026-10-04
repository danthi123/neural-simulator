# Review: the local model's N-channel basal-ganglia work (branch `research/draft-seam-map`)

*2026-09-29, Claude (build agent, lane A, branch `research/selection-seam-need-reversal`). Reviewed commits
`cee57e083`, `cd8f33d67`, `62d47eaa9`, `340178654` and the files they add
(`research/runners/_tone_bg_nchannel_*.py`, finding `research/findings/2026-09-27-tone-bg-nchannel-tonic-firing-wall.md`,
raw `research/findings/raw/_gpi_fix_6seed.json`). My own review probes are DEV, not a verdict (dev seed 7, numpy);
scripts and output are in `docs/drafts/selection-seam/raw/`.*

## Verdict in one paragraph

The "N=5 tonic-firing wall" is **not a biological wall and not a tonic-firing problem**. In the N=5 build, every
commit pool's membrane potential is driven to **-377 mV within the first 60 steps**, and the pools then "fire" at the
refractory ceiling (0.5 spikes/step) **even with the arousal drive switched off and the thalamus silent**. That is a
numerical failure: four un-normalized `commit_fs -> commit` GABA-A projections (weight 40 each, 4x the N=2 total)
overshoot the membrane far past any reversal potential in one explicit step. Scaling the cross-inhibition to keep the
N=2 total per pool (40/(N-1) = 10) removes the saturation at the **default** tonic (gpi 1000 / thal 900) and the race
then commits on 0.83-1.0 of trials on dev seed 7. The GPi-tonic "fix" (gpi 500, thal 45) works only because it keeps
the thalamus nearly silent, so the commit FS pools rarely fire and the overshoot is not triggered; it is a
workaround of the artifact, not a faithful GPi operating point. Separately, the finding's headline claim that
"0% of the selection is host-owned" is **wrong**: the candidate values are computed by the host
(`congruence_saliences`, a Python formula over the held mood) and injected as a current, so the host owns the scoring
and, through it, which candidate wins. The lesion arms cannot show otherwise (see below). The six gate-seed run was
made without a reviewed preregistration and from a dirty tree (`git_dirty: true` in its provenance), so it stays an
**unregistered DEV result**.

## What I checked

### 1. The mechanism of the saturation (DEV, seed 7, `raw/xinh_probe.py`, output `raw/review_xinh_probe_seed7.jsonl`)

Full 600-step action window, equal saliences, v2 config, the local model's `build_nchannel_bridge` imported verbatim:

| build | cross-inh weight | arousal | commit rate per pool (spikes/neuron/step) | thalamus | commit_0 min V, first 60 steps |
|---|---|---|---|---|---|
| N=2 | 40 | on | 0.015, 0.002 | 0.016, 0.005 | -65.4 mV |
| N=3 | 40 | on | 0.021, 0.001, 0.015 | ~0.018 | -67.7 mV |
| **N=5** | **40** | on | **0.5 x5** | 0.012-0.021 | **-377.4 mV** |
| **N=5** | **40** | **off** | **0.5 x5** | **0.0** | **-377.4 mV** |
| N=5 | 10 (=40/(N-1)) | on | 0.005-0.013 | 0.012-0.021 | -64.2 mV |
| N=5 | 0 | on | 0.014-0.026 | 0.012-0.021 | -67.4 mV |

Reading: at N=5 the commit pools saturate with the thalamus silent (arousal off), so the saturation is not driven by
the basal ganglia, the tonic currents or the salience. The membrane at -377 mV is non-physical (the most negative
reversal in the engine is the K+ reversal, about -90 mV). The finding's own "ruled out" list (NMDA, commit
recurrence, thalamic drive, salience) never varied the cross-inhibition magnitude, which is the one variable that
moves it. The tonic-floor sweep's apparent result ("scale <= 0.025 silences the pools") is the same artifact seen from
the other side: with no thalamic drive the commit FS pools never fire and nothing overshoots.

Why the GPi-tonic point "works" (`raw/gpifix_probe.py`, N=5, equal drive, arousal on, cross weight 40, seed 7):

| gpi / thal tonic (pA) | commit rate per pool | commit FS rate | thalamus rate | commit min V |
|---|---|---|---|---|
| 1000 / 900 (default) | 0.5 x5 | 0.26-0.32 | 0.012-0.021 | -389 mV |
| 500 / 45 (the "fix") | 0.0 x5 | 0.0 | 0.0 | -60 mV |
| 50 / 45 (the "sub-threshold window") | 0.45 x5 | 0.24-0.28 | 0.004-0.005 | -386 mV |

At 500 / 45 the thalamus is silent unless the direct path disinhibits a channel, so at most one commit FS pool fires
and no pool receives the summed inhibition that overshoots. The "sub-threshold window" of the tonic sweep (pools
~8000 spikes and "differentiating") is the same -386 mV overshoot.

### 2. The race with the cross-inhibition normalized (DEV, seed 7, `raw/norm_probe.py`, 12 trials per mood)

At the default tonic with `commit_fsi_cross_weight=10`: intact commit rate 0.92 / 0.83 / 1.0 / 1.0 / 1.0 at
v_held = -0.9 / -0.45 / 0 / +0.45 / +0.9; no-arousal 0.0 at every mood. Winner selectivity is soft: at v_held=+0.9
the host-argmin candidate won 6/12 (next 4/12); at v_held=0 8/12. With five candidates and a salience contrast of
0.05-0.2 (x600 pA) the race is a noisy soft choice. So the commit bar is reachable without touching the GPi tonic;
selectivity at low contrast is the open parameter. One seed, 12 trials: a lead, not a result.

### 3. "0% host-owned" does not follow from the lesion arms

- The salience injection sits **inside** the `if arousal:` branch (`_run_nchannel_trial`), so the no-arousal lesion
  removes the host's salience current as well as the bulk drive. A zero commit rate there shows the race needs its
  input, not that the host does not choose.
- The direct-path lesion shows the race is the only route to a commit (the host has no bypass). That rules out a host
  fallback; it says nothing about who decided which candidate should win.
- Plan section 2 names the correct measurement: hold the neural state fixed and change the host decision. Here the
  host decision is the salience vector. Swapping it moves the winner by construction, because the salience is the
  only candidate-specific input the race receives. Scoring (plan section 1: "candidate scoring") is host-owned; the
  race owns only the noisy winner-take-all over host scores. The host-share of this seam is high, not zero.
- `winner_tracks_host_argmin` is scored as a success criterion. Agreement with the host argmin is the decoding screen
  (plan section 2 says it is never evidence that the neural system decided).

### 4. Process

- Gate seeds 42/43/44/100/101/102 were run (`_gpi_fix_6seed.json`) although the runbook reserves them for a
  Claude-reviewed preregistration. Provenance: `git_sha 62d47eaa9`, `git_dirty: true`, corpus check stale (260k s).
  The operating point (gpi 500 / thal 45) was chosen on dev seed 7 immediately before. Treat the numbers as DEV.
- The N=2 "byte-identical to the two-channel production selector" claim is stated in the docstring but has no test
  (the committed CI guard tests the host-share probe, not the topology hash).
- The motor-gain sweep's flat result (64x gain, no change) and the pathway trace were both taken at gpi 50 / thal 45,
  where the commit pools sit in the -386 mV overshoot (table above): the commit and motor counts there are driven by
  the artifact, not by the D1 -> GPi -> thalamus pathway, so no salience gain could move them. (The near-silent GPi
  at that tonic adds to it: extra D1 inhibition has little left to remove.)

## What is sound and salvageable

1. **`build_nchannel_bridge`** is a reasonable N-way generalization of the Gate-A v2 topology. Salvage it with the
   cross-inhibition normalized per pool (40/(N-1)) and a topology-hash test that N=2 equals `build_selector_bridge`.
2. **The pathway trace method** (per-stage firing D1 -> GPi -> thalamus -> commit -> motor) is the right instrument; it
   should also record the membrane range per stage, which would have exposed the -377 mV pools on day one.
3. **The host-share tone probe's structure** (decoding, substitution, intervention kept separate) matches plan
   section 2 and can be reused once the scoring is neural.
4. **Not salvageable as a finding:** "N=5 tonic-firing wall", "the GPi's unfaithful near-silence is the mechanism",
   "the GPi-tonic method replaces the wall's method", "100% of the selection is owned by the spiking race". The
   finding should be amended to point here; the wall is an integration artifact of un-normalized inhibition.

## Consequences for this lane

- My seam build (need-state reversal) uses the N=2 Gate-A v2 topology, where the overshoot does not occur (commit
  V min -65 mV), and it records the membrane range every trial. A new helper, `tools/lab.py::membrane_bound_check`,
  raises when any neuron leaves [-150, +100] mV, and the failure is logged in `research/FAILURE_LOG.md`.
- The tone seam's real open problem is not the race but the scoring: `congruence_saliences` is host arithmetic. The
  need-reversal design in `DESIGN.md` is the template for removing it (the value reaches the striatum through
  synapses from a neural state, and the host supplies no candidate-specific input).
- Engine-level follow-up for the owner (not done here; `sim/` is out of scope): conductance-based inhibition is
  integrated with an explicit step that can overshoot the reversal potential when the summed conductance is large.
  Any build with strong convergent inhibition can produce spurious max-rate firing. Options: clamp V at the
  inhibitory reversal per step, or integrate the conductance term exponentially. Flag-gated default-off with a
  byte-identical-off test, per the production rule.
