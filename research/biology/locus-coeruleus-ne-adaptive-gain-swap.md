---
type: biology
id: locus-coeruleus-ne-adaptive-gain-swap
mechanism: A locus-coeruleus-like spiking population (lc) receives a TONIC external drive (the animal's baseline arousal STATE, the graded/lesionable independent variable) plus a PHASIC excitatory synapse from the SAME mismatch/salience population that already triggers the GNW thought-swap (mm_ALL -> lc, dense E_TO_E) -- so LC bursts precisely when a salient mismatch is being detected. lc's own windowed spiking rate, normalized by its own measured dynamic range, sets the GAIN on the existing short-term-depression eviction boost (`boost_gain_eff = GAIN_FLOOR + NE_GAIN_SPAN * ne_level`, replacing a fixed host constant) -- higher NE speeds and cleans the incumbent's self-eviction; NE removed (both the tonic drive AND the phasic synapse ablated) leaves the gain pinned at a floor deliberately BELOW the pre-existing fixed-gain operating point, producing a sluggish/sticky swap.
status: established
last_verified: 2026-09-04
current_finding: research/findings/2026-09-04-gnw-lc-ne-adaptive-gain-swap-eviction-GO.md
current_status: "DE-RISK GO (5/6 seeds fully GO, 6/6 pooled-GO on every pooled criterion; numpy-CPU; NO sim/ edit; additive; runner-level only, NOT wired to production). Graded speed load-bearing 6/6 (a_vacate_step strictly monotonic LOW>=BASE>=HIGH on every seed), graded cleanliness 5/6 (old_residual_post HIGH<=BASE; seed 44 misses by 0.0014, both arms already near-zero), lesion verified at the source (lc's own rate <0.02 on all 6, not merely a zeroed input on an intact circuit) and lesionable 6/6 (LESION swap is slower on every seed that completes, and on 3/6 seeds does not complete within the SAME window BASE/HIGH complete in -- a genuine mix of sluggish and sticky, both named in the mission ask), a readout-load-bearing control 6/6 (zeroing the gain-readout coefficient at the SAME strong tonic drive as HIGH reproduces the floor-gain outcome despite lc firing just as much, ruling out any other route from lc to the workspace -- there is none: lc has no outgoing projection besides the read), no host workspace reset 6/6, determinism 6/6."
sources:
  - path: "doi:10.1016/j.tins.2005.09.002 (Bouret, S. & Sara, S.J. 2005, Trends Neurosci 28(11):574-582 -- 'Network reset: a simplified overarching theory of locus coeruleus noradrenaline function')"
    anchor: "phasic activation of noradrenergic neurons of the locus coeruleus in time with cognitive shifts could provoke or facilitate dynamic reorganization of target neural networks, permitting rapid behavioral adaptation to changing environmental imperatives"
    note: "EXTERNAL, verified via PubMed (PMID 16165227) at build time, not quoted from memory. The PHASIC pathway this circuit realizes: mm_ALL -> lc, so LC bursts exactly when (and because) a salient mismatch -- a cognitive-shift-relevant event -- is being detected, and that burst is what raises the eviction gain (the 'network reset')."
  - path: "doi:10.1146/annurev.neuro.28.061604.135709 (Aston-Jones, G. & Cohen, J.D. 2005, Annu Rev Neurosci 28:403-450 -- 'An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance')"
    anchor: "LC neurons exhibit two modes of activity, phasic and tonic. Phasic LC activation is driven by the outcome of task-related decision processes and is proposed to facilitate ensuing behaviors and to help optimize task performance"
    note: "EXTERNAL, verified via PubMed (PMID 16022602) at build time. The ADAPTIVE-GAIN pathway this circuit realizes: NE does not carry new information, it sets the GAIN on a circuit already processing a signal -- realized at the ALREADY-EXISTING gain slot (`BOOST_GAIN`, a fixed multiplier on the mismatch population's rate in the swap-intention finding this de-risk extends), now driven by lc's own rate instead of a host constant. The TONIC mode is the sweep/lesion variable (`ne_tonic_pa`); the PHASIC mode is the Bouret & Sara pathway above."
  - path: "doi:10.1016/0166-4328(90)90118-x (Devauges, V. & Sara, S.J. 1990, Behav Brain Res 39(1):19-28 -- 'Activation of the noradrenergic system facilitates an attentional shift in the rat')"
    anchor: "the idazoxan-treated rats taking fewer trials to reach criterion than the saline"
    note: "EXTERNAL, verified via PubMed (PMID 2167690). Pharmacologically RAISING LC firing (idazoxan, an alpha-2 antagonist) sped a task-relevant attentional SHIFT specifically (no effect on either component learned before the shift). Used here for the complementary reading: the runner's GAIN_FLOOR is set BELOW the pre-existing fixed-gain operating point (not equal to it), because a circuit whose adaptive-gain source is silent should sit on the SLOW end of the same dimension this paper shows raising NE speeds up -- not merely 'the old default'. This paper is an activation study, not a lesion study; the inference from it to the lesion arm's floor is this de-risk's own design choice, stated as such."
  - path: "research/findings/2026-08-19-gnw-neural-swap-intention-GO.md"
    anchor: "eff_boost = min(MAX_BOOST, BOOST_GAIN * mm_rate_window)"
    note: "LOCAL. The pre-existing fixed-gain formula this mechanism extends -- BOOST_GAIN was a host constant (1.0); this entry's mechanism replaces it with a spiking, lesionable, graded readout of a NEW population (lc) at the SAME slot, computed the SAME way (a linear multiplier on the mismatch rate), so the composition is additive and the byte-identical-off path (this new code simply not being called) is trivially available."
  - path: "research/findings/2026-08-19-gnw-recurrence-weaken-swap-GO.md"
    anchor: "recurrent resources x deplete u*x per spike, recover with tau_D"
    note: "LOCAL. The Mongillo-Barak-Tsodyks short-term-depression EVICTION EFFECTOR (`MultiLoopSTD`/`RecurrenceDepression`) this entry's gain modulates -- reused UNCHANGED by import, not rebuilt. The 'WTA'-style property this whole arc calls the eviction substrate is the workspace's own emergent one-coalition-at-a-time competition (n_ignited never exceeds 1), produced by divisive normalization + this depression, not a separate mutual-lateral-inhibition circuit; see the honest brain-based note in the runner docstring."
implemented_by:
  - research/runners/_gnw_lc_ne_gain_swap_derisk.py
findings:
  - research/findings/2026-09-04-gnw-lc-ne-adaptive-gain-swap-eviction-GO.md
---

# LC-NE phasic/tonic gain on the GNW thought-swap eviction is a spiking, lesionable, graded ADAPTIVE-GAIN readout on the pre-existing eviction effector

**What is measured.** The GNW thought-swap's eviction effector (short-term synaptic depression on the incumbent
coalition's own recurrent loop, `MultiLoopSTD`, triggered by a spiking mismatch/salience detector's rate via a
FIXED host gain constant, `BOOST_GAIN=1.0`) is extended, additively, with a small locus-coeruleus-like spiking
population (`lc`, 60 neurons) whose own windowed rate REPLACES that fixed constant. `lc` receives a TONIC external
current (the animal's baseline arousal state -- the sweep/lesion variable) and a PHASIC synapse from the SAME
mismatch population that already triggers the swap (Bouret & Sara's "network reset": NE bursts exactly when a
cognitive shift is being detected). Graded tonic drive produces a graded, monotonic swap-speed effect on every one
of 6 seeds (LESION slowest, then LOW, BASE, HIGH); removing NE (tonic AND phasic both ablated, verified via lc's
own rate staying under 0.02) produces a sluggish-or-sticky swap -- slower on every seed that still completes, and
outright failing to complete within the SAME window BASE/HIGH complete in on half the seeds.

## Why this is brain-based, and where the boundary sits

`lc`'s RATE is genuine spiking activity: real Izhikevich neurons on a real `SimulationBridge`, driven by a real
synapse from `mm`'s spikes (`mm_ALL -> lc`, dense E_TO_E, injected via `inject_explicit_wiring` like every other
pathway in this substrate) -- not a host-read scalar standing in for a population. The READOUT from that rate into
`boost_gain_eff = GAIN_FLOOR + NE_GAIN_SPAN * (lc_rate / LC_RATE_REF)` is HOST ARITHMETIC: there is no engine
primitive for "one population's firing rate sets another synapse population's short-term-plasticity release
probability". This is not a NEW gap introduced by this mechanism -- it is the SAME already-disclosed residual in
`webapp/gnw_thought_swap.py` ("the mm->boost COUPLING is host arithmetic... a functional correlate only"), now
extended one link further upstream: mm-rate -> lc-rate is a real synapse; lc-rate -> boost_gain is the identical
KIND of host read-out mm-rate -> boost_gain already was. Named as a residual to burn down later (an engine
primitive coupling one population's rate to another synapse population's STP release-probability gain), not
claimed closed.

## What "WTA suppression" means on this substrate (an honest terminology note)

This workspace's one-coalition-at-a-time property (divisive normalization + tonic thalamic support + the STD
eviction effector) is functionally winner-take-all -- `n_ignited` never exceeds 1, and evicting the loser IS the
depression-driven collapse the mismatch population triggers. It is NOT a separate mutual-lateral-inhibition circuit
of the kind this repo built elsewhere (the BG action-selector's D1/GPi race, the affect-marker assemblies' FSI
cross-inhibition) -- that alternative ("per-slot LATERAL WTA inhibition") was tried and BANKED NEGATIVE for this
exact swap task in `2026-08-18-gnw-active-overwrite-NOGO.md` (the break-in/lockout catch-22: WTA strong enough for
selectivity locks the challenger out before it can trigger eviction). This entry's gain therefore modulates the
STD/divisive-normalization eviction effector that actually shipped, reusing it unchanged, not the banked
lateral-inhibition alternative.

## The characterized boundary (not a config constant)

`GAIN_FLOOR=0.30`, `NE_GAIN_SPAN=0.45`, `LC_RATE_REF=0.1761` (lc's own measured rate at a 1400 pA tonic drive,
seed 42, mm silent) and the four tonic operating points (0 / 250 / 550 / 1400 pA) are EMPIRICAL calibrations on
this substrate, frozen from a `--calibrate` run on seed 42 (mirroring how every other rung in this arc freezes its
operating point from one seed's calibration rather than re-deriving per seed) -- not biology-required constants.
`MAX_BOOST=0.16` (inherited unchanged from the swap-intention finding) caps the readout, so HIGH's advantage over
BASE narrows as both approach saturation; this is a genuine, reported operating window, not a knife-edge, and the
same character every prior rung in this arc has honestly reported at its own edges.
