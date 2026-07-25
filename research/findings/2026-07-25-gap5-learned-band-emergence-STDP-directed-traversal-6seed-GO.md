# gap#5 learned replay band — the directional traveling-replay band EMERGES from experience via STDP (6-SEED GO, structural + functional): the hand-wired band of the Ecker replay model is now GROWN from directed traversal, and the learned band replays in the TRAINED direction — the emergence-bar version of gap#5's replay structure (2026-07-25)

## Headline
The gap#5 Ecker replay model (`_gap5_ecker_recurrent_replay.py`, 6-seed GO earlier today) used a **hand-wired**
forward-biased near-diagonal recurrent band. That band now **EMERGES from experience**: a weak *symmetric* plastic
near-diagonal band + STDP, driven by a bump sweeping the track in one direction, develops a **forward-asymmetric** band
(Mehta-Blum-Abbott experience-dependent asymmetric place-field expansion), and that **learned** band produces a
**directional traveling replay whose direction follows the training direction** — forward-trained → forward replay,
reverse-trained → reverse replay. Both halves are 6-seed GO. This converts the last hand-designed piece of gap#5's replay
structure into a learned, emergent one (the emergence-bar version).

## Biology
Mehta et al. 1997/2000 (Blum-Abbott model): a rat running a track repeatedly in one direction fires place cells in
sequence; the causal (asymmetric) STDP window potentiates the i→i+1 connection (pre-before-post = LTP) and depresses
i+1→i (post-before-pre = LTD), so the recurrent connectivity becomes **forward-biased** and place fields expand/shift
opposite to motion. That learned forward-asymmetric connectivity is exactly the directional structure a moving-bump
replay needs.

## STRUCTURAL emergence — 6-SEED GO (seeds 42/43/44/100/101/102)
Weak symmetric plastic near-diagonal band (w₀=1, plastic=True, STDP-on) → sweep a drive bump N laps → measure developed
forward/backward mean-weight ratio:
- **FORWARD traversal + STDP:** ratio [1.662, 1.663, 1.662, 1.666, 1.664, 1.664] — forward-bias **6/6**.
- **REVERSE traversal + STDP:** ratio [0.60, 0.60, 0.601, 0.601, 0.601, 0.601] — backward-bias **6/6** (opposite).
- **NO-STDP control:** ratio [1.0 ×6] — stays symmetric **6/6** (Δ=0, no learning).
⇒ the asymmetry **emerges from experience** and **tracks the traversal direction**; without the plasticity rule the band
stays symmetric. Remarkably tight across seeds (the traversal + STDP learning is near-deterministic). (At the broader
sigma=25 replay-matched band the developed asymmetry is even stronger, ratio ~4.3 fwd / ~0.23 rev.)

## FUNCTIONAL emergence — the learned band REPLAYS in the trained direction — 6-SEED GO
Learn the band (sigma=25, replay-matched) → freeze STDP → reset neuron state + scale to operating strength (uniform gain,
preserves the learned asymmetry ratio) → cue the interior → Bayesian population decode:
- **FWD-learned band → FORWARD replay:** DECODE_r [0.991, 0.983, 0.982, 0.985, 0.988, 0.984] — forward replay **6/6**
  (localized packet, width ~3.5, traversing ~49% of the track).
- **REV-learned band → REVERSE replay:** DECODE_r [−0.985, −0.989, −0.982, −0.982, −0.984, −0.988] — reverse replay **6/6**
  (position decreases with time = backward travel).
The **same protocol with opposite training gives opposite replay direction** (fwd/rev contrast) — proving the **learned
asymmetry** (not the band per se) sets the replay direction. **VERDICT: GO.**

## GOTCHA (reusable, silent-failure class) — direct step-calls don't advance the STDP clock
A raw `bridge._run_one_simulation_step()` loop does **NOT** advance `runtime_state.current_time_ms` (only
`step_simulation()` does, at `bridge.py:4127`). STDP timestamps each spike from `current_time_ms`, so without advancing
it **every spike gets the same timestamp → delta_t=0 for all pairs → STDP silently no-ops** (first run: 56k spikes, Δweight
exactly 0). The machinery to detect this existed (the spike-count + Δweight diagnostic caught it); the fix is one line —
advance `current_time_ms += dt_ms` per step in any direct-step harness that uses STDP. Any prior direct-step STDP harness
should be audited for this.

## Verdict + next (per THE LAW)
- **The gap#5 replay band's directional STRUCTURE is EMERGENT, 6-seed GO both halves** — the asymmetric band grows from
  directed traversal via STDP (structural), and the learned band functionally replays in the trained direction (functional).
  This closes the "hand-wired band" caveat of the gap#5 replay GO — the directional structure is now GROWN from experience,
  not designed.
- **HONEST SCOPE:** (1) the operating-strength **gain** is a uniform host scalar (a maturation gain; the learned STRUCTURE
  = the asymmetry ratio is what emerged — the absolute efficacy could instead be reached by more STDP laps at higher
  `stdp_w_max`, a follow-on); (2) the traversal drive is external (a moving sensory/idiothetic bump — legitimate as the
  world/body driving the place code); (3) still a standalone bridge (merge onto the one-brain = the remaining gap#5
  closure item, with the neural reader).

## Provenance
`research/runners/_gap5_learned_band_emergence.py` (modes: default 3-arm structural, `sixseed`, `func6`; logs
`learned_band{,2,3,_6seed}.log`, `learned_func{,2,3,4,6}.log`). Reuses the ECKER_CA3_PC preset (`d707bf34`) + the
committed replay decoder (`decode_and_width`, `d6e140bf`). NO `sim/` edit. GPU.
