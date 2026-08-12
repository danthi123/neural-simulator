---
type: finding
status: qualified
date: 2026-08-12
mechanism: btsp
---
# Gap#4 / E3 — a per-turn BTSP PLATEAU write is a LASTING trace: it is STILL RECALLABLE after an intervening decay window where a non-plateau write has decayed below recall (6-seed GO). Spiking WRITE + spiking RECALL on the real bridge; the LASTING side is a runner-level tag-and-capture MODEL (next rung = a guarded default-off sim/ kernel). NO new sim/ edit.

**2026-08-12.** Burn-down **E3**: production LEARN today writes the RF store (a synaptic write), but the fuller "the turn
writes synapses" is a **behavioral-timescale-plasticity (BTSP) plateau that leaves a LASTING trace** — one that
PERSISTS through later activity and stays RECALLABLE, where an ordinary write decays. This de-risk closes the
persistence rung: a plateau-written fact is still recalled (the post cell fires to the cue) after a decay window of
intervening activity, while a non-plateau write that recalled at t0 has decayed below recall.

## What was ALREADY closed (corpus-first — this is NOT a re-derivation)
- **2026-07-18 on-bridge BTSP GO** measured the **WRITE**: a HELD bistable plateau potentiates the co-active
  pre→post synapse one-shot over a seconds-long window (held_dw ~110 vs transient ~13). It did NOT test whether that
  write PERSISTS and stays recallable.
- **2026-07-18/20 recall-BIAS probe** measured recall **immediately** after storing (a partial cue drives held-out
  partners). It inserted NO decay window / intervening activity before recall.
- So the genuine UNCLOSED residual = does the plateau write **LAST**? Measured here as a BEHAVIORAL/spiking recall
  (post firing to the cue), never a weight read (docs/TERMS.md: a weight read is a proxy; behaviour is the capability).
  This is distinct from the gap#5 episodic-completion lane (attractor pattern-completion), which is not duplicated.

## Mechanism (genuinely-spiking WRITE + RECALL on a real `SimulationBridge`; runner-level MODEL of the LASTING side)
- **WRITE** — the REAL on-bridge BTSP block (`enable_btsp`, `fused_btsp_update`) reads the REAL bistable BDSP apical
  plateau (`bdsp_apical_bistable`, self-regen + KIR) as the instructive signal and potentiates pre→post-TARGET
  one-shot. Held plateau ⇒ supra-barrier write; transient plateau ⇒ sub-barrier write; silent apical ⇒ no write.
- **DECAY WINDOW** — the bridge KEEPS STEPPING (intervening activity = later turns); BTSP/BDSP learning OFF, so only a
  maintenance rule moves the pre→post weights. The maintenance rule is **synaptic TAG-AND-CAPTURE**: a synapse above a
  capture threshold (barrier) is stabilized (resists decay); a sub-barrier synapse passively decays (`w *= 1-beta`).
  A plateau is what naturally drives a synapse over the barrier (the large one-shot BTSP potentiation); ordinary weak
  plasticity does not. This is a runner-level model applied to `cp_connections.data` — NOT yet a sim/ kernel.
- **RECALL (spiking)** — after the window, fire the pre cue only (learning + maintenance OFF), count post-TARGET
  spikes. Post firing rate ∈ [0, ~0.09/step] (the cell's intrinsic rate ceiling), ~0 when afferents are silent.

## The pre-registered GO gate (6-seed 42/43/44/100/101/102, ALL seeds)
<!--derived-->
(Per-seed values below are rounded from the cited raw JSON; means/ratios are computed over it.)
"DECAYED below recall" is graded PER SEED against that seed's own persisted (plateau) trace (recall ≤ 0.4× the
plateau AND below the "fires" line) — robust to the ~0.003–0.006/step spontaneous floor. "still fires" is ABSOLUTE
(≥ `RECALL_HI` = 0.015). Params: n_pre 64, w_max 10, btsp_lr 0.04, barrier 2.0, beta 0.04, static_w 1.5,
window 200 steps, recall 200 steps.

| seed | PLATEAU t0 / after / distr | TRANSIENT t0 / after | MOAT after | NO-CAPTURE after | STATIC t0 / after | plateau W: write→post-window |
|---|---|---|---|---|---|---|
| 42  | 0.091 / 0.092 / 0.003 | 0.036 / 0.005 | 0.005 | 0.005 | 0.040 / 0.001 | 1129 → 1101 |
| 43  | 0.074 / 0.079 / 0.003 | 0.026 / 0.006 | 0.006 | 0.006 | 0.033 / 0.003 | 1079 → 1018 |
| 44  | 0.069 / 0.069 / 0.004 | 0.024 / 0.003 | 0.003 | 0.003 | 0.033 / 0.003 | 1129 → 1062 |
| 100 | 0.051 / 0.054 / 0.004 | 0.016 / 0.003 | 0.003 | 0.003 | 0.024 / 0.000 | 1065 → 990 |
| 101 | 0.072 / 0.071 / 0.001 | 0.026 / 0.003 | 0.003 | 0.003 | 0.034 / 0.001 | 1174 → 1125 |
| 102 | 0.061 / 0.065 / 0.000 | 0.019 / 0.003 | 0.003 | 0.003 | 0.028 / 0.000 | 1136 → 1095 |

- **L1 LASTING** — plateau STILL FIRES after the window (0.054–0.092/step, all ≥ 0.015); transient + moat have DECAYED.
- **L2 CRUX (persistence, not write-strength)** — the STATIC and TRANSIENT writes BOTH recalled at t0 (0.024–0.040
  and 0.016–0.036, all ≥ 0.015), then DECAYED after the window ⇒ the after-window failure is DECAY, not a failed write.
- **L3 LESION (capture is load-bearing)** — the **identical** big plateau write (weight ~1065–1174) decays to ~0.3 and
  recall fails (0.003–0.006) once the tag-and-capture maintenance is removed (`plateau_nocapture`). This is the single
  strongest arm: same write, capture on → weight ~1100 and post fires; capture off → weight ~0.3 and post silent.
- **L4 ANTI-MAGNITUDE** — a large sub-barrier STATIC weight (1.5/synapse, summed 384) recalls at t0 then decays: a
  trivially-large static weight does NOT last; it must cross the barrier, which in the write phase only the plateau does.
- **L5 ATTRIBUTABILITY** — the post-DISTRACTOR half (never plateaued) does not recall (0.000–0.004): only the
  plateau-targeted cells persist.
- **Attribution** — 95.1% of plateau recall-after is attributable to the capture manipulation (`attributable_to`,
  plateau 0.0717 vs no-capture lesion 0.0035); ≤4.9% is in the control.

## The instrument was verified — and shown capable of FAILING
- `off_dw == 0` all 6 seeds (`enable_btsp=False` write path byte-identical); the maintenance rule is inert when
  `beta=0`; the spiking recall readout distinguishes a huge weight (≥0.015) from a zero weight (≤0.008).
- **Falsifications (3-seed) that CORRECTLY report BOUNDARY**: `--beta 0.0` (no decay → the controls do not decay →
  L1–L4 fail) and `--barrier 100` (nothing captured → the plateau trace also decays → plateau falls below HI). The
  gate is not a check that cannot fail.

## Honest scope / caveats (a boundary is a next rung, never a stop)
<!--derived-->
- The WRITE (on-bridge BTSP + the bistable BDSP apical) and the RECALL (post spiking) are on the real spiking
  substrate. The **LASTING side (tag-and-capture stabilization) is a RUNNER-LEVEL MODEL** applied to
  `cp_connections.data` — the interesting non-linearity (barrier-gated stabilization vs passive decay) is host code,
  not a spiking/synaptic kernel yet. This is a de-risk of the CONCEPT, honestly placed; it is NOT the production path.
- This is **NOT "consolidation"** in the docs/TERMS.md sense (no replay/reactivation path executes) — it is a
  bistable-synapse / synaptic-tag maintenance model. "LASTING" here means: recallable after a 200-step decay window of
  intervening activity where a non-plateau write drops below recall.
- Absolute post firing is low (≤ ~0.09/step, the cell's rate cap); the separation is plateau ~0.07 vs decayed ~0.004
  (~17×). A single feedforward BTSP synapse drives only modest downstream firing — robust recall needed n_pre=64 and a
  strong cue; this echoes why the recall-bias probe used recurrence. Seed 100's transient-t0 (0.016) is the thinnest
  margin above HI.

## NEXT RUNG (the real burn-down)
Port the tag-and-capture maintenance to a **guarded, default-OFF, byte-identical-when-off `sim/` kernel** — a
per-synapse bistable-weight (double-well / capture-threshold) update alongside `hebbian_weight_decay`, so the LASTING
side runs ON the substrate, not in the runner. Then wire it under production LEARN so a taught fact's per-turn write is
a genuine spiking BTSP plateau + on-substrate capture. Biology to bind in `research/biology/`: Bittner & Magee 2017
(BTSP, the supra-threshold one-shot plateau write); Frey & Morris 1997 synaptic tag-and-capture (Kandel 6e Ch 67, the
"capture" of a lasting trace); Lisman 1985 CaMKII "perpetuating switch" (the bistable maintenance).

## Repro
```
SIM_BACKEND=numpy python -m research.runners._gap4_btsp_lasting_trace_recall_after_delay_derisk \
    --seeds 42 43 44 100 101 102
```
Runner: `research/runners/_gap4_btsp_lasting_trace_recall_after_delay_derisk.py`.
Raw: `research/findings/raw/_gap4_btsp_lasting_trace_recall_after_delay.json`.
