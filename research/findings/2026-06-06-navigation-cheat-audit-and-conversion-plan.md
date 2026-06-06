# Navigation / gridworld cheat audit + conversion plan (roadmap step 2 kickoff) — 2026-06-06

**Status:** Evidence-based, read-only audit of the `g11_bg_runner.py` navigation path (code-explorer subagent,
`file:line` verified against the actual code, not the CLAUDE.md notes). This is the foundation for the navigational
cheat-removal arc (roadmap step 2). Conversion priority + recommended first target at the bottom.

## Two flagship configs (the audit's first finding — they differ in how many cheats they use)

- **Config A — "perception-arc complete" (2026-04-27), 4.08 ± 0.49 (6-seed).** Biologically STRONGER: no heuristic
  (`--cue-reflex-replaces-heuristic`), no direct goal coords, sensed (beacon-gradient) reward (`--sensed-reward`).
- **Config B — "K v2 visual scaling" (2026-05-05), 2.57 ± 0.11 (6-seed, 32×32).** Performance CHAMPION but
  biologically WEAKER: REVERTS to the action heuristic + Manhattan-distance reward.

So "the production path" is two-headed: the best *score* (B) uses MORE cheats than the most-biological config (A).
The cheat-removal goal is a config with NO cheats that still performs — partly "lift A's biology to B's score,"
partly "convert B's remaining cheats." Several cheats are in BOTH (N6, N7, N8, N9) — those are the universal targets.

## The 12 cheats (file:line, active-where, why-a-cheat)

| # | cheat | file:line | A? | B? | why it's a shortcut | biological alternative |
|---|---|---|---|---|---|---|
| N1 | action heuristic: raw (gx,gy,x,y) → 800 pA into the winning cortex pool | `g11_bg_runner.py:3372-3398` | no | **yes** | reads raw coords, overrides all learned signal; no perception/STDP/cascade | K v2 visual + cue-reflex (STDP); beacon gradient; landmark bearing |
| N2 | goal position painted into the rendered image | `g11_bg_runner.py:3635-3664` | no | **yes** | V1/IT "see" the goal for free — no perceptual problem to solve | separate reward-presence sensor / salient learned stimulus, goal not baked into pixels |
| N3 | learned perception driven by raw goal-relative vector dx,dy | `g11_bg_runner.py:3439-3446` | opt-in | no | agent "knows" the signed vector to goal | visual-cortex perception; landmark-relative displacement |
| N4 | place/goal cells driven by raw (x,y),(gx,gy) | `g11_bg_runner.py:3548-3566` | opt-in | no | bypasses sensory encoding | allothetic cues (visual landmarks, path integration from motor efference) |
| N5 | Manhattan-distance reward (+1/−1 on dist change) | `g11_bg_runner.py:3782-3806` | no | **yes** | needs (gx,gy),(x,y); host-computed | sensed beacon-gradient reward (exists, validated); ultimately spiking SNc RPE |
| N6 | host-side argmax action decode (D2H + CPU dict + argmax every substep) | `g11_bg_runner.py:3759-3763` | **yes** | **yes** | not spiking WTA; host counts spikes + argmax | on-GPU WTA via MSN lateral inhibition + GPi competition / readout pool |
| N7 | V1 Gabor weights analytically pre-initialized | `sim/visual_cortex.py` (`apply_v1_gabor_weights`), called `g11_bg_runner.py:1699-1706` | yes* | **yes** | weights set by formula, not learned | developmental critical-period STDP on natural images (Olshausen-Field); labeled scaffold |
| N8 | thalamic relay driven by tonic 300 pA (NOT GPi disinhibition) | `g11_bg_runner.py:3329-3336` | **yes** | **yes** | thalamus externally paced; the genuine GPi→thal disinhibition chain is never used | **genuine cascade — ALREADY VALIDATED in `gated_compose_bg_genuine_demo.py`; port it** |
| N9 | tonic GPe/GPi/STN/SNc external drives (110–300 pA) | `g11_bg_runner.py:3329-3336` | **yes** | **yes** | hard-wired operating point; bypasses closed-loop pacing | closed STN loop (cluster A exists); genuine SNc RPE (hard) |
| N10 | cue-reflex beacon: hand-coded 8-direction angular sensors | `g11_bg_runner.py ~2445-2510` | A only | no | geometric bearing, not learned features | IT feature detectors for goal-associated stimuli |
| N11 | beacon intensity = 1/(1+Manhattan d) | beacon falloff | A only | no | needs d from coords | physically-simulated emission (rendered brightness), not a formula |
| N12 | landmark sensors: hand-coded angle/distance to fixed anchors | `g11_bg_runner.py ~2510-2600` | A only | no | raw-coordinate geometry | visual-landmark recognition via V1→V2→IT; optic-flow displacement |

\*N7 active in A only when the visual path is enabled; it is always on in B.

## NOT a cheat (audit correction to the owner's mental model)

**Cross-projection "cheat #5" (`--bg-cross-projections`) is NOT a shortcut to remove.** It is an *unimplemented
biological capability* (diffuse cortical→striatal input enabling action switching), parked because all tested
variants were NEGATIVE. Neither flagship enables it. So it leaves the cheat-removal list entirely — it would be a
future biological ADDITION (capability work), not a conversion. (The original "cheat #5" framing was about the
heuristic bypass, N1, not cross-projections.)

## What is already biological (do NOT re-litigate)

The BG cascade architecture (per-action D1/D2/GPe/GPi/thal/motor + direct/indirect/hyperdirect pathways), STDP /
R-STDP corticostriatal learning, MSN lateral inhibition, D1/D2 asymmetric reward modulation, striatal PV-FSIs,
cluster-A closed loop (cortex→STN hyperdirect, thal→cortex feedback), the V1→V2→IT hierarchy *structure*, spiking
SNc dynamics — all genuinely on-substrate. The cheats are in the *drives, perception inputs, reward computation,
and action decode*, not the core architecture.

## Conversion priority (controller synthesis) + recommended FIRST target

**▶ RECOMMENDED FIRST: N8 — thalamic tonic drive → genuine GPi→thal disinhibition.** Best ROI, by a lot:
- **Most impactful single biological correction** (the thalamus is the cascade's output gate; tonic-driving it
  short-circuits the whole BG selection logic).
- **No new science** — the genuine GPi-tonic / D1-silences-GPi / thal-disinhibited pattern is ALREADY VALIDATED in
  `gated_compose_bg_genuine_demo.py` (the weight scales `GPI_TONIC_PA`, D1→GPi ≈ 15, GPi→thal ≈ 8 are known).
- **Exact analog of the resolved conversational cheat #2** (we already did this move once, in the composer).
- In BOTH flagships → universal win.
- Clean, well-scoped, gated (does selection still work in spikes once thalamus is released by disinhibition rather
  than externally paced?).

Then, in increasing difficulty:
1. **N8** thalamic disinhibition (port the validated pattern) — *recommended start*.
2. **N6** host argmax → on-GPU spiking WTA readout (medium; the selection should be readable from thalamic/motor
   firing without D2H).
3. **N5** Manhattan reward → sensed beacon-gradient (a validated flag switch for the intermediate; spiking SNc RPE
   is the longer pure-biology target).
4. **N2** goal-in-image → a real perceptual problem (egocentric view + salient learned stimulus). Architecturally
   non-trivial but scientifically essential (otherwise V1 has nothing to learn).
5. **N1** action heuristic — load-bearing (8× worse without it at 16×16), removal requires perception quality first
   (Config A removed it only at 8×8/16×16; B keeps it even with K v2). Hardest; do after perception (N2/N7) improves.
6. **N3/N4** raw coords in place/learned-perception — opt-in, not in the champion config; low urgency.
7. **N7** Gabor pre-init → developmental STDP (least urgent; biologically-motivated scaffold).
8. **N9** tonic BG drives (GPe/GPi/STN/SNc) → closed-loop pacing + genuine SNc RPE (hardest neuroscience; defer to last).

## Method (same rigor as the conversational arc)

Each conversion: cheap-first de-risk → gate on the actual benchmark (cheat-5 multi-goal navigation score, NOT a
proxy) → controls (the cheat-on baseline + a no-op control) → multi-seed (6) → honest GO/BOUNDARY → both remotes.
Protected `sim/` edits only with owner approval + byte-for-byte review. The navigation runs on the SAME core
`SimulationBridge` (different `BrainRegion`/`RegionPathway` combinations) — not a separate brain.

## Key reference files
- `research/runners/g11_bg_runner.py` — every cheat instantiated here.
- `research/runners/gated_compose_bg_genuine_demo.py` — the validated GPi→thal disinhibition (N8 reference impl).
- `research/runners/_framework_inhibition_minimal_probe.py` — inhibitory-conductance weight-scale control (N8 needs it).
- `sim/visual_cortex.py` — Gabor pre-init + V1→IT hierarchy (N7, N2).
- `research/findings/2026-06-04-pure-biology-cheat-removal-backlog.md` — standing backlog (this audit = item #7).
- `research/findings/2026-05-01-tier0-no-heuristic-perception-bottleneck.md` — heuristic removal needs perception first (N1).
