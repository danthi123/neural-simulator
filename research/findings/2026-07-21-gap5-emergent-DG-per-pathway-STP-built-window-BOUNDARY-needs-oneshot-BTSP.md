# gap#5 emergent-DG detonator — a reusable per-pathway STP-disable `sim/` mechanism BUILT (byte-identical), but it is necessary-NOT-sufficient; the sparse+separated window is a BOUNDARY needing one-shot BTSP-during-encoding (the gap#4↔#5 unification)

**2026-07-21.** The gap#5 emergence-bar close (self-organized DG-selected CA3 assemblies, replacing the pre-assigned
random masks the CLOSED completion + SWR readout use) was pinned to a mossy-detonator fix: STP-OFF on mossy dg→ca3 (so
it detonates) SIMULTANEOUSLY with STP-ON on ca3→ca3 (so the recurrent doesn't avalanche). STP was a GLOBAL toggle only,
so this needed a new substrate capability. Built it; it works but does NOT open the window — a precisely-characterized
boundary that names the next mechanism.

## (1) The `sim/` mechanism — per-pathway STP-disable (additive / guarded / DEFAULT-OFF / BYTE-IDENTICAL, verified)
`RegionPathway.stp_disabled: bool = False` (`sim/regions.py`, +16) → wiring-plan flag → `cp_stp_disabled_mask` in
`sim/bridge.py` (+57), applied at the STP effective-strength site in `_run_one_simulation_step`: when the mask is None
the code is the verbatim original `base·stp_u·stp_x` (byte-identical); else `base·where(mask,1.0,stp_u·stp_x)` so flagged
synapses skip depression (their STP u/x still evolve; only the multiplier is overridden). Mirrors the existing
`transmission_gate`/`graded` mask pattern. **A genuine reusable capability the substrate lacked: opposite STP states on
co-resident pathways.** VERIFIED (controller trust-but-verify, not the subagent's word): `tests/test_stp_disabled_pathway.py`
**4/4** (default → mask is None; two default bridges same-seed → bit-identical driven trajectory; a flagged pathway →
mask exactly its synapses; the detonator FIRES: gated `stp_u·stp_x=0.037` → 0 downstream spikes vs STP-disabled → 6);
`tests/test_determinism.py` **9/9** (the step edit preserves determinism); the 4 pre-existing `test_regions.py`
numpy-cupy failures are in untouched code (confirmed identical on pristine `sim/`).

## (2) It fires but does NOT open a sparse+separated+stable window — BOUNDARY
At the real operating point (mossy w=200, amplify/bistable recurrent, n_ca3=2000, DG-direct + sync):
- Boundary reproduced: global STP-ON → assemblies `[0,0,0]` (silent); global STP-OFF → `[2000,2000,2000]` sep=1.0 (full
  avalanche).
- The mechanism FIRES decisively: mossy-STP-disable takes CA3 detonation **12 → 1114 ever-fired cells** (13 → 1460
  spikes) — the mossy is definitively unblocked from STP depression.
- BUT no sparse+separated+stable SETTLED window: mossy-STP-disable gives `[0,0,0]` settled (theta≥0.3) across the full
  sweep mossy_w{40,80,150,200} × ca3_ff_inhib{None,60,150} × dg_ffi{6,12} × drive{500,700}. E%-max FFI doesn't cap the
  avalanche either.

## Root cause (the named next mechanism)
The settled assembly needs a **sustained sparse** state, but the bistable recurrent is ALL-OR-NONE in the read-only
selection pass: a sparse novel UN-POTENTIATED set can't trigger the plateau (→ transient detonation decays → 0 settled),
and once the recurrent is un-depressed the plateau ignites GLOBALLY (→ 2000-cell avalanche). Per-pathway STP
(mossy-off/recurrent-on) cannot thread this needle — disabling mossy STP only makes the detonation denser+more transient
(1114 diffuse cells ~1.3 spikes each), not a stable re-detonated sparse set. **⇒ the binding constraint is the
sustained-sparse-attractor problem; the missing ingredient is ONE-SHOT BTSP DURING ENCODING** (plateau-gated recurrent
potentiation among the detonated set — the detonated cells POTENTIATE their mutual recurrents in one shot so the sparse
set becomes a stable attractor). **That is the gap#4↔#5 unification** (`2026-07-18-gap4-gap5-UNIFICATION-BTSP-...`,
`2026-07-19-gap5-emergent-DG-ROOT-CAUSE-...` "Problem B"), NOT mossy STP-depression. Per THE LAW: a verdict on the METHOD
(mossy STP-off), the capability (self-organized assemblies) proceeds via the named next mechanism.

## Corroboration (both gap#5 subagents this cycle)
The prior "emergent-DG SELECTION 6-seed GO" (n_ca3=400) does **NOT reproduce on the current codebase at n_ca3=2000** —
the committed `_select_emergent` gives `[0,0,0]`; the read-pass sparse-selection window is scale-fragile independent of
STP (the GO rested on now-deleted scratchpad). So the emergent-DG piece is genuinely OPEN, and the honest next step is
the one-shot-BTSP encoding (gap#4↔#5), not more STP/mossy tuning.

## a-1 RAG on the named next mechanism (2026-07-21) — it is a SUBSTANTIAL build, not a quick wiring
Checked `2026-07-18-gap4-gap5-UNIFICATION-BTSP-stores-bistable-CA3-completes-mechanism-6seed-GO.md`: the existing BTSP
unification (6/6 mechanism, cue ~0.18) drove the plateau on a **PRE-ASSIGNED** assembly (`encode_btsp` path in
`_riii_ca3_synchronous_assembly_derisk.run`). So the EMERGENT one-shot-BTSP does NOT transfer directly — it needs the
mossy/DG to first SELECT a SPARSE (6-40) separated CA3 code (the pattern-separation the mossy sweep could NOT achieve —
the STP-unblocked detonation was DIFFUSE, 1114 cells ~1.3 spikes each), and only THEN BTSP-store that emergent set. ⇒ the
real chain is DG-pattern-separation (sparse code) → emergent-BTSP-store → bistable-complete; the binding blocker is the
UPSTREAM DG sparsification (get the detonated set sparse), then the (validated) BTSP-store + bistable-complete follow.
A substantial intricate frontier for a careful fresh pass, not a next-tick wiring.

## Deliverable
BOUNDARY + a verified reusable `sim/` mechanism. The per-pathway STP-disable is committed (byte-identical, tested,
reusable for any circuit needing opposite STP states on co-resident pathways — e.g. the SWR phase-2 Schaffer STP-off
could use it in place of the global-toggle hack). The emergent-DG window's next mechanism is precisely named.
