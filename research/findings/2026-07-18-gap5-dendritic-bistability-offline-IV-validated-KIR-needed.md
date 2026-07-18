# Gap #5 — intrinsic dendritic bistability is achievable (OFFLINE I-V validated); it REQUIRES a KIR load line, which the point soma lacks

**2026-07-18.** The completion trilemma root-cause was: the single-compartment point soma has NO intrinsic bistability,
so a recurrent attractor strong enough to complete self-sustains. The deep-research gate (Antic 2010, Major-Larkum-Schiller
2013, Sanders-Berends-Major-Goldman-Lisman 2013, Schiller 2000) returned the mechanism + a ranked de-risk whose FIRST
step is a pure offline I-V test — done here, decisive.

## The test (the kernel's EXACT Mg-block, no spiking, seconds)

Steady-state net current `f(v) = I_NMDA(v) + I_load(v)`, zeros = fixed points.
- `I_NMDA(v) = g_res · mg_block(v) · (E_e − v)`, `mg_block(v) = 1/(1 + (Mg/3.57)·e^(−0.062v))` — the Jahr-Stevens block
  from `sim/kernels.py:275`, Mg=1 mM, E_e=0 mV.
- Two load lines compared: a LINEAR leak `g_L·(V_L − v)` vs a KIR (inward-rectifier K⁺) `g_K/(1+e^((v−v_kir)/k))·(E_K − v)`,
  E_K=−90 mV, v_kir=−50 mV (Sanders "perfect couple": K⁺ conductance HIGH at hyperpolarized, LOW at depolarized).

## Result — LINEAR leak has NO bistable band; KIR gives a WIDE robust one

| load line | g_res/g sweep | outcome |
|---|---|---|
| **linear leak** | 0.3 → 8.0 | monostable-DOWN ("boosting") up to ~2.0, then flips straight to monostable-UP ("self-trigger") at ~3.0 — **no 3-fixed-point window** (the down↔up flip has no bistable band at these samples) |
| **KIR** | g_res 2 → 14, gK 3 & 5 | **BISTABLE at EVERY point** — 3 fixed points: down ~−85 to −89 mV, unstable trigger ~−30 to −59 mV, up ~0 mV (plateau at E_e) |

This is the Sanders-2013 result reproduced with the project's own kernel: a linear leak makes the bistable regime a
knife-edge (essentially absent); the KIR's complementary voltage-dependence anchors a robust silent down state without
fighting the up state, opening a wide bistable band (g_res 2–14 all bistable). The three regimes
(boosting → **bistable** → self-triggering, Schiller-Schiller 2001) are exactly the single-cell analogue of the
network trilemma: "boosting" = the current transient dAP, "self-triggering" = the always-on point-attractor,
"bistable" = the target the point soma cannot reach without KIR.

## What this proves + the implementation path

- **Intrinsic dendritic bistability IS achievable** on this substrate — the up state is a true attractor (needs no
  continued input) and the down state is a robust silent rest. This resolves the magnitude-vs-bistability opposition:
  COMPLETION becomes a one-shot coincidence trigger crossing the unstable middle point (set by `k_thresh`), and
  SELF-SUSTAINING becomes intrinsic to each cell (its own plateau), so the recurrent weight can be SUB-CRITICAL
  (specific, silent rest) while completion still holds.
- **The kernel change (research Rank 1 + Rank 2), now offline-designed:** (1) split the plateau conductance into an
  input `trigger` term + a v-gated self-regenerating `sustain` term (`g_regen · mg_block(v) · sigmoid(k_v·(v − v_hold))`)
  with a slow reservoir decay, so the plateau HOLDS after the volley ends; (2) add a KIR-shaped down-state stabilizer
  so the bistability is robust (linear leak fails). Both additive / default-off / byte-identical when off.
- **Next de-risk (research-ordered):** implement the kernel change → single-cell latch-and-hold probe (ignite with a
  volley, remove input, verify HOLD; no-cue → silent; permuted/desync → no latch) → sweep the bistable band on-substrate
  → wire into the CA3 completion network with SUB-CRITICAL W_rec + `structural_sep` + `selective_inhib`, re-run the
  frozen + no-cue + permuted anti-cheats (prediction: completion now survives permuted-recall because sustaining is
  intrinsic, not loop-driven).

Sources: Antic 2010 (10.1002/jnr.22444), Major-Larkum-Schiller 2013 (10.1146/annurev-neuro-062111-150343),
Sanders et al 2013 "perfect couple" (10.1523/JNEUROSCI.1854-12.2013), Jadi et al 2012 (10.1371/journal.pcbi.1002550).
Probe: `research/findings/raw/` (offline I-V, inline). Kernel to modify: `fused_coincidence_plateau` (`sim/kernels.py:253`).
