---
type: finding
status: smoke
date: 2026-08-11
mechanism: MULTI-SLOT variable-binding WORKING MEMORY — R banks of the D3 slow-NMDA bistable HOLD slot on ONE bridge sharing ONE FS pool, each bank latching one role-filler bind; holds >=2 bindings simultaneously across novel fillers with no cross-talk
lane: emergence engine / working memory (scaling the single-slot variable-binding WM toward multi-variable language)
verdict: 1-SEED SMOKE-GO (indicator; run the 6-seed sweep) — a spiking WM composed of R disjoint D3 NMDA banks sharing one FS pool HOLDS k=1,2,3,4 role-filler bindings SIMULTANEOUSLY and recovers EACH across NOVEL fillers (held-out ALL-correct 1.000, per-slot all 1.000, >> chance 1/6), then degrades GRACEFULLY (k=5 all-correct 0.900, k=6 0.775; per-slot stays ~0.96). CAPACITY CEILING (all-correct>=0.80) = k=5; clean (1.000) to k=4. All teeth bite: lesion-the-hold collapses (k>=2 -> 0.000), the SUPERPOSED-single-slot baseline collides at exactly ~1/k (item 0.500 at k=2 down to 1/6 at k=6 = the ~2-cap regime), filler-swap cross-talk ~0, referent-shuffle ~0, hold-alive positive with external input ASSERTED zero. The break past k=5 is modest register SUPPRESSION under the shared FS (collapse-rate 0.02@k5 -> 0.0375@k6), NOT catastrophic cross-talk. Residual (unchanged from single-slot): the role-by-position gate is a host MARKER (the LEARNED spiking role-gate is gap#4, per 739a8867); the BIND + verb readout are host. k>6 is confounded by the RUNG6c binder's _K=6 slot cap (guarded NOT-RUNNABLE) — a bigger binder is the lever to probe higher k.
seeds: [42]
runner: research/runners/_multi_slot_binding_derisk.py
artifacts:
  - research/findings/raw/_multi_slot_binding/multi_slot_smoke.json
instrument: composition of THREE banked GOs (each re-read at source this session) — HOLD = R banks of the D3 slow-NMDA bistable persistent slot on ONE SimulationBridge sharing ONE FS inhibitory pool (`build_persistent_slot(seed, K=R*n_slot)`; register r = pools [r*n_slot,(r+1)*n_slot); the exact keystone step-2a coexistence substrate — separate assemblies, shared inhibition, NOT R separate brains); BIND = the RUNG6c content-agnostic Hebbian fast-weight binder (`HebbianBinder`; each fixed entity -> a stable local slot index); WRITE-GATE = a role-by-position MARKER (subject->reg0, object->reg1, ...). Task = a k-role agreement clause [e_r]+[L fillers] per role then k agreeing verbs, verb r agrees with the entity bound in register r; scored on HELD-OUT disjoint NOVEL filler tuples. SIM_BACKEND=numpy; NO sim/ edit.
---

# Multi-slot variable-binding WORKING MEMORY: R disjoint NMDA banks (one shared FS) HOLD k role-filler bindings simultaneously, each recovered across novel fillers — no cross-talk to k=4 (1-seed SMOKE); the superposed single slot collides at ~1/k (the ~2-cap)

Conversation needs holding MULTIPLE variables at once — a SUBJECT *and* an OBJECT (or a small stack), each agreeing with
its own verb across an arbitrary intervening span. The single-slot variable-binding WM GO
(`2026-08-11-variable-binding-working-memory-gated-slot-surpasses-HTM-heldout-1.000-vs-0.000-6seed-GO.md`) latches ONE
variable (held-out 1.000) with a LOAD-BEARING spiking hold. This de-risk asks the multi-variable question: does the SAME
composition hold k>=2 bindings WITHOUT CROSS-TALK, generalising to novel fillers, and at what k does it break?

## Our own record first (read, cited, COMPLEMENTED — not re-derived)

The ~2 cap is a SUPERPOSITION-SNR artifact of storing all binds in ONE shared register (EDGE-5 4-rung refutation;
`2026-05-12-cumulative-binding-fixed-capacity.md`; `2026-06-05-B-innetwork-superposition-NEGATIVE.md`) — a real brain
ALLOCATES a distinct near-orthogonal SLOT per bind, converting capacity from SNR-limited (~2) to slot-count-limited. The
gap#2 KEYSTONE arc (`2026-07-17-keystone-2-spiking-slot-binder-STEP1-prereq-GO.md`) already showed, on
`build_persistent_slot`: (step 2a) multi-slot COEXISTENCE is GO (P=3 pools coexist via NMDA persistence, no-recur
collapses 3/3->0/3); (retrieval) a competitive-slot LTM store recovers P=2=1.00 / P=3=1.00 / P=4=0.79 vs the shared
store's ~2 cap. **But that GO is an LTM STORE** (plastic slot->filler, read by RE-DRIVING; its no-recur was 1.00, i.e. the
NMDA hold was NOT load-bearing). **This de-risk is the complementary, un-done piece:** multi-slot as WORKING MEMORY where
the D3 NMDA HOLD is LOAD-BEARING (read the HELD bump, zero input), on the HELD-OUT AGREEMENT STREAM, with per-slot
recovery + a cross-talk control + the memory teeth biting PER SLOT.

## Result — 1-seed SMOKE-GO (`research/findings/raw/_multi_slot_binding/multi_slot_smoke.json`)

<!--derived-->
Seed 42, n_ent=6 (chance 0.167), n_slot=6, n_fill=10, L=2 per role, held-out NOVEL filler tuples:

| k | held-out ALL-correct | per-slot (mean) | LESION-the-hold ALL | SUPERPOSED-1slot (all / per-item) | filler-swap | collapse | referent-shuffle |
|---|---|---|---|---|---|---|---|
| 1 | **1.000** | 1.000 | 0.267 | n/a | 0.000 | 0.000 | 0.000 |
| 2 | **1.000** | 1.000 | **0.000** | **0.000 / 0.500** | 0.000 | 0.000 | 0.000 |
| 3 | **1.000** | 1.000 | **0.000** | **0.000 / 0.317** | 0.000 | 0.000 | 0.000 |
| 4 | **1.000** | 1.000 | 0.000 | **0.000 / 0.250** | 0.000 | 0.000 | 0.000 |
| 5 | 0.900 | 0.975 | 0.000 | **0.000 / 0.200** | 0.005 | 0.020 | 0.000 |
| 6 | 0.775 | 0.963 | 0.000 | **0.000 / 0.167** | 0.000 | 0.037 | 0.000 |

**How many concurrent bindings hold cleanly: k=1,2,3,4 all at 1.000** (each register recovers ITS OWN entity's agreeing
verb; ALL-correct requires EVERY register right, so it is the product of k near-perfect per-slot reads). **CAPACITY
CEILING (all-correct>=0.80) = k=5**; the break past it is GRACEFUL (k=5 all-correct 0.900, k=6 0.775) with per-slot
recovery still ~0.96 — the fall is the product of k terms plus modest register SUPPRESSION under the shared FS
(collapse-rate 0.020@k5 -> 0.037@k6), NOT a catastrophic cross-talk collapse at 2. This is far past the superposition
~2 cap, which the SUPERPOSED-single-slot control reproduces (it collides at exactly ~1/k every k). k=6 is the largest k
testable at n_ent=6 (distinct entities per clause), which is also the RUNG6c binder's _K=6 slot cap.

**All anti-cheat teeth bite** (this is the surpass, not a scaffold artifact):
- **LESION-the-hold** (recur=0, the stateless bridge) -> k>=2 ALL-correct **0.000** (k=1: 0.267) — the spiking slow-NMDA
  banks are genuinely HOLDING the k bindings across the span, not a host store. **This is the WM difference from the
  keystone LTM store** (whose no-recur was 1.00).
- **SUPERPOSED-single-slot collide baseline** -> cram all k bindings into ONE bank (superpose): ALL-correct **0.000** and
  per-item recovery **exactly ~1/k** (0.500 / 0.317 / 0.250 at k=2/3/4) — one bump wins the 1-of-K WTA. This reproduces
  the ~2-cap collision and proves the multi-register SEPARATION is load-bearing (not the attractor alone).
- **CROSS-TALK (filler-swap) 0.000** every k — a register never returns another role's filler. (Disjoint banks: each
  register reads only its own pools, per the keystone "separate slots never share a value pool"; see the honesty note.)
- **REFERENT-SHUFFLE ~0** — no topic->answer leakage, per slot.
- **HOLD-NOT-RE-READ** — external input ASSERTED zero across the whole hold+read span (hold-alive positive per register)
  — the banks SUSTAIN, they do not re-read a host store per step.

## Honesty / scope — what is genuinely solved+spiking, and the precisely-named residuals (per brain-based-only)

<!--derived-->
NO-EXTERNAL-NEEDED: grounded in our OWN verified GOs (D3 hold, RUNG6c bind) + the keystone gap#2 arc, each re-read at
source this session; the biology (multi-item bump-attractor WM; Compte/Wang; Bouchacourt-Buschman) is corroborating
context, not load-bearing.

- **SOLVED + spiking + load-bearing:** the multi-slot MEMORY composition — R disjoint slow-NMDA banks on ONE bridge with
  ONE shared FS coexist and each SUSTAINS its own bind (lesion-load-bearing at k>=2), so k concurrent role-filler bindings
  are carried invariantly across NOVEL fillers with per-slot recovery 1.000 and zero filler-swap. This directly scales the
  single-slot WM GO to multiple concurrent variables.
- **The capacity picture, stated precisely (the honest part):** with DISJOINT pool banks, cross-register filler-swap is
  near-impossible by construction (each register reads only its own pools) — filler-swap stays ~0.000 to k=6. The only
  SHARED resource is the FS inhibition; it does NOT limit coexistence at k<=4 (all banks hold, all-correct 1.000), and
  begins to SUPPRESS a bank only at k=5-6 (collapse-rate 0.020 -> 0.037) as k pools drive the one shared FS harder —
  matching the keystone step-2a observation that the shared FS is not the coexistence-limiting factor at sparse loads. So
  within the valid regime the capacity is register-count-limited with a soft edge at k~5-6 from shared inhibition, NOT
  cross-talk-limited — which is exactly why the SUPERPOSED-single-slot control (shared value pool) collides at ~1/k every
  k while the separated banks hold to k=4-5. This is far past the superposition ~2 cap.
- **A confound I caught + guarded (the anti-cheat discipline):** an initial probe at n_ent=8/n_slot=8 read a FALSE
  degradation (per-slot ~0.77 even at k=1, filler-swap up to 0.20). Root cause on reading my own substrate: the RUNG6c
  `HebbianBinder` is hardcoded to `_K=6` slots (`slot()` caps at `min(free, _K-1)`), so the 7th/8th entities COLLIDE onto
  the last slot — a binder-capacity artifact, NOT a WM interference limit. The runner now BLOCKS `n_ent > _BINDER_K` as
  NOT-RUNNABLE. The valid regime is n_ent<=6; probing k>6 needs a binder with more slots (a named lever, not a wall).
- **THE RESIDUALS (unchanged from the single-slot GO — not re-opened here):**
  1. **The gate's ROLE/TIMING is a host MARKER** (role-by-position: subject->reg0, object->reg1). `739a8867` established
     that even a host position-ORACLE fails to induce role at 6 seeds -> the residual is CREDIT ASSIGNMENT (gap#4). The
     LEARNED, EMERGENT, SPIKING multi-register role-gate is the open problem.
  2. The BIND is host numpy (RUNG6c fast-weight; its spiking-STP realisation is a banked next rung); the verb read-out is
     a host deref of the held bank.
- **Named next steps (dependency-ordered):** (a) the 6-seed sweep (decisive; 1-seed is only an indicator); (b) to probe
  the ceiling past k=6, raise the RUNG6c binder's slot count (`_K`) so more distinct entities/registers are addressable,
  then re-sweep to see whether the soft shared-FS suppression edge (k~5-6 here) is the true limit or moves with capacity;
  (c) the learned multi-register role-gate (gap#4); then (d) the spiking bind + wire into the emergence stream.
  Reuse-by-import; NO `sim/` edit.

## The exact 6-seed command (decisive)

(writes to `multi_slot_6seed.json` under `research/findings/raw/_multi_slot_binding/`)
```
SIM_BACKEND=numpy python -m research.runners._multi_slot_binding_derisk --seeds 42 43 44 100 101 102 \
  --ks 1 2 3 4 5 6 --n-ent 6 --n-slot 6 --n-fill 10 --distance 2 --n-test 60 --out multi_slot_6seed.json
```
