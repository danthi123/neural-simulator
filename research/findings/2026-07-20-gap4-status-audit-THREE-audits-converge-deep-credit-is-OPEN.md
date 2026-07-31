---
type: finding
status: live
date: 2026-07-20
mechanism: gap4-credit
---

# gap#4 status audit — THREE independent audits converge: deep credit on spikes is OPEN; the board overstates closure

**2026-07-20.** Read-only audit triggered before starting gap#4 work (so as not to build on an invalidated result).
Three independent read-only agents traced every gap#4 claim to its runner, its seed status, and its verdict. They
**converge on the substance** — and the third **corrects** the second on an important detail, which is why all three
are recorded rather than just the tidiest one.

## 1. The headline: "FULLY RESOLVED" on the board is a REDEFINITION, not a result

`GAP_CLOSURE_MISSION.md` carries **three mutually incompatible gap#4 verdicts, all dated 2026-07-19, in one file**:

| verdict | location |
|---|---|
| "gap#4 itself **FULLY RESOLVED**" | `:333-334` |
| "gap#4 supervised **PARKED** (capability met by the unsupervised method)" | `:423` |
| "gap#4 keystone (DEEP biological local-credit to accuracy) **remains the honest, field-hard OPEN frontier** … I initially over-claimed 'substantially achieved' — corrected" | `:481-483` |

**No experiment closed gap#4.** The supervised-deep-credit-to-accuracy *method* was parked after failing, and the
*capability* was reassigned to the pre-existing unsupervised stream cortex (EMERGE-30..55, 2026-07-02). By the board's
**own** definition of CLOSED (`:44-47`, criterion (e) "wired into the actual system the owner uses"), gap#4 fails:
`enable_btsp` appears only in de-risk runners, `sim/`, and its test — **no console, no agent**.

## 2. What is GENUINELY established on the real spiking substrate, clean-seeded, 6-seed

| mechanism | status |
|---|---|
| **BTSP** — local, one-shot, plateau-gated credit (`fused_btsp_update`, real `SimulationBridge`) | **GO, 6-seed, clean-seeded.** Its own finding scopes it honestly: *"local one-shot credit, NOT multi-layer/deep credit (confirmed-hard, not claimed)"* |
| **Two-compartment dAP** (`cp_v_apical`) | GO, 6-seed — but as a *read-out/completion* primitive on a hand-installed attractor |
| **Dendritic bistability** | GO for gap#5 completion; explicitly **NEGATIVE as a gap#4 credit booster** (*"when forced on, it HURTS"*) |

## 3. What is NEGATIVE / BOUNDARY / RETIRED — i.e. the actual deep-credit frontier

Every method that would constitute **deep multi-layer credit through the spiking substrate**:

| mechanism | verdict |
|---|---|
| BDSP / Burstprop → accuracy on-bridge | **NEGATIVE** (6-seed depth-2; 6-seed dense-redundant "decisive"; graded credit 0/6; at scale credit == lesion == wrong-sign, all below chance) |
| e-prop feedforward on-bridge | **NOT-GO** (4/5 blind seeds negative, GO gate falsified) |
| e-prop recurrent language | **REFUTED by its own must-run controls** (loses to a proper trigram) |
| Node perturbation | **RETIRED** (off-bridge GO, but REFUTED at 12 seeds vs Kolen-Pollack; on-bridge variance wall) |
| MDGL off-diagonal | **DECISIVE NEGATIVE** on spikes (adds magnitude, not sign-correct credit) |
| Feedback alignment / Kolen-Pollack | **family exhausted-negative**; KP lift was a **dev-seed artifact** (3 dev seeds strong, 6-blind kills it) |
| Urbanczik-Senn | never tested in the July arc; May arc terminated **VOID** |
| "Replay replaces BPTT" | **GO, 6-seed — but numpy RATE, not on spikes** |

## 4. The seed confound — what it actually invalidated (audit #3 CORRECTS audit #2)

Audit #2 concluded *"the headline gap#4 positive is 3-seed on the most-confounded unseeded runner and was never re-run."*
**Audit #3 checked the code and refuted that:** the D1/BDSP headline numbers (held-out **0.664**, microcircuit **0.964**)
come from **Stage-B, a pure-numpy reference net** seeded by `np.random.default_rng(seed)` — *no `SimulationBridge` in
that path*. Only three small **Stage-A** bridge smokes (burst detector / apical→P read / cancellation) were unseeded.
The genuinely-confounded deep-credit result (`_semantic_inheritance_onbridge_spiking_derisk.py` via
`_onbridge_eprop_port_derisk.py`) **was** fixed and re-run clean → the **NOT-GO** verdict above.

Audit #3 also found **two imprecisions in the seed doc itself**: (a) it says `cp.random.seed()` is "NEVER CALLED" — in
fact `_initialize_rng` *is* called and falls back to a **wall-clock** seed (same observable consequence, different
causal chain); (b) its "0.664/0.521 are in the marginal regime where unseeded heterogeneity can bite" exposure claim
does not bite, because those are numpy-path numbers.

**ACTION TAKEN THIS SESSION:** the three still-unseeded runners (`_gnw_d1_spiking_bdsp_derisk` ×3 sites,
`_d1_apical_soma_coupling_probe`, `_batched_onbridge_forward_derisk`) — flagged 2026-07-17, **still unfixed 3 days
later with zero commits touching them** — now set `cfg.seed`, **verified by a two-process hash** of
`cp_neuron_firing_thresholds` (`actual_seed_used`-only → hashes DIFFER; `cfg.seed` set → IDENTICAL).

## 5. Systemic defect classes this arc keeps producing (worth naming)

- **dev-seed selection** — named in the record as *"a recurring failure in this project, not a one-off"*: a "6-seed GO"
  that was 3 dev seeds; a KP lift that died on blind seeds; two independent node-perturbation seed artifacts.
- **metrics lifted from runs whose own verdict was NEGATIVE** — the K=8 0.877 "GO" was `SIGNAL=False` on every seed.
- **controls that exist in the code but were never invoked** — the frozen-hidden control that revealed the "deep credit
  GO" was ~80% a fixed random reservoir.
- **filenames that still assert retracted claims** (`...-3seed-GO.md` whose body opens "❌ REFUTED"; `...-net-learns.md`
  superseded as "subset noise"; a "3 arms crashed" title that was a monitor false-positive).

## 6. The honest next steps for gap#4 (what the record itself points to)

1. **The BTSP one-shot TASK — named 2026-07-18, NEVER RUN.** The BTSP finding's own "NEXT" is *"(b) a one-shot TASK
   (association/place-field) the substrate LEARNS via BTSP"*; item (c) was pursued, **(b) was not** (no such runner
   exists). This is what converts BTSP from *"a weight moves 8.4× more"* into *"the substrate LEARNS a behaviour"* —
   the actual gap#4 capability claim. Cheap: the on-bridge `enable_btsp` block is committed + CI-pinned and
   `_gap4_btsp_onbridge_behavioral_timescale_derisk.py` is a ready 6-seed template.
   **Required anti-cheats** (each earned by a burn in this arc): frozen-weight control (a fixed reservoir + trained
   read-out must NOT match it); wrong-sign/permuted-plateau must go BELOW chance; no-plateau moat → dw 0.000;
   `enable_btsp=False` byte-identical; `cfg.seed` set + two-process hash verified; 6 seeds with **blind seeds reported
   separately**.
2. **Wire BTSP into something the owner actually uses** — criterion (e) of the board's own CLOSED definition.
3. **M3 (calibrated end-to-end surrogate-BPTT) is NOT the cheap option** — the record calls it *"a genuine DEEP arc
   (substantial), NOT a one-step clean-up"*.

⚠️ **Stale next-action in the board:** `:1435-1441` still names the off-diagonal on-bridge test "(→ MDGL warranted)".
That was superseded by the 2026-07-16 MDGL DECISIVE NEGATIVE. **Do not run it.**

## Bottom line

**gap#4's capability — a substrate that learns deep representations by a biological local rule — is OPEN.** What is
real is *local, one-shot, plateau-gated credit on real spikes* (BTSP), and it is not wired into anything the owner
uses. The board should say so.
