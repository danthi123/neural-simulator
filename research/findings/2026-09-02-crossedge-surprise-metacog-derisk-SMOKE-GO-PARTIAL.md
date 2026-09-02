---
status: live
type: finding
lane: onebrain-integration-design
date: 2026-09-02
mechanism: crossedge-surprise-metacog-error-to-confidence
verdict: SMOKE-GO (2-seed numpy indicator) — PARTIAL pending the 6-seed cupy verify
---

# Cross-edge #2 — D2 surprise -> E1 metacog (ERROR -> CONFIDENCE): 2-seed numpy DE-RISK, SMOKE-GO, pending 6-seed cupy

**This is a DE-RISK SMOKE (2 seeds, numpy CPU), not a full GO and not a production wire-in.** It builds and
measures the rank-#2 cross-edge from
`research/findings/2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md` (the surprise->metacog
coupling that design confirms UN-BUILT): a genuine LEARNED spiking synapse from the D2 surprise pool onto the E1
metacog confidence read, so a VIOLATED prediction / high surprise LOWERS the confidence margin the substrate reads
off itself — not via a host if-statement. The 6-seed cupy verify (42/43/44/100/101/102) is QUEUED separately; this
finding is PARTIAL until that lands.

## What was un-built (verify-first)

The DESIGN doc states the surprise<->metacog coupling is UN-BUILT (its §"What already exists" — surprise's shipped
learned edges target `source_provenance` and an episodic-encode-decision gate, never metacog). A grep of
`research/findings/` for a surprise->metacog edge returns nothing. Confirmed absent before building.

## The mechanism (brain-based; emergent; ONE brain; NO sim/ edit)

- **ONE shared spiking bridge** holds D2 SURPRISE + E1 METACOG, built from the single-pool RECONCILED family
  (`_onebrain_twopool_merge_organread_verify._recon_descriptors`, filtered to {surprise, metacog} — reuse-by-import
  of the single source of truth the 2026-09-02 single-pool organ-read GO validated). This is the exact substrate
  the design says the edge should span (surprise + metacog co-resident on one `merge_organs` pool), not a bespoke
  pool. Two scoped, behavior-preserving adjustments let a LEARNED cross-edge enter a reconciled-frozen region:
  global `enable_hebbian_learning=False` (nothing drifts at build; my train window re-enables it) and metacog
  `freeze_regions=()` (the build's step-7 guard rejects any single-endpoint edge into a frozen region as an
  "unintended cross-synapse" — which a declared surprise->workspace edge is; the whitelist `apply_cross_edge_freeze`
  re-imposes the identical gain-0 freeze on metacog's internal edges instead). Conduction is untouched; only WHICH
  synapse may learn changes.
- **ONE plastic cross-edge** `surprise -> metacog.workspace member[1]` (the RUNNER-UP first-order assembly),
  declared as a framework `CrossEdge` row (init_weight W0=0.05, the SOLE plastic synapse via
  `apply_cross_edge_freeze()`'s whitelist inversion, `freeze_rest=True`). E_TO_E onto the competing assembly: when
  the prediction is violated the surprise pool EXCITES the runner-up, raising g_nmda(asm1) toward g_nmda(asm0) so
  the divisively-normalized winner-vs-runner-up margin d=(g0-g1)/(g0+g1+eps) — the metacog confidence — DROPS. This
  is the Yu & Dayan (2005) "unexpected uncertainty broadens the posterior / down-weights the current model" arm
  realized as a spike-driven synapse; the net functional read is a LOWER confidence margin, exactly the design's
  "error -> lower confidence -> the reply hedges more" claim.
- **The edge GROWS by the substrate's own rate-Hebbian rule** over training episodes that co-drive a CONTRADICT
  (mismatch) surprise trial (cue block c + a false assertion block c'!=c, which the FIXED surprise wiring turns into
  surprise firing specifically in block c') WITH a tonic teaching current into member[1], so member[1] reliably
  co-fires with the block-c' surprise slice and the Hebbian rule binds that slice's edges to member[1]. The
  teaching current is a DECLARED host-supervised co-drive (the same class every cross-edge in this codebase uses:
  R1/R4/surprise->provenance/surprise->encode-decision) — NOT self-organized (the host supplies the post-synaptic
  co-activation target); the WEIGHT itself is grown by the local rule from W0=0.05, not hand-set.

## The confidence read

The SIGNED divisive-normalized NMDA-conductance margin d=(g0-g1)/(g0+g1+eps) over the two workspace member
assemblies (higher = more confident in the correct answer asm0). This is the metacog organ's own `nmda_norm` read
(`metacog_production_organ.nmda_norm_margin`, Carandini & Heeger 2012 divisive normalization off
`cp_conductance_g_nmda`) with the SIGN kept rather than |.|: in this de-risk asm0 is the winner by construction (the
higher evidence drive), so the signed winner-dominance margin is the faithful "confidence in the correct answer",
monotone in member[1]'s activity. Both signed and |.| forms are reported; the gate keys on the signed drop.

## Results (2-seed numpy smoke + seed-7 calibration; artifact below)

<!--derived-->
The numbers below are a rounded human summary of the committed smoke artifact
(`_crossedge_surprise_metacog_numpy_smoke.json`, cited in Files); the seed-7 row is the calibration console output
(not persisted — calibration only). The raw per-seed values live in the artifact.

The de-risk harness is `onebrain_crossedge_gate.run_gate` + `CrossEdgeGateSpec`, reused UNMODIFIED (no bespoke
F-gate). Per-seed GO = emergence PASS (edge grew, no non-edge corruption) AND interaction PASS (confidence drops
intact, drop vanishes on lesion, attributable) AND byte-off PASS AND anti-cheat (other surprise blocks stay ~W0).

| seed | edge grown (W0=0.05) | conf low (CONFIRM) | conf high (CONTRADICT) | drop intact | drop lesion | % attributable | byte-off | GO |
|---|---|---|---|---|---|---|---|---|
| 7 (calib) | 0.678 | +0.1032 | -0.0088 | -0.1120 | — | — | — | — |
| 42 | 0.615 | +0.0961 | -0.0092 | -0.1053 | -0.0051 | 95.2% | PASS | True |
| 43 | 0.659 | +0.1078 | +0.0033 | -0.1045 | -0.0018 | 98.3% | PASS | True |

- **(a) EMERGENCE** — the surprise->metacog weight GROWS from W0=0.05 to 0.615/0.659 (>12x, well above the
  grow_factor*W0=0.25 bar) by the substrate's own rate-Hebbian rule; no non-edge synapse moved (`no_corruption`
  True, frozen-weight maxdrift < 1e-6).
- **(b) VARY / LESION** — a surprising (CONTRADICT) turn drops the metacog confidence margin by ~-0.105 relative to
  the low-surprise (CONFIRM) control (surprise raises the runner-up g_nmda from ~97 to ~127, flattening the
  winner-vs-runner-up balance to a near-tie / slight flip). FREEZING the edge (the gate's in-place lesion, zeroing
  exactly the declared synapses; plasticity is OFF at read, so the zero holds) collapses the drop to ~-0.005: the
  confidence STAYS HIGH on the surprising turn. The LEVER helper confirms the lesion actually moved the
  high-condition read on both seeds (not a vacuous no-op).
- **(c) ATTRIBUTABLE** — `attributable_to` assigns 95.2% / 98.3% of the confidence drop to the surprise-driven
  cross-edge synapse (the residual is the small control-present component).
- **(d) BYTE-OFF** — the pool built WITHOUT the cross-edge has base connectivity byte-identical (exact map compare,
  `verify_byte_off`) to the with-edge pool once the declared edge's own (pre,post) slots are excluded: integration
  added ONLY the edge, so with the flag off the metacog confidence reads exactly as today.
- **ANTI-CHEAT** — the other (never-mismatched) surprise blocks' edges into member[1] stay at W0=0.050 before and
  after training: the edge tracks THIS seed's randomly-assigned (`_assign_blocks`) surprise block, not every block.

## Honest scope / residuals

- **PARTIAL, not GO** — 2 numpy seeds is an INDICATOR; the canonical verdict needs the 6-seed cupy verify
  (42/43/44/100/101/102), QUEUED on `gpu_queue`. Do not read this as a 6-seed GO or generalize past 2 seeds.
- **Host-supervised co-drive, declared** — the edge grows under a tonic teaching current into member[1] (the
  post-synaptic target the host supplies during training), the same declared boundary every cross-edge here uses;
  this is NOT self-organized. What is emergent is the WEIGHT (grown by the local rule from near-zero), and the
  RUN-TIME drive is load-bearing + lesion-attributable, not a hand-set coupling.
- **Fixed evidence + 2AFC read, not a live chat turn** — the confidence read uses a fixed high-evidence 2AFC drive
  (asm0 the intended winner) on the metacog workspace, the organ's own `nmda_norm` operating point, not an arbitrary
  live-chat answer. Binding the drop onto a real turn's confidence read (and downstream, onto the honest hedge) is a
  separate, later, reviewed production rung — NOT claimed here. No production wiring, no default flip, no `sim/`
  edit.
- **Functional correlate, not phenomenal** — this measures + reports a confidence CORRELATE shifting under a
  prediction-error signal. It makes NO claim of subjective experience.

## Sources (external, verify-first)

- Yu & Dayan (2005), "Uncertainty, neuromodulation, and attention", *Neuron* 46(4):681-692 — unexpected uncertainty
  (ACh/NE) down-weights top-down prediction and broadens the posterior (the alternatives gain), the biological arm
  this edge realizes as a spike-driven synapse. Logged via `tools/record_external_search.sh`.
- Carandini & Heeger (2012), "Normalization as a canonical neural computation", *Nat Rev Neurosci* 13(1):51-62 —
  divisive normalization, the balance-of-evidence read the metacog organ's `nmda_norm` confidence already uses.

## Files

- Runner: `research/runners/_crossedge_surprise_metacog_derisk.py`
- Smoke artifact (2 seeds + provenance sidecar):
  `research/findings/raw/_crossedge_surprise_metacog_numpy_smoke.json`
- Harness reused UNMODIFIED: `research/runners/onebrain_crossedge_gate.py` (`run_gate` / `CrossEdgeGateSpec`).
- Substrate reused by import: `research/runners/_onebrain_twopool_merge_organread_verify.py` (`_recon_descriptors`),
  the single-pool reconciled family; `research/runners/onebrain_single_pool_production.py` (the flag it flips).
- 6-seed cupy verify: QUEUED on `gpu_queue` (its 6-seed artifact does not exist yet — the exact `--out` path is in
  the queued command; the CONTROLLER harvests it and updates this finding to a full GO / NO-GO when it lands).
