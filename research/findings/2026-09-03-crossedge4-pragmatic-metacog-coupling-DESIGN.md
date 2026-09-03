---
type: finding
status: design
claim_check: synthesis
date: 2026-09-03
mechanism: DESIGN — cross-edge #4 pragmatic↔metacog coupling; DIRECTION PINNED metacog(confidence)→pragmatic (confidence precision-weights the RSA divisive-normalization gain / illocutionary commitment), spiking learned CrossEdge workspace→item_fs, cheapest vary/lesion/shuffle+run_gate de-risk
lane: onebrain-integration-design
seeds: [42, 43, 44, 100, 101, 102]
---

# Cross-edge #4 — pragmatic ↔ metacog coupling: DIRECTION PINNED + mechanism spec + de-risk (DESIGN, nothing built)

**This is a DESIGN / SCOPING doc — no runner, no `sim/` edit, no GO.** It closes the one blocker the ranked
cross-edge design left open: cross-edge #4's DIRECTION was UNPINNED
(`2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md`, the rank-4 row: *"metacog→pragmatic is
the canonical direction; verify the functional story"*). It pins the direction with a biological + load-bearing
warrant, specifies the spiking coupling on the already-shipped pool-#2 substrate (reusing existing hooks, no new
seam), and scopes the cheapest vary/lesion de-risk with a pre-registered GO gate and the failure modes it must
survive. Each claim is a hypothesis until its 6-seed cupy `run_gate` GO. Functional read-outs only; no phenomenal
claim.

## The two organs, as they actually are in this codebase (verify-first)

- **E1 metacog** — `research/runners/metacog_production_organ.py`. A spiking balance-of-evidence CONFIDENCE
  monitor. It maps an answer's evidence scalar (the composer's mean role-decode `margin_snr`, host-derived) to a
  graded drive on a 2-assembly workspace WTA, settles it, and reads confidence as the divisive-normalized
  NMDA-conductance balance `|g_nmda(asm1)−g_nmda(asm0)| / (g_nmda(asm1)+g_nmda(asm0)+eps)` (`nmda_norm_margin`,
  Carandini & Heeger 2012). Low margin → an honest hedge is PREPENDED. Regions on the shared bridge: `workspace`
  (the two `ASSEMBLY_SIZE=80` decision assemblies, `metacog_idx()["member_dev"]`), `workspace_fs`, `meta_schema`.
  Default-ON; `BRAIN_METACOG=0` oracle; `BRAIN_METACOG_LESION=1` removes the evidence differential (margin→0).
- **D pragmatic** — `research/runners/pragmatic_production_organ.py`. A spiking scalar-implicature listener: for
  "some" it forms a graded RSA-L1 belief over {none, SBNA, all} ≈ [0, 0.73, 0.27] (the W4 depth-2 RSA posterior,
  Frank & Goodman 2012). The SBNA-vs-"all" split IS a graded discourse-commitment signal: `implicature_margin`
  (SBNA−all) and `residual_all_prob` ("all still X-possible" — the calibrated hedge the one-hot WTA destroys).
  Read off the shared bridge's `item` slice (`pragmatic_item_dev()`) via `_rsa_recursion`. Regions: `item`
  (`RSA_ITEM_SIZE*3=120` excitatory, the {none,SBNA,all} states) and **`item_fs`** (`RSA_FS_N=40`, the FS
  divisive-normalization pool). The pragmatic LESION is exactly the `item_fs→item` normalization weight
  `RSA_FS_EXC_W=22.0 → 0`: kill the normalization and the belief FLATTENS to [0,0.5,0.5], the implicature vanishes.
  Default-ON; `BRAIN_PRAGMATIC=0` oracle.

**They already co-reside byte-identically on ONE spiking bridge** — pool #2 `MergedSubstrate2(organs=("metacog",
"pragmatic"))` (`research/runners/onebrain_merge_production2.py`, default-ON per `BRAIN_ONEBRAIN_MERGE2`), and on
the 4-core single pool (`onebrain_single_pool_production`, `merge_organs([surprise, worldmodel, metacog,
pragmatic], wire=True)`). A cross-synapse between `workspace` and `item_fs` therefore needs NO new substrate — the
rank-4 "CHEAPEST, no new seam" note is confirmed in code.

## PINNED DIRECTION: metacog(confidence) → pragmatic — confidence precision-weights the pragmatic commitment

The coupling is **metacog CONFIDENCE gating the STRENGTH of the pragmatic commitment** (the illocutionary force of
the reading): when the brain is confident, it COMMITS to the enriched "some but not all" reading; when it is
unsure, the reading stays SOFTER — the residual "all is still possible" grows, the hedge widens. Direction and
sign are set by two independent warrants that agree.

### Biological warrant (the clearest, best-cited arm)

1. **Confidence is computed to gate communicative action.** Metacognitive confidence is a decision variable read
   out of rostral ACC / medial PFC (Fleming & Dolan 2012, *Phil Trans R Soc B* 367:1338; Bang & Fleming 2018,
   *PNAS* 115:6082 — mPFC encodes decision confidence). The *supra-personal* account (Shea, Boldt, Bang, Yeung,
   Heyes & Frith 2014, *Trends Cogn Sci* 18:186) argues confidence exists precisely to be SHARED and to gate
   communicative acts — confidence is an input to speaking, not a bystander of it.
2. **Speech-act felicity is confidence-gated.** The sincerity/preparatory condition of an assertion (Searle 1969,
   *Speech Acts*) and the Gricean Quality maxim (Grice 1975, *Logic and Conversation*: do not assert what you lack
   adequate evidence for) make the illocutionary force of a communicative act a function of the speaker's evidence
   / confidence. Low confidence → downgrade the force: hedge, keep alternatives open, or ask rather than assert.
   That is exactly "widen the residual 'all is still possible'."
3. **The substrate implements gating as precision-weighted normalization.** Confidence acts as PRECISION that
   weights the gain of the downstream inference (Feldman & Friston 2010, *Front Hum Neurosci* 4:215 — precision as
   the gain on prediction/likelihood). The pragmatic organ's own decisiveness knob is a divisive normalization
   (`item_fs`, Carandini & Heeger 2012, *Nat Rev Neurosci* 13:51). So "confidence gates the pragmatic commitment"
   maps cleanly onto "the metacog confidence signal sets the GAIN of the pragmatic FS normalization pool" — high
   precision sharpens the posterior (commit), low precision leaves it near the flatter prior (hedge). The
   neuromodulatory form of the same idea is LC-NE / ACh adaptive gain and unexpected-uncertainty broadcast
   (Aston-Jones & Cohen 2005, *Annu Rev Neurosci* 28:403; Yu & Dayan 2005, *Neuron* 46:681).

### Load-bearing conversational warrant (why this direction, not merely a plausible one)

- The project ALREADY ships a HOST version of this exact direction: the **#94 confidence→forthcomingness**
  reach-cap (`webapp/confidence_forthcoming_chat.py`; `2026-09-01-confidence-forthcomingness-ltm-elaboration-
  load-bearing-GO.md`) — confidence gates HOW MUCH the brain volunteers, default-ON and user-visible. That is a
  metacog(confidence)→discourse-behaviour coupling computed by a Python formula. Converting a host coupling into a
  genuine spiking cross-synapse is precisely the brain-based-only mandate ("a prediction error computed by a
  Python formula … is a shortcut"). The host coupling being real, load-bearing, and shipped is direct evidence the
  DIRECTION is right; cross-edge #4 makes it neural on the pragmatic organ's own graded surface.
- It satisfies "faculties must DRIVE not observe": VARY the metacog evidence (high vs low) and the pragmatic
  reading changes (commit vs hedge); LESION the edge and the change vanishes (the reading reverts to the fixed
  default, no longer tracking confidence). Both are measurable on the pragmatic organ's OWN `implicature_margin` /
  `residual_all_prob`, i.e. in the surfaced `pragmatic_notice` text.

### Why not the reverse (pragmatic → metacog), and why not "bidirectional" for the FIRST build

The reverse story — "a direct question (an information-request speech act) raises the metacog monitoring bar" — is
a task-demand / cognitive-control account (criterion shifts with stakes; ACC need-for-control). It is weaker here
for two concrete reasons: (a) the *scalar-implicature* pragmatic organ has no natural spiking EMITTER of a
"monitoring-demand" signal — it represents a belief over {none,SBNA,all}, not an illocutionary act type, so there
is no clean source population to wire; (b) the biological literature on confidence→communication is far thicker
than on discourse-act→metacognitive-gain. **Pin metacog→pragmatic; DOCUMENT the reverse as a deferred second arm**
(it would first need a discourse-act-type detector emitting a "question/answerability-demand" rate — a separate
organ rung, not a cross-synapse on the current pair). Declaring it bidirectional now would be an un-grounded
over-claim; the honesty boundary holds.

## Coupling mechanism on the spiking substrate (reused hooks; no invented parts)

**Edge:** metacog `workspace` DECISION assembly → pragmatic `item_fs` (the RSA FS normalization pool), a LEARNED
excitatory cross-synapse. This is the canonical feedforward-inhibition / gain-control motif: an excitatory
projection onto an FS interneuron pool that then normalizes the target excitatory pool. Precision-weighting is the
function; the FS pool is the gain knob the pragmatic organ ALREADY uses (its lesion is `RSA_FS_EXC_W→0`).

**Source = the confidence-graded rate, not a host scalar.** The metacog decision assembly (the one driven by
`base+sig(evidence)`) fires at a rate that is MONOTONIC in evidence/confidence — high evidence → the winner fires
harder and the loser is more suppressed (that IS the growing NMDA margin the organ reads). So the source
population's spike rate is a genuine on-substrate confidence signal; nothing is host-computed downstream of the
already-declared evidence-encoding boundary. Select it with `source_idx_fn` returning
`metacog_idx()["member_dev"][k_decision]` (a sub-slice of `workspace`, exactly as R4's `author` is a sub-slice of
`self_schema`, `onebrain_xedge_selfschema_production.py`).

**Sign / operating point (precision-weighting):** confident (high source rate) → EXTRA excitatory drive to
`item_fs` → stronger divisive normalization → SHARPER implicature (higher `implicature_margin`, lower
`residual_all_prob` = committed). Uncertain (low source rate) → little extra FS drive → the belief stays at the
softer baseline (higher residual = hedged). The observable coupling is the DIFFERENCE (confident − uncertain), and
it must VANISH under lesion. This keeps the edge EXCITATORY and Hebbian-growable (matching every shipped edge). An
alternative operating point — a dedicated UNCERTAINTY population that fires MORE when confidence is low and
SUPPRESSES `item_fs` (uncertain→flatten→hedge as the primary swing) — is biologically also defensible (NE
unexpected-uncertainty) but needs a new source population or a 2-synapse relay; it is a documented residual, not
the first build.

**Declarative wiring (reuse `onebrain_merge_framework.CrossEdge` + `merge_organs`):**

```python
# de-risk: build the pool with the ONE declared edge, then run the generic gate.
from research.runners.onebrain_merge_framework import CrossEdge, merge_organs
from research.runners._onebrain_twopool_merge_organread_verify import _recon_descriptors

K_DECISION = 1  # the workspace assembly the read drives with base+sig(evidence): the 'confident' pole

def _decision_members_of(bridge):
    # workspace member indices of the decision assembly (its rate is monotone in confidence) — a sub-slice of
    # "workspace"; resolved off the pool's metacog_idx()["member_dev"][K_DECISION] (dict k -> abs neuron indices)
    ...

ce = CrossEdge(
    key="metacog_confidence__to__pragmatic_norm",
    source_key="metacog",   source_region="workspace",   # documentation label; real endpoint via source_idx_fn
    target_key="pragmatic", target_region="item_fs",      # the RSA divisive-normalization FS pool
    init_weight=0.05,        # near-zero — MUST grow (grow_factor*W0 = 0.25 bar), never pre-wired
    plastic=True, learn_rule="rate_hebbian", freeze_rest=True,
    source_idx_fn=_decision_members_of,                   # -> metacog_idx()["member_dev"][K_DECISION]
    # target_idx_fn=None -> region_manager.indices("item_fs")
)
pool = merge_organs(_recon_descriptors(), seed, wire=True, cross_edges=[ce])
```

**The load-bearing READ must drive the source DURING the pragmatic settle (co-temporal), inside one isolation
guard.** The edge only transmits when `workspace` is actively driven during the RSA recursion — so `read_fn(pool,
condition)` HOLDS the metacog decision assembly at `base+sig(evidence[condition])` throughout `_rsa_recursion` on
`item`, then reads the belief. Because this is a prime-then-read spanning NMDA-recurrent dynamics, it MUST run
inside `pool.sequence_isolation()` (NOT the per-call `read_isolation()`): the wider `_SEQ_EXTRA_STATE` set
(NMDA-recurrent rise/recurrent conductances + `cp_synapse_pulse_timers/_progress`) plus the 4 per-neuron reset
arrays must be restored, or the read leaks state across conditions and the lesion baseline inflates above zero
(this is the exact bug that flipped cross-edge #2 and curiosity→d6-WM; see Failure modes). Conditions:
`("uncertain", "confident")` with `control="uncertain"` (evidence 0.0) and `expected={"confident": {"sign": +1,
"floor": 0.008}}` reading `implicature_margin` (or `−residual_all_prob`).

**Flags (mirror the shipped edge conventions, default-OFF until an owner-gated flip):**
`BRAIN_ONEBRAIN_XEDGE_METACOG_PRAGMATIC` (build the frozen pre-grown edge into the live pool #2 read),
`BRAIN_ONEBRAIN_XEDGE_METACOG_PRAGMATIC_LESION=1` (zero the edge — the load-bearing lesion control),
`..._DECLARATIVE` (optional: build via the declarative `CrossEdge`/`merge_organs` path vs a bespoke pool),
exactly the `onebrain_xedge_selfschema_production.py` pattern. Production wire-in (later, owner-gated) makes it a
FROZEN pre-grown edge on the live `MergedSubstrate2` / single pool, so a live turn's confidence read drives a
co-temporal pragmatic read; the surfaced effect is the hedge-strength of `pragmatic_notice`.

## Cheapest de-risk + GO gate (a vary/lesion "drive not observe" test)

**Runner (proposed):** `research/runners/_crossedge_metacog_pragmatic_derisk.py` — additive, no `sim/` edit,
CPU numpy for the smoke, cupy 6-seed for the verdict. It composes the SHIPPED
`onebrain_crossedge_gate.run_gate(pool, CrossEdgeGateSpec(...))` (generic emergence + interaction + byte-off) with
the vary/lesion/shuffle structure of `_curiosity_metacog_lowconfidence_coupling_derisk.py` (the closest existing
template — it already couples metacog low-confidence to another organ with monotonic/lesion/shuffle bars). This is
the CHEAPEST edge: both organs co-reside on the default-ON pool #2, only ONE `CrossEdge` row + a `train_fn` +
`read_fn` are new; everything else is reused by import.

**Pre-registered GO bar (all must hold, 6/6 cupy seeds 42/43/44/100/101/102 — a 2-seed numpy run is a SMOKE
INDICATOR labelled PARTIAL, never generalized):**

- **G1 EMERGENCE.** The edge GREW from `W0=0.05` above `grow_factor*W0 = 0.25` by the substrate's own rate-Hebbian
  rule over the confident-drive episodes; `no_corruption` True with frozen-weight maxdrift `< 1e-6` over every
  non-edge synapse (`verify_emergence`).
- **G2 INTERACTION (the crux).** With the edge intact, the CONFIDENT condition's `implicature_margin` exceeds the
  UNCERTAIN control by `> INTACT_FLOOR = 0.008` in the predicted (+) direction. LESION the edge (zero its
  synapses) → the confident−uncertain gap collapses to `< lesion_ratio(0.34) * |intact gap|`, i.e.
  `frac_attributable → ~1.0` (`verify_interaction` / `tools.lab.attributable_to`).
- **G3 CROSSES A NEW TURN CLASS.** At the confident condition the pragmatic reading is measurably MORE committed
  (residual drops / margin rises) than the pragmatic organ produces with NO metacog drive — the edge exercises a
  coupling today's wiring never connects (the G2 of the curiosity template, adapted).
- **G4 SHUFFLE / NO-CONFOUND.** Driving the source at a MATCHED total rate but with the confidence LABEL scrambled
  (or permuting evidence↔condition) collapses the systematic shift below the floor — the effect tracks the
  confidence CODE, not raw drive magnitude. Plus the anti-cheat that never-driven metacog blocks' edges stay
  `~W0` (`< 5*W0`), so the edge tracks only the decision assembly.
- **G5 BYTE-OFF.** A pool built WITHOUT `cross_edges` has base connectivity byte-identical to the with-edge pool
  once the declared edge's (pre,post) slots are excluded (`verify_byte_off`) — integration added ONLY the edge.
- **G6 PRAGMATIC-UNCHANGED-OFF.** With the edge lesioned/absent, the pragmatic organ's own belief is byte-identical
  to its standalone read (the coupling is purely additive; `BRAIN_PRAGMATIC=0` still the oracle).

**Verdict plumbing:** pass the seed OUTCOME (`n_go==len(seeds)`) ONLY to `Vd.decide(go=...)`, NEVER as a
`Vd.require(...)` precondition (that latent-harmless mistake collapses a genuine 3/6 to UNDEFINED instead of an
honest NO-GO). Preconditions are the validity checks (lesion-removes-bias, byte-identical-off) only.

## Failure modes this de-risk MUST survive (from the banked cross-edge record)

1. **The read-isolation bug class (the dominant, verdict-flipping failure).** A shared-pool read that does not
   restore ALL mutated state leaks hard-gate + homeostatic state across conditions, inflating the lesion baseline
   above zero and flipping GOs to NO-GOs (curiosity→d6-WM "GO 6/6" corrected to NO-GO 3/6;
   `2026-09-02-onebrain-crossedge-curiosity-to-d6wm-read-isolation-fix-corrects-GO-to-NOGO-3-6.md`). The leak works
   WITH the effect on some seeds and AGAINST it on others — not a one-way bias. **Because this read is a
   prime-then-read over NMDA-recurrent dynamics, use `pool.sequence_isolation()`** (the 4 per-neuron reset arrays
   `cp_refractory_timers`/`cp_prev_firing_states`/`cp_neuron_activity_ema`/`cp_neuron_firing_thresholds` AND the
   `_SEQ_EXTRA_STATE` NMDA/pulse-timer set), NOT the per-call `read_isolation()` — the 4-array C2 fix alone was
   insufficient for a load-then-read edge (`2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`,
   `2026-09-02-c2-metacog-read-isolation-fix-GO.md`). Add a `--selftest` asserting bitwise-identical repeat-reads
   AND verified to FAIL on a deliberately un-isolated read (fails-in-failing-direction); small 2–3-call probes MISS
   this — the residue only compounds over the full sweep.
2. **Metacog is a fragile near-floor endpoint — and here it is the SOURCE.** Cross-edge #2 (surprise→metacog) went
   NO-GO purely because the metacog read's incomplete reset left the discriminating quantity (`delta_lesion`
   ~0.005–0.05) at the SAME scale as the leak; after the full reset, `delta_lesion` went exactly 0.0 and
   `frac_attributable` 1.0 (`2026-09-02-c2-metacog-read-isolation-fix-GO.md`). As the SOURCE, the metacog decision
   rate feeding the edge sits at that same fragile operating point, so the DRIVE is seed-variable unless the source
   hold carries the full sequence isolation. Average over `READ_REPS` jittered holds (as the organ's own
   `nmda_norm_margin` does) to de-noise the drive.
3. **Do NOT port a sibling edge's learning rule without checking its structural precondition.** The attempt to fix
   #2 by porting C1's prediction-error-GATED three-factor update REGRESSED 6/6→3/6, because C1's gate self-limits
   via a CLOSED feedback loop that C2's fixed surprise circuit does not have
   (`2026-09-02-c2-metacog-error-gated-port-second-negative.md`). Cross-edge #4's plain `rate_hebbian` growth is
   correct for a monotone confidence→gain map; only add a third-factor gate if a measured selectivity failure
   demands it, and only after checking the precondition holds.
4. **`plastic=False` is NOT enforced at runtime.** A `RegionPathway(plastic=False)` still drifts toward
   `hebbian_max_weight` when `enable_hebbian_learning` is globally True, unless held at a NAMED gain-0
   `plasticity_gate` (`2026-09-02-read-isolation-audit-29runner-followup-map-plus-a-bigger-adjacent-bug.md`). The
   edge's `freeze_rest=True` (via `apply_cross_edge_freeze`) already does this for the OTHER edges; the production
   wire-in must FREEZE the grown edge (a gain-0 gate), exactly as `onebrain_xedge_selfschema_production` freezes R4.
5. **Numpy is a smoke only.** numpy and cupy have diverged on these exact edges; the verdict is cupy 6/6
   (`2026-09-02-integration-program-6seed-harvest-singlepool-GO-C1-GO-C2-NOGO.md`).
6. **Live-turn caveat.** The de-risk drives the metacog workspace at the organ's `nmda_norm` operating point on a
   fixed scalar-implicature item, NOT an arbitrary live chat turn. Binding the edge onto a real turn's confidence
   AND onto the surfaced hedge text is an un-validated LATER rung (the same residual R4 and #2 both declare) — the
   de-risk GO is "the confidence→commitment drive is real, spiking, and lesion-attributable", not "it changes live
   production text".

## Honest scope

Nothing is built. The direction is pinned with a biological + load-bearing warrant; the mechanism reuses the
shipped pool-#2 substrate, the `CrossEdge`/`merge_organs`/`run_gate` declarative framework, the metacog decision
assembly, and the pragmatic FS-normalization knob — no invented parts. The de-risk is one `CrossEdge` row + a
`train_fn`/`read_fn` over the generic gate, gated 6/6 cupy. "Coupling" means a genuine LEARNED spiking
cross-synapse that carries confidence into the pragmatic commitment — not a host formula relabeled. Functional
read-outs only; no phenomenal-experience claim. The reverse arm (pragmatic→metacog) and the
uncertainty-suppresses-normalization operating point are documented residuals, not this build.

## Files / citations

- **Builds on:** `2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md` (the rank-4 row this
  pins); `2026-09-02-c2-metacog-read-isolation-fix-GO.md`,
  `2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`,
  `2026-09-02-onebrain-crossedge-curiosity-to-d6wm-read-isolation-fix-corrects-GO-to-NOGO-3-6.md` (the failure
  modes); `2026-09-01-confidence-forthcomingness-ltm-elaboration-load-bearing-GO.md` (the #94 host coupling this
  neuralizes).
- **Raw artifacts grounding the failure modes + the co-residence substrate this design rides on** (this design
  states no NEW measurement — every number is a code constant or a proposed threshold, marked `claim_check:
  synthesis`): `research/findings/raw/_onebrain_twopool_merge_organread_6seed.json` (metacog + pragmatic co-reside
  byte-identically on ONE bridge — the substrate cross-edge #4 spans);
  `research/findings/raw/_crossedge_surprise_metacog_readfix_numpy6seed.json` (the metacog read-isolation fix / the
  near-floor fragility of failure mode #2); `research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_readfix_6seed.json`
  (the curiosity→d6-WM read-isolation correction of failure mode #1).
- **Code reused (no edits):** `research/runners/metacog_production_organ.py`,
  `research/runners/pragmatic_production_organ.py`, `research/runners/onebrain_merge_production2.py`
  (`MergedSubstrate2`, `metacog_idx()`, `pragmatic_item_dev()`), `research/runners/onebrain_merge_framework.py`
  (`CrossEdge`, `merge_organs`, `read_isolation`/`sequence_isolation`, `apply_cross_edge_freeze`),
  `research/runners/onebrain_crossedge_gate.py` (`CrossEdgeGateSpec`, `run_gate`, `verify_emergence` /
  `verify_interaction` / `verify_byte_off`), `research/runners/_curiosity_metacog_lowconfidence_coupling_derisk.py`
  (the vary/lesion/shuffle template), `research/runners/onebrain_xedge_selfschema_production.py` (the flag +
  freeze + live-consumer wire-in template), `research/runners/_recursive_tom_rsa_derisk.py` (`RSA_FS_EXC_W`,
  `_rsa_recursion`). Proposed new runner (NOT built): `research/runners/_crossedge_metacog_pragmatic_derisk.py`.
- **External sources:** Fleming & Dolan 2012 *Phil Trans R Soc B* 367:1338; Bang & Fleming 2018 *PNAS* 115:6082;
  Shea, Boldt, Bang, Yeung, Heyes & Frith 2014 *Trends Cogn Sci* 18:186; Searle 1969 *Speech Acts*; Grice 1975
  *Logic and Conversation*; Feldman & Friston 2010 *Front Hum Neurosci* 4:215; Carandini & Heeger 2012 *Nat Rev
  Neurosci* 13:51; Frank & Goodman 2012 *Science* 336:998; Aston-Jones & Cohen 2005 *Annu Rev Neurosci* 28:403; Yu
  & Dayan 2005 *Neuron* 46:681; Pouget, Drugowitsch & Kepecs 2016 *Nat Neurosci* 19:366 (confidence as a
  computable variable used by downstream systems).
