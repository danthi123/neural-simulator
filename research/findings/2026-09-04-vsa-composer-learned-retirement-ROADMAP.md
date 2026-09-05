---
type: finding
status: design
claim_check: synthesis
date: 2026-09-04
mechanism: ROADMAP — retire the VSA composer's host-computed exact-inverse binding algebra (the "composer-as-idealization" residual) toward a learned/emergent on-substrate one-brain composer; staged from the already-built SlotBinderComposer + the July RUNG arc + the already-built (default-off / onebrain-default-on) on-substrate composer mechanisms
lane: scaffold-retirement (owner 2026-09-04 stage 3: "VSA composer -> learned") + one-brain (substrate consolidation) + E-language
seeds: [42, 43, 44, 100, 101, 102]
verdict: >
  ROADMAP / DESIGN NOTE (no new measurement). Maps EXACTLY which composer operations are host-exact-algebra vs
  already-spiking/already-learned; reconciles the June learned-binder scoping with the July+August arc that already
  RAN it; and stages a cheap-first, gated de-risk ladder toward a learned/emergent one-brain composer. The
  load-bearing findings that reframe the task: (1) a learned-FROM-SCRATCH bind MEMORIZES multi-attribute bundling
  but does NOT generalize — even the true-gradient ORACLE fails — so the multiplicative COINCIDENCE bind is a
  biology-grounded STRUCTURAL primitive, not a shortcut to be learned away; (2) a mature LEARNED replacement for
  the WHOLE FHRR surface already exists and is 6-seed GO with an on-bridge Hebbian WRITE — SlotBinderComposer —
  declared "FHRR exact-inverse algebra FULLY RETIRABLE", but its production wire-in was scoped and never executed,
  blocked on an unmeasured dense-pathway scale question — ⚠️ NOW MEASURED 2026-09-04c: NO-GO at live scale on the
  dense wiring (968M synapses / ~463GB by the consumer-hardware gate, does not fit the 3090; incumbent FHRR is
  ~1000× lighter + correct), so L2 sparsification is the real next rung, NOT the direct wholesale wire-in; (3) the LIVE deployed brain bundles still run
  composer_kind=rf (pure host exact-inverse algebra), so the shortcut is fully in production today. The retirement
  therefore splits into two PATHS (A: retire the host COMPUTATION around a kept structural FHRR bind; B: replace
  FHRR wholesale with the learned SlotBinder) and the first tractable de-risk is the SlotBinder SCALE measurement,
  not another feasibility de-risk. No sim/ file and no runner is edited by this doc.
artifacts:
  - research/findings/raw/_onsubstrate_coincidence_systematicity.json
  - research/findings/raw/_onsubstrate_systematicity_scale_hardening.json
  - research/findings/raw/_learned_bilinear_binder_systematicity.json
  - research/findings/raw/_fixedbind_systematicity.json
  - research/findings/raw/cortex_learned_binder_systematicity_multiseed.json
  - research/findings/raw/_deep_eprop_binder_bundling.json
  - research/findings/raw/_gap2_deltarule_binder_production_scale_recheck.json
  - research/findings/2026-06-06-composer-vsa-idealization-known-limitation.md
  - research/findings/2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md
  - research/findings/2026-06-20-binding-structure-self-organization-scoping.md
  - research/findings/2026-07-14-deep-eprop-binder-multiattribute-CONFIRMED-BOUNDARY.md
  - research/findings/2026-07-15-onsubstrate-bind-onbridge-bdsp-readout-RUNG3-BOUNDARY.md
  - research/findings/2026-07-20-composer-factstore-host-persistence-is-the-VSA-idealization-scoping.md
  - research/findings/2026-07-21-gap2-spiking-learned-binder-6seed-GO-emergence-bar-close.md
  - research/findings/2026-07-22-gap2-attribute-slot-GO-FHRR-retirement-step1.md
  - research/findings/2026-07-22-gap2-pointer-clause-GO-FHRR-fully-retirable.md
  - research/findings/2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED.md
  - research/runners/rf_phasor_composer.py
  - research/runners/one_brain_composer.py
  - research/runners/slotbinder_composer.py
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
---

# ROADMAP — retire the VSA composer's host exact-inverse binding algebra toward a learned/emergent one-brain composer

**This is a ROADMAP / DESIGN NOTE, not a measured result.** It takes the owner's 2026-09-04 priority — the ordered
scaffold-retirement arc, whose **stage 3** names this task verbatim: *"per-faculty host-shortcut retirement
[affect-coupling->neural, num/den->spiking, **VSA composer->learned**, other organ internals]"* (commit `70166962c`;
`GAP_CLOSURE_MISSION.md` L51-52; `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §8) — and turns it into a
staged, dependency-ordered de-risk ladder with a GO gate + honest residual per rung. It **builds ON** the prior
scoping (2026-06-06 idealization note; 2026-06-11/06-16 learned-binder research; 2026-06-20 self-organization
scoping) and, critically, on the **July+August arc that already ran the de-risks and already built a mature
replacement** — it does not re-derive them. It edits no `sim/` file and no runner.

**Owner nuance that governs the forks (verbatim, commit `70166962c`):** *"'emergent'/self-organized can require a
thin slice of DEVELOPMENTAL learning-mechanism as a prerequisite — allowed where a retirement genuinely needs it;
continuous learning-OVER-TIME stays deferred."* A learned/offline-trained binder or codes are therefore in scope as
a developmental prerequisite; continuous learning-through-use is stage 5, not this arc.

---

## 0. TL;DR — the honest state (it is further along, and more forked, than "replace the algebra")

The composer binds conversational role-filler pairs by a **Fourier Holographic Reduced Representation (FHRR)**
vector-symbolic algebra: bind = a per-component complex product (`z_bound[k] = z_role[k]*z_filler[k]`), unbind = the
exact conjugate (`z_role^-1 = conj(z_role)`), cleanup = a matched-filter argmax over the codebook, abstention = a
host loop returning `None` when no stored fact matches. CLAUDE.md names this the *"VSA composer's clean exact-inverse
algebra ... a host shortcut for what a learned cortex would do."*

**Four facts, established by direct code-read + the July/August arc, set the actual state:**

1. **The shortcut is fully in PRODUCTION today.** The live deployed developed-brain bundles
   (`bridges/developed/scale787/day_{9,33}`) are saved `composer_kind: "rf"` → `RFPhasorComposer` with a FIXED,
   host-designed diagonal complex bind (`self.roles[role]`, never learned) and every on-substrate flag OFF (host
   `np.conj` unbind, host `argmax` cleanup, host `list` store). The webapp *code* default is `onebrain`
   (`webapp/server.py:3695` `_COMPOSER_KIND_DEFAULT="onebrain"`), and `OneBrainComposer` defaults
   `enable_spiking_cleanup=True` + `local_reciprocal_unbind=True` — but a persisted bundle keeps the
   `composer_kind` it was built under, so the user talks to the pure host-algebra `rf` composer
   (`2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED.md` L77-83).

2. **A learned-from-scratch BIND is a CHARACTERIZED BOUNDARY, not a TODO.** When the arc actually *trained* a bind
   (deep e-prop, BPTT, the true-gradient ORACLE, and a dendritic two-compartment version), every learned bind
   MEMORIZED multi-attribute bundling but did NOT generalize (held-out 0.002-0.007), while the fixed ±1/FHRR
   coincidence primitive generalizes to novel role-filler combos for free (0.993). ⇒ the multiplicative coincidence
   bind is a **biology-grounded STRUCTURAL primitive** (dendritic/σ-π conjunction), *not* a host shortcut to be
   learned away (`2026-07-14-deep-eprop-binder-multiattribute-CONFIRMED-BOUNDARY.md`;
   `2026-06-16-...single-attr-GO-bundling-NEGATIVE.md`).

3. **A mature LEARNED replacement for the WHOLE FHRR surface already exists, 6-seed GO, and is unwired.**
   `SlotBinderComposer` (`research/runners/slotbinder_composer.py`, `composer_kind="slotbinder"`) sidesteps FHRR
   superposition entirely: it allocates SEPARATE spiking WTA slots per (fact, role), writes each slot→filler by a
   REAL on-bridge spike-driven Hebbian potentiation (a per-slot plasticity gate + `_run_one_simulation_step()`,
   NOT a host formula), recalls by a neural content-addressable scan, and abstains intrinsically (the moat is the
   scan, not a VSA-cleanup shortcut). It covers the COMPLETE deployed FHRR capability set — flat SVO + polarity +
   multi-hop + single-attribute + depth-1 embedded clause (by pointer indirection) — 6-seed GO with anti-cheats
   (permuted-pointer, lesion-second-hop, wrong-clause distractor, moat). Its own finding declares *"the FHRR
   exact-inverse algebra is now FULLY RETIRABLE"* and names the next step (make it the production default) — which
   was **never executed**, blocked on an unmeasured dense-pathway scale question
   (`2026-07-22-gap2-pointer-clause-GO-FHRR-fully-retirable.md`; `2026-08-25-...NOT-WIRED.md` L89-107).

4. **The genuinely host-computed residual is a SET of small pieces, most already built.** The host `np.conj`
   unbind, the host `argmax` cleanup, the host `if`-loop abstention, the host `list` store, and the host-random
   codes each have an on-substrate/learned replacement that is built-but-default-off (in `rf`) or default-on (in
   `onebrain`) or, for the store, blocked on a `sim/` single-array constraint.

**⇒ "retire the exact-inverse binding algebra" is two PATHS + one boundary:**
- **Path A — retire the host COMPUTATION around a KEPT structural FHRR bind.** Flip the on-substrate host-op
  replacements (spiking cleanup, reciprocal-conjugate unbind, substrate store) + learned codes; keep the
  multiplicative coincidence bind as a fixed structural primitive. `OneBrainComposer` is already partway.
- **Path B — REPLACE FHRR wholesale with the learned SlotBinder.** The bind, write, recall and moat are all
  learned/on-substrate; superposition (and its multi-attribute boundary) is sidestepped by separate slots. Most
  mature at SMALL scale. **⚠️ NO LONGER the recommended path AS-IS (UPDATED 2026-09-04c):** the named blocker (the
  scale de-risk) came back **NO-GO** (`2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md`) — the
  dense all-to-all slot→filler wiring is 968M synapses at 404-fact scale (~463GB by the consumer-hardware gate),
  does NOT fit the 3090, while incumbent FHRR measured 334MB / 0.9s / 3-of-3-correct at the same scale (~1000×
  lighter). Banked = the DENSE-WIRING method (NOT the capability; the small-scale 6-seed GO stands). Real next rung
  = **L2: sparsify the slot→filler pathway** (~40× fewer synapses, a fixed small fan-out per slot) then re-de-risk;
  FHRR stays the efficient+correct incumbent meanwhile (its exact-inverse algebra = the host-idealization
  honest-negative).
- **The boundary (both paths inherit or sidestep it):** a *learned* multiplicative bind does not systematically
  generalize (even the oracle fails). Path A keeps the fixed structural bind (accepts the primitive); Path B
  sidesteps superposition with separate slots. Neither requires "learn the bind operator from scratch" — the record
  says that is a characterized boundary, so the deliverable there is the boundary, not a GO.

**Nothing here is "closed"** (TERMS.md: integrated = wired + on-by-default + scaffold-retired). The three relevant
ledger rows — `semantic-recall`, `moat-verify`, `open-ended-generation` — are all `retire_status: BLOCKED` today.
This roadmap is the plan to move them BLOCKED → RETIRED.

---

## 1. THE SHORTCUT MAP — which composer operations are host-exact-algebra, precisely (code-read)

Line numbers are `research/runners/rf_phasor_composer.py` unless noted (the production shard engine; `OneBrainComposer`
in `research/runners/one_brain_composer.py` shares the op set, `_compose_phases` ≈ L253-273). "rf default" =
`RFPhasorComposer.__init__` (L62) with no opt-in flag; "onebrain default" = `OneBrainComposer.__init__` (L107-123).

| # | composer sub-part | code (default path) | host-exact-algebra? | on-substrate / learned replacement | status |
|---|---|---|---|---|---|
| S1 | **concept + role CODES** | `rng.uniform(0,1,D)` per word/role (L178,L197); runtime-grown via `_growth_rng` (L183,L413) | **partly** — host-random draw (a genome-style developmental wiring rule, accepted per `sim/dendritic_neuron.py:25` / catalog F.12/D.18); NOT grounded/learned by default | learned rate-Hebbian **stream-cortex codes** (`grounded_codes=` kwarg, L189-192); `2026-06-15-on-bridge-hebbian-co-occurrence-...GO` | de-risked; interface wired; **DEFAULT-OFF** |
| S2 | **bind OPERATION** | `_bind` diagonal complex synapse, weight = role phasor; RF resonate + complex matvec on the bridge (L384-393) | **NO** — genuinely spiking; the weight VALUES are host-injected once via `rf_set_complex_weights`, but the OP is on-substrate | (needs no learned replacement — a learned bind is a boundary, §3); the STRUCTURE is a fixed developmental wiring rule | on-substrate already |
| S3 | **unbind = conj(role)** | `_unbind_phases`: rf default `np.conj(...)` host-computed + injected (L678) | **YES (the exact-inverse residual)** | `local_reciprocal_unbind=True` (L663-682): a LOCAL per-component quadrature-flip wiring rule (`_reciprocal_conjugate`, L372-382); byte-identical, host-free at runtime | built; **onebrain default-ON, rf default-OFF** |
| S4 | **cleanup (nearest concept)** | `_cleanup`: rf default host `np.cos(...).mean` matched-filter + `argmax` over the codebook (L844-856) | **YES** (CLAUDE.md's "argmax over spike counts" class) | `enable_spiking_cleanup=True`: on-bridge NEF matched-filter + spiking Izhikevich WTA argmax-over-firing (`_spiking_cleanup`, L712-760); == numpy at 320 | de-risked; **onebrain default-ON, rf default-OFF** |
| S5 | **no-confab / abstention moat** | `query_agent`/`query_patient`: host `for`-loop over `self.kb`, `==`, `return None` (L1311-1314) | **YES** — abstention is host bookkeeping | learned Bogacz-Brown **familiarity gate** (novelty energy), de-risked +0.982, lesionable (`2026-06-10-cortex-learned-cleanup-derisk-PARTIAL` TEST 3); OR SlotBinder's intrinsic content-addressable scan | de-risked; **NOT wired to default** |
| S6 | **fact STORE (data persistence)** | `self.kb`=host list (L294); OneBrain `store_conns`=host list (one_brain_composer.py:345) | **YES** — fact data lives host-side; the bridge synapses are installed transiently per read | needs a `sim/` PERSISTENT store-tensor DISJOINT from the per-op `cp_rf_w_*` (the single-array constraint: `rf_set_complex_weights` REPLACES, not appends). `enable_substrate_store` builds per-fact bridges from host data (not truly-in-synapses) | **BLOCKED on a scoped `sim/` change** (`2026-07-20-...factstore-host-persistence`) |
| S7 | **the WRITE (fact → store)** | host `_bundle`/`np.outer` composite written into the store | **YES (open for FHRR)** | gap#2 local Hebbian outer-product (READ 6-seed GO, WRITE still host-numpy); **OR SlotBinder's real on-bridge Hebbian slot-write** (spike-driven, GO) | FHRR write host; **SlotBinder write on-bridge (GO)** |

**The exact-inverse "algebra" = S3 (unbind conjugate) + S4 (argmax cleanup) + S5 (if-abstention) over S1 (clean
unit-phasor codes), with S2 (the multiplicative bind) as the structural primitive that makes it exactly invertible.**
The June idealizations map on: **I-1 = the exact-inverse FORM** (S2+S3); **I-2 = the clean-code demand** (S1);
**I-3 = host cleanup + host abstention** (S4+S5). Path B (SlotBinder) replaces S2-S7 with a slot/scan architecture;
Path A retires S1,S3,S4,S5,S6,S7 around a kept S2.

---

## 2. WHERE IT SITS IN THE PROJECT TODAY (ledger + owner arc + two stale pointers)

- **Owner arc (2026-09-04, commit `70166962c`):** stage 3 = "VSA composer -> learned"; stage 4 = shared-substrate
  integration (organs on ONE persistent spiking substrate w/ cross-synapses); stage 5 (continuous learning)
  deferred. The end-state is a learned/emergent composer on the shared one-brain substrate.
- **Ledger (`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`):** `one-brain-substrate` (row ~188) is `scaffold_retired:
  YES` but ONLY for the recall MECHANISM (resonate store vs numpy `_scan`; same answers by design) — *where* the
  algebra runs, not *what* it is. `semantic-recall` (row 254, VSA bind/unbind): `scaffold_retired: NO`,
  `retire_status: BLOCKED:neural-render`, host residual = "host VSA unbind (enable_substrate_store=False -> numpy
  composite + host bind/unbind)". `moat-verify` (row 265): `BLOCKED:neural-render` (host abstain gate + host clause
  decomposition). `open-ended-generation` (row ~159): `BLOCKED: host RF-composer exact-inverse VSA moat + ...`.
- **Two stale pointers to flag for `sync-documentation` (not edited here):** (a) the MASTER ROADMAP wall-ledger
  binder row (`docs/plans/2026-07-23-...` L1079) still names "dendritic multiplicative binding" as the next lever —
  but that was ruled out as the WORST generalizer (2026-06-24 / the 2026-07-14 correction) and predates the
  2026-07-22 SlotBinder "FHRR fully retirable" GO. (b) the three ledger rows point their frontier at `neural-render`
  (the mouth) — but the bind/unbind→learned retirement is a DISTINCT frontier the mouth does not unblock; this arc
  is that retirement.

---

## 3. THE CURRENT FRONTIER — the de-risks were run; the residual is scale + wire-in, not feasibility

The June scoping recommended a cheapest-first de-risk (extend the systematicity probe with a spiking binder). The
*question* was pursued (via separate runners, not by literally extending that probe — it is still host-only), and
the July/August arc closed it. Ground-truth results, read directly from the raw artifacts under
`research/findings/raw/` (e.g. `research/findings/raw/_deep_eprop_binder_bundling.json` and
`research/findings/raw/_gap2_deltarule_binder_production_scale_recheck.json`):

| de-risk | tests | held-out | verdict |
|---|---|---|---|
| `_fixedbind_systematicity.json` | fixed ±1/FHRR bind (reference) | ~0.857 (permuted 0.286) | GO — the systematic reference |
| `_learned_bilinear_binder_systematicity.json` | numpy learned ADDITIVE bilinear bind | ~0.571 (memfloor ~0.43) | GO=false — additive learned bind does not match |
| `_onsubstrate_coincidence_systematicity.json` + scale-hardening | on-substrate MULTIPLICATIVE coincidence bind (RUNG1) | 0.67-0.86, permuted → ~0.14 | RUNG1 GO (5/6 at scale) |
| `cortex_learned_binder_systematicity_multiseed.json` | learned binder, correlated vs decorrelated codes | `NEGATIVE_ON_CORRELATED` | needs decorrelated codes |
| `_deep_eprop_binder_bundling.json` | DEEP e-prop / BPTT / ORACLE bind, multi-attribute bundling | **0.002-0.007** (fixed ±1 = 0.993; oracle also fails) | **CONFIRMED BOUNDARY** |
| `_gap2_deltarule_binder_production_scale_recheck.json` | delta-rule fast-weight binder, P=1..48 | delta == additive == 1.000 | delta NOT load-bearing; WRITE host-numpy |

**The July RUNG arc, honestly closed:** RUNG1 (GO) — the spiking coincidence bind computes the product + extrapolates
to held-out combos (5/6 at scale). RUNG2 (GO) — the read-out over the fixed spiking bind is biologically learnable
(transport-free feedback alignment). RUNG3 (BOUNDARY) — a from-scratch on-bridge RATE read hits the rate-coded-SNR
wall, BUT the placed NEF/phasor spiking read is already validated (so the fully-spiking read is achievable; the
the named surpass is phasor/population-coordinate learning — the placed NEF/phasor read is already validated, so the
fully-spiking read is achievable without a from-scratch rate read). gap#2 (6-seed GO, READ) —
a learned local Hebbian outer-product J, read on the RF resonate loop, = the fixed-FHRR 1.000 ceiling, permuted-role
→ 0.000; residual = the WRITE is host-numpy, per-fact multi-ROLE not multi-FACT.

EXTERNAL-SEARCH-RAN: the learned-bind systematicity boundary (the "STRUCTURAL primitive" verdict) is NOT new to
this doc — it is grounded in the external systematicity literature already reviewed in the cited June/July findings
(Lake & Baroni, Nature 2023; "Fodor and Pylyshyn's Legacy", arXiv:2506.01820, 2025) and the dendritic/decorrelation
literature (Mel/Larkum apical-basal coincidence; Mikulasch & Priesemann, PNAS 2021). This ROADMAP asserts NO new
boundary; it cites an existing, externally-reviewed one, and its own recommendation (SlotBinder, Path B) SIDESTEPS
that boundary rather than banking it.

**The SlotBinder supersedes the RUNG/delta line for production.** The delta-rule fast-weight binder is not
load-bearing over plain additive Hebbian to P=48 (12x production's real P=4) and its WRITE is still host-numpy, so
it is not the path (`2026-08-25-...NOT-WIRED`). The SlotBinder is: a genuinely on-bridge Hebbian WRITE, the complete
FHRR surface (incl. embedded clauses the delta binder never covered), 6-seed GO, "FHRR fully retirable."

**⇒ the frontier is NOT "can a learned composer work" (SlotBinder says yes, capability-complete + on-bridge write) —
it is three concrete residuals:** (a) the SlotBinder's **dense-pathway SCALE** at the live corpus (404 facts / full
vocab: `K = 5*max_facts` slot pools × a dense plastic pathway to every filler pool = O(K·KF) — flagged, unmeasured);
(b) the **wire-in** as production default + a 320-scale GPU re-verify + retiring the FHRR/rf fallback to an oracle;
(c) the **fact-store DATA persistence** (S6), which needs a `sim/` persistent store-tensor for BOTH paths.

---

## 4. THE CANDIDATE MENU (RAG-checked; organized by the two paths)

Biology grounded in `research/biology/coincidence-binding.md`, `dendritic-plateau-coincidence-burst.md`,
`urbanczik-senn-dendritic-prediction.md` (+ `dg-ca3-sparse-index.md`).

**Path B — the learned SlotBinder (PRIMARY, most mature):**
- **`SlotBinderComposer`** — competitive WTA slot-per-(fact,role) + a real on-bridge spike-driven Hebbian
  slot→filler write + pointer indirection for clauses + an intrinsic content-addressable scan (the moat). Biology:
  competitive assemblies + Hebbian binding + a Neural-Blackboard-Architecture / Frankland-Greene factored-register
  pointer (point-don't-copy). 6-seed GO, complete FHRR surface, "FHRR fully retirable." The multi-attribute
  bundling boundary is SIDESTEPPED (separate slots, no superposition to invert). Blocker = the dense-pathway scale.

**Path A — retire the host COMPUTATION around a kept structural FHRR bind:**
- **S2 (bind op) — keep as a FIXED STRUCTURAL coincidence primitive** (Mel/Larkum σ-π / NMDA-plateau; catalog
  G.02/J.08; `dendritic-plateau-coincidence-burst.md`). The July boundary makes this the honest answer, not a
  learned-bind TODO. A faithful SPIKING dendritic bind via temporal BAC coincidence is itself GO
  (`2026-08-10-faithful-spiking-dendrite-bind-temporal-BAC-coincidence-GO`, contributing).
- **S3 (unbind) — a local reciprocal-conjugate WIRING RULE** (`_reciprocal_conjugate`; a reciprocal/transpose
  cortical motif applied once at construction; byte-identical, host-free at runtime; neuromorphic-portable;
  `2026-06-20-...self-organization-scoping` Mechanism 1). Built (`local_reciprocal_unbind`).
- **S1 (codes) — learned sparse/decorrelated stream-cortex codes** (`2026-06-15-...GO`, cos ≈ 0.05; DG/F.12
  expansion). The binder consumes decorrelated codes; it need not produce them (Mikulasch-Priesemann: point-neuron
  pairwise Hebbian cannot discover decorrelation).
- **S4 (cleanup) — the placed NEF/TPAM localist spiking read** (Stewart-Tang-Eliasmith Spaun cleanup; == numpy at
  320; common-mode-immune). Built (`enable_spiking_cleanup`).
- **S5 (abstention) — a learned Bogacz-Brown familiarity/novelty gate** (perirhinal recognition; +0.982,
  lesionable). De-risked, not wired.
- **S6/S7 (store + write) — a `sim/` persistent store-tensor + Crawford-Eliasmith / Hebbian LTP write** disjoint
  from the per-op `cp_rf_w_*`. A scoped `sim/` change (the single-array constraint).

**Banked (not the path):** the delta-rule full-matrix fast-weight binder (delta≈additive to P=48; host write); a
learned-from-scratch multiplicative bind (memorizes, does not generalize — the boundary); VTB (learned-but-algebra-
shaped; lifts the clause D-floor, does not test genuine learned systematicity).

---

## 5. THE DE-RISK LADDER — cheap-first, single-variable, gated

Acceptance harness for every rung: `research/runners/vocab_ceiling_probe.py` verbatim at V=320, 6 seeds, moat 100%.
**Path B is the primary line (L1-L4); Path A rungs (LA1-LA3) are the interim/complement while B's scale is
resolved.**

| rung | replaces | change | GO gate | anti-cheats | honest residual |
|---|---|---|---|---|---|
| **L1 (first)** | the SlotBinder scale blocker | MEASURE `SlotBinderComposer` at the live scale (404 facts / full vocab): the O(K·KF) dense slot→filler pathway synapse count, RSS, build+recall latency | fits consumer-GPU memory + acceptable latency, OR a clear sparse-wiring fix is identified | — (a measurement) | if it does not fit, L2 is required first |
| **L2** | dense slot→filler wiring | if L1 fails, sparsify the slot→filler pathway (per-slot sparse fan-in / DG-style expansion) | recall matrix unregressed 6-seed at scale; RSS O(K) not O(K·KF) | permuted-pointer → chance; moat 100% | a scale lever, may bound max_facts |
| **L3** | S2-S7 FHRR (the wire-in) | wire `composer_kind="slotbinder"` as the production default; 320-scale GPU re-verify; demote FHRR/rf to a verify-only oracle | vocab_ceiling_probe 6-seed == the FHRR surface at 320; **LESION**: disable the slot store → recall collapses | permuted-pointer / wrong-clause / shuffled-fact collapse; moat 20/20 | S1 codes (if still random); one-brain co-residency (stage 4) |
| **L4** | S1 codes | pass learned stream-cortex codes by default (both paths) | matrix unregressed 6-seed on learned codes; cos ≈ 0.05 unit-check (NOT median-bipolarized — the false-negative trap) | code lesion/shuffle → recall collapses | codes GIVEN by encoding (developmental), not continuous |
| **LA1** | S3+S4 (interim, rf path) | flip `enable_spiking_cleanup=True` + `local_reciprocal_unbind=True` on any default `rf` path + re-tag deployed bundles; already the `onebrain` default | matrix 6-seed == host; **LESION**: disable spiking cleanup → answer changes (not byte-identical only) | permuted-fact collapse; moat 100% | keeps FHRR bind + host store |
| **LA2** | S5 abstention | wire the learned familiarity gate as the moat (both paths can share) | abstention floor: max-known novelty < min-unknown clean, 6-seed; **LESION**: zero gate weights → abstention collapses | never-stored cues abstain 20/20 | clause decomposition/coverage stays host (moat-verify row) |
| **LA3** | S6/S7 store | a scoped `sim/` persistent store-tensor disjoint from `cp_rf_w_*` + a Hebbian/Crawford-Eliasmith write | held-out recall ≥ host-store parity 6-seed; **LESION**: store lesion → recall collapses | shuffled-fact → chance; write persists under live plasticity | a genuine `sim/` arc; not runner-level |

**Ordering rationale:** L1 (the SlotBinder scale measurement) is the single unclosed gate on the most mature
retirement — it is the spawned `task_5c54ca7f`. If L1/L2 clear, L3's wire-in retires S2-S7 wholesale (Path B) —
the highest-leverage move. LA1 is a near-zero-risk interim retirement of the two most conspicuous host ops on any
`rf` path (and simply confirms/propagates the `onebrain` default). LA2/LA3 + L4 are shared by both paths. L4 (codes)
and stage-4 one-brain co-residency are the deepest.

---

## 6. THE FIRST TRACTABLE DE-RISK (run this next)

**L1: measure `SlotBinderComposer` at the live corpus scale (404 facts / full vocab).** This is the single unclosed
gate on the most mature FHRR-retirement (Path B), it is already scoped as a spawned follow-on (`task_5c54ca7f`,
`2026-08-25-...NOT-WIRED` L132), and it directly serves the owner's stage-3.

Concretely (design; no code in this doc):
1. Instantiate `SlotBinderComposer(max_facts≈420, vocab=<live vocab>)` and measure: the slot→filler pathway synapse
   count (`K=5*max_facts` slot pools × KF filler pools, 20 neurons each — the O(K·KF) dense wiring), the build RSS,
   and the store+recall latency for the 404-fact corpus. Compare against the consumer-hardware reference (a single
   24GB RTX 3090) per the standing principle.
2. **GO** (fits memory + acceptable latency) ⇒ proceed to L3 (the wire-in + 320-scale GPU re-verify), retiring the
   FHRR/rf fallback to a verify-only oracle. **NO-GO** (O(K·KF) blows up) ⇒ L2 first: sparsify the slot→filler
   pathway (per-slot sparse fan-in / DG-style expansion), then re-measure.
3. Route the compute correctly (cost-routing): this is a mechanical CPU/GPU measurement + sweep, NOT agent work —
   `tools/sweep_pool.sh` / `tools/gpu_queue.sh`, 0 Claude tokens. A 6-seed re-verify at 320 is a `--seeds`
   controller-fanned GPU run.

**The wall-reframe question to hold at L1's edge** (CLAUDE.md's first-question-at-a-wall): if the dense pathway
does not fit, what companion process did the SlotBinder replace with a constant — is the dense all-to-all
slot→filler pathway standing in for a *sparse, developmentally-wired* connectivity (DG/F.12 expansion), which is both
cheaper AND more biologically faithful? That reframes L2 from "a scale hack" to "the correct developmental wiring."

**Why NOT "run the systematicity probe / build a learned bind":** those were run (§3) and the learned-bind route is
a characterized boundary. Re-deriving them is the redundant-run failure `corpus_check_required` exists to catch. The
unclosed work is the SlotBinder scale + wire-in, plus the LA-line flips.

---

## 7. GENUINE OWNER-FORKS (decisions this roadmap surfaces but does not make)

1. **PATH A vs PATH B — the primary fork.** Path B (SlotBinder) is a genuinely different architecture (learned
   slots + Hebbian write + pointer indirection) that retires FHRR *wholesale* and is capability-complete + 6-seed
   GO; it sidesteps the multi-attribute bundling boundary. Path A keeps the FHRR structural coincidence bind and
   retires only the host computation around it (and inherits the FHRR's own 2-attribute/depth-2 boundaries).
   **Recommendation: Path B primary** (it is the more complete "VSA composer -> learned", already built), with the
   LA-line as a near-zero-risk interim on the `rf` path. **Fork for the owner:** is wholesale replacement by the
   slot/pointer architecture the intended "learned composer", or is the intent to keep the FHRR VSA and learn only
   its codes/store/read (Path A)?
2. **Is a FIXED structural coincidence bind an acceptable end-state (either path)?** The record is decisive that a
   learned-from-scratch bind operator MEMORIZES and does not generalize (even the oracle fails). So "VSA composer ->
   learned" is honestly satisfied by learning the codes + write + read/recall THROUGH a fixed structural primitive
   (or by the slot architecture), NOT by a plastic bind operator. **Fork:** accept the fixed structural primitive as
   "learned enough", or require the bind operator itself to be plastic — in which case the deliverable is the
   documented systematicity boundary (bank the method, per the no-defer law).
3. **Offline/developmental-trained vs continuous.** The owner's 2026-09-04 nuance allows a thin developmental slice.
   The SlotBinder's Hebbian write and the learned codes are developmental (trained as facts arrive / from a corpus),
   not continuous-learning-over-time. **Recommendation:** developmental is in scope for L3/L4/LA3; continuous is
   stage 5, out.
4. **Keep the exact FHRR as an ORACLE vs delete.** The exact algebra is a clean regression reference + a fast CPU
   path. **Recommendation:** demote to a verify-only test oracle (the pattern the one-brain-substrate row already
   used for the numpy `_scan`: demote, don't delete), per the ledger's own `scaffold_retired` definition.
5. **The fact-store `sim/` tensor (LA3/S6):** build the persistent store-tensor now, or accept the host-list store
   as the documented idealization for this arc? It needs a scoped `sim/` change (single-array constraint), a
   genuine build. **Fork:** in-scope for this arc, or deferred to the one-brain substrate-consolidation arc (stage
   4)?

---

## 8. HONEST RESIDUALS / WHAT COULD STILL GO NEGATIVE

- **The SlotBinder scale question (L1) is unmeasured.** The dense O(K·KF) slot→filler pathway at 404 facts / full
  vocab could blow up on consumer hardware; the sparse-wiring fix (L2) is un-de-risked and may bound `max_facts`.
  This is the real gate, and it is why the "FHRR fully retirable" wire-in has sat undone for a month.
- **The multi-attribute-bundling systematicity boundary is real and characterized.** A learned bind operator
  memorizes but does not generalize (even the oracle). Path B sidesteps it (separate slots); Path A inherits the
  FHRR's own 2-attribute (~29% F3) and depth-2 boundaries. If the owner requires a *learned bind operator*, the
  deliverable is this boundary.
- **The deployed bundle is still `rf`.** Even after a code-default flip, persisted developed bundles keep their
  `composer_kind`; the retirement is not real until the deployed bundle is re-tagged / rebuilt (part of L3/LA1). A
  byte-identical flip earns zero (TERMS.md); the lesion must show the spiking/learned path is load-bearing.
- **The fact-store (S6) needs a `sim/` change for BOTH paths.** `enable_substrate_store` builds per-fact bridges
  from host data; truly-in-synapses persistence needs a store-tensor disjoint from `cp_rf_w_*` (the single-array
  constraint). Not a runner-level win.
- **One-brain integration (owner stage 4) is a separate axis.** A retired composer must then live on the ONE
  persistent shared spiking substrate with cross-synapses (not a co-resident separate bridge) — a further arc.
- **This doc reports no new measurement.** Every quantitative claim is cited to a prior artifact/finding; the raw
  de-risk numbers in §3 are read directly from the cited `.json` artifacts.

---

## 9. Sources (project record + literature)

- Findings: `2026-06-06-composer-vsa-idealization-known-limitation`; `2026-06-11-cortex-core-learned-binder-research`;
  `2026-06-16-onsubstrate-learned-binder-deep-research-scoping`;
  `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE`;
  `2026-06-20-binding-structure-self-organization-scoping`;
  `2026-07-14-deep-eprop-binder-multiattribute-CONFIRMED-BOUNDARY`;
  `2026-07-15-onsubstrate-coincidence-bind-systematicity-RUNG1-GO` / `-bind-learned-readout-RUNG2-GO` /
  `-bdsp-readout-RUNG3-BOUNDARY`; `2026-07-20-composer-factstore-host-persistence-is-the-VSA-idealization-scoping`;
  `2026-07-21-gap2-spiking-learned-binder-6seed-GO-emergence-bar-close`;
  `2026-07-22-gap2-attribute-slot-GO-FHRR-retirement-step1` / `-pointer-clause-GO-FHRR-fully-retirable`;
  `2026-08-10-faithful-spiking-dendrite-bind-temporal-BAC-coincidence-GO`;
  `2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED`.
- Code: `research/runners/rf_phasor_composer.py`, `one_brain_composer.py`, `slotbinder_composer.py`,
  `_keystone2_spiking_slot_binder_derisk.py`, `vocab_ceiling_probe.py`, `brain_conversational_agent.py`,
  `webapp/server.py`; `sim/compose_temporal_bind.py`, `sim/dendritic_neuron.py`, `sim/dendritic_plasticity.py`;
  `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`.
- Biology: `research/biology/coincidence-binding.md`, `dendritic-plateau-coincidence-burst.md`,
  `urbanczik-senn-dendritic-prediction.md`, `dg-ca3-sparse-index.md`.
- Literature (from the June/July reviews): Eliasmith Spaun / Semantic Pointer Architecture; Stewart-Tang-Eliasmith
  NEF cleanup; van der Velde-de Kamps Neural Blackboard Architecture + Frankland-Greene factored registers (the
  pointer-indirection grounding); Gosmann-Eliasmith 2019 (VTB); Crawford-Gingerich-Eliasmith 2016 (human-scale
  VSA-on-spikes / the store); Frady-Sommer TPAM; Bellec 2020 (e-prop three-factor); Mikulasch-Priesemann 2021
  (dendritic balance / decorrelation limit); Lake-Baroni 2023 + arXiv 2506.01820 2025 (systematicity).

**No banking — the boundaries (learned-bind systematicity; the SlotBinder dense-pathway scale; the fact-store `sim/`
constraint; the deployed-bundle-is-rf gap) are reported as found. This is a DESIGN/ROADMAP; no result is asserted
beyond the cited prior artifacts.**
