# Nav read-out geometry — deep-research gate: decoding a 2-D retinotopic bump into the correct movement cardinal is POINT-NEURON-FEASIBLE (population-vector read-out + ring WTA); the SC #6 fix is a BUILD, not a deferral (2026-06-20)

**Type:** READ-ONLY deep-research gate (the project's standing deep-research-first step before committing build/GPU
resources). NO code, NO experiments. Single deliverable = this doc. Stayed on `main`, read-only.
**Owner directive (2026-06-20):** a spiking shortcut/honest-negative must be PROPERLY BIOLOGIZED on the point-neuron
substrate where feasible, NOT left as a deferred follow-on. The boundary audit localized the nav read-out NEGATIVEs
(#6 SC orienting, #9 place-code) to a point-neuron-feasible read-out-geometry fix — so they must be ATTEMPTED. This
gate is the deep-research FIRST step before that build.
**The boundary under review:** `2026-06-20-nav-sc-drive-reorient-derisk.md` (commits `e7ca4655`/`e66944bf`/`e333d771`)
+ the `#6`/`#9` rows of `2026-06-20-boundary-ledger-dendritic-audit.md`.
**Method:** built the diagnosis from the project's OWN SC/place machinery (read in full + verified the load-bearing
read-out math line-by-line against `g11_bg_runner.py`), cross-checked the canonical biology catalog
(`sim-catalog/references/feature-catalog.md`, entries verified against their actual text + Kandel citations), and the
current spiking-SC / population-decode / spiking-actor-critic literature (WebSearch, sources listed at the end). Every
load-bearing "this is point-neuron-feasible" claim is anchored to a published point-neuron model, not asserted.

---

## TOP-LINE ANSWER (the honest call)

**The nav read-out geometry is CLOSABLE on point neurons — and the mechanism is named, canonical, and already
half-present in the codebase. It is a BUILD, not a deferral.**

| boundary | closable on point neurons? | the mechanism | the genuinely-dendritic residual (if any) |
|---|---|---|---|
| **#6 SC retinotopic→cardinal orienting** | **YES — fully** | a **population-vector read-out** (each `sc_map` site has a preferred (dx,dy) vector; the four cardinal pools read the cosine projection of that vector — the Goossens-Van Opstal SC "spike-vector" decode / Georgopoulos H.17) **+ normalization by total bump mass** (`input_divisive_norm`, already in `sim/`) **+ a competitive WTA between the four cardinals** (the project's own `enable_spiking_wta_readout` ring, #4) | **NONE.** The decode is a feedforward weighted sum of preferred vectors on LIF point neurons; the bump itself is already a point-neuron continuous-attractor (the `sc_map<->sc_fs` Mexican-hat). No dendrite implicated. |
| **#9 place→value/cardinal** | **the READ-OUT half: YES** | a **graded rate read-out** of the critic (replace the all-or-none coincidence-plateau that over-clamps the SNc → a modest near>far weight gradient gives a modest near>far rate → a graded GABA_B δ) — the Frémaux-Sprekeler-Gerstner spiking-actor-critic critic | **the FIELD-CARVING half: dendritic-FLAVORED.** A sparse+selective place code from heavily-overlapping egocentric sensors plausibly needs per-cell nonlinear input integration (Mikulasch-Priesemann). That is the *deeper* cause; the *immediate* δ blocker is the point-neuron-fixable read-out regime. |

**The decisive fact for #6:** the SC read-out the project ships (a signed half-plane linear ramp, verified at
`g11_bg_runner.py:262-263`) is **not** the SC's canonical decode. The SC's canonical decode — and the host positive
control the NEGATIVE measures against (`sc_orienting_cardinal_from_image`, `:149-155`) — is a **centroid /
population-vector** of the bump position. The ramp read-out is provably position-INVARIANT in the deployed regime for
two structural reasons (unpacked below), and both are fixed by the canonical population-vector + normalization +
WTA, all of which are point-neuron mechanisms with published precedents and partial in-codebase implementations.
**⇒ #6 becomes a BUILD.** #9's read-out half rides along; #9's field-carving half stays the (legitimately deferred)
dendritic frontier.

---

## 1. DIAGNOSIS — why the current `sc_map → cortex_X` read-out is position-invariant

### 1a. What the deployed read-out actually computes (verified against the code)

`install_spiking_sc_wiring` (`g11_bg_runner.py:201-303`) builds three stages:
1. **`retina(ON) → sc_map`** retinotopic 2×2 pooling (`:229-241`) — a faithful retinotopic map (each `sc_map` site
   pools its 2×2 egocentric-ON block). The egocentric render (`render_egocentric_goal`, `:183-198`) paints the goal
   as a single dim blob at `(c + (goal−agent)·ppc)`, so the **bump position in `sc_map` directly encodes the goal's
   egocentric bearing.** This stage is correct.
2. **`sc_map ↔ sc_fs` Mexican-hat + `sc_map` recurrent** (`:242-255` + framework-built) — local excitation + surround
   inhibition → a clean single activity bump. This is a point-neuron continuous-attractor; it is correct.
3. **`sc_map → cortex_{N,E,S,W}` "weighted-quadrant pooling"** (`:256-273`) — **the read-out, and the flaw.** For
   each cardinal `a`, every `sc_map` site `(sx,sy)` projects to `cortex_a` with weight

   ```
   ddx, ddy = sx - sc_center, sy - sc_center
   wv = {"E": max(0, ddx), "W": max(0, -ddx), "N": max(0, ddy), "S": max(0, -ddy)}[a]
   w  = w_sc_cortex * wv          # w_sc_cortex = SC_CORTEX_W, default 18
   ```

   So the drive into the East pool is `Σ_sites max(0, sx − c) · sc_map_activity(sx,sy)`, the West pool
   `Σ_sites max(0, c − sx) · activity`, etc. **This is a signed half-plane LINEAR RAMP** — a projection of the
   activity-weighted horizontal/vertical coordinate onto a fixed cardinal axis.

### 1b. Why a half-plane ramp is NOT a position decode (the two structural flaws)

A population-vector / centroid decode reports *where the bump is*. The half-plane ramp does not, for two reasons that
compound:

- **(i) No normalization by bump mass.** The pool drives are an *un-normalized* weighted SUM, so they scale with the
  *total* `sc_map` activity, not with the *location* of the mass. A centroid is `Σ(pos·activity) / Σ(activity)` — the
  division by `Σ(activity)` is what turns "total weighted mass" into "where". The ramp omits the denominator, so a
  brighter/bigger bump raises all the half-plane sums together; the read-out conflates *how much* SC fires with
  *where* it fires. This is exactly why **raising `SC_CORTEX_W` 18→150 only over-drives all four pools** (the NEGATIVE's
  "near-uniform at strong drive", grid-8 `[121,117,105,107]`): a global gain on an un-normalized sum cannot sharpen a
  position read-out.
- **(ii) No competition between the four cardinals.** Each `sc_map_to_cortex_a` pathway is an INDEPENDENT, purely
  additive projection — there is no lateral inhibition *between* `cortex_N/E/S/W`. So the read-out has no mechanism to
  convert "East's drive slightly exceeds the others" into "East wins, the rest are suppressed". The four pools' drives
  rise and fall together with bump size, and the **winner's MARGIN over the runners-up does not widen** with either
  bump position or drive magnitude. The cascade's intrinsic structural N-bias (the agent pins to the top edge,
  pos-row 31) then dominates the tiny, un-sharpened margin → the **stuck-N, goal-invariant action distribution** the
  NEGATIVE documented (N ~0.45-0.52 in EVERY phase, every drive level).

Contrast the **host positive control** (`sc_orienting_cardinal_from_image`, `:144-155`), which the NEGATIVE shows
re-orients cleanly: it reads the goal-blob **centroid** (`goal_xs.mean()`, `goal_ys.mean()`), subtracts the agent
centroid, and returns `argmax(|dx|,|dy|)` → cardinal. That IS a center-of-mass position decode followed by a
hard argmax (the competition). The spiking read-out is missing *both* halves of what the host does for free: the
position normalization AND the argmax/competition.

**⇒ The flaw is a read-out-GEOMETRY problem, exactly as the NEGATIVE classified it (operating-point floor / under-
selective, non-goal-tracking read-out) — and it is fully attributable to two missing point-neuron mechanisms
(normalization + inter-cardinal competition), neither of which is dendritic.** The retinotopy and the bump are
already correct; only the bump→cardinal decode is wrong.

### 1c. #9 — separate the point-neuron read-out from the dendritic field-carving

`#9` (the place-code → value/cardinal read-out, `2026-06-19-place-code-sparsify-default-BOUNDARY.md`) is a COMPOUND,
and it must be split:

- **The read-out sub-part (POINT-NEURON-feasible):** the critic afferent uses an **all-or-none weighted
  coincidence-plateau** read-out (`coincidence_detector=True`, `g11_bg_runner.py:1839`). That read-out has only two
  reachable regimes (under-discriminating at low weight; over-clamping the SNc GABA_B to 0 at high weight) — neither
  GRADES. A **graded rate read-out** (a modest near>far weight gradient → a modest near>far critic *rate* → a graded
  GABA_B δ, without the binary over-clamp) is the Frémaux-Sprekeler-Gerstner critic geometry, and it is point-neuron.
  The boundary doc itself names this as the fix ("a graded rate read-out that scales smoothly with V … without the
  over-clamp", `:111-112`).
- **The field-carving sub-part (DENDRITIC-FLAVORED):** the deeper cause is that the point-neuron `place` pool **cannot
  form many distinct, location-selective sparse codes from heavily-overlapping egocentric landmark sensors** in the
  FS-ping-open read regime (a few dominant cells fire at MANY locations; read cos ≈ 0.42-0.78 regardless of self-org
  sparsity). A genuinely sparse+selective place code (real place cells ~1-5% AND selective) plausibly needs per-cell
  nonlinear input integration to carve selective fields — the Mikulasch-Priesemann analog/dendritic limit the project
  repeatedly hits. **This is the one residual with a real dendritic flavor, and it is the deeper cause; it is NOT on
  the conversational path and is legitimately deferred.**

**The honest #9 framing:** the read-out fix (graded rate critic) is point-neuron-feasible and worth attempting; even
done, the δ-lift is capped by the field-carving limit unless the place code is made selective. The host-Gaussian
`vs_place_context` (position-specific by construction) stays the better-δ scaffold for #9 until the field-carving
frontier is taken up. **#9 is therefore a PARTIAL build (the read-out) over a deferred dendritic floor (the fields),
distinct from #6 which is a FULL point-neuron build.**

---

## 2. RANKED OPTIONS for the read-out geometry

Each option: the mechanism, the biology source (verified), the point-neuron-vs-dendrite call, the expected failure
mode. Ranked by leverage × cheapness for the #6 SC orienting read-out (the place read-out #9 reuses options B+D for
its read-out half).

### Option A (RECOMMENDED, top rank) — population-VECTOR read-out: each SC site reads its preferred cardinal cosine, normalized by bump mass

- **Mechanism:** replace the half-plane ramp with the SC's canonical decode. Each `sc_map` site `(sx,sy)` has a
  **preferred direction vector** `u = (ddx, ddy)/|.|` (its retinotopic bearing). The four cardinal pools read the
  **cosine projection** of each site's preferred vector onto the cardinal axis (E:`+x̂`, W:`−x̂`, N:`+ŷ`, S:`−ŷ`),
  i.e. `w(site→a) = max(0, û_a · u_site)` (a cosine-tuned weight, NOT a linear ramp). Then **normalize by total bump
  activity** so the read-out reports the bump's *direction* not its *mass*: route the four cardinal pools through
  `input_divisive_norm` with the `sc_map` total drive as the normalization pool (`drive/(σ + gain·mean_bump)`), the
  Carandini-Heeger gain-control that the project already ships at D=2048.
- **Biology source (verified):** the SC "spike-vector" / weighted-averaging population decode — Goossens & Van Opstal
  spiking SC models (each recruited cell contributes a fixed location-determined movement vector; the saccade =
  dynamic linear sum / weighted average of all cell vectors over the bump). Catalog **H.25** (Superior colliculus
  saccade map — topographic motor map, Kandel 6e Ch 35 p 875-882) + **E.03** (Population coding & vector averaging,
  Kandel 6e Ch 17 p ~458-464) + **H.17** (Georgopoulos population vector, Kandel 6e Ch 34 p 825-840 — *and the catalog
  H.17 entry already flags the project's motor pools as "categorical (one pool per action) rather than vector-tuned …
  could be tested by adding cosine-tuned input layer; would naturally yield population vector readout"* — this gate
  is the cash-out of that note). Normalization: **E.05** (lateral inhibition / center-surround, Kandel 6e Ch 22) is
  the same algorithmic motif; the `input_divisive_norm` primitive is the divisive realization.
- **Point-neuron vs dendrite:** **PURE POINT-NEURON.** The decode is a feedforward weighted sum of preferred-vector
  cosines over LIF point neurons — the published LIF population-vector result confirms "the net synaptic current
  driving a neuron's spike generator is a weighted sum of post-synaptic currents" with cosine tuning curves. No
  dendrite. The cosine weights are a one-line change to the weight formula at `:262-263`; the normalization reuses an
  existing `sim/` primitive (a region flag, no new kernel).
- **Expected failure mode:** under-normalization (if the `input_divisive_norm` pool/gain is mis-set the read-out
  reverts toward mass-coding) — checkable directly via the per-phase action distribution. A cosine read-out WITHOUT
  the inter-cardinal competition (Option D) may still tie when two cardinals are near-equal (a diagonal goal) — so
  A is necessary but B/D sharpen it.

### Option B (top rank, COMPLEMENTARY) — competitive WTA between the four cardinals (the project's own #4 ring)

- **Mechanism:** add lateral inhibition *between* `cortex_N/E/S/W` so the cardinal with the largest (normalized)
  population-vector drive suppresses the others — turning a small winner-margin into a decisive choice. The project
  ALREADY HAS this: `enable_spiking_wta_readout` (`g11_bg_runner.py:448`) builds `sel_X` pools with recurrent
  self-excitation (`sel_recurrent_weight`, the Wang-2002 NMDA attractor) + cross-pool inhibition + a `commit_X`
  Lo-Wang commit-burst stage (`:478`), the project's #4 fully-spiking decision read-out (default-on,
  `2026-06-19-spiking-decision-default-on-GO.md`, 1.16× host). Route the SC population-vector drive INTO the `sel_X`
  competition instead of (or before) the categorical `cortex_X` pools.
- **Biology source (verified):** the **ring of action neurons with local excitation + global inhibition → a single
  activity bump** is the Frémaux-Sprekeler-Gerstner (2013, PLOS Comput Biol) spiking-actor read-out ("each neuron
  excites the neurons with similar tuning and inhibits all other neurons … the lateral connectivity ensures a single
  bump of activity"). Catalog **A.04** (BG output disinhibition is selective — competitive WTA at GPi/SNr) +
  **E.03/H.17** (the same population-vector frame). Wang 2002 / Lo-Wang (the accumulator + commit burst) is the
  project's documented #4.
- **Point-neuron vs dendrite:** **PURE POINT-NEURON** (the `sel_X` ring is LIF Izhikevich; it is the deployed #4
  default). No new mechanism — a wiring/routing change to feed the SC drive into the existing competition.
- **Expected failure mode:** if the ring gain (`sel_recurrent_weight`) is set too hard it locks onto the first winner
  and resists re-orient (a hysteresis cost) — exactly the re-orient metric the gate measures, so it is directly
  falsifiable; the #4 GO already tuned this operating point (`sel_recurrent_weight=0.3`).

### Option C (lower rank) — sharper pooling KERNEL (narrow the half-plane to a quadrant/Gaussian-weighted wedge)

- **Mechanism:** keep the additive read-out but replace the broad linear ramp with a **narrow directional kernel** —
  each cardinal pool reads only a Gaussian-weighted wedge centered on its axis (sites near the East meridian weighted
  ~1, off-axis sites ~0), so off-target sites stop leaking into the wrong cardinal.
- **Biology source:** the cosine-tuning width of E.03/H.17 (a narrow tuning curve = sharper discrimination, the
  "tuning-curve width vs discrimination acuity" trade-off in E.03's validation). Center-surround E.05.
- **Point-neuron vs dendrite:** **POINT-NEURON.** A weight-formula change only.
- **Expected failure mode:** sharpening the kernel WITHOUT normalization (A) or competition (B) still leaves the
  read-out mass-coding within the wedge — likely a partial improvement that doesn't reach host (the kernel narrows
  *which* sites vote but not *how* the votes become a position). This is why C is ranked below A+B: it treats the
  symptom (leak) not the cause (no normalization, no competition). Useful as an ablation, not the primary fix.

### Option D (for #9's read-out half) — graded rate critic read-out (replace the all-or-none coincidence plateau)

- **Mechanism:** for the place→value critic, replace the all-or-none weighted coincidence-plateau read-out with a
  **graded rate read-out**: a modest near>far weight gradient → a modest near>far critic firing RATE → a graded
  GABA_B δ at the SNc, without the binary over-clamp. (The boundary doc names exactly this, `:111-112`.)
- **Biology source (verified):** the **Frémaux-Sprekeler-Gerstner (2013) spiking actor-critic critic** — "a pool of
  critic neurons encode the expected future reward at the agent's current position … the change in predicted value …
  leads to a TD error broadcast to synapses." Catalog (place→value): D.06 (place cells) feeding a graded value pool.
- **Point-neuron vs dendrite:** **POINT-NEURON for the read-out** (a rate-coded critic pool is LIF). The residual
  field-carving (a selective sparse place code) is the dendritic-flavored deferred part — see §1c.
- **Expected failure mode:** even a perfect graded read-out is δ-capped by the non-selective place code (§1c) — so D
  alone lifts δ only partway; the full #9 lift needs the deferred field-carving. D is worth doing (it removes the
  over-clamp pathology) but is honestly a PARTIAL fix over a deferred floor.

**Ranking summary:** for #6 the primary build is **A (population-vector read-out + normalization) + B (the existing
#4 WTA ring)** — both pure point-neuron, both with published precedents AND partial in-codebase implementations. C is
an ablation. D is the #9 read-out half (partial, over a deferred dendritic floor).

---

## 3. REUSABLE PROJECT MACHINERY (what already exists — minimal new code)

- **The SC build itself** — `install_spiking_sc_wiring` (`g11_bg_runner.py:201-303`): the retinotopic `retina→sc_map`
  2×2 pooling + the `sc_map↔sc_fs` Mexican-hat + the `sc_map→cortex_X` read-out. **Only stage 3 (the read-out weight
  formula, `:262-263`) changes** for Option A — stages 1+2 (retinotopy + bump) are already correct.
- **`input_divisive_norm`** (`sim/bridge.py:6048-6057`, `cfg.enable_input_divisive_norm` + `BrainRegion.input_divisive_norm`):
  the Carandini-Heeger divisive gain control `drive/(σ + gain·mean_pool)` — the Option-A normalizer, already shipping
  at production D=2048, GUARDED NO-OP when off (byte-identical). A region flag, no new kernel.
- **The #4 spiking-WTA ring** — `enable_spiking_wta_readout` + `sel_X` (Wang attractor) + `commit_X` (Lo-Wang commit
  burst), `g11_bg_runner.py:448-479`, default-on per `2026-06-19-spiking-decision-default-on-GO.md`: the Option-B
  inter-cardinal competition. A routing change (feed the SC population-vector drive into `sel_X`), not a new build.
- **The host positive control** — `sc_orienting_cardinal_from_image` (`:124-155`) (centroid decode + argmax) and the
  graded sibling `sc_salience_offset_from_image` (`:158-180`, the continuous (dx,dy) offset): the host scaffold the
  build must approach, AND a ready-made graded "where" signal to validate the population-vector read-out against.
- **V1/Gabor + retina render** — `sim/visual_cortex.py`: `build_v1_simple_weights` (`:76`), `render_gridworld_to_image`
  (`:155`), `image_to_retina_drive` (`:210`); `render_egocentric_goal` (`g11_bg_runner.py:183`). The perception front
  end is intact and unchanged.
- **The place self-org + critic** — `nav_critic_place_selforg` builder path (`place_sensors → place` plastic competitive
  + `place → striosome_value` coincidence critic, `g11_bg_runner.py:1789-1840`), the `_n5_place_sparsify_probe.py`
  iteration harness, the host-Gaussian `vs_place_context` scaffold: the #9 machinery for the Option-D read-out swap.
- **The anti-cheat lesion already wired** — `install_spiking_sc_wiring(scramble=True)` (`:221-226`) permutes the SC-site
  target assignment (destroys retinotopy) → the build's lesion control is built in.

---

## 4. RECOMMENDED CHEAP-FIRST DE-RISK

**The smallest experiment that answers the fork: does a population-vector read-out (Option A) + normalization make
`cortex_X` TRACK the bump position, so the agent RE-ORIENTS after a goal change?**

- **The build (minimal):** change ONLY the `sc_map→cortex_X` weight formula at `g11_bg_runner.py:262-263` from the
  half-plane ramp `wv = max(0, ±dd)` to the cosine-tuned preferred-vector projection
  `wv = max(0, û_a · (ddx,ddy)/|.|)`, AND flag the four `cortex_X` pools `input_divisive_norm=True` with the `sc_map`
  total as the normalization pool. (Option B — routing into the existing #4 `sel_X` ring — is the immediate follow-on
  if A alone ties on diagonal goals; it is already-built, so it costs only a routing change.) This is a runner-local
  read-out change + an existing-primitive flag — small, additive, default-preservable behind an env/kwarg so the
  documented SC op-point reproduces byte-identical when off.
- **The decisive control (the SAME metric the NEGATIVE used):** the **per-goal-phase re-orient finalQ + the per-phase
  action distribution** on the faithful **grid-32 / 1800 / warmup-600** schedule (4 goal phases, 3 re-orients), the
  exact `--spiking-sc` config of the NEGATIVE. PASS = the population-vector arm's post-goal-change finalQ approaches
  the host control (the NEGATIVE's gap was ~73× on post-change) AND the per-phase action distribution **TRACKS the
  goal** (W-heavy for the far-west goal, E-heavy for the SE goal) instead of the stuck-N (N ~0.45-0.52 every phase)
  the ramp produces.
- **Why this is the right cheap-first shot:** it is the single decisive hypothesis (the read-out geometry is the
  cause, per §1), it reuses `run_moving_goal_episode` by import, the grid-8/480 smoke gives a seconds-scale early read
  (the NEGATIVE's smoke already shows the stuck-N there), and the faithful grid-32 confirm is the same scale the
  NEGATIVE used — so the comparison is apples-to-apples against a known floor.

---

## 5. THE ANTI-CHEATS (mandatory, all carried from the NEGATIVE + the build's own lesion)

1. **Host positive control** — `sc_orienting_cardinal_from_image` (centroid + argmax), same grid/schedule, anchors the
   population-vector arm's residual gap (the NEGATIVE's host re-orients to post-change finalQ ~0.5, gate 2.19).
2. **The re-orient-after-goal-change metric** (NOT just static acquisition) — the per-phase finalQ on phases 1..3
   (post-change), because the NEGATIVE showed the ramp acquires the FIRST goal fine and fails ONLY on re-orient.
   A read-out fix must move the *re-orient* metric, not the static-hold metric (which `SC_CORTEX_W` already moved
   without fixing re-orient).
3. **The per-goal-phase action distribution** (the datum that diagnosed the NEGATIVE) — the (N,E,S,W) fraction per
   phase MUST track the goal's location (shift W-heavy↔E-heavy across phases), not stay goal-invariant. This is the
   direct read of "does the read-out track the bump's retinal position", and it is the clincher the NEGATIVE used.
4. **The retinotopy-scramble LESION** — `install_spiking_sc_wiring(scramble=True)` (already wired): a scrambled-
   retinotopy population-vector read-out MUST regress to chance (proves the orienting is carried by the *retinotopic*
   decode, not a non-retinotopic leak / a cascade prior).
5. **Drive non-confound** — re-run the population-vector arm at the SAME `SC_CORTEX_W` as the host-pA equivalent so the
   improvement is attributable to the read-out GEOMETRY, not a covert drive increase (the NEGATIVE proved drive alone
   does not fix it; the build must beat that at matched drive).
6. **Perception NOT stripped** — `enable_visual_cortex` on, warmup honored (the actor keeps its vision drive), as the
   NEGATIVE did.

---

## 6. HONEST TOP-LINE

**#6 (the SC retinotopic→cardinal orienting read-out) is CLOSABLE on point neurons, and the mechanism is named: a
population-VECTOR read-out (each SC site's preferred-cardinal cosine projection) NORMALIZED by total bump mass
(`input_divisive_norm`) and sharpened by a COMPETITIVE WTA between the four cardinals (the project's own already-built
#4 `sel_X`/`commit_X` ring).** The deployed half-plane linear-ramp read-out is provably position-invariant in the
deployed regime because it (i) omits the bump-mass normalization that turns "weighted mass" into "where" and (ii) has
no competition between cardinals to widen the winner's margin — and the canonical SC decode, the host positive
control's own centroid+argmax, and the published LIF population-vector / spiking-actor-critic results all confirm the
fix is feedforward weighted sums + a ring WTA on point neurons, **with no dendrite implicated.** The retinotopy and the
attractor bump are already correct point-neuron structures; only the bump→cardinal decode is wrong, and it is a
weight-formula change + an existing-primitive flag + a routing change into existing machinery.

**⇒ #6 is a BUILD, not a deferral** — exactly what the owner directive asks for, and exactly what the boundary audit
predicted ("the read-out fix is point-neuron"). The cheap-first de-risk (population-vector read-out + the re-orient-
after-change control on grid-32) is the next step.

**#9 (the place→value/cardinal read-out) splits:** its READ-OUT half (a graded rate critic replacing the all-or-none
coincidence-plateau that over-clamps the SNc) is point-neuron-feasible and worth the partial build; its FIELD-CARVING
half (a sparse+selective place code from overlapping egocentric sensors) is the one residual with a **genuine dendritic
flavor** (per-cell nonlinear input integration; the Mikulasch-Priesemann limit) and is legitimately deferred — it is
the *deeper* cause, the δ remains capped by it until the place fields are made selective, and it is NOT on the
conversational critical path. The host-Gaussian `vs_place_context` stays #9's better-δ scaffold meanwhile.

**The one genuinely-dendritic part of this whole nav read-out boundary is #9's field-carving (selective sparse place
fields) — NOT #6's bump decode (which is fully point-neuron) and NOT #9's critic read-out (which is point-neuron).**

---

## Sources (verified against the actual text)

**Project code (read in full, load-bearing math verified line-by-line):**
- `research/runners/g11_bg_runner.py` — `install_spiking_sc_wiring` (`:201-303`, the read-out flaw at `:256-273`,
  weight formula `:262-263`); `sc_orienting_cardinal_from_image` (`:124-155`, the host centroid+argmax control);
  `sc_salience_offset_from_image` (`:158-180`); `render_egocentric_goal` (`:183-198`, bump-position = goal bearing);
  the spiking-WTA `sel_X`/`commit_X` read-out (`:446-479`); the neural-critic place→value path (`:1789-1880`);
  the SC drive injection (`:6730-6739`).
- `sim/bridge.py` — `input_divisive_norm` Carandini-Heeger gain control (`:6048-6057`, guarded no-op).
- `sim/visual_cortex.py` — `build_v1_simple_weights` (`:76`), `render_gridworld_to_image` (`:155`),
  `image_to_retina_drive` (`:210`).

**Project findings (the boundary under review + the ledger):**
- `research/findings/2026-06-20-nav-sc-drive-reorient-derisk.md` (the #6 NEGATIVE: drive sweep, stuck-N action
  distribution, operating-point-floor classification).
- `research/findings/2026-06-20-boundary-ledger-dendritic-audit.md` (#6 point-neuron, #9 "dendritic-flavored but the
  read-out fix is point-neuron").
- `research/findings/2026-06-19-place-code-sparsify-default-BOUNDARY.md` (#9: the read-out vs field-carving split;
  the "graded rate read-out" named fix at `:111-112`).
- `research/findings/2026-06-20-nav-loop-closure-derisk.md` (the SC-drive gap localization; the reentrant-arc
  refutation that pointed here).
- `research/findings/2026-06-19-spiking-decision-default-on-GO.md` (the #4 `sel_X`/`commit_X` WTA ring, default-on,
  1.16× host — the reusable Option-B machinery).

**Catalog (`sim-catalog/references/feature-catalog.md`, entries verified against their text + Kandel citations):**
- **H.25** Superior colliculus saccade map — topographic motor map (Kandel 6e Ch 35 p 875-882): the SC's
  spatial-map→motor decode; "Stimulating a SC site evokes a saccade of fixed amplitude/direction matching that site."
- **E.03** Population coding & vector averaging (Kandel 6e Ch 17 p ~458-464): "downstream vector sum … extracts the
  value … decode angle from population vector; tuning-curve width vs discrimination acuity."
- **H.17** Georgopoulos population vector (Kandel 6e Ch 34 p 825-840): cosine-tuned preferred directions; population
  vector = Σ rᵢ·θᵢ. The entry itself flags the project's pools as "categorical … could be tested by adding cosine-
  tuned input layer; would naturally yield population vector readout" — this gate cashes that out.
- **E.04** Topographic / retinotopic maps (Kandel 6e Ch 17 p ~460-462); **E.05** Lateral inhibition / center-surround
  (Kandel 6e Ch 22, the sharpening / divisive motif); **A.04** competitive WTA at GPi/SNr; **A.07** SNr→SC.
- **D.06** Place cells (O'Keefe 1971); the place→value reference for #9.

**Literature (WebSearch; the point-neuron-feasibility anchors):**
- Goossens & Van Opstal — spiking SC models: the cell "spike vector" (location-determined movement contribution)
  summed/weighted-averaged over the bump = the SC population decode. ["A spiking neural network model of the midbrain
  superior colliculus that generates saccadic motor commands"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5506246/);
  ["A spiking neural network model of the Superior Colliculus that is robust to changes in the spatial-temporal
  input"](https://www.nature.com/articles/s41598-022-10991-6).
- Population-vector decode on LIF point neurons (weighted sum of preferred directions; net synaptic current = weighted
  sum of PSCs; cosine tuning): ["The accuracy of the population vector estimate in networks of integrate-and-fire type
  neurons"](https://www.sciencedirect.com/science/article/abs/pii/S092523120100399X); ["Bayesian population decoding of
  spiking neurons"](https://pubmed.ncbi.nlm.nih.gov/20011217/).
- **Frémaux, Sprekeler & Gerstner (2013), PLOS Comput Biol — continuous-time actor-critic with spiking neurons** (the
  canonical point-neuron place→action + critic): the actor = a ring of direction-coding neurons (local excitation +
  global inhibition → single bump), action = population vector (direction-weighted firing-rate sum); the critic = a
  place-reading value pool → TD error. ["Reinforcement Learning Using a Continuous Time Actor-Critic Framework with
  Spiking Neurons"](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003024).

_Read-only deep-research deliverable. No code, no experiments. Load-bearing read-out math verified line-by-line
against `g11_bg_runner.py`; every "point-neuron-feasible" claim anchored to a published point-neuron model + a catalog
entry verified against its text._
