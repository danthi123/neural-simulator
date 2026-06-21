# Shortcut #6 — the UPSTREAM orienting residual: SURPASS round (2026-06-22)

**Type:** READ-ONLY deep-research SURPASS scoping (no code edits — this doc is the only write; no GPU run beyond
reading existing JSONs + a 5-line CPU render-clipping calculation). The gated next step after BOTH accumulator-stage
fixes came back NEGATIVE — FIX-A (divisive normalization at the `sel_X` input,
`2026-06-20-shortcut6-FIXA-divnorm-accumulator.md`) and FIX-B (opponent-pair the `sel_X` accumulators, same doc /
`7d86fc59`). The 3-run convergence (FIX-A seed-42, FIX-A seed-43, FIX-B seed-42 — in every run the retinotopy-scramble
lesion SCRAM ≈ the intact read-out) **re-located the residual UPSTREAM of the Wang-2002 `sel_X` accumulator**: the
goal-direction superior-colliculus (SC) orienting signal is too weak at the selection ring to overcome the cascade's
relative bias in the first place. The accumulator was never the problem. This is the canonical "deep-research at a
multiply-confirmed boundary" move.

**Owner standard (load-bearing):** BRAIN-BASED-ONLY; the verdict is grid-32 (never grid-8); a boundary is not an exit.
The no-confab moat is array-disjoint from the nav cascade (the nav SC read-out is `cp_*` nav-cascade state —
`cp_connections` / `cp_membrane_potential_v` / `cp_firing_states`; the conversational composer's complex `cp_rf_w_*`
synapses are a separate allocation) and is untouched throughout; nothing here weakens it. Protected `sim/` edits are
APPROVED for the eventual fix — flagged for byte-review where applicable, not gated.

**Terms defined once.** *SC* = superior colliculus, the midbrain retinotopic orienting map. *Retinotopic / topographic
map* = adjacent retinal locations map to adjacent SC neurons. *Egocentric render* = the world drawn from the agent's
own eye (the goal's bearing relative to where the agent is looking). *`sc_map`* = the spiking SC sheet (16×16 neurons)
that holds an activity *bump* at the goal's egocentric retinal location. *Population-vector / pop-vector decode* = each
SC site has a preferred (dx,dy) direction; the four cardinal pools read the cosine projection of the bump's location.
*sel ring / `sel_X` accumulators* = the four Wang-2002 NMDA-recurrent "selection" pools (one per cardinal N/E/S/W) that
integrate evidence and commit to an action. *Margin* = the firing-rate gap between the winning cardinal and the
runners-up. *Tie-break* = how the host read of the spiking decision resolves a 4-way tie. *FIX-1* = the
already-validated stochastic tie-break (replaces the N-first `max()` that resolved every `[40,40,40,40]` tie to N).
*HOST* = the host orienting scaffold (centroid + argmax position decode), the ceiling. *SCRAM* = the
retinotopy-scramble lesion (permutes the SC-site → target assignment, destroying retinotopy).

---

## TL;DR — the verdict in three sentences

The 3-run accumulator-fix convergence correctly re-located the residual upstream, but the upstream residual is **not
"the SC→sel projection is too weak"** — it is, one stage further up still, that **the egocentric retina the SC reads
cannot represent the goal at all when the goal is more than ~4 grid cells away.** At grid-32 the moving-goal schedule's
four corner goals (30+ cells from the agent) render **entirely OFF the 32-pixel egocentric retina** — `sc_retina` mass
= **0.0**, the SC bump is **absent**, so there is nothing for the pop-vector decode, the divnorm, the WTA ring, or any
sel-stage operation to act on. The fix is the SC's canonical **log-polar / foveal-magnified retinotopy** (eccentricity
compressed-but-always-represented; Ottes-Van Gisbergen, Hafed lab 2019) so an eccentric goal lands on a peripheral
`sc_map` site instead of clipping off-image — a runner-side render change (rank-1), and **this boundary is
SURPASSABLE, not irreducible.**

---

## MOVE 1 — ISOLATE + QUANTIFY: where exactly is the signal too weak, and by how much?

The brief lists four candidate loci for the upstream weakness: (i) the SC orienting *output* itself, (ii) the
`sc_map → cortex_X` SC→sel *projection strength*, (iii) the *perception → SC* pathway, or (iv) the common-mode N-lead
swamping the differential. The evidence localizes it to **(iii), and within (iii) to the most upstream point possible —
the egocentric *render* that feeds the SC retina** — with the others ruled out as the binding constraint.

### 1a. The accumulator stage is exonerated (the 3-run convergence, from the existing JSONs)

Reading the FIX-A / FIX-B arm-3 result JSONs (`scpv_FIXA_arm3_seed{42,43}.json`, `scpv_FIXB_arm3_seed42.json`) — and
note every arm in these runs already has **FIX-1 (the validated tie-break) ON** (the arms are named HOST / FIX1 / FIX1A
/ FIX1B / SCRAM), so this arc was attacking the *post-tie-break* residual:

| run | HOST post-Σ | **FIX1** post-Σ (tie-break only) | FIX1A/B post-Σ (+ sel op) | FIX1 dom-per-phase | FIX1A/B dom-per-phase |
|---|---|---|---|---|---|
| FIX-A s42 | 1.93 | **68.7** (best spiking) | 115.8 (WORSE) | N,W,W,E (tracks 3/4) | E,E,E,E (stuck) |
| FIX-A s43 | 1.59 | **78.3** (best spiking) | 111.4 (WORSE) | N,W,E,W (tracks 3/4) | N,N,N,N (stuck) |
| FIX-B s42 | 2.21 | **69.8** (best spiking) | 82.0 (WORSE) | W,N,W,E (tracks 3/4) | E,W,N,N |

**Reads (decisive):**
- **FIX1-alone is the BEST spiking arm in all three runs; both sel-stage operations make it WORSE.** Divisive
  normalization (FIX1A) and opponent-pairing (FIX1B) each *degraded* a working tracking arm — FIX1A flipped seed-42 to
  stuck-E and seed-43 back to stuck-N; FIX1B's tie-fraction dropped (0.30 → 0.18) yet its score got worse, i.e. it
  reduced ties by *throwing away the small real SC margin*. A sel-stage bias-rejection operator cannot help because the
  thing it would reject (a count-level common-mode surplus) is not the binding constraint; the binding constraint is
  the *presence and magnitude of the goal-direction signal*, which these operators shrink.
- **SCRAM ≈ FIX1A/B in every run** (FIX-A s42: SCRAM 118.2 ≈ FIX1A 115.8; FIX-B s42: SCRAM 81.1 ≈ FIX1B 82.0). The
  retinotopy-scramble lesion is statistically identical to the intact decode ⇒ under these sel-stage fixes the
  orienting is **not carried by the retinotopic decode at all.** This is the same signature the grid-32 CLOSE verdict
  reported, and it is the formal proof the residual is upstream of `sel_X`.

⇒ **(iv) common-mode at the accumulator is REJECTED as the binding constraint** (the operator that removes it makes
things worse), and **the accumulator is exonerated.**

### 1b. The signal that DOES reach the sel ring tracks on STRONG margins and fails on WEAK ones — a magnitude problem, precisely located in the per-phase pattern

The decisive per-phase finalQ pattern (from the FIX1 arms above; goal schedule phase0 NE / phase1 far-W / phase2 SW /
phase3 SE):

| arm | phase0 NE (diagonal) | phase1 far-W (pure lateral) | phase2 SW (diagonal) | phase3 SE (diagonal) |
|---|---|---|---|---|
| HOST | ~0.6 | ~0.5 | ~0.6 | ~0.5 |
| FIX1 s42 | **25.4** | **1.1** | **20.3** | **47.3** |
| FIX1 s43 | **42.4** | **1.3** | **33.5** | **43.5** |
| FIX1 s42 (FIX-B run) | **31.7** | **1.0** | **27.4** | **41.4** |

**The signal is present and correct on exactly ONE phase — phase1 (far-W, a pure due-West goal with no N/S component) —
where FIX1 reaches finalQ ≈ 1.0-1.3 (host-level) in all three runs.** On the three *diagonal* phases (NE / SW / SE,
each requiring two cardinals to be distinguished) it fails (finalQ 20-47). The earlier cascade-north-bias-FIX doc
confirmed the complement: with FIX1 on, the agent reaches its strong-margin goal cleanly AND **SCRAM then COLLAPSES**
(SCRAM +23..+81% worse than FIX1 across 3 seeds) — i.e. when there *is* a margin, the retinotopic decode IS
load-bearing. So the post-tie-break residual is unambiguously a **margin-MAGNITUDE / SNR problem**, not a bias-rejection
problem: the SC produces enough margin on a pure-lateral goal and not enough on a diagonal goal.

The prior docs (CLOSE; cascade-north-bias-FIX) attributed this to "the far goal-blob is dim/small in the 16×16
`sc_map`." That is directionally right but **understated** — the next probe shows the blob is not dim, it is **absent.**

### 1c. The ROOT — the egocentric render CLIPS the goal entirely off-image at grid-32 (the genuine residual, quantified)

`render_egocentric_goal((x,y),(gx,gy), image_size=32)` (the SC's eye drive, called at `g11_bg_runner.py:7042-7050`)
paints the goal as a `radius=2` (5×5-pixel) blob at pixel `c + (goal − agent)·ppc` with the defaults **`ppc=4`,
`radius=2`, `image_size=32`** (`g11_bg_runner.py:183`). The render loop only writes pixels with `0 ≤ px,py < 32`. A
5-line CPU replay of that exact function, for the four schedule goals with the agent at the documented top-edge pin
(row ~31) AND at the foveal-neutral centre:

| goal (cell) | blob centre pixel | on 32×32 image? | pixels painted | `sc_retina` mass |
|---|---|---|---|---|
| NE (30,30) | (72, 12) / (72, 72) | **NO** | **0 / 25** | **0.0** |
| far-W (1,30) | (−44, 12) / (−44, 72) | **NO** | **0 / 25** | **0.0** |
| SW (1,1) | (−44, −104) / (−44, −44) | **NO** | **0 / 25** | **0.0** |
| SE (30,1) | (72, −104) / (72, −44) | **NO** | **0 / 25** | **0.0** |

**The result is decisive: at grid-32, all four schedule goals render ENTIRELY off the 32-pixel egocentric retina —
0/25 pixels, retina mass 0.0, for every goal, at both agent positions.** The representable window is image-half-width /
ppc = 16/4 = **±4 cells**; including the blob radius, anything beyond ~6 cells clips fully off-image. The grid-32
schedule places goals at the far corners (30+ cells away), so for the goals that matter the SC has **no input and
therefore no bump.** The pop-vector decode, the divnorm, the WTA ring, the tie-break, and any sel-stage operator are all
acting on an **empty SC map** — they cannot manufacture a goal-direction signal that the retina never received.

**Why phase1 (far-W) nonetheless tracks:** it is a *pure-lateral* goal. Once the agent (under the unbiased tie-break)
random-walks to within ~4 cells of the goal's *x* or *y* line, the blob re-enters the retina along that axis and a
1-D margin appears — enough to home in on a single-axis goal. The diagonal goals need *two* axes simultaneously inside
the ±4-cell window, which essentially never co-occurs from a far start, so they never acquire a stable bump. This
exactly reproduces the CLOSE doc's sharpest observation — "x DOES drift west under the SC signal (combo phase-2 reached
x=11) but y stays glued at row ~31" — the agent can correct ONE axis when that axis's blob is briefly on-retina, but
cannot hold a two-axis (diagonal) bump.

### 1d. The other candidate loci, ruled out as the binding constraint

- **(i) SC output / decode geometry** — the pop-vector cosine decode is built and verified correct
  (`install_spiking_sc_wiring(popvector=True)`, `g11_bg_runner.py:287-296`); it is not the binding constraint because
  it has nothing to decode when the bump is absent (1c).
- **(ii) `sc_map → cortex_X` projection strength** — the `SC_CORTEX_W` drive sweep (18 → 60 → 150) is a documented
  NEGATIVE that *saturates* (`2026-06-20-nav-sc-drive-reorient-derisk.md`); a stronger projection of an absent/empty
  bump cannot create position information (more gain on zero is zero).
- **(iv) common-mode at the accumulator** — exonerated in 1a.

**⇒ The genuine residual, pinned to the byte: the egocentric render's LINEAR `ppc=4` mapping over a FIXED 32-pixel
field clips every eccentric (> ~4-cell) goal off-image, so the SC has no bump for the grid-32 schedule's goals.** The
"signal too weak at the sel ring" framing is correct in its conclusion (the orienting does not reach selection) but the
*mechanism* is one stage further up than the SC→sel projection: the signal is not weak, it is **absent at the SC's own
input** for far goals. This is the smallest, most upstream, and most fixable description of the residual.

---

## MOVE 2 — REFRAME: how does real biology route a STRONG, SELECTIVE goal-direction signal to action selection?

The project has been testing the WRONG stage. Patching the accumulator (FIX-A/B), then the read-out geometry
(pop-vector + divnorm + cortex-WTA, all NEGATIVE at grid-32), then the sel ring again — all assume the orienting
*signal* exists at the SC and the problem is how to *select* on it. Biology says the failure is upstream: the SC's
**input representation** is the thing that, in a real brain, guarantees an eccentric target is still represented.

### 2a. The SC retinotopic map is LOG-POLAR / foveal-magnified — peripheral targets are compressed, never clipped

The decisive biological fact (verified against current literature + the catalog):

- The intermediate/deep SC holds a **retinotopically organized saccadic map** where **eccentricity is mapped along the
  rostral-caudal axis with strong foveal magnification** — the foveal/central representation is hugely over-represented
  "at the expense of peripheral locations," but the periphery (out to ~45-90° eccentricity) is **still represented**,
  just compressed (log-polar). The standard Ottes-Van Gisbergen-Eggermont afferent mapping is logarithmic; the Hafed
  lab's 2019 model and the human-SC eccentricity work confirm small eccentricities are magnified and the map covers the
  full hemifield (Sources below). The mapping is **non-linear** — over-representation of small saccades and of polar
  angles near the horizontal meridian.
- Catalog **E.04** (Topographic / retinotopic maps, Kandel 6e Ch 17): maps are explicitly **"warped by behavioral
  importance (cortical magnification — fingertips, fovea)"** — the magnification is the canonical property, and a
  *linear* map is the non-biological special case. Catalog **H.25** (SC saccade map): the SC is the topographic map of
  "saccade target relative to fovea" covering the visual hemifield; "stimulating a SC site evokes a saccade of fixed
  amplitude/direction matching that site" — eccentric sites exist and drive large saccades.

**The rig's render is the non-biological special case.** A linear `ppc=4` over a fixed 32-pixel field is a *flat,
truncated* retina: it has no foveal magnification AND it has a hard ±4-cell horizon past which the target simply
vanishes. A log-polar SC retina would place a 30-cell-distant goal at a compressed peripheral site (still inside the
map), with a clean bump the decode can read. **We have been testing whether a flat-and-truncated SC can re-orient; the
biology-faithful SC never truncates.** That is the reframe: the wrong stage was being patched (selection / decode) when
the **input representation** (foveal-magnified, full-hemifield) is the missing biology.

### 2b. The SC IS the priority/salience map that feeds selection — not a sensor that hands off to a downstream BG re-derivation

A secondary reframe, lower-leverage but worth recording. In biology the SC intermediate/deep layers ARE the orienting
*priority map*: "the SC integrates visual + auditory + cognitive inputs into a 'where to look next' decision; output →
pontine saccade generator" (catalog H.25), with **buildup cells** that ramp to a movement command and selection by
**SNr disinhibition** (catalog A.07: SNr → SC tonic inhibition, released to permit the saccade). The orienting decision
is *in* the SC map and gated *out* by the BG; it is not re-derived by a four-pool argmax downstream of a long
cortico-striatal cascade. The rig routes the SC drive through `sc_map → cortex_X → str_D1_X → … → thal_X → sel_X` — a
long path where the (already weak, or absent) SC margin is swamped by the cascade's own dynamics before the WTA ring
sees it (the CLOSE doc's `[40,40,40,40]` saturation). This is a *placement* mismatch (the competition is too far
downstream of the salience map), and it is the reason the cortex-WTA sweep helped a little (it moved competition
earlier) but not enough. It is, however, the *second-order* issue: with no bump (2a) there is nothing to route, so
fixing the input representation must come first.

### 2c. Re-orienting in biology uses a fixation/omnipause RESET — relevant only AFTER the bump exists

The earlier scoping flagged Option E (collicular fixation-zone / inhibition-of-return reset on goal change; Munoz;
catalog A.07). The grid-32 CLOSE diagnosis explicitly found the residual is **swamping, not hysteresis** (the bump
re-renders fresh each step), so Option E is NOT indicated for the *current* residual — and the present analysis sharpens
that: there is no persisted-old-bump to reset because for far goals there is **no bump at all.** Option E remains a
reserve for a *future* residual that may surface once the bump exists (a real log-polar bump attractor can have
hysteresis), but it is not the current lever.

---

## MOVE 3 — RANK the cheap-first SURPASS mechanisms (the path PAST the residual)

All ranked by leverage × cheapness × reuse, and by *directness of attacking the genuine residual* (the absent bump for
far goals). Each: the mechanism, the reusable project machinery, the cheap-first de-risk, the anti-cheats, and whether
it needs a protected `sim/` edit.

### RANK 1 (RECOMMENDED) — biology-faithful eccentricity remapping of the egocentric SC render (log-polar / foveal-magnified retina)

- **Mechanism.** Replace the linear, hard-clipping `render_egocentric_goal` mapping `pixel = c + Δcell·ppc` with a
  **monotone compressive (log-polar) mapping** that keeps every eccentric goal *inside* the 32-pixel retina: e.g.
  `r_pixel = R_max · log(1 + |Δcell|/d0) / log(1 + R_cell/d0)` along the goal's bearing, with a foveal scale `d0` that
  magnifies small eccentricities and a `R_max` that caps the most-eccentric goal at the retina edge (never past it).
  This is the SC's canonical afferent map (Ottes-Van Gisbergen): central goals get a large, sharp central bump
  (preserving the fine near-goal discrimination FIX1 already exploits), peripheral goals get a compressed-but-present
  peripheral bump (so a 30-cell goal lands ON the map). The 16×16 `sc_map` and the whole downstream decode/divnorm/WTA
  stack are unchanged — only the position-encoding of the input is made biology-faithful.
- **Why it attacks the genuine residual directly.** It converts retina mass from **0.0** (current, for all four
  schedule goals) to a clean bump for every goal — restoring the very signal the 3-run convergence proved is missing.
  Everything downstream that already works on a strong margin (FIX1 reaches host-level finalQ + SCRAM collapses on the
  one pure-lateral goal that currently *does* render) should now have a margin on the diagonal goals too.
- **Reusable machinery.** `render_egocentric_goal` (`g11_bg_runner.py:183`) — a ~10-line runner-side weight/position
  formula change; `install_spiking_sc_wiring` (the SC sheet + Mexican-hat bump + pop-vector decode, all unchanged); the
  FIX1 tie-break (keep ON); the `_nav_sc_popvector_readout_derisk.py` harness with HOST / sc_popvector / SCRAM arms; the
  `SC_SCRAMBLE` lesion; the grid-32 goal schedule. The pop-vector decode is the *correct* decode for a log-polar input
  (it reads bump *bearing*, which the log-polar map preserves; it compresses only *eccentricity*, which the cardinal
  decode discards anyway).
- **Cheap-first de-risk.** (1) **Render-unit smoke (CPU, seconds, no GPU):** replay the new `render_egocentric_goal`
  for the four schedule goals at agent (16,31) and confirm retina mass > 0 and the bump *bearing* is correct (NE blob in
  the NE quadrant, etc.) for every goal — i.e. the off-image clipping is gone. (2) **grid-32 faithful confirm
  (GPU, the verdict):** the EXACT FIX1 NEURAL config (`_nav_sc_popvector_readout_derisk.py`, grid-32 / 1800 /
  warmup-600, the merged-het-off SC op-point, FIX1 ON, pop-vector decode, the #4 WTA ring) with the log-polar render,
  vs HOST and vs SCRAM. **PASS = the per-phase dom-cardinal now tracks on the DIAGONAL phases too (not just far-W), the
  post-change Σ drops materially toward HOST, AND SCRAM clearly collapses.**
- **Anti-cheats (all carried + the new one).** (a) **HOST positive control** (centroid+argmax) anchors the gap. (b)
  **SCRAM lesion MUST collapse** relative to the intact log-polar decode (the discriminator the accumulator fixes never
  passed) — proves the orienting is carried by the *retinotopic* decode, not a cascade prior. (c) **Per-phase
  per-cardinal action distribution must TRACK the moving goal on the diagonal phases**, not stay fixed. (d)
  **Matched-everything-else** (same `SC_CORTEX_W`, same divnorm, same tie-break) so any lift is attributable to the
  render geometry, not a covert drive/calibration change. (e) **NEW — the magnification must not smuggle (x,y):** the
  render takes ONLY `(agent, goal)` egocentric bearing exactly as now (it is still the legitimate environment "render
  the agent's sensory input" operation, channel-1 of the BRAIN-BASED-ONLY bar; a compressive *visual* mapping is what a
  real retina/SC does, not a coordinate read-out) — assert the decode still consumes only `sc_map` firing, never
  `(gx,gy)`. (f) **grid-32, never grid-8** (the documented false-GO scale). (g) **tie-fraction reported** (a GO needs
  the diagonal decisions driven by the SC margin, few ties — not a lucky random-walk). (h) **6-seed on a GO** (the
  standing rule). (i) **moat untouched** (no conversational regions in the nav run; the nav cascade is array-disjoint
  from `cp_rf_w_*`). (j) **FIX1-OFF/render-OFF byte-identical** (default-off guard; regression `test_nav_conv_merged_agent`
  8/8 + `test_nav_conv_step2b_coresident` 7/7).
- **`sim/` edit?** **NO.** `render_egocentric_goal` and `install_spiking_sc_wiring` are in the research runner
  (`g11_bg_runner.py`), not protected `sim/`. A runner-side render-position formula change behind a default-off kwarg
  (parallel to `sc_popvector_readout`). This is the cheapest possible path PAST the residual *and* needs no protected
  edit — the strongest combination.

### RANK 2 — enlarge the egocentric retina field (raise `visual_image_size` and/or lower `ppc`) so far goals fit

- **Mechanism.** The cheapest *blunt* version of RANK 1: keep the linear map but widen the window so 30-cell goals fit —
  e.g. drop `ppc` from 4 to ~1 (so ±16 cells fit in 32 pixels), or raise `visual_image_size` so ±30 cells fit. This
  removes the clipping without the log-polar non-linearity.
- **Why ranked below RANK 1.** It trades the foveal magnification away — a `ppc=1` map spreads the whole grid across the
  retina, so the *near-goal* discrimination FIX1 currently relies on (the sharp central bump when the agent is close)
  *coarsens*, and a far goal becomes a faint few-pixel blob (the "dim/small bump" the prior docs feared, now actually
  realized rather than absent). It is the un-magnified flat map at a larger scale — it fixes clipping but reintroduces
  the SNR worry at the periphery that log-polar magnification specifically solves. It is biology-INFAITHFUL (real SC is
  magnified, not uniformly scaled). Raising `visual_image_size` also enlarges the `sc_map`/retina neuron count
  (`2·N²`), a GPU/memory cost.
- **Reusable machinery / de-risk / anti-cheats.** Identical harness and anti-cheats to RANK 1 (it is the same
  experiment with a different render). Useful primarily as an **ABLATION** alongside RANK 1: if a uniformly-enlarged
  linear retina (RANK 2) tracks far goals but worse than log-polar (RANK 1), that attributes the lift specifically to
  the *magnification*, not merely to *un-clipping*. `visual_image_size` is an existing kwarg (`g11_bg_runner.py:760`);
  `ppc` is the `render_egocentric_goal` arg. **NO `sim/` edit.**

### RANK 3 — move the competition EARLIER (feed the SC bump closer to the sel ring / sharpen the cortex-WTA on a real bump)

- **Mechanism.** The 2b placement reframe: with a real bump present (after RANK 1), re-test whether routing the
  pop-vector drive closer to `sel_X` (or the inter-cardinal cortex-WTA) now wins, instead of being swamped over the long
  `cortex_X → str_D1 → … → thal_X → sel_X` path. The cortex-WTA at FS-weight 8 already *broke the phase-0 N-pin* in the
  CLOSE sweep (dom flipped to E) — proving the competition-placement direction is right; it just had no margin to
  sharpen because the bump was absent.
- **Why ranked below RANK 1/2.** It is the *second-order* fix — it presupposes a bump to compete on. On the current
  empty-bump input it is a documented NEGATIVE (the CLOSE cortex-WTA sweep at three strengths). Its value is conditional
  on RANK 1 first; it is the natural *follow-on* if RANK 1 un-clips the bump but the diagonal margin is still swamped by
  the downstream cascade.
- **Reusable machinery.** `enable_cortex_lateral_inhibition` + the FS-weight knobs (`g11_bg_runner.py:945-961,
  3250-3255`); the `sel_X`/`commit_X` ring. **NO `sim/` edit** (builder flags). De-risk/anti-cheats as RANK 1, run on
  top of the RANK-1 render.

### RANK 4 (reserve) — Option E goal-change fixation/omnipause reset

- Only if RANK 1 surfaces a *new* residual where a real log-polar bump attractor exhibits hysteresis on goal change
  (the current residual is swamping/absence, NOT hysteresis — 2c). Machinery exists (`sc_rostral` foveation read-out,
  `g11_bg_runner.py:311-335`, as the on-substrate goal-change detector; a transient inhibitory current into
  `sc_map`/`sel_X`). **NO `sim/` edit** (a runner-side current injection). Held in reserve; not the current lever.

---

## MOVE 4 — VERDICT: SURPASSABLE, and the cheapest path is the log-polar render (RANK 1)

**The boundary is SURPASSABLE; it survives the SURPASS round as a precisely-located, non-irreducible residual with a
cheap, biology-faithful, no-`sim/`-edit fix.** The accumulator was correctly exonerated (FIX-A/FIX-B NEGATIVE, the
operator removes the wrong thing); the upstream re-location was correct in *direction* (the orienting signal does not
reach selection) but the *mechanism* is one stage further up than "the SC→sel projection is too weak" — it is that **the
egocentric render clips every eccentric goal off the 32-pixel retina, so for the grid-32 schedule's far-corner goals the
SC has no bump at all** (retina mass 0.0, quantified). No selection-stage, decode-stage, or projection-strength
operation can act on an absent signal — which is exactly why seven prior read-out/sel-stage mechanisms (pop-vector +
divnorm, drive sweep, cortex-WTA ×3, FIX-A, FIX-B) all converged on the same NEGATIVE.

**The genuinely-irreducible part is TINY and is NOT a substrate limit:** the only thing "irreducible" about the current
setup is that a *flat, truncated* retina cannot represent a target outside its window — and that is a non-biological
modeling choice, not a point-neuron limit. The biology-faithful SC retina (log-polar foveal magnification) represents
the full hemifield without truncation; once the bump exists, the rest of the stack already demonstrably works on a
strong margin (FIX1 reaches host-level finalQ AND SCRAM collapses on the one goal that currently renders). There is no
dendritic frontier here and no graded-read-out / point-neuron-limit family wall — it is a **representation-coverage**
fix.

**Recommended rank-1 de-risk (the precise next move):** implement the log-polar / foveal-magnified mapping in
`render_egocentric_goal` behind a default-off kwarg; (1) CPU render-unit smoke to confirm retina mass > 0 + correct
bearing for all four schedule goals at the pinned agent; then (2) the grid-32 faithful confirm
(`_nav_sc_popvector_readout_derisk.py`, FIX1 ON + pop-vector + the #4 WTA ring + the log-polar render) against HOST and
SCRAM. **GO bar = the per-phase dom-cardinal tracks on the DIAGONAL phases (not just the pure-lateral far-W), the
post-change Σ drops materially toward HOST, AND SCRAM clearly collapses** (the decode is load-bearing on the diagonals).
On a seed-42 GO, run the 6-seed confirmation. The host orienting heuristic retires only if the spiking SC, with a
biology-faithful input representation, re-orients within the deploy bar across seeds — a question this rank-1 de-risk
answers and the accumulator fixes never could.

**This is the canonical deep-research-at-a-boundary outcome: a comfortable "the substrate's orienting signal is just too
weak / honest-negative" verdict was the START of the research, not the end. The ISOLATE step pinned the residual to a
specific 10-line render formula (retina mass 0.0, measured); the REFRAME identified the right upstream stage (the SC's
log-polar input representation, not the selection it feeds); and the RANK named a cheap, no-`sim/`-edit, biology-faithful
fix. The boundary is surpassable.**

---

## Provenance + machinery (file:line, for the controller's trust-but-verify)

- **The residual, quantified:** `render_egocentric_goal` (`g11_bg_runner.py:183`, defaults `ppc=4, radius=2,
  image_size=32`); the SC eye-drive call site (`g11_bg_runner.py:7042-7050`, `image_size=int(visual_image_size)`,
  default 32); the off-image clip (the render loop's `0 <= px,py < image_size` guard). CPU replay (this session):
  retina mass **0.0** for all four schedule goals (NE/far-W/SW/SE) at agent (16,31) and (16,16); representable window =
  ±4 cells (16px / 4 ppc).
- **The accumulator exoneration:** `scpv_FIXA_arm3_seed{42,43}.json`, `scpv_FIXB_arm3_seed42.json` (FIX1 best spiking;
  FIX1A/B worse; SCRAM ≈ FIX1A/B); `2026-06-20-shortcut6-FIXA-divnorm-accumulator.md` (the FIX-A + FIX-B verdicts + the
  3-run convergence).
- **The post-tie-break per-phase pattern:** FIX1 tracks only the pure-lateral far-W phase (finalQ ~1.0-1.3) and fails
  the three diagonal phases (20-47) in all three runs; `2026-06-20-cascade-north-bias-FIX.md` (FIX1 tracks + SCRAM
  collapses 3/3, the margin-SNR residual named).
- **The decode + bump machinery (unchanged by RANK 1):** `install_spiking_sc_wiring(popvector=True)`
  (`g11_bg_runner.py:287-296`, cosine projection); the `sc_map↔sc_fs` Mexican-hat bump; `sc_rostral` foveation read-out
  (`:311-335`, the Option-E reserve trigger); the `SC_SCRAMBLE` lesion (`:244-249`); the #4 `sel_X`/`commit_X` WTA ring
  (`:446-479`); the harness `_nav_sc_popvector_readout_derisk.py`.
- **Biology (verified):** catalog **E.04** (topographic maps "warped by behavioral importance — cortical
  magnification — fovea"), **H.25** (SC saccade map, full-hemifield "where to look next"), **A.07** (SNr→SC
  disinhibition gate). Literature: SC log-polar / foveal magnification — eccentricity along the rostral-caudal axis,
  strong foveal magnification, full-hemifield coverage, non-linear mapping (Ottes-Van Gisbergen-Eggermont; Hafed lab
  2019; human-SC eccentricity work). Sources:
  - [The foveal visual representation of the primate superior colliculus (Current Biology / bioRxiv 2019, Hafed lab)](https://www.biorxiv.org/content/10.1101/554121v1.full)
  - [New model of superior colliculus topography — Hafed Lab (2019)](https://hafedlab.org/2019/04/24/new-model-of-superior-colliculus-topography/)
  - [Topography of covert visual attention in human superior colliculus (J Neurophysiol)](https://journals.physiology.org/doi/full/10.1152/jn.00283.2010)
  - [Eccentricity-dependent saccadic reaction time: foveal magnification and attentional orienting (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12281140/)
  - [Polar-angle representation of saccadic eye movements in human superior colliculus (bioRxiv)](https://www.biorxiv.org/content/10.1101/169003.full.pdf)

_READ-ONLY SURPASS scoping. This doc is the only write; no code edited, no protected `sim/` touched. grid-32 IS the
verdict (never grid-8). The no-confab moat is array-disjoint from the nav cascade and untouched. Load-bearing claims
cited to `g11_bg_runner.py` line numbers + the existing result JSONs + the catalog + the SC-magnification literature;
the render-clipping residual was confirmed by a CPU replay of the exact render function this session._
