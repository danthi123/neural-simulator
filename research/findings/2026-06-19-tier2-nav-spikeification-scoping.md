# Tier-2 scoping — the navigation sensorimotor half's remaining HOST shortcuts → spikes (deploy the validated spiking SC onto the merged "one brain"; the orienting/reward organ is the highest-leverage lever)

**Date:** 2026-06-19
**Type:** READ-ONLY deep-research + catalog scoping (no code edited, no jobs run). Tier-2 thread, scoped in parallel. **This is NOT a directive to build now** — per the owner's TRUE-ONE-BRAIN ordering ([[feedback_move_everything_to_shared_spiking_substrate]]) the SHARED limbic systems (reward/value/dopamine, in flight as #6 limbic→composer) come FIRST; the navigation sensorimotor loop is scoped here so it is ready when its turn arrives.
**Owner standard:** BRAIN-BASED-ONLY ([[feedback_brain_based_only_standard]]) — any cognitive computation *between sensation and action* done by host (non-neural) code is a shortcut, even if the host formula is biologically correct. Host code is legitimate ONLY for (1) the environment (world state + rendering the agent's sensory input) and (2) the body (acting on the motor output).

---

## 0. The single most important framing correction (read this first)

**The premise that the navigation half still needs a spiking superior colliculus BUILT FROM SCRATCH is out of date.** The spiking SC for orienting (N1) was **already built, de-risked, and 6-seed validated GO on the standalone nav** on 2026-06-10 (`2026-06-10-N1-spiking-superior-colliculus-CLOSED.md`: SC-on mean 3.607 vs host-reflex 4.085 = SC/host 0.883, 12% BETTER, 5/6 seeds win, scrambled-retinotopy lesion regresses 2.4×). The neural approach-reward (N5) was de-risked on the merged bridge on 2026-06-18 (`2026-06-18-merged-limbic-core-lift.md`: merged-tuned SC op-point, corr(ecc, reward_us) = −0.81, SNc burst 1.45×, lesion collapses).

So the **highest-leverage remaining navigation spike-ification is not a new organ — it is DEPLOYING the already-validated spiking SC (orienting + neural-approach-reward) as the DEFAULT on the merged "one brain" bridge**, where navigation *currently still orients with the host Manhattan heuristic and rewards with the host `sign(distance)` formula*. The pieces exist (`enable_spiking_sc`, `enable_spiking_sc_approach`, `nav_critic_spiking_sc`); they are wired but **default-off and not yet 6-seed validated as the merged default**, and they carry a documented **co-residence operating-point risk** (the SC bump "starves" on the heterogeneity-OFF merged bridge). That deployment-and-validation is the recommended scoped target below. This is honest and concrete: the science is largely done; the remaining work is integration + a 6-seed A/B + op-point hardening.

---

## 1. The remaining-host-shortcut map (nav cognitive computations: HOST vs SPIKING)

A crucial distinction runs through this table: the runner has a **LIBRARY default** (the function-signature default used when the merged "one brain" calls `run_moving_goal_episode` directly) and a **CLI default** (the argparse default for the documented standalone `python -m research.runners.g11_bg_runner` benchmark). They differ for the action read-out, and the merged bridge inherits the LIBRARY defaults.

| Nav cognitive computation | HOST or SPIKING (default) | Flag = default | file:symbol/line | Note |
|---|---|---|---|---|
| **Action selection / read-out** | **SPIKING (lib) / HOST (CLI)** | `readout_source`: lib `"spiking_wta"` `g11_bg_runner.py:3734`; CLI `"motor"` `:7986` | decision `:6968-6988` | Wang-2002 accumulator + Lo-Wang commit-burst; the host `max()` only OBSERVES which `commit_X` pool bursted. **Merged bridge = spiking by default.** Default-on flip 2026-06-19 (CLAUDE.md headline). **CLOSED.** |
| **Dopamine / RPE (δ = r − V)** | SPIKING when on | `spiking_snc=False` `:3262`; merged forces it via `nav_critic_*` | build `:4304`, drive `:5696` | `snc` (IZH2007_DOPAMINE) firing encodes δ; host RPE bypassed. **Validated; closed on the path that turns it on.** |
| **Value / critic V(s)** | SPIKING when on | `enable_neural_critic=False` `:324`; merged forces `True` `nav_conv_merged_bridge.py:582` | build `:4646`, drive `:6605` | `striosome_value` MSN learns V; subtracts at SNc via GABA_B/GIRK. **Closed on the merged path.** |
| **Reward DELIVERY (US→SNc burst)** | SPIKING when on | `spiking_reward_us=False` `:330`; merged forces `True` `:582` | wire `:2541`, drive `:5699` | `reward_us` (PPN-like) FIRES into SNc = the reward burst is a synapse. |
| **Reward CONTENT (r = "did the goal get closer?")** | **HOST** (the live shortcut on the merged bridge) | host `sign(dist_after − dist_before)` (else-branch, no flag) `:7055-7061`; image-graded host `perceived_approach_reward=False` `:7016-7040`; **neural** = `sc_rostral`→`reward_us` via `enable_spiking_sc_approach` | reward logic `:7016-7061` | The *value* "the goal got closer" is host arithmetic. The **neural** version (SC rostral-pole proximity → `reward_us`) is **de-risked but default-off**; on the merged bridge the host `sign(Manhattan)` still drives `reward_us` (`nav_conv_merged_bridge.py:568`). **OPEN (deploy).** |
| **ORIENTING / salience (which way to the goal)** | **HOST** (the live shortcut on the merged bridge) | host heuristic `heuristic_strength=1.0` (always on unless weaned) `:6386-6411`; host pixel-reader `sc_orienting_reflex=False` `:6747-6770`; **neural SC** `enable_spiking_sc=False` `:2488`/`:6730` | host fn `sc_orienting_cardinal_from_image` `:124-155` | **The spiking SC EXISTS + is 6-seed GO standalone**, but is **OFF on the merged bridge**, which orients with the host Manhattan heuristic. **OPEN (deploy) — HIGHEST LEVERAGE.** |
| **Place / position code** | **HOST** (a characterized substrate BOUNDARY) | host Gaussian goal/critic bumps (default); `neural_place_selforg=False` `:388` (only meaningful with the critic) | self-org build `:4095`, drive `:6621` | Self-org place LEARNS a V gradient and COMPOSES, but its read-out δ underperforms the host Gaussian (δ ~1.04 vs host ~1.3): **honest NEGATIVE / dendritic-flavored wall** (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`). **OPEN but BLOCKED on substrate; not the cheap win.** |
| **Perception (object/where code)** | SPIKING when on (defensible-host front end) | `enable_visual_cortex=False` `:692`; merged forces it via `nav_critic_spiking_sc` | build `:2428`, drive `:6700` | Gabor V1 → V2 → `cortex_it`, all spiking. The image *render* (`render_gridworld_to_image`) is legitimate environment/sensory rendering (the bar's channel-1). **N7 defensible; closed.** |
| Exploration / last-resort tie-break | HOST (negligible) | none (only fires when BOTH commit AND sel silent) | `:6986-6987` | A genuinely-undriven-trial fallback; not a cognitive computation worth converting. |
| Body movement (`np.clip` move) | HOST — **legitimate** | n/a | `:6999-7007` | The body acting on the motor decision = channel-2 of the bar. Not a shortcut. |

### Leverage ranking of the remaining OPEN host shortcuts

1. **ORIENTING (the host Manhattan heuristic on the merged bridge) → the spiking SC orienting read-out.** Highest leverage: the heuristic is the agent's *primary goal-direction teacher*, it reads raw `(gx, gy, x, y)` (the most flagrant remaining cheat), the spiking replacement is **already built + 6-seed-GO standalone**, and the same organ also carries the reward signal (#2). Closing it on the merged bridge is the difference between "the one brain navigates by a host distance heuristic" and "the one brain orients by a spiking retinotopic map." **← the depth target of this doc.**
2. **REWARD CONTENT (the host `sign(Manhattan)` on the merged bridge) → the SC rostral-pole neural approach-reward.** Same organ, same deployment — N1 and N5 are two read-outs of one SC bump (orienting = *where* the bump is; approach = *how* the bump moves toward the rostral/foveal pole). De-risked on the merged bridge already; ships with #1.
3. **PLACE code → self-organized sparse place cells.** Real but **blocked on a substrate wall** (point neurons can't carve many distinct location-selective sparse fields from overlapping egocentric landmark sensors; the read-out δ over-clamps or can't discriminate). The host Gaussian stays the better-δ scaffold; this is the documented honest NEGATIVE, not a cheap engineering win. Re-prioritize only with the dendritic substrate ([[feedback_dendritic_substrate_fair_game]]) or a fundamentally different graded read-out.

---

## 2. Mechanism map for the highest-leverage target: the spiking superior colliculus (orienting + approach), and the merged-bridge deployment

### 2a. What the spiking SC does (the salience-map → WTA → orient mechanism — the crux)

The superior colliculus (catalog **A.07** SNr→SC orienting `feature-catalog.md:169-179`; **H.25** "SC saccade map — topographic motor map" `:3209-3219`; **H.24** EBN/IBN/OPN saccade generator `:3197-3207`) holds a **single retinotopic map** that does two things from one activity bump:

- **(N1) orienting** = *where the bump is* → which way to move.
- **(N5) approach reward** = *how the bump moves toward the rostral/foveal pole over frames* → "the target is foveating / getting closer" (Munoz & Wurtz 1993 rostral-pole fixation cells; NOT looming — this gridworld's goal blob doesn't expand, so the faithful homologue is eccentricity-decrease/foveation, not an LGMD collision detector — `2026-06-10-N1-N5-spiking-superior-colliculus-research.md` §1).

The mechanism, as built (`g11_bg_runner.py:2488-2544` build, `install_spiking_sc_wiring` `:201-303` post-init wiring, `:6730-6739` per-step drive):

1. **Egocentric retina (the salience input).** A dedicated `sc_retina` (2·W·W neurons, the SC's OWN eye, separate from the allocentric `retina` the visual cortex/critic use) is driven each nav step by `render_egocentric_goal((x,y),(gx,gy))` — the goal painted as a dim ON blob at its *bearing* relative to the foveal centre (`:183-198`). This is legitimate environment rendering (channel-1); the *cardinal/eccentricity* is NOT computed host-side.
2. **Retinotopic salience map.** `retina_to_sc_map`: each `sc_map` site (a 16×16 sheet for image 32) pools its 2×2 ON block → the goal blob lights up the matching SC site. This is the retinotopic projection that the host centroid-read replaces.
3. **Mexican-hat winner-take-all.** `sc_map` short-range recurrent excitation (radius-1) + a `sc_fs` inhibitory surround (`sc_map↔sc_fs`, declared with REAL density so `inject_explicit_wiring` marks `sc_fs` INHIBITORY — a documented gotcha: a density-0 + `set_pathway_weights` route leaves the trait mask unset and `sc_fs` acts excitatory and floods the map). Strong-short-range-excite + weak-long-range-inhibit = a dynamic soft WTA → a **single sharp activity bump** at the most salient site (Marino et al. spiking-SC model; Trappenberg neural-field SC).
4. **(N1) Topographic orienting read-out — BY FIRING.** `sc_map_to_cortex_{N,E,S,W}`: four fixed weighted-quadrant pooling pathways (the northern half of the sheet → `cortex_N`, etc., weight ∝ distance from sheet centre). Whichever SC quadrant wins drives the matching cortex action pool **synaptically** — the orienting cardinal is *which cortex pool fired*, not a host argmax over pixels. This replaces the host current injection that `sc_orienting_cardinal_from_image` performed.
5. **(N5) Approach read-out.** `sc_map_to_sc_rostral`: a broad-Gaussian foveal-centre pool that fires graded with how central/close the bump is → `sc_rostral → reward_us` (excitatory) gates the reward burst **neurally**, replacing the host `sign(eccentricity_after − eccentricity_before)`. (The temporal-difference "is it *getting* closer" is left to the dopamine RPE δ = r − V — the correct actor-critic factorization — so N5 only needs the proximity rate, validated by `sc_n5_rpe_probe.py`: corr(distance, SNc) = −0.99, omission dip, lesion `sc_rostral→reward_us` collapses it.)

The orienting drive then **competes** with the BG cascade + spiking-WTA accumulator inside the existing action-selection layer — the SC biases, the BG commits. This is biologically faithful (SC receives BG/SNr tonic inhibition; selection by SNr disinhibition — catalog A.07/A.56).

### 2b. Tie to SC biology

- **Salience map + topographic motor map:** catalog H.25 — "stimulating a SC site evokes a saccade of fixed amplitude/direction matching that site; the SC integrates visual + cognitive inputs into a 'where to look next' decision by winner-take-all." That is exactly steps 2-4.
- **The Mexican-hat soft-WTA** is the established spiking-SC mechanism (PMC5506246, PMC3704631, PMC4699154).
- **The rostral-pole approach read-out** is Munoz-Wurtz fixation/buildup cells; the population bump sits rostral as a target foveates.
- **Residual idealization (documented, NOT a hidden cheat):** the `sc_map → cortex_X` topographic read-out is a *fixed*, genetically-specified-style projection (chemoaffinity / ephrin-Eph map formation), not a learned map — the same accepted status as the innate V1 Gabor RFs (N7). No cognitive quantity is host-computed; it is innate structure, which the bar permits.

### 2c. The named reuse (everything already exists)

- **The neural retina + visual hierarchy** — `sim/visual_cortex.py` (`render_gridworld_to_image`, `image_to_retina_drive`, the Gabor V1 bank `build_v1_simple_weights`/`apply_v1_gabor_weights`). The SC's egocentric eye reuses the same render.
- **Retinotopic 2D sheets + topographic connectivity** — `sim/regions.py` `BrainRegion.coordinate_dim/coordinate_extent` + `RegionPathway.distance_sigma` Gaussian connectivity. NO `sim/` edit needed for the map.
- **The saccade-WTA / commit-burst idiom** (the orienting selection, in spikes) — `g11_bg_runner.py:1959-2068` `sel_X`/`sel_FS_X` (Rutishauser selective inhibition)/`commit_X` (Lo-Wang burst)/`commit_OPN` (omnipause) — cited in-code as "SC / saccade-generator EBN analogue, H.24/H.25." The read-out convention ("the host argmax merely OBSERVES which commit pool bursted") is the anti-cheat template.
- **The N1 host reflex as the innate-teacher scaffold** — `sc_orienting_reflex` + the wean ramp (`sc_reflex_wean_start/_steps`) is the project's reflex-teaches-a-learned-circuit pattern; the spiking SC is the matured organ it hands off to.
- **The already-spiking reward leg** — `reward_us`→`snc`→GABA_B striosome critic; N5 only changes *who* fires `reward_us` (the `sc_rostral` pool instead of the host scalar).
- **The slow channels for any TD refinement** — `nmda_slow`, `gaba_b`/GIRK, `coincidence_detector` are merged + runner-enabled (`cfg.enable_gabab` etc.).

---

## 3. Reuse-vs-new + the `sim/` edit surface

**Net new protected-`sim/` surface for the recommended deployment: ZERO.** Every piece is runner-side region/pathway vocabulary over already-merged machinery (confirmed by the standalone CLOSED finding: "ZERO protected `sim/` edits"). The build, the Mexican-hat WTA, the afferent, the orienting read-out, the approach read-out, and the merged-bridge hook (`nav_critic_spiking_sc`) all already exist.

What "deploy onto the merged one brain" concretely means (all runner-side, in `nav_conv_merged_bridge.py` + env-var op-points already present in `g11_bg_runner.py`):

| Piece | Status | What deployment needs |
|---|---|---|
| Spiking SC region build (`sc_retina`/`sc_map`/`sc_fs`/`sc_rostral`) | EXISTS, forwarded by `nav_critic_spiking_sc` (`nav_conv_merged_bridge.py:580-587`) | Flip the merged default ON after the 6-seed A/B passes. |
| `install_spiking_sc_wiring` (retinotopy + Mexican-hat + quadrant pooling + rostral) | EXISTS (`:201-303`), called post-init | Ensure it runs AFTER the merged bridge's V1/SC post-init CSR rebuild (the `finalize_conv_for_nav_gate` index-discipline pattern, `nav_conv_merged_bridge.py:1046`). |
| Het-off op-point (the co-residence starvation fix) | EXISTS as env vars `SC_RET_DRIVE`/`SC_RET_SC`/`SC_REC`/`SC_ROS_US`/`SC_CORTEX_W` (`g11_bg_runner.py:4433-4459`, `:6736`) | Promote the merged-tuned values (de-risked 160/12 + drive 3500 + `sc_rostral→reward_us` 40 — `2026-06-18-merged-limbic-core-lift.md`) from env-var to the merged builder's default so it is reproducible without the env. This is the ONE small additive runner change; default-off byte-identical to standalone. |
| Replace the host orienting (heuristic) + host reward (`sign(Manhattan)`) on the merged path | The host paths are still the merged default | Gate them OFF when `nav_critic_spiking_sc` is on (the orienting comes from `sc_map→cortex_X`, the reward `r` from `sc_rostral→reward_us`). |

**No new mechanism, no new `sim/` edit, no new rule** — this is a deploy-and-validate, exactly the kind of integration the standalone-vs-merged gate (`test_nav_conv_step2b_coresident`) was built for.

---

## 4. The cheapest-first de-risk (the SMALLEST load-bearing test) + GO bar

The standalone SC is already 6-seed GO and the merged-bridge SC approach-reward is already CPU-de-risked. So the cheapest-first de-risk for the *deployment* is a **two-step ladder**, smallest first, matching the project's established methodology:

### Step 0 (the smallest, ~minutes, CPU/single-seed GPU smoke) — co-residence op-point check

Before any multi-seed run, confirm the SC bump is NOT starved on the merged bridge at the promoted op-point: build the merged bridge with `nav_critic_spiking_sc=True` at the merged-tuned op-point (160/12, drive 3500, `sc_rostral→reward_us` 40) and assert, on a handful of hand-set (agent, goal) renders, that **(a) `sc_map` forms a clean single bump** (peak site ≫ background), **(b) the winning `cortex_X` by FIRING matches `sc_orienting_cardinal_from_image` on ≥ 7/8** positions, and **(c) `reward_us` crosses threshold and `corr(eccentricity, reward_us) < −0.6`**. This is the `sc_map_orienting_probe.py` / `sc_n5_rpe_probe.py` falsifiers re-run on the *merged* bridge (not the standalone) to catch the documented starvation. Cheap, fully diagnostic, and it gates the expensive run.

### Step 1 (the decisive test, GPU, 6 seeds) — merged-bridge nav A/B, SC-on vs host-heuristic+host-reward

On the merged "one brain" bridge, the moving-goal nav episode, head-to-head:
- **SC-on:** `nav_critic_spiking_sc=True`, host orienting heuristic + host `sign(Manhattan)` reward gated OFF.
- **Host control:** the current merged default (host heuristic orienting + host distance reward).
- **Metric:** `nav_sum` = Σ `final_quarter_mean_distance` over the 4 moving-goal phases (LOWER = better), the established metric.
- **Seeds:** 42/43/44/100/101/102 (6 seeds, [[feedback_6seed_validation]]).

**GO bar:** SC-on mean `nav_sum` ≤ 1.25 × host-control mean (the project's "within 25% of the host it replaces" deploy bar, consistent with the 2026-06-19 spiking-decision default-on GO at 1.16× host), with **no conversational regression** (`test_nav_conv_merged_agent` 8/8 + `test_nav_conv_step2b_coresident` 7/7 still pass — the SC slice is array-disjoint from the parser/composer, so the no-confab moat is preserved by construction). Per the BRAIN-BASED-ONLY standard, a clean honest NEGATIVE here (the spiking SC underperforms the host heuristic on the merged bridge by > 25%) **is the deliverable** — it maps the co-residence operating-point limit.

### Anti-cheat controls (must hold or the GO is rejected)

1. **Image-only afferent (provenance assertion).** `(x,y)`, `(gx,gy)`, Manhattan distance, and the host `sc_orienting_*`/`sc_salience_offset_*` outputs NEVER enter the SC drive — the SC reads only the egocentric render. (The render may use `goal_pos` — that is the world's visible goal, N2, legitimate.)
2. **Scrambled-retinotopy lesion (the decisive one).** `SC_SCRAMBLE=1` permutes the `sc_retina→sc_map` target assignment → orienting accuracy → chance AND `reward_us` sign-agreement → chance (standalone showed 2.4× regression). If navigation survives a scrambled map, the signal leaks from somewhere non-retinotopic → reject.
3. **Relay lesion.** Zero `sc_map→cortex_X` (orienting) and `sc_rostral→reward_us` (reward) → the corresponding capability vanishes (proves synaptic transmission carries it, not host arithmetic).
4. **Winner-by-firing.** The orienting cardinal is read as *which cortex pool fired*, the reward as `sc_rostral`/`reward_us` *firing rate* — never a host argmax over pixels or a host distance.
5. **Conversational moat untouched.** The 8/8 + 7/7 co-resident tests (incl. the three `is None` no-confab assertions) pass with SC-on.

---

## 5. Honest risk + the clean cheap-first GO vs NEGATIVE

### The biggest wall

**Co-residence operating-point starvation (the documented, partially-handled risk).** The standalone-tuned SC weights (`w_ret_sc=80`/`w_sc_rec=6`/`SC_RET_DRIVE=2500`) make `sc_map` fire ~2 Hz and `reward_us` never cross threshold on the heterogeneity-OFF merged bridge ("the standalone organ fires ~6-10× weaker co-resident" — `2026-06-18-merged-limbic-core-lift.md`). The merged-tuned op-point (160/12, drive 3500, rostral→US 40) is de-risked to fix it, but it is currently **env-var-gated, not the builder default** — so a deployment that forgets the env would silently get a starved SC and look like a NEGATIVE for the wrong reason. **Mitigation:** Step 0's bump-and-threshold check explicitly verifies the bump is alive at the promoted op-point before the 6-seed run; promote the op-point to the merged builder default so it is reproducible. This is the single most likely cause of a spurious NEGATIVE.

**Secondary walls (same family the nav decision hit):** (a) rate-coded WTA on a 16×16 sheet with a 3×3 goal blob can give a too-broad bump that ties on diagonals — mitigation: the discrete 4-pool `commit_X` fallback (Option B in the deep-research doc), which the existing WTA layer is already tuned for; (b) the SC orienting drive and the `sel_X` accumulator can fight inside the action-selection layer (the integration-vs-isolation gap that bit the spiking-SNc) — mitigation: the `SC_CORTEX_W` pooling-strength sweep (non-monotonic optimum ~18 standalone; re-tune on the merged bridge) + the 6-seed A/B is mandatory before claiming GO.

**The place-code wall is a DIFFERENT, deeper problem (do NOT conflate it with the SC deploy).** The self-org place code is a genuine substrate/dendritic-flavored BOUNDARY: the FS-PING-open read regime is non-location-selective and the all-or-none coincidence-plateau read-out either can't discriminate or over-clamps the SNc (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`). The host Gaussian stays the better-δ scaffold. This is NOT a cheap engineering win and should be left BENCHED behind the SC deploy; re-prioritize only with the dendritic substrate or a genuinely graded rate read-out.

### The clean cheap-first GO vs NEGATIVE

- **GO** (the likely outcome, given the standalone 6-seed GO + the merged-bridge CPU de-risk): Step 0 shows a live bump + threshold-crossing `reward_us` at the merged op-point; the Step-1 6-seed merged A/B lands SC-on ≤ 1.25× host with the scrambled-retinotopy lesion regressing and the conversational tests green → **flip the merged default to `nav_critic_spiking_sc=True`**, retiring the host orienting heuristic AND the host `sign(Manhattan)` reward on the one brain in one move (N1 + N5 closed on the merged bridge). The navigation half is then *fully brain-based* on the merged bridge by the strict bar (spiking SC orienting + neural approach reward + spiking commit-burst decision + spiking SNc RPE + defensible perception), with only the place code's host Gaussian remaining as a documented substrate-limited scaffold.
- **NEGATIVE** (the honest deliverable if it fails): the spiking SC, alive at a non-starved op-point, still underperforms the host heuristic on the merged bridge by > 25% even after the `SC_CORTEX_W` re-tune and the 4-pool fallback → record the co-residence operating-point limit (the spiking organ that matches/beats host standalone cannot, on the het-off merged substrate, within the tested op-point window) as the deliverable, keep the host orienting/reward as the documented scaffold on the merged bridge, and surface the het-on / op-point-widening question to the owner.

---

## Appendix — the precise file:line anchors (for the eventual builder)

- Host orienting reader: `g11_bg_runner.py:124-155` `sc_orienting_cardinal_from_image`; injected `:6747-6770` (gated `sc_orienting_reflex`).
- Host approach reader: `g11_bg_runner.py:158-180` `sc_salience_offset_from_image`; reward logic `:7016-7061`.
- Host orienting teacher (the Manhattan heuristic, the flagrant cheat): `:6386-6411`.
- Egocentric render (legitimate environment): `:183-198` `render_egocentric_goal`.
- Spiking SC build: `:2488-2544` (regions `sc_retina`/`sc_map`/`sc_fs`/`sc_rostral` + Mexican-hat pathways).
- Spiking SC post-init wiring: `:201-303` `install_spiking_sc_wiring` (retinotopy + recurrent + `sc_map→cortex_NESW` + `sc_map→sc_rostral`).
- Per-step SC drive + the het-off op-point env vars: `:6730-6739` (`SC_RET_DRIVE`), `:4425-4459` (`SC_CORTEX_W`/`SC_RET_SC`/`SC_REC`/`SC_ROS_US`/`SC_SCRAMBLE`).
- Merged-bridge hook: `nav_conv_merged_bridge.py:457`/`580-587`/`1263-1278` (`nav_critic_spiking_sc`), the host-reward note `:564-574`, the post-init index discipline `:1046` `finalize_conv_for_nav_gate`, the nav-episode call `:1618-1644`.
- Place-code self-org: `g11_bg_runner.py:4095-4217` build, `:6621-6626` drive (`neural_place_selforg`); BOUNDARY `2026-06-19-place-code-sparsify-default-BOUNDARY.md`.
- Standalone SC GO: `2026-06-10-N1-spiking-superior-colliculus-CLOSED.md`. Deep research: `2026-06-10-N1-N5-spiking-superior-colliculus-research.md`. Merged approach-reward de-risk: `2026-06-18-merged-limbic-core-lift.md`. Probes: `sc_map_orienting_probe.py`, `sc_n5_rpe_probe.py`, `sc_approach_td_probe.py`.
- Catalog: A.07 (`feature-catalog.md:169-179`), H.24 (`:3197-3207`), H.25 (`:3209-3219`), E.04 retinotopy, E.13 dorsal where/how (flags the `(gx,gy)` cheat), B.04 multisensory SC salience (`:1599-1600`).
