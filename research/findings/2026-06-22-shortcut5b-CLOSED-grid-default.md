# Shortcut #5b CLOSED — the grid-cell place front end is the merged-nav PRODUCTION default; the host-Gaussian `vs_place_context` RETIRES (2026-06-21)

**Status: #5b CLOSED.** The last HOST shortcut on the navigation place-code path — the host-Gaussian
`vs_place_context` position injection (a Python Gaussian-bump formula computing the agent's place code) —
is **retired by production default** for the merged "one brain" navigation. The brain now does the place
code: a self-organized spiking `place` pool that carves locally-SELECTIVE fields off a decorrelated
spatial-phase grid-cell metric. **NO `sim/` edit** (runner/builder-side, reuse-by-import). **The no-confab
moat was NEVER weakened** (8/8 + 7/7 = 15/15 with the flip ON, incl. the three `is None` no-confab
assertions). GPU (`SIM_BACKEND=cupy`, RTX 3090). PATHSPEC, both remotes.

This supersedes the production-wiring nav-chunk Item 2 disposition
(`2026-06-22-production-wiring-nav-chunk.md`), which shipped the grid front end as **opt-in (default OFF)**
because at the time the full host-Gaussian retirement was framed as "gated on the δ-readout stabilization."
The TD-read de-risk (`2026-06-22-shortcut5b-td-read-derisk.md`, `6e9e970d`/`2dc71d60`) resolved that
disposition: **#5b closes on R1 grounds; the close does NOT depend on the δ/TD read; the value-read residual
is the characterized dendritic frontier, NOT a host shortcut and NOT a blocker.**

---

## Why the close is sound (the de-risk verdict, R1 grounds)

The host-Gaussian retires because the grid front end produces a **genuinely-neural, value-gradable place
code** — both halves validated 3/3 multi-seed on real spikes:

- **Afferent selectivity (R1) — GO 3/3** (`2026-06-22-shortcut5b-R1-grid-frontend-derisk.md`): the
  spatial-phase grid metric (catalog D.07, the named missing medial-EC metric) is decorrelated by
  construction (adjacent-cell afferent cos 0.58 vs the landmark render's 0.9954 R1-cap), so a plain
  feedforward competitive `place` pool carves locally-SELECTIVE fields — NO dendrite. On real
  `cp_firing_states`, near-neighbour place cos 0.137 (< 0.30). The graded plateau read-out then grades the
  place value **V n/f 4.5–12.3× on every one of 3 seeds** vs the render's 1.02× R1-cap.
- **Learned near/far value — GO 3/3** (`-deltabar-3of3-close.md` + the TD-read finding's `grid`-arm table):
  the value-train learns a real near/far ratio (`w_n/f` 1.27→2.59); the TD δ HOLDS on the genuine learned
  value 3/3 (td1 graded-V 1.66 / 4.12 / 1.94).

So the place code is self-organized spiking (no Python formula) AND it supports a learned value — the two
things the host-Gaussian scaffold was standing in for. The host-Gaussian retires.

## The residual is the DENDRITIC FRONTIER (a neural-substrate limit), NOT a host shortcut

The TD-read de-risk isolated the genuine residual precisely (per the CLAUDE.md SURPASS sharpening — the
"gated on the δ-readout stabilization" framing was a *disguised boundary* the deep research overturned):
the value-READ operator cannot fully **separate** the learned-value increment from the structural place-code
magnitude on a point-neuron substrate, because they are the same physical quantity (afferent drive
magnitude) differing only in how it was set. The magnitude-matched `shuffle_v` control (destroy the learned
gradient, hold the structural magnitude) collapses the TD δ on only 1/3 seeds — the structural place-code
asymmetry survives the weight-shuffle (it lives upstream in the grid code's per-location drive density, not
in the shuffled place→value weights) and dominates the graded-V read.

This is the **dendritic frontier**: a two-compartment neuron (apical = structural place drive, basal =
learned value) could route the structural magnitude away from the learned-value read-out; a point neuron
cannot. It is a **point-neuron-substrate LIMIT, NOT a host shortcut**, and **NOT a blocker for the close**:

- The close does NOT depend on the TD read (the TD δ is a characterization read only; the existing
  graded-plateau read stays as the production read-out).
- The value/RPE δ is BEHAVIORALLY INERT on the orient-solvable immediate-reward gridworld anyway (the #9
  lesson / the merged-nav-critic BOUNDARY) — closing R1 retires the host-Gaussian without changing the
  navigation score.

The algorithmic positive control confirms the TD principle is correct in the abstract (pure array math:
`td` vrmse 0.003 / scale-free 0.997 vs `no_bootstrap` 182.3 / 0.203); the on-bridge fallback is the
substrate property above. The clean separation is a deferred deep-frontier item (a dendritic two-compartment
read-out), carried honestly — not a reason to keep the host-Gaussian as the default.

---

## The flip (the de-risk's recommended flip-path)

`MergedNavConvAgent` (`research/runners/nav_conv_merged_bridge.py`) — the production "one brain" agent —
`nav_critic_place_selforg` + `nav_critic_grid_frontend` constructor defaults **`False` → `None`**, where
`None` = "production default ON, but ONLY when the spiking critic is actually co-resident (it builds the
`place → striosome_value` arm the self-org/grid afferent feeds)":

```python
if nav_critic_place_selforg is None:
    self.nav_critic_place_selforg = bool(self.co_resident_nav_critic)   # ON when the critic is resident
else:
    self.nav_critic_place_selforg = bool(nav_critic_place_selforg)      # explicit True/False wins
if nav_critic_grid_frontend is None:
    self.nav_critic_grid_frontend = bool(self.nav_critic_place_selforg) # grid mirrors place_selforg
else:
    self.nav_critic_grid_frontend = bool(nav_critic_grid_frontend)
```

This follows the CYCLE-209 `co_resident_nav_critic` `None`-sentinel **auto-yield precedent** exactly (Item 5
of the nav chunk): the production agent default brings up the spiking limbic critic by default
(`co_resident_nav_critic` resolves ON unless a mutually-exclusive critic was requested), and the place
front end rides on it.

**Dependency-aware resolution (the design):**
- The grid front end **requires** the self-org place pool (it IS the `place_sensors` afferent; the builder
  asserts `grid ⇒ place_selforg`), so grid's default **mirrors** place_selforg.
- place_selforg **requires** the spiking critic (it builds the `place → striosome_value` arm; without the
  critic there is no `vs_place_context` to retire and nothing to drive). So when the critic is NOT resident
  — an explicit `co_resident_nav_critic=False`, OR a mutually-exclusive critic was requested so the nav
  critic auto-yields OFF — the place flags **auto-resolve OFF** (no crash, no silently-dropped flag).

**The conservative escapes (revertible, CPU-portable):**
- The **low-level `build_merged_nav_conv_bridge` default STAYS `False`** (the research runners that compose
  their own critic config keep the assert protecting a genuine double-request).
- The **`g11_bg_runner.py` function defaults** (`build_bg_brain_regions` / `run_moving_goal_episode`
  `nav_critic_grid_frontend: bool = False`) **STAY `False`** so the documented standalone CLI benchmarks
  reproduce byte-identically (the CYCLE-219 pattern: the CLI/library default stays the documented config;
  the production AGENT default flips). The comments were updated to record the close.
- An EXPLICIT `nav_critic_place_selforg=False` opts back into the legacy host-Gaussian `vs_place_context`
  afferent (the revertible escape).

---

## The HARD gate — all GREEN with the flip ON (GPU, seed 42)

| gate | required | result |
|---|---|---|
| **Moat 0-FA (HARD)** — `tests/test_nav_conv_merged_agent.py` | 8/8 PASS, incl. the `is None` no-confab assertions | **8/8 PASS** (92.6 s) |
| **Moat 0-FA (HARD)** — `tests/test_nav_conv_step2b_coresident.py` (composer co-resident) | 7/7 PASS, incl. the `is None` no-confab assertions | **7/7 PASS** (87.9 s) |
| **Host-Gaussian retired** — production bridge inventory | `vs_place_context` ABSENT; self-org place + grid present | **PASS** (`place_sensors` sized 198 = grid dim; `place`/`place_fs` present; `vs_place_context` absent) |
| **Nav not regressed** — `nav_on_merged_smoke` | A1 co-reside · A2 parser byte-frozen (gains==0, nnz same) · A3 parses post-episode · navigates | **PASS** (byte-unchanged by the flip; the smoke runs `run_moving_goal_episode` directly = the standalone-config reproduction path, function defaults stay False) |
| **Revertible escape** — `MergedNavConvAgent(nav_critic_place_selforg=False)` | the legacy host-Gaussian path returns | **PASS** (`vs_place_context` 200 neurons restored; self-org `place` absent) |
| **No-critic auto-OFF** — `MergedNavConvAgent(co_resident_nav_critic=False)` | place flags auto-resolve OFF, no crash | **PASS** |
| **Mutual-exclusivity** — `MergedNavConvAgent(co_resident_limbic=True)` | the limbic research config builds; nav critic auto-yields; place flags OFF | **PASS** (no mutual-exclusivity crash) |

The 8 + 7 = 15/15 conversational acceptance assertions pass with the place/critic/grid regions co-resident
on the merged bridge — the place/critic arrays (`cp_connections` / `cp_firing_states` /
`cp_conductance_g_graded_plateau`) are **array-disjoint** from the composer's complex `cp_rf_w_*` synapses,
so the no-confab moat is preserved by construction. The three `is None` no-confab assertions
(`what_does`/`elaborate`/`describe`) hold on both gates.

**The host-Gaussian retirement is real (the proof):** the production `MergedNavConvAgent(seed=42)` bridge now
builds with `nav_critic_place_selforg=True` + `nav_critic_grid_frontend=True`; its region inventory contains
`place_sensors` (198), `place`, `place_fs` and **does NOT contain `vs_place_context`** — the host-Gaussian
scaffold is genuinely absent from the production bridge. The legacy host-Gaussian (`vs_place_context`, 200
neurons) returns only via the explicit-`False` escape.

---

## HONEST closure statement

**The host-Gaussian `vs_place_context` shortcut is CLOSED** (the grid-cell front end is the production
default place code; the place code is genuinely neural — a self-organized spiking place pool over a
decorrelated grid metric, value-gradable). **The value-READ structural/learned separation is the
CHARACTERIZED DENDRITIC FRONTIER** — a point-neuron-substrate limit (a two-compartment apical/basal neuron
would separate the structural place drive from the learned-value read-out; a point neuron cannot), **NOT a
host shortcut**. The close does not depend on it (the existing graded-plateau read stays the production
read-out; the nav δ is behaviorally inert anyway). The clean-learned-δ dendritic read-out is the recorded
deep-frontier item.

**sim/-edit flag: NONE.** Every piece is runner/builder-side reuse-by-import — the grid reference helper
(`_n9_make_grid_code`), the self-org place pool, the graded plateau read-out, the spiking critic all already
ship. The flip is a constructor-default change (`False → None`-sentinel) in `MergedNavConvAgent` plus comment
updates recording the close.

---

## Provenance / files
- `research/runners/nav_conv_merged_bridge.py` — the `MergedNavConvAgent` `nav_critic_place_selforg` /
  `nav_critic_grid_frontend` `None`-sentinel production-default-ON + the dependency-aware auto-yield; the
  low-level builder default stays `False` (comment updated to record the close).
- `research/runners/g11_bg_runner.py` — the function-default comments updated to record the close (the
  function defaults stay `False` for standalone-CLI reproducibility).
- Findings (the close rests on): `2026-06-22-shortcut5b-td-read-derisk.md` (the VERDICT: close on R1 grounds,
  the dendritic-frontier characterization), `2026-06-22-shortcut5b-R1-grid-frontend-derisk.md` +
  `-deltabar-3of3-close.md` (the R1 selectivity + learned value the close rests on),
  `2026-06-22-production-wiring-nav-chunk.md` (Item 2 — the now-superseded opt-in disposition).

## Reproduce
```bash
# the two moat gates (HARD, 0-FA) with the flip ON:
SIM_BACKEND=cupy python -m pytest tests/test_nav_conv_merged_agent.py -v          # 8/8
SIM_BACKEND=cupy python -m pytest tests/test_nav_conv_step2b_coresident.py -v     # 7/7
# host-Gaussian retired (production-default bridge inventory): vs_place_context ABSENT, place+grid present.
# nav not regressed (the standalone-config reproduction path, byte-unchanged):
SIM_BACKEND=cupy python -m research.runners.nav_conv_merged_bridge --nav-on-merged-smoke --seed 42 --n-steps 400 --grid-size 8
```
