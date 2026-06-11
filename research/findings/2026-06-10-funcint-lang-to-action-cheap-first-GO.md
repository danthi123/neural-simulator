# Functional integration cheap-first de-risk — LANGUAGE→ACTION: GO

**Date:** 2026-06-10 (overnight Thread, functional-integration arc).
**Verdict:** **GO** — a parsed command SYNAPTICALLY biases the navigation cascade, the bias is gated by the
parser's FIRING, and it is lesion-confirmed. Multi-seed (42/43/44), all 4 cardinal directions.
**Probe:** `research/runners/funcint_lang_to_action_probe.py` (CPU, `SIM_BACKEND=numpy`, ~minutes).
**Result data:** `research/findings/raw/funcint_lang_to_action_probe.json`.
**Design:** `docs/plans/2026-06-10-functional-integration-one-brain-design.md` §3 (mechanism), §4 (this
de-risk), §5 (anti-cheats).

---

## 0. Terms (defined once)

- **navigation cascade** — the basal-ganglia action-selection circuit's action cortex pools
  `cortex_{N,E,S,W}` (M1-equivalent per-direction pools; the body moves in the direction whose pool wins).
- **parser** — the conversational comprehension network (`parse_conj` 6 conjunction units + `parse_role`
  3×40 role-ensemble neurons split into agent/action/patient blocks). It assigns each word of an SVO command
  ("dog go north") to its grammatical role; for "go" at the verb position, the **action** role ensemble fires.
- **transmission gate** — a per-synapse multiplier in [0,1] on a pathway's synaptic CURRENT, set at runtime by
  `bridge.set_transmission_gate(name, value)`. Pre-wire a route, hold it closed (gate 0 → no current), open it
  on command.
- **gate-from-firing coupling** — `couple_gate_to_indices`/`couple_gate_to_pool`: each step the gate opens iff a
  control population's smoothed firing rate (EMA) exceeds a threshold. **In-substrate** — no Python reads a
  value; the firing of one region opens a route into another (`_apply_gate_couplings`, `sim/bridge.py:3002`).
- **command_route** — the ONE transmission gate this probe adds, on the `language_input → cortex_{N,E,S,W}`
  pathways. Closed at rest; coupled to the parser's action-role ensemble.

---

## 1. The load-bearing question (design §4)

Does a parser-opened SYNAPTIC route measurably bias the navigation cascade's action pools — with NO Python
value crossing the navigation↔conversational halves? Concretely: when the parser comprehends a command verb
(its action ensemble FIRES), does that firing open `command_route` so the commanded direction word's
`language_input → cortex_{direction}` current steers the navigation action cortex?

This is the cheapest test that shows ANY cross-region synaptic influence, run BEFORE any task harness or
episode loop. The full spoken-instruction-navigation behavioral task (the GPU 6-seed A/B) is gated on it.

## 2. The mechanism under test (design §3.1, fully synaptic, no `sim/` edit)

- **WHICH direction (identity):** the navigation cascade's OWN `language_input → cortex_X` channel binds a
  direction word's code to its action pool. The probe drives `language_input` with the direction word's
  orthogonal code (a legitimate sensory render — the environment presents the command as text).
- **WHEN to listen (a gate, NOT a value):** ONE `command_route` transmission gate on
  `language_input → cortex_X`, held CLOSED at rest, COUPLED to the parser's ACTION-role sub-block firing via
  the in-substrate `couple_gate_to_indices` primitive (the action block is a sub-range of `parse_role`, not its
  own region, so the index-based variant is used — design §3.2 option ii; the IDENTICAL coupling dict
  `couple_gate_to_pool` builds). Each step `_apply_gate_couplings` opens `command_route` iff the parser action
  ensemble is firing. **The parser's comprehension, in spikes, opens the route** — exactly the mechanism
  step-2's `hear_synaptic` used (parser ensemble → gate → composer), now pointed at the navigation cortex.
- **COMPREHEND → LATCH → ACT (the validated timing, design §3.1 step 3):** drive the action-verb parser
  conjunction for a PRE-WINDOW until the parser fires and opens the gate, THEN run the cortex readout window
  holding the parser conjunction (so the gate stays latched) and the direction's `language_input` drive.

### Cheapest faithful substrate (design §4 explicitly allows it)

A fresh brain-region-framework `SimulationBridge` with ONLY the action cortex pools `cortex_{N,E,S,W}` +
`language_input` + the parser slices — no striatum/GPi/thalamus/dlPFC/RF (none are needed to read the
cortex-pool bias). It exercises the §3 route verbatim, on CPU, in minutes. Reuse-by-import:
`nav_conv_merged_bridge` (parser slices/train/read) + `unified_brain_bridge`
(`couple_gate_to_indices`). **No `sim/` edit.**

---

## 3. Result — GO on all 3 seeds, all 4 directions

Per-direction cortex-pool firing fraction (mean over the 120-step readout). **OPEN** = parser fires → gate
open; **CLOSED** = no parse → gate held closed; **LESION** = `command_route` weights zeroed + parser fires.
"target" = the commanded pool `cortex_d`; "maxOther" = the strongest of the other three. A "bias" requires the
commanded pool to lead the runner-up by ≥ 0.01 firing-fraction (`MIN_BIAS_MARGIN`) — so a floating-point
residual does NOT count.

### Seed 42

| dir (word)  | OPEN target | OPEN maxOther | OPEN margin | winner | CLOSED target | CLOSED margin | winner |
|-------------|------------:|--------------:|------------:|:------:|--------------:|--------------:|:------:|
| N (north)   | 0.0667      | 0.0000        | +0.0667     | ✅     | 0.0004        | +0.0004       | —      |
| E (east)    | 0.0665      | 0.0000        | +0.0665     | ✅     | 0.0000        | +0.0000       | —      |
| S (south)   | 0.0501      | 0.0000        | +0.0501     | ✅     | 0.0000        | +0.0000       | —      |
| W (west)    | 0.0642      | 0.0000        | +0.0642     | ✅     | 0.0001        | +0.0001       | —      |

### Seed 43

| dir (word)  | OPEN target | OPEN margin | winner | CLOSED target | CLOSED margin | winner |
|-------------|------------:|------------:|:------:|--------------:|--------------:|:------:|
| N (north)   | 0.0664      | +0.0664     | ✅     | 0.0000        | +0.0000       | —      |
| E (east)    | 0.0671      | +0.0671     | ✅     | 0.0000        | +0.0000       | —      |
| S (south)   | 0.0671      | +0.0671     | ✅     | 0.0004        | +0.0004       | —      |
| W (west)    | 0.0501      | +0.0501     | ✅     | 0.0014        | +0.0014       | —      |

### Seed 44

| dir (word)  | OPEN target | OPEN margin | winner | CLOSED target | CLOSED margin | winner |
|-------------|------------:|------------:|:------:|--------------:|--------------:|:------:|
| N (north)   | 0.0583      | +0.0583     | ✅     | 0.0000        | +0.0000       | —      |
| E (east)    | 0.0667      | +0.0667     | ✅     | 0.0000        | +0.0000       | —      |
| S (south)   | 0.0500      | +0.0500     | ✅     | 0.0000        | +0.0000       | —      |
| W (west)    | 0.0628      | +0.0628     | ✅     | 0.0011        | +0.0011       | —      |

**Roll-up (winners = meaningful-margin biases):**

| seed | OPEN | CLOSED | LESION | parser parses correctly |
|------|:----:|:------:|:------:|:-----------------------:|
| 42   | 4/4  | 0/4    | 0/4    | ✅ (`dog go north`, action=go, agent=dog) |
| 43   | 4/4  | 0/4    | 0/4    | ✅ |
| 44   | 4/4  | 0/4    | 0/4    | ✅ |

**The bias appears ONLY when the parser has fired** (OPEN: every commanded pool fires at 0.05–0.067 while the
other three sit at exactly 0.0000), and it **VANISHES** both when the gate is held closed (CLOSED: the
commanded pool collapses to a ≤0.0014 residual leak, far below the 0.01 threshold) and when the route is
lesioned (LESION: command_route weights zeroed → no pool wins even with the parser firing).

---

## 4. The two anti-cheat controls (design §5) — both pass

1. **Gate-closed control.** With no parser conjunction drive, the action ensemble never fires, the coupling EMA
   stays below threshold, `command_route` stays closed, and only the direction's `language_input` drive is
   applied. The directional bias **collapses** (commanded-pool firing ≤ 0.0014 across all 12 direction×seed
   cells). → the OPEN bias is not ambient leakage from the language drive; it requires the open gate.
2. **Lesion control.** A fresh bridge with every `command_route` synapse's weight zeroed (≈4.6–5.1k synapses
   per seed), gate coupled, parser firing: the bias **vanishes** (0/4 winners every seed). → the OPEN bias is
   carried by the `command_route` synapses specifically, not by any non-route path.
3. **Provenance (no Python value-copy).** By construction the ONLY navigation-side current write is the
   orthogonal direction code into `language_input` (a legitimate sensory render); the parser conjunction drive
   is conversational-side. No parser-DERIVED quantity (a parsed `{role: word}`) is written into any
   cortex/striatum/motor drive. The cross-region coupling is the `couple_gate_to_indices` gate (transmits a
   0/1 GATE STATE from the action ensemble FIRING, not a value) + the pre-existing `language_input → cortex_X`
   synapses (carry the word identity the environment legitimately presented). The probe asserts these
   structural facts; the report records them.

So the influence is the SYNAPTIC route opened by parser firing — not ambient leakage and not a Python path.

---

## 5. What it took to get a clean signal (honest engineering notes)

Three tuning issues surfaced in the cheap CPU de-risk; none change the mechanism, all are documented in the
probe:

- **Gate flicker (the documented step-2 regression).** The parser action ensemble fires burstily (sustained
  mean ≈ 0.017 of the 40 ensemble neurons under continuous verb drive). The default coupling threshold (0.05)
  sits ABOVE that, so the EMA hovers at threshold and the gate flickers open <½ the readout. Fix: set the
  coupling threshold to 0.008 (below the sustained rate) + a slightly stickier EMA (alpha 0.2). The gate then
  HOLDS open ~67% of the readout. This is a coupling parameter, NOT a hand-set gate — the gate still opens only
  because the parser SPIKES. (The OPEN margin ≈ 0.065 is ~67% of the gate-forced-open margin ≈ 0.083, exactly
  consistent with the 67% open fraction.)
- **The untrained `language_input → cortex_X` route is not direction-selective.** The nav cascade's text-IO
  route is a UNIFORM per-pool projection; its "north"→cortex_N selectivity is what TRAINING grows (the
  concept-pool/b3 word→action mapping; Pulvermüller action-word somatotopy, catalog G.20). This cold probe does
  not train, so the WHICH-direction selectivity is installed STRUCTURALLY as a per-direction topographic
  labeled line (direction d's orthogonal `language_input` band → `cortex_d`) — the standard topographic prior
  the project uses (`apply_topographic_bias`), a stand-in for the trained mapping. The thing being de-risked is
  the GATING (does parser firing synaptically open the route), not the learning of the mapping; the full task
  build will use the trained/learned route.
- **The labeled line must be EXCITATORY→excitatory.** `language_input` is 20% inhibitory, and each orthogonal
  band happens to contain 4–6 inhibitory neurons. Wiring the whole band (exc+inh) into cortex made the route
  SUPPRESS its pool, and worse, raising the weight scaled the inhibition up too → cortex went silent (the
  inversion: higher weight ⇒ MORE inhibition ⇒ no firing; bands with more inhibitory neurons, e.g. N=6, were
  fully suppressed while E/W=4 squeaked through). Fix: filter each band to its excitatory neurons before wiring
  band→cortex. (The orthogonal DRIVE still hits the whole band as a sensory render; only the band→cortex ROUTE
  is excitatory.) After this fix all 4 pools fire selectively at ≈ 0.083 (gate forced open).

These are exactly the kind of bounded substrate-tuning facts the de-risk exists to find cheaply — the gate's
load-bearing behavior (parser-firing-gated, lesion-confirmed, direction-correct) is robust once they are set.

---

## 6. Verdict and next step

**GO.** The synaptic language→action influence is real, gated by the parser's firing, and lesion-confirmed,
on all 3 seeds and all 4 cardinal directions, with both anti-cheat controls collapsing the bias and a clean
provenance audit. This is the first demonstrated cross-region SYNAPTIC interaction between the navigation
brain and the conversational brain on the (minimal) merged substrate — a parsed sentence, by neuron firing and
synaptic current alone, biases the navigation action cortex.

**Proceed to the full spoken-instruction-navigation behavioral task** (design §7 step 2–3, GPU): present
3-word instructions to the conversational channel on the merged bridge, wire `command_route` on the trained
`language_input → cortex_X` route, run the episode loop with the comprehend→latch→act order, and score
commanded-direction following over a multi-phase instruction schedule. The 6-seed A/B then runs the four
behavioral anti-cheats: coupled vs route-lesioned (must collapse to chance), nav-only and conv-only isolated
controls (must fail), provenance (no Python value-copy), instruction-scramble (must regress). That is the
moment "functional integration" is demonstrated behaviorally AND the standing nav-reward residual is resolved
(the route/policy becomes load-bearing, lesion-confirmed).

No banking: this de-risk is GO, so the next action is the task harness, not a stop.

---

## 7. Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners.funcint_lang_to_action_probe \
    --seeds 42 43 44 --out research/findings/raw/funcint_lang_to_action_probe.json
# exit 0 = GO ; 2 = PARTIAL ; 1 = NEGATIVE
```
