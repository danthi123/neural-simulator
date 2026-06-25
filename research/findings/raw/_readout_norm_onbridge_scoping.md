# Read-out normalization → on-bridge spiking circuit — RESEARCH-GATE scoping (read-only, 2026-06-25)

**Purity backlog #7 / biology-fidelity audit class-(b) #6.** The stream/PPMI cortex's READ-OUT NORMALIZATION
is currently a HOST log-domain double-centring scaffold (per-concept + per-hub mean-subtraction in host numpy,
NOT on the bridge). The CYCLE-93b prescription (per-concept feedforward inhibition + per-hub adaptation, POST-f-I)
is the named on-bridge circuit. This doc scopes closing it.

**READ-ONLY. No edits, no runs, no webapp.** Standing 4-move deliverable (diagnose → reframe → rank → de-risk +
verdict). Every load-bearing claim cross-checked against code (file:line) + the finding + the catalog.

---

## 0. TL;DR for the controller

- **The genuine host residual is TINY and PRECISELY LOCATED.** It is **two lines** of host numpy at *offline
  cortex-code GENERATION* time: `double_center(log1p(M*100))` where `M` is the learned `hub→target` weight block
  read off the bridge (`_phaseB_onbridge_stream_cortex_derisk.py:160-165` and the conversation runner
  `:116-120`). `double_center` = `X − X.mean(0) − X.mean(1) + X.mean()` (per-hub mean over concepts +
  per-concept mean over hubs + a global constant). It is **NOT** in the runtime who/what turn — the production
  agent loads pre-computed `.npy` codes — so it is a **learning-pipeline / offline preprocessing** op, the
  lowest-stakes class of shortcut.
- **The correctness gate is ALREADY GREEN.** Burndown #5 (2026-06-20) proved the NEURAL normalization (per-hub
  spike-frequency ADAPTATION + per-concept FEEDFORWARD INHIBITION, with realistic rate-coded-pool noise on the
  means) reproduces the **who/what == host EXACTLY** and keeps the **no-confab moat at 0 false-accepts**,
  multi-seed, *through the actual conversational pipeline* (not just the structure proxy). So this scoping is NOT
  about whether it's biologizable — that's settled — it is about the **literal on-bridge CIRCUIT realization** of
  the validated `neural_norm` specification.
- **MOST of the circuit machinery is already SHIPPED in `sim/`** as guarded, default-off, byte-identical-when-off
  primitives, both wired into the step loop:
  - **per-hub mean → SUBTRACTIVE adaptation:** `cp_input_mean_ema` + `cp_input_mean_adapt_mask`
    (`BrainRegion.input_mean_adapt`, `cfg.enable_input_mean_adapt`; step block `bridge.py:6257-6269`). **EXACT
    match** to the `neural_norm` per-hub op — already on-substrate, BRAIN-BASED (EMAs the neuron's OWN input
    current, not a host x-mean), 6-seed GO (`2026-06-15-slow-perhub-mean-primitive-deep-research.md`).
  - **per-concept mean → normalization pool:** `cp_input_divisive_mask` (`BrainRegion.input_divisive_norm`,
    `cfg.enable_input_divisive_norm`; step block `bridge.py:6209-6218`). **PARTIAL match** — it is **DIVISIVE**
    (`x/(σ+gain·mean)`, Carandini-Heeger), whereas `neural_norm`'s per-concept op is **SUBTRACTIVE feedforward
    inhibition** (`a − mean`). Same per-concept-pool wiring, different arithmetic.
- **The ONE genuine residual to build is the SUBTRACTIVE per-concept feedforward-inhibition primitive** (the
  divisive twin exists; the subtractive twin does not). Two paths: (A) **reuse the existing per-hub SUBTRACTIVE
  primitive on the per-concept axis** (the per-hub block already does `x − mean`; a per-concept SUBTRACTIVE pool
  is the same arithmetic on a different mask) — this is a **small guarded `sim/` edit** (clone the divisive block,
  swap divide→subtract), or (B) test whether the SHIPPED **divisive** per-concept op + per-hub adaptation already
  carry the conversation (divisive ≈ subtractive-in-log-domain may be good enough; **zero `sim/` edit** if so).
- **VERDICT: closeable on-bridge. Cheapest-first = (B) zero-`sim/`-edit divisive+adapt test; fallback = (A) a
  small guarded subtractive-FFI `sim/` clone (byte-review required).** Neither is a boundary; the deep
  whitening/dendritic wall does NOT apply (this is the separable diagonal+DC half, not cross-neuron decorrelation).

---

## 1. DIAGNOSIS — exactly what the host double-centring does + the genuine residual (which bytes)

### 1.1 The op, verified against code

The read-out (the same in the mechanism runner and the conversation runner):
```
# _phaseB_onbridge_stream_cortex_derisk.py:51-52, 160-165
def double_center(X):
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()
...
W   = to_host(bridge.cp_connections.todense())          # the LEARNED hub→target weights (on-bridge)
blk = W[ix_(hub, tgt)].reshape(...).mean(axis=(1,3))     # population block-mean → M[target, hub]
code = double_center(np.log1p(M * 100.0))                # ← THE HOST SCAFFOLD (the residual)
```
Decomposing `double_center(L)` where `L = log1p(M·100)` (`M` = the on-bridge learned co-occurrence block,
`corr(M,C)≈0.88–0.94` on-substrate):
1. **`log1p(M·100)`** — the f-I / Weber-Fechner read-out (the spiking neuron's log-ish frequency-vs-input curve).
   CYCLE-93b validated this half on-bridge separately (it is what the neuron's own f-I does). **NOT the residual.**
2. **`− L.mean(0)`** (per-HUB mean, over concepts) — subtract each context-hub's average activation across all
   concepts = remove the hub's DC/baseline level. This is the **axis-0 / per-feature** centering.
3. **`− L.mean(1)`** (per-CONCEPT mean, over hubs) — subtract each concept's average over its hubs = remove the
   concept's total-drive offset. This is the **axis-1 / per-concept (row-marginal)** centering = PPMI's
   per-concept normalization.
4. **`+ L.mean()`** (global constant) — the additive double-centring correction term.

### 1.2 The genuine host residual (which bytes)

- **The residual = items (2)+(3)+(4): the two mean-subtractions + the global constant, in host numpy.** Two source
  lines: `double_center()` (`:51-52`) and its call at `:164` (mechanism) / `:120` (conversation, gated by
  `--readout-norm`, default `host`). The neuron's f-I log (item 1) is NOT host cognition — it's the substrate's own
  transfer function; the residual is strictly the **centring**.
- **It is OFFLINE, not runtime.** It runs ONCE during cortex-code generation; the production conversational agent
  loads the resulting `.npy` codes. Per the burndown inventory: *"#5 is an OFFLINE learning-pipeline op, not a
  runtime conversational shortcut … it does not sit in the live who/what turn"* (`2026-06-20-shortcut-burndown-
  inventory.md:164-166`). Class-(b) thin — host doing a cognitive normalization the cortex should do with neurons,
  but at code-generation time, not in the live turn. The fidelity audit lists it as residual **#6 / I-10**
  (`_biology_fidelity_audit_2026-06-24.md:103-104`): *"the on-bridge normalization circuit … is scoped, not built."*
- **Why it's load-bearing (so it can't just be deleted):** burndown #5's anti-cheat showed a NO-normalization
  control (raw `L`, unit-normed) is **worse** through the pipeline (who/what drops or the moat leaks). The centring
  does real work; it must be REPLACED by neurons, not removed.

### 1.3 What is already NOT residual (so we don't re-build it)

- **Item (1) log f-I:** the neuron's own transfer function (CYCLE-93b validated separately).
- **The per-hub mean (item 2):** **already an on-bridge spiking primitive** (`input_mean_adapt`, §3) — it just isn't
  yet WIRED at the cortex read-out (it was built/validated for the L1 learned-cortex axis-0 centering).
- **The LEARNING of `M`:** on-substrate rate-Hebbian, `corr(M,C)≈0.88` (CYCLE-95, GO). The block read
  `to_host(cp_connections.todense())` is a legitimate **body/inspection read** of learned weights (bookkeeping, not
  cognition) — the residual is the `double_center` MATH applied to it, not the read.

---

## 2. REFRAME via biology — divisive normalization / feedforward inhibition / SFA as the on-bridge circuit

### 2.1 The biology of the two centring ops (catalog-cited)

| host op | what it removes | canonical neural mechanism | catalog / Kandel cite |
|---|---|---|---|
| `− L.mean(0)` per-HUB (over concepts) | the hub's own slow DC/baseline level | **subtractive spike-frequency ADAPTATION** (a per-neuron high-pass removing the slow mean) | **I.08** M-current/Kv7 (Kandel 6e Ch 10 p227,231); **I.13** SK/sAHP (Ch 10 p229); Benda-Longtin-Maler 2005 J.Neurosci 25:2312 ("purely subtractive"); point-neuron predictive coding Lee/Pennartz 2024 PMC11045951 |
| `− L.mean(1)` per-CONCEPT (over hubs) | the concept's total-drive offset (common mode across its hubs) | **feedforward INHIBITION** by a per-concept interneuron reading the pool's mean drive (a global interneuron / center-surround common-mode removal) | **B.06** PV+ FS feedforward inhibition (Kandel 6e Ch 38 p935); **E.05** lateral inhibition / center-surround ("decorrelates output", Ch 22 p588–593) |

Both are **subtractive common-mode removal** — the legitimate, point-neuron-realizable side of the
Mikulasch-Priesemann boundary. The slow-per-hub deep-research (`2026-06-15-slow-perhub-mean-primitive-...md`)
established the load-bearing scope distinction with HIGH confidence: **per-neuron / per-pool MEAN-subtraction is the
separable diagonal+DC half of whitening; the cross-neuron de-correlation (off-diagonal) is the expensive half MP
forbids on point neurons.** This op is the cheap half — burndown #5's GREEN confirms it lands cleanly on point
neurons.

### 2.2 An alternative biological framing for the per-concept op — DIVISIVE (Carandini-Heeger)

The SHIPPED `input_divisive_norm` block implements the **canonical cortical normalization model** (Carandini-Heeger
2012 *Nature Rev Neurosci* "Normalization as a canonical neural computation" — cited in the bridge code at
`bridge.py:6198`): `r_i = x_i/(σ + gain·mean_j x_j)`. The S5 finding (`2026-06-20-S5-divisive-norm-derisk.md`) and
the bridge comment argue that **divisive per-concept normalization + the neuron's log-ish f-I ≈ PPMI's
per-concept (row-marginal) normalization** ("the neuron's log-ish f-I then makes it the log-ratio";
`bridge.py:6202-6204`). This is the key reframe opportunity (§3 Option B): the per-concept op the host does
SUBTRACTIVELY *in the log domain* may be reproducible by the SHIPPED DIVISIVE op *pre-log* — `log(x/m) = log x −
log m`, i.e. a divisive gain BEFORE the log f-I equals a subtractive shift AFTER it. If the f-I log is faithful,
the divisive primitive already realizes the per-concept centring with **zero new `sim/` code** — the same identity
the in-code comment asserts and S5 partially validated for the cleanup-score seam.

### 2.3 Is the EXISTING machinery reusable? (the decisive question for the verdict)

**YES for the per-hub op (exact), PARTIAL for the per-concept op (divisive-vs-subtractive arithmetic gap).**

| `neural_norm` op | shipped bridge primitive | match | reuse class |
|---|---|---|---|
| per-hub `a = L − hub_mean` (subtractive) | `input_mean_adapt` — `cp_input_mean_ema` subtractive EMA of own input, masked, `bridge.py:6257-6269` | **EXACT** (subtractive, per-neuron, BRAIN-BASED, 6-seed GO) | **wiring only** — flag the read-out region `input_mean_adapt=True`, slow α |
| per-concept `a − con_mean` (subtractive FFI) | `input_divisive_norm` — `cp_input_divisive_mask` DIVISIVE `x/(σ+gain·mean)`, `bridge.py:6209-6218` | **PARTIAL** (per-concept-pool wiring identical; arithmetic divisive≠subtractive) | **wiring only IF divisive≈subtractive-in-log (Option B); else a small guarded subtractive clone (Option A)** |

So the per-hub half is a pure **wiring** reuse (no `sim/` edit). The per-concept half is the one place a NEW `sim/`
primitive *might* be needed — and even there, the divisive twin already exists as the exact template (the
subtractive clone is `divide → subtract`, ~6 lines mirroring `:6209-6218`). There is also a SECOND independent
divisive pool already shipped (`cp_input_divisive_mask_2`, `bridge.py:6229-6238`) demonstrating the
clone-a-normalization-pool pattern is established and byte-reviewed.

---

## 3. RANKED cheap-first options (with `sim/`-edit flags)

Ordered cheapest-first. **`sim/`-edit flag is called out explicitly; a `sim/` edit needs byte-review per the
BRAIN-BASED-ONLY standard.**

### Option B (LEAD, ZERO `sim/` edit) — wire the SHIPPED divisive + per-hub-adapt primitives at the cortex read-out
- **What:** build the stream cortex with a read-out region flagged BOTH `input_mean_adapt=True` (per-hub
  subtractive, the exact op) AND `input_divisive_norm=True` (per-concept divisive, *pre-log*). Drive the learned
  block `M` (or the population code) as input to that region; the per-concept divide + per-hub subtract + the
  neuron's log f-I produce the code on-substrate. Read `cp_firing_states`/membrane as the code (no host
  `double_center`).
- **Biology:** Carandini-Heeger divisive (per-concept) + SFA subtractive (per-hub) + log f-I. The log identity
  (`log(x/m)=log x−log m`) is the bridge code's own stated rationale (`:6202-6204`) and S5's validated mechanism.
- **`sim/` edit:** **NONE.** Both primitives shipped + byte-reviewed (2026-06-15). Pure runner wiring (flag two
  `BrainRegion` booleans + two cfg flags + slow α). This is the literal *"flip the demo default to `--readout-norm
  neural` after producing the codes"* gate the burndown named (#5, P3), now realized as the on-bridge circuit
  rather than the numpy `neural_norm` proxy.
- **Risk:** **LOW-MEDIUM.** The open question is whether **divisive per-concept ≈ subtractive per-concept** for THIS
  read-out (the `neural_norm` GREEN used subtractive; divisive is a different arithmetic that the log identity makes
  *approximately* equivalent but not exactly). If the divisive per-concept op + per-hub adaptation carries the
  who/what == host with the moat at 0-FA, **the shortcut is closed with no `sim/` edit at all.** If it falls short
  (divisive ≠ subtractive enough in this regime), fall back to Option A.

### Option A (FALLBACK, SMALL guarded `sim/` edit) — add a SUBTRACTIVE per-concept feedforward-inhibition primitive
- **What:** clone the divisive per-concept block into a SUBTRACTIVE twin: a new guarded
  `cp_input_subtractive_mask` + `BrainRegion.input_subtractive_inhib` + `cfg.enable_input_subtractive_inhib`,
  step block `total_input_current_pA −= mask·gain·mean_pool(total_input_current_pA)` (the per-concept mean over the
  flagged pool, subtracted) — the EXACT `neural_norm` per-concept op (`a − con_mean`) on-substrate. Compose with the
  shipped `input_mean_adapt` for the per-hub half.
- **Biology:** B.06 PV+ feedforward inhibition / E.05 center-surround common-mode removal (subtractive). This is the
  literal CYCLE-93b "per-concept feedforward inhibition" prescription.
- **`sim/` edit:** **YES — small, guarded, default-off (BYTE-REVIEW REQUIRED).** ~6 step-loop lines mirroring the
  divisive block at `bridge.py:6209-6218` (divide → subtract) + one mask alloc mirroring `:1335-1348` + 3 config
  fields mirroring `:499-503` + 1 region field mirroring `regions.py:235`. Guard pattern: `cp_input_subtractive_mask
  is None` ⇒ block unreached ⇒ byte-identical when off (the established `input_divisive`/`input_mean` contract). The
  edit is strictly smaller than the already-reviewed divisive/per-hub primitives.
- **Risk:** **LOW.** It is a literal clone of a reviewed primitive on a new mask; the arithmetic is the exact
  validated `neural_norm` op; the OFF==byte-identical guard makes regression impossible for existing runs.

### Option C (literal-circuit fidelity, MEDIUM `sim/` work) — explicit per-concept FS interneuron POOL via region framework
- **What:** instead of a per-neuron masked-mean primitive, build the per-concept inhibition as an **actual
  interneuron pool**: a small FS region per concept-pool that pools the concept's drive and projects GABAergic
  feedforward inhibition back onto the read-out neurons (a real B.06 microcircuit), using the existing
  `RegionPathway` + inhibitory connectivity. Per-hub via `input_mean_adapt` as before.
- **Biology:** the most literal B.06 PV+ FS feedforward-inhibition circuit (real interneurons, real GABA synapses).
- **`sim/` edit:** **probably NONE** (region framework + inhibitory pathways already exist) but **MEDIUM wiring +
  tuning** — building/parameterizing the FS pools, ensuring the pooled inhibition computes the *mean* (not a
  saturating WTA), and the slow vs fast timescale. Higher variance than A/B.
- **Risk:** **MEDIUM-HIGH.** A real interneuron pool computing a clean per-concept *mean* (vs a winner-take-all or a
  saturating common-mode) is the same membrane-timescale + rectification challenge the slow-per-hub research flagged
  for the graded-shadow option (`2026-06-15-...md` Option C). More faithful, but slower to land and not needed for
  correctness. **Deprioritize unless A/B both fail or maximal literal fidelity is wanted.**

### Option D (NOT recommended) — the cross-neuron `graded_lateral` decorrelation machinery
- **What:** the shipped `cp_graded_lateral_M` anti-Hebbian K×K decorrelation (`bridge.py:1972-2013`).
- **Why NOT:** this is the **off-diagonal / cross-neuron whitening** machinery (the expensive half). The read-out
  centring is the **diagonal+DC half** (per-feature + per-concept MEAN). Using `graded_lateral` for it is the wrong
  tool — overkill, re-opens the locality wall, and the means it would remove are per-neuron/per-pool scalars, not a
  covariance. Listed only to rule it out (it IS the right tool if the residual +0.31→+0.44 de-correlation gap is
  ever pursued — a SEPARATE, deferred frontier).

---

## 4. ANTI-CHEATS, cheap-first de-risk, GO bars, VERDICT

### 4.1 Anti-cheats (mandatory — BRAIN-BASED-ONLY + project standards)

1. **On-bridge, not host:** the normalization MUST be computed from the bridge's own neuronal state (the read-out
   region's own input current / pool drive), subtracted/divided as a neuronal current BEFORE threshold — NOT a host
   `double_center` written into the codes. The shipped primitives already satisfy this (`input_mean_adapt` EMAs the
   neuron's OWN current, `bridge.py:6249-6250` comment; divisive pools the flagged set's OWN drive). The GO must
   assert the host `double_center` call is REMOVED from the code path (read `cp_firing_states`/membrane, not numpy).
2. **== the host-normalized read to tolerance:** the on-bridge code's `Pearson(cos, S_true)` ≥ **0.90×** the host
   `double_center` ceiling (the burndown-#5 / biologization-sweep bar; numpy proxy hit 96%). At the conversational
   level: **who/what recall == host** and **moat abstain with 0 false-accepts** (the burndown-#5 HARD bars).
3. **Generalization / recall preserved:** held-out generalization (the `heldout_generalization` helper) preserved vs
   host; the moat familiarity gap preserved (burndown #5: neural +0.401 vs host +0.416, well above the 0.10 gate).
4. **Both ops load-bearing (ablation):** drop the per-hub op OR the per-concept op → the structure/who-what drops
   (burndown #5: adapt-only +0.148, FFI-only +0.305, both ≪ combined; no-norm control worse). Asserts the
   normalization is doing the work, not the bind tolerating anything.
5. **OFF == byte-identical:** any `sim/` edit (Option A/C) must leave every existing run byte-identical with the new
   flag off (the established guarded-no-op contract; assert a second OFF bridge steps identically).
6. **Slow-α load-bearing (per-hub):** re-assert the per-hub α is SLOW (the mean spans concept presentations); a
   fast/wrong-τ must not pass by accident (slow-per-hub research §4).
7. **6 seeds** for any GO (project standard; the de-risks already use 42–47).

### 4.2 Cheap-first de-risk (BEFORE any `sim/` edit)

**De-risk R0 (Option B, ZERO `sim/` edit, run first):** on a SMALL stream-cortex bridge (the CYCLE-95
`build_stream_bridge` harness, e.g. 64 concepts / population code), flag the read-out region `input_mean_adapt=True`
+ `input_divisive_norm=True` (slow α, tuned σ/gain), drive the learned block, read the on-bridge code, and run the
exact CYCLE-90 who/what + moat pipeline on it. **GO ⇒ the divisive+adapt SHIPPED primitives already realize the
read-out normalization on-bridge with NO `sim/` edit — the shortcut closes outright.** PARTIAL/NEGATIVE ⇒ divisive ≠
subtractive enough here; localizes that the per-concept op specifically needs the subtractive twin → Option A.

**De-risk R1 (Option A justification, only if R0 < bar):** a numpy/bridge smoke confirming the SUBTRACTIVE
per-concept op (the validated `neural_norm` arithmetic) on-substrate clears the bar where divisive fell short — the
empirical justification for the small `sim/` clone. (The `neural_norm` numpy proxy is already 6-seed GO at 96%; R1
is the on-bridge confirmation that subtractive specifically is the needed arithmetic.)

### 4.3 GO bars (for the closure claim)

- The on-bridge read-out code (NO host `double_center`) reproduces **who/what recall == host** and **moat abstain
  with 0 false-accepts**, **6 seeds**, on the real stream-learned codes (the burndown-#5 bars, now on-bridge not
  numpy-proxy).
- Structure recovery ≥ 0.90× host; generalization + familiarity gap preserved.
- Both ops ablation-load-bearing; on-bridge-not-host asserted (the `double_center` call removed from the path).
- IF a `sim/` edit (Option A/C): OFF == byte-identical for every existing run.

### 4.4 VERDICT

**CLOSEABLE ON-BRIDGE.** This is NOT a boundary and NOT a deep-frontier item — it is the **literal circuit
realization of an already-correctness-GREEN neural specification**, on the legitimate (separable, diagonal+DC) side
of the Mikulasch-Priesemann line, with MOST of the machinery already shipped + byte-reviewed.

- **Cheapest path = Option B: ZERO `sim/` edit** — wire the SHIPPED `input_divisive_norm` (per-concept) +
  `input_mean_adapt` (per-hub) primitives at the cortex read-out and run R0. The per-hub half is an exact reuse; the
  per-concept half rides on the divisive≈subtractive-in-log identity the bridge code already asserts and S5
  partially validated. If R0 is GO, the shortcut closes with no core edit.
- **Fallback = Option A: a SMALL guarded SUBTRACTIVE per-concept feedforward-inhibition `sim/` primitive
  (BYTE-REVIEW REQUIRED)** — a literal clone of the reviewed divisive block (`divide → subtract`) on a new mask, the
  exact `neural_norm` arithmetic, default-off byte-identical. Low risk, only triggered if R0 shows divisive ≠
  subtractive enough.
- **Option C (real FS interneuron pools)** is the maximal-fidelity build, deprioritized (MEDIUM tuning risk, not
  needed for correctness). **Option D (`graded_lateral`)** is the WRONG tool (cross-neuron, the deferred
  off-diagonal frontier) — ruled out.

The genuine residual is two host lines at offline code-gen, not a runtime turn op; the correctness is already
proven; the per-hub op is already on-substrate; the one possible new primitive is a ~6-line guarded clone of an
existing reviewed one. This is a clean conversion, ranked low-risk, with a zero-`sim/`-edit best case.

---

## 5. Files / evidence map

- **The host residual:** `research/runners/_phaseB_onbridge_stream_cortex_derisk.py:51-52,160-165` (mechanism),
  `research/runners/_phaseB_onbridge_stream_conversation_derisk.py:~116-120` (conversation, `--readout-norm`
  default `host`).
- **The validated neural spec (numpy proxy):** `research/runners/_phaseB_biologize_readout_norm_derisk.py`
  (`neural_norm` = per-hub subtractive + per-concept subtractive FFI, rate-coded-pool noise).
- **Correctness GREEN (end-to-end):** `research/findings/2026-06-20-burndown-5-ppmi-norm-onbridge.md` (who/what ==
  host, moat 0-FA, 3 seeds); `tests/test_ppmi_readout_norm_conversation.py` (11 tests).
- **Structure-proxy GO + the 4-piece map:** `research/findings/2026-06-16-biologization-sweep-conversational-
  pipeline.md` (piece 4 = 96% of host, both ops load-bearing).
- **The fork diagnosis (per-neuron mean is the separable diagonal, NOT the MP wall):** `research/findings/2026-06-15-
  slow-perhub-mean-primitive-deep-research.md` (Option A = the shipped `input_mean_adapt`).
- **SHIPPED bridge primitives (reuse targets):**
  - per-hub subtractive: `sim/bridge.py:6240-6269` (step), `:294-306,1305-1328` (alloc/mask),
    `sim/config.py:464-488` (cfg), `sim/regions.py:219-234` (`input_mean_adapt`).
  - per-concept divisive: `sim/bridge.py:6196-6218` (step), `:1330-1348` (alloc/mask), `sim/config.py:490-503`
    (cfg), `sim/regions.py:235-...` (`input_divisive_norm`); SECOND pool clone `bridge.py:6220-6238` (the
    clone-a-normalization-pool precedent).
  - cross-neuron decorrelation (ruled OUT for this op): `sim/bridge.py:1905-2013` (`graded_lateral`).
- **S5 divisive-norm precedent (zero `sim/` edit, the same primitive for the cleanup-score seam):**
  `research/findings/2026-06-20-S5-divisive-norm-derisk.md`.
- **Catalog cites:** E.05 lateral inhibition/center-surround (`feature-catalog.md:1391-1401`, Kandel 6e Ch22
  p588-593); B.06 PV+ FS feedforward inhibition (`:488-503`, Ch38 p935); I.08 M-current (`:~3320-3330`), I.13
  SK/sAHP (`:3379-3392`, Ch10). Carandini-Heeger divisive normalization: no dedicated catalog entry; cited in-code
  at `bridge.py:6198` (Nature Rev Neurosci 2012).
- **Burndown framing:** `research/findings/2026-06-20-shortcut-burndown-inventory.md:60,133-135,164-166` (#5,
  CLEAN-CONVERSION, P3, "offline learning-pipeline op").
