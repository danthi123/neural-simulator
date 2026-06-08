# GABA_B → GIRK slow inhibitory conductance — protected `sim/` edit design

**Date:** 2026-06-08
**Type:** Deep-research / pre-edit design doc (READ-ONLY; no code changed, no GPU run). Scopes the
**protected `sim/` edit** before it is written, per the standing practice ("deep research + catalog
review FIRST at roadblocks and new directions") and the protected-edit bar (owner byte-reviews every
`sim/` change; the surface must be additive, default-off, and byte-identical when unused).
**Predecessors (read in full):**
- Stage-B anatomy/scoping: `research/findings/2026-06-08-spiking-snc-stageB-striosome-critic-research.md`
- Cheap-first de-risk + the failure mode: `research/findings/2026-06-08-spiking-snc-stageB-critic-derisk.md`
- Option-B′ circuit research: `research/findings/2026-06-08-spiking-snc-stageB-Bprime-value-subtraction-circuit-research.md`
- The de-risk probe (to extend): `research/runners/snc_stageb_critic_probe.py`

**The problem this edit fixes (one paragraph).** The spiking-SNc actor-critic's neural value critic
*learns* a cue-gated, state-dependent value `V` robustly (validated multi-seed: V rises 20–25 → 90–104 Hz
on the CS, cue-gated 3/3, omission-dip 3/3). What does **not** emerge is a **strong, sign-correct
subtraction of that value at/around the SNc dopamine cell** — the `state-specific SNc gap` gate failed
0/3 across three circuit variants (direct membrane GABA, B′-EXC excitatory relay, B′-SNr disinhibition).
Root cause, multiply confirmed against both biology and source: **the engine models only GABA_A** — a
single inhibitory current `I_syn = g_i·(E_GABA − V)` with a per-region `E_GABA`
(`sim/kernels.py:208–215`; `sim/bridge.py:5322–5331`). The SNc dopamine cell lacks the KCC2 chloride
exporter, so its `E_GABA ≈ −55 mV` (depolarized, faithfully modeled at `g11_bg_runner.py:859`); direct
GABA_A onto it is weak/shunting and sign-flips with the operating point (the de-risk: raising the
`striosome→snc` weight moved the predicted burst the *wrong way*). Real biology subtracts the expected
reward onto DA cells through a **non-chloride** mechanism the sim does not have: local GABA interneurons
recruit **GABA_B (metabotropic) receptors → G-protein inwardly-rectifying K⁺ (GIRK) channels, reversal
E_K ≈ −90 mV** — a genuinely, strongly hyperpolarizing, slow (~100–300 ms) conductance (Eshel 2015;
Cohen 2012; Tepper & Lee PBR-160 ch 11; the baclofen-evoked DA IPSP reverses at −90 mV). The B′ research
explicitly named this as the **one place a protected edit changes the trade-off**: "the GABA_B/GIRK arm …
is a *future protected feature* (a second inhibitory conductance with `E ≈ −90 mV`) … If the owner ever
wants the *direct* membrane subtraction to be strong, that conductance — not a circuit workaround — is
the principled addition." This doc designs exactly that conductance.

---

## 1. Executive summary + recommendation

**The recommended edit is a second, slow, hyperpolarizing inhibitory conductance `g_gabab`** with its
own reversal `E_gabab ≈ −90 mV` (the GIRK potassium reversal) and a slow decay time constant
(`gabab_tau_decay ≈ 150 ms`, configurable), added to the membrane as `I_gabab = g_gabab·(E_gabab − V)`,
incremented per step by firing routed **only through GABA_B-designated synapses**, and **default-off /
per-pathway opt-in** so it is byte-identical when unused. This is structurally the **NMDA pattern run in
reverse**: the engine already carries a second *excitatory* conductance (`g_nmda`) with its own decay,
its own reversal, a per-neuron mask, and an additive current term (`sim/bridge.py:5405–5425`). GABA_B is
the inhibitory mirror — same allocation, same decay-caching, same per-step increment, same additive-current
shape — with three differences: (a) the reversal is −90 mV not 0 mV; (b) the increment comes from a
**presynaptically-tagged subset** of synapses (a designated GABA_B pathway), not from all excitation; (c)
the decay is slower than even NMDA (GABA_B/GIRK is the slowest fast-ligand-gated conductance in the model).

**Why a conductance, not a circuit.** The three B′ circuit workarounds all failed for the same reason:
on a GABA_A-only substrate, the *only* strong action available is excitation (`g_e·(0 − V)`, full driving
force) or GABA onto a *normal-reversal relay* — but the value must ultimately reach the **DA cell**, whose
depolarized reversal makes the *delivered* GABA weak no matter how the upstream circuit is arranged (the
B′-SNr "final hop still lands GABA on the depolarized SNc" caveat). A new conductance with E_K = −90 mV is
the **postsynaptic** fix the circuit cannot supply: the GABA reversal is a property of the *postsynaptic*
neuron's ion gradient, and GIRK opens a **potassium** channel (E_K ≈ −90 mV), not a chloride channel — so
it hyperpolarizes the DA cell *strongly* and *sign-correctly* regardless of operating point. This is the
biology (§2) and it is the minimal honest model of it.

**The protected surface is small, additive, and provably byte-identical-when-off.** Concretely (full
audit in §3–§4): **one kernel** (a new `fused_gabab_decay_and_current`, a 3-line function alongside the
existing `fused_conductance_decay_and_current` — the existing kernel is **byte-unchanged**); **one bridge
current term** (a guarded `if self.cp_conductance_g_gabab is not None:` block mirroring the NMDA block,
unreached unless the feature is on); **one bridge allocation** (`cp_conductance_g_gabab` + a per-neuron
`E_gabab` array + a per-synapse `cp_gabab_synapse_mask`, all `None` by default); **one bridge increment**
(a guarded GABA_B matvec restricted to tagged synapses); **config fields** (5 new defaults beside the NMDA
block, all OFF/zero); **one `RegionPathway` field** (`receptor: str = "gaba_a"`, default preserves
current routing) **or** a per-region `enable_gabab_target` flag (§3.4 weighs both). Every Izhikevich/HH/AdEx
dynamics line, the existing `fused_conductance_decay_and_current`, and every current run are **untouched**.
Estimated diff: **~70–110 lines added, ~0 lines of existing logic modified** (the only edits to existing
lines are additive: new fields in two dataclasses, a new array reset to `None` in `__init__`, a new term
appended to `total_input_current_pA`).

**Cheap-first de-risk (§5).** Extend `snc_stageb_critic_probe.py` (CPU/numpy, no GPU, no nav build) to
tag the `striosome_value → snc` pathway as `receptor="gaba_b"` and re-run the existing 4-gate falsifier.
The gate the direct GABA_A projection FAILED 0/3 — **state-specific SNc gap** (predicted reward burst <
unpredicted by a robust margin, multi-seed, value still cue-gated) — is the single discriminator. The
anti-cheat is a **conductance lesion**: zero `g_gabab` (or the GABA_B mask) → the subtraction must vanish,
proving it is carried by the new conductance, not host arithmetic or the weak GABA_A path.

**Honest uncertainty, flagged up front.** (Established biology) DA cells lack KCC2 → depolarized GABA_A
reversal; GABA_B → GIRK opens a K⁺ conductance reversing at E_K ≈ −90 mV that strongly hyperpolarizes DA
cells (the baclofen −90 mV reversal is directly measured); local VTA GABA neurons subtract expected reward
(Eshel/Cohen). (Modeling choices, stated as such) (a) a **single-state** conductance with one decay τ is a
*phenomenological* model of GABA_B/GIRK — the receptor is metabotropic with a multi-step G-protein cascade
and **supralinear (cooperative) dose-dependence** (Destexhe-Sejnowski 1995); the recommended first version
drops the cooperativity (a sigmoidal/Hill term is a ranked option in §6); (b) the **routing-by-presynaptic-
pathway** is a modeling convenience — in biology GABA_B vs GABA_A is set by *postsynaptic receptor
expression* and *spillover/volume transmission*, not by which axon fired; the design supports both a
per-pathway and a per-postsynaptic-region opt-in and recommends the per-pathway form for the critic
(§3.4); (c) the slow τ is set from GIRK-IPSC measurements (rise ~tens of ms, decay ~150–500 ms); the exact
value is a tunable, not a hard datum.

---

## 2. Biology of GABA_B / GIRK on dopamine neurons

Terms defined once. **GABA_A receptor:** an ionotropic (ligand-gated ion channel) receptor permeable to
chloride (Cl⁻); fast (ms); its reversal `E_GABA` is set by the cell's chloride gradient (the KCC2
exporter sets a low intracellular [Cl⁻] → hyperpolarized E_GABA in most mature neurons). **GABA_B
receptor:** a *metabotropic* (G-protein-coupled, no intrinsic pore) receptor; binding activates a Gi/o
G-protein whose βγ subunits open a separate channel — the **GIRK** (G-protein inwardly-rectifying K⁺;
Kir3) channel. **GIRK:** a *potassium* channel; its reversal is the potassium reversal **E_K ≈ −90 mV**,
well below threshold, so opening it **strongly hyperpolarizes** the cell. The whole cascade is slow
(metabotropic): the IPSC rises over tens of ms and decays over ~150–500 ms (vs ~10 ms for GABA_A).

**2.1 Why direct GABA_A fails on the SNc — the established datum.** SNc dopaminergic neurons **do not
express KCC2** (Gulácsi et al. 2003; catalog **B.15**, `feature-catalog.md:351–358`), so their chloride
reversal `E_Cl ≈ −55 mV` sits near AP threshold. GABA_A IPSPs are therefore "only weakly hyperpolarizing
and frequently shunting near threshold," and DA cells are "remarkably resistant to direct striatal/
pallidal GABA inhibition" (B.15). This is the exact wall the de-risk hit (raising `striosome→snc` weight
moved the predicted burst the *wrong* way; high tonic flipped the sign but the gap stayed tiny).

**2.2 GABA_B/GIRK is the strong, non-chloride hyperpolarizing arm — the established fix.** Because GIRK
is a **potassium** channel, its reversal (E_K ≈ −90 mV) is **independent of the chloride gradient** and is
genuinely, strongly hyperpolarizing on the DA cell. Direct intracellular recordings from SNc/VTA DA
neurons show **GABA_B IPSPs of 10–20 mV**, and **baclofen (a selective GABA_B agonist) evokes a slow
hyperpolarization that reverses at −90 mV** (the slice literature, e.g. the dopamine-modulates-GABA_B
work, PMC2290228; Beckstead & Williams show GABA_B activates the *same* GIRK conductance in DA neurons as
the D2 autoreceptor). So the strong value-subtraction biology uses on DA cells is **GABA_B → GIRK**, not
more GABA_A. This is precisely the conductance the engine lacks.

**2.3 The local-interneuron prediction arm (why this is exactly Stage B).** The direct analogue of what
the critic wants is the VTA/SN local GABA interneuron:
- **Cohen et al. 2012 (Nature 482:85):** VTA GABA neurons show **persistent activity during the
  cue→outcome delay, parametrically proportional to expected reward, and *not* modulated by reward
  delivery/omission** (16/17 encode reward size, P<0.001). They "encode expectation about rewards" and
  "synapse preferentially onto dendrites of dopaminergic neurons" — local, carrying the *prediction* `V`.
- **Eshel et al. 2015 (Nature 527:398):** optogenetically exciting these neurons *subtracts* the DA reward
  response; inhibiting them *increases* the DA response to expected reward; the DA response function shows
  a **linear (subtractive), not divisive, shift** — the canonical demonstration that DA cells compute
  `δ = r − V` by **subtractive inhibition from a local GABA population that carries V**.
That subtraction is delivered (in part) via GABA_B/GIRK onto the DA dendrites — the slow, hyperpolarizing,
non-chloride arm. The critic's `striosome_value` population is the project's stand-in for this
prediction-carrying GABA population; routing its projection through GABA_B/GIRK (E_K = −90 mV) is the
faithful translation of the Eshel/Cohen subtraction onto this substrate.

**2.4 Kinetics — what to model.** Established/measured: reversal **E_K ≈ −90 mV** (baclofen reversal,
directly measured on DA cells); amplitude 10–20 mV slow IPSP; **slow** time course — the GIRK conductance
that GABA_B gates has a **rise of tens of ms (time-to-peak ~150–200 ms)** and **decay τ of ~150–500 ms**
(the GIRK-mediated D2-IPSC on the same channel rises in ~200 ms and decays in ~500 ms; GABA_B IPSCs on
the same conductance are comparably slow). Mechanistically the metabotropic cascade is **cooperative**:
the G-protein activation is supralinear in agonist, which is *why* hippocampal/thalamic GABA_B responses
have a sigmoidal stimulus-response and a delayed onset (Destexhe & Sejnowski 1995, the canonical
four-state cooperative kinetic model). **Modeling recommendation (first version):** a *single*
exponentially-decaying conductance with `gabab_tau_decay ≈ 150 ms` (and, if a rise is wanted, a
dual-exponential like NMDA's `g_slow − g_rise` reusing the existing NMDA-style decay caching). Drop the
G-protein cooperativity in v1 (it is a sigmoidal gain on the increment — a ranked option in §6, not needed
for the value-subtraction de-risk). Document the single-state model as a phenomenological abstraction of a
metabotropic receptor.

**2.5 Honest scope of the biology claim.** Established: the depolarized DA GABA_A reversal; GABA_B→GIRK as
the strong non-chloride hyperpolarizing arm (E_K ≈ −90 mV); local GABA interneurons subtract expected
reward. Modeling choices: single-state vs cooperative kinetics; routing GABA_B by presynaptic pathway
(biology sets it by postsynaptic receptor expression + spillover); the exact τ. None of these blocks the
edit; all go into the result's "modeling choices" table.

---

## 3. The minimal protected `sim/` edit — design

**Design principle: mirror the NMDA conductance, inverted.** The engine *already* carries a second
synaptic conductance with its own reversal, its own decay, a per-neuron mask, and an additive current term
— NMDA (`sim/bridge.py:5405–5425`, allocation `:1124–1144`, decay-cache `:1501`). GABA_B is the
inhibitory mirror. Reusing that exact shape keeps the edit minimal, reviewable, and consistent with a
mechanism the owner already trusts. The **one** structural difference from NMDA: NMDA's increment piggybacks
on `g_e_increase` (all excitation gets NMDA), but GABA_B must be **selective to designated synapses** (only
the value pathway), so it needs its own restricted matvec + a per-synapse tag (the `cp_synapse_action_tag`
pattern, `:2242–2257`).

### 3.1 The kernel — `sim/kernels.py` (ADD one function; existing kernel byte-unchanged)

The existing inhibitory kernel is (verified `sim/kernels.py:207–215`):
```python
@fuse()
def fused_conductance_decay_and_current(g_e, g_i, decay_e, decay_i, v, E_e, E_i):
    g_e_new = g_e * decay_e
    g_i_new = g_i * decay_i
    I_syn = g_e_new * (E_e - v) + g_i_new * (E_i - v)
    return g_e_new, g_i_new, I_syn
```
**Do NOT modify it.** Add a sibling (keeps the GABA_A path byte-identical; the new kernel is unreached
unless the feature is on):
```python
@fuse()
def fused_gabab_decay_and_current(g_gabab, decay_gabab, v, E_gabab):
    """Slow GABA_B -> GIRK K+ inhibitory conductance (E_gabab ~ -90 mV, the
    potassium reversal). Metabotropic/slow: decay_gabab = exp(-dt/tau) with
    tau ~150 ms, far slower than GABA_A (~10 ms). Mirrors the AMPA/NMDA pattern
    inverted: a hyperpolarizing K+ current independent of the chloride gradient,
    so it strongly inhibits KCC2-lacking DA cells where GABA_A is weak/shunting."""
    g_gabab_new = g_gabab * decay_gabab
    I_gabab = g_gabab_new * (E_gabab - v)
    return g_gabab_new, I_gabab
```
*(Alternative dual-exponential form mirrors `fused_nmda_update_and_current` with `g_slow − g_rise` for a
finite rise; recommend the single-exponential above for v1 — GABA_B's rise is slow but a single decay
captures the dominant slow IPSC, and it is the smaller surface.)*
**Diff:** +~10 lines, 0 modified. Add `fused_gabab_decay_and_current` to the bridge import block
(`sim/bridge.py:80–93`) — +1 line.

### 3.2 The config fields — `sim/config.py` (ADD beside the NMDA block; all OFF/zero by default)

Beside `enable_nmda` (`sim/config.py:132–139`), add (all defaults make the feature inert):
```python
# GABA_B -> GIRK slow K+ inhibitory conductance (metabotropic). Default OFF;
# byte-identical when disabled. E_gabab is the POTASSIUM reversal (~-90 mV),
# independent of the chloride gradient, so it strongly hyperpolarizes KCC2-lacking
# DA cells where GABA_A (E_GABA ~ -55 mV) is weak/shunting. See catalog B.15, J.11.
enable_gabab: bool = False
gabab_reversal_potential: float = -90.0   # E_K (GIRK), mV
gabab_tau_decay: float = 150.0            # slow decay (ms); GABA_B/GIRK >> GABA_A's 10 ms
gabab_propagation_strength: float = 0.105 # per-spike conductance increment scale (mirrors inhibitory_propagation_strength)
```
**Diff:** +4 fields. (No existing field changes.) These are read only inside guarded `enable_gabab`
blocks, so a default config is byte-identical.

### 3.3 The bridge — `sim/bridge.py`

**(B1) `__init__` array declarations** (beside `cp_conductance_g_nmda = None` at `:231`): add
```python
self.cp_conductance_g_gabab = None          # slow GABA_B/GIRK conductance (None unless enable_gabab)
self.cp_gabab_reversal_per_neuron = None     # per-neuron E_gabab (~-90 mV on GABA_B targets)
self.cp_gabab_synapse_mask = None            # bool per-synapse: True for GABA_B-routed synapses
```
**Diff:** +3 lines (all `None`).

**(B2) Allocation** (inside `_initialize_simulation_data`, beside the NMDA allocation at `:1124–1144`),
guarded by `enable_gabab`:
```python
if getattr(cfg, "enable_gabab", False) and n > 0:
    self.cp_conductance_g_gabab = cp.zeros(n, dtype=cp.float32)
    # Per-neuron E_gabab: -90 mV (GIRK K+) on GABA_B target regions, NaN/unused else.
    # (Mirror the per-neuron E_GABA build at :1086-1101.)
    self.cp_gabab_reversal_per_neuron = cp.full(n, cfg.gabab_reversal_potential, dtype=cp.float32)
    # Per-synapse GABA_B routing mask: True where the synapse is GABA_B (set in inject_explicit_wiring).
    # Allocated there alongside cp_synapse_action_tag; left None here.
```
The **per-synapse GABA_B mask** is built in `inject_explicit_wiring` next to `cp_synapse_action_tag`
(`:2242–2257`), keyed on the pathway's `receptor` field (or post-region flag, §3.4):
```python
if getattr(cfg, "enable_gabab", False) and self.region_manager is not None:
    self.cp_gabab_synapse_mask = cp.zeros(nnz, dtype=bool)
    # mark synapses belonging to GABA_B-designated pathways (post-region in a GABA_B set,
    # or pathway.receptor == "gaba_b"); see wiring-plan threading in §3.4.
```
**Diff:** +~12 lines, all inside `enable_gabab` guards.

**(B3) Decay-factor caching** (beside `_cached_decay_nmda` at `:1501` and `:6744`):
```python
self._cached_decay_gabab = float(cp.exp(-cfg.dt_ms / cfg.gabab_tau_decay)) if getattr(cfg, "gabab_tau_decay", 0) > 0 else 0.0
```
**Diff:** +1 line in each of the two cache sites (init `:1501`, reconfigure `:6744`).

**(B4) The per-step current term** (the load-bearing edit — a guarded block mirroring the NMDA block at
`:5405–5425`, inserted right after it). The GABA_B increment comes from a **restricted matvec** over only
the tagged synapses:
```python
# --- 2.3b. GABA_B -> GIRK slow K+ inhibition (metabotropic; E_K ~ -90 mV) ---
if getattr(cfg, "enable_gabab", False) and self.cp_conductance_g_gabab is not None:
    # Increment from GABA_B-tagged synapses only (restricted matvec; the value pathway).
    if (self.cp_gabab_synapse_mask is not None and effective_connections_matrix.nnz > 0 and _prev_any):
        _nnz = self.cp_connections.nnz
        _gb_data = effective_connections_matrix.data * self.cp_gabab_synapse_mask[:_nnz]
        _gb_mat = csp.csr_matrix((_gb_data, self.cp_connections.indices, self.cp_connections.indptr),
                                 shape=self.cp_connections.shape)
        gabab_increase = (_gb_mat.T @ self.cp_prev_firing_states.astype(cp.float32)) * cfg.gabab_propagation_strength
        self.cp_conductance_g_gabab += gabab_increase
    self.cp_conductance_g_gabab, I_gabab = fused_gabab_decay_and_current(
        self.cp_conductance_g_gabab, self._cached_decay_gabab,
        self.cp_membrane_potential_v, self.cp_gabab_reversal_per_neuron)
    total_input_current_pA = total_input_current_pA + I_gabab
```
This is exactly the NMDA pattern (decay the conductance, compute the additive current, add to
`total_input_current_pA`), plus the action-tag-style restricted matvec for the GABA_B subset. **When
`enable_gabab` is False, `cp_conductance_g_gabab is None`, the whole block is skipped, and
`total_input_current_pA` is byte-identical to today.**
**Diff:** +~14 lines, fully guarded.

**(B5) Checkpoint save/load (optional, for completeness).** Add `cp_conductance_g_gabab` to the saved
array set only if persistence is wanted; not required for the de-risk or nav (the conductance self-recovers
in ~one τ of free running, like the existing conductances which are *not* checkpointed). **Recommend
deferring** — keeps the surface smaller. (The per-neuron reversal `cp_gabab_reversal_per_neuron` and the
mask `cp_gabab_synapse_mask` are rebuilt from config + wiring on load, like `cp_syn_reversal_potential_i_per_neuron`
and `cp_synapse_action_tag`, so no checkpoint change is needed for correctness.)

### 3.4 How a pathway is tagged GABA_B vs GABA_A (the one real design decision)

The current inhibitory routing is **presynaptic-trait-based**: every inhibitory neuron's firing flows into
the *single* `g_i` via one matvec (`sim/bridge.py:5346–5358`); there is **no per-pathway split today**.
GABA_B needs a *subset* (the value pathway → SNc), so a tag is required. Two clean options, both reusing
existing patterns:

- **Option (a) — per-`RegionPathway` `receptor` field (RECOMMENDED).** Add to `RegionPathway`
  (`sim/regions.py:221–248`):
  ```python
  receptor: str = "gaba_a"   # "gaba_a" (default, unchanged routing) | "gaba_b" (slow GIRK, E_K=-90mV)
  ```
  Thread it through the wiring plan (`build_wiring_plan` / `_build_pathway`) so `inject_explicit_wiring`
  can mark the synapses of any `receptor=="gaba_b"` pathway in `cp_gabab_synapse_mask`, AND set those
  *post* neurons' `cp_gabab_reversal_per_neuron` to −90 mV. **Pros:** the critic→SNc projection is the
  natural unit to tag; mirrors how `transmission_gate`/`plasticity_gate`/`neuromodulator_gates` are
  per-pathway; default `"gaba_a"` is byte-identical. **Cons:** the wiring-plan dict needs to carry one
  more key (a few lines in `regions.py`).
- **Option (b) — per-`BrainRegion` `enable_gabab_target` flag (alternative).** Mirror `enable_nmda`
  (`sim/regions.py:112`): a post-region flag meaning "inhibitory input to this region uses GABA_B." Build
  the mask from `cp.isin(cp_connections.indices, post_set)` exactly like the NMDA mask (`:1132–1141`) and
  the action tag (`:2242–2257`). **Pros:** the *smallest* surface (no `RegionPathway` change, no
  wiring-plan threading — pure `cp.isin` on post indices, the literal action-tag code). **Cons:** routes
  **all** inhibition into the region through GABA_B, not just the value pathway — fine for the SNc (whose
  only modeled inhibition we *want* to be GABA_B-like) but coarser; and biologically GABA_B is
  receptor-expression-based, which is post-region-level, so (b) is arguably the *more* faithful framing.

**Recommendation:** ship **(a)** `receptor` on `RegionPathway` as the primary (it isolates the value
pathway and matches the per-pathway idiom), and note (b) as the smaller-surface fallback if the
wiring-plan threading proves fiddly. Both are zero-modification to existing routing (default preserves
GABA_A everywhere).

---

## 4. Byte-level protected surface (the audit)

Every protected file + function touched, the nature of the change, and why it is byte-identical when the
feature is off. **All edits are additive; the only modifications to *existing* lines are new dataclass
fields, new `None` declarations in `__init__`, and one new term appended to `total_input_current_pA`
inside a guard.**

| File | Function / location | Change | Byte-identical when off? |
|---|---|---|---|
| `sim/kernels.py` | new `fused_gabab_decay_and_current` (after `fused_conductance_decay_and_current`, `:215`) | **ADD** ~10-line kernel. The existing `fused_conductance_decay_and_current` is **unchanged**. | Yes — new function never called unless `enable_gabab`. |
| `sim/bridge.py` | import block `:80–93` | **ADD** `fused_gabab_decay_and_current` to the import tuple | Yes — import only. |
| `sim/config.py` | `CoreSimConfig`, beside NMDA `:132–139` | **ADD** 4 fields (`enable_gabab=False`, `gabab_reversal_potential=-90`, `gabab_tau_decay=150`, `gabab_propagation_strength=0.105`) | Yes — read only inside `enable_gabab` guards; default config has the feature off. |
| `sim/bridge.py` | `__init__` `:231` | **ADD** 3 `None` array declarations | Yes — `None` ⇒ all guards skip. |
| `sim/bridge.py` | `_initialize_simulation_data` alloc `:1124–1144` | **ADD** `enable_gabab`-guarded allocation of `cp_conductance_g_gabab` + per-neuron `E_gabab` | Yes — guard is False by default; arrays stay `None`. |
| `sim/bridge.py` | decay caches `:1501`, `:6744` | **ADD** `_cached_decay_gabab` line at each | Yes — value computed but unused unless block (B4) runs. |
| `sim/bridge.py` | `inject_explicit_wiring` `:2242–2257` (next to `cp_synapse_action_tag`) | **ADD** `enable_gabab`-guarded build of `cp_gabab_synapse_mask` + set GABA_B post-neuron reversals to −90 mV | Yes — guard off ⇒ mask stays `None`. |
| `sim/bridge.py` | `_run_one_simulation_step`, **after** the NMDA block `:5425` | **ADD** the guarded GABA_B current block (B4); appends `I_gabab` to `total_input_current_pA` | Yes — `cp_conductance_g_gabab is None` ⇒ block skipped ⇒ `total_input_current_pA` unchanged. |
| `sim/regions.py` | `RegionPathway` `:221–248` (Option a) | **ADD** `receptor: str = "gaba_a"` field + thread through `build_wiring_plan`/`_build_pathway` | Yes — default `"gaba_a"` ⇒ no synapse tagged ⇒ identical routing. |

**Explicit byte-identity argument.** The Izhikevich/HH/AdEx dynamics kernels and their call sites
(`sim/bridge.py:5448–5500+`) are **not touched**. `fused_conductance_decay_and_current` is **not touched**
(the GABA_A current is computed exactly as today). The single new term in `total_input_current_pA` lives
inside `if ... self.cp_conductance_g_gabab is not None:`, which is `None` for every existing run (no
config sets `enable_gabab`), so the assembled current is **bit-for-bit** what it is today. The new
dataclass fields default to off/zero. New `__init__` arrays default to `None`. This satisfies the
protected-edit bar (additive, default-off, byte-identical-when-unused). **Diff estimate: ~70–110 lines
added across 4 files; existing-line modifications limited to additive field/declaration insertions.**

**Per-synapse-array growth gotcha (noted, handled).** The repo has a documented hazard: structural
plasticity / synapse growth can leave per-synapse arrays (e.g. `cp_d1_d2_sign`, `cp_transmission_gain`)
shorter than `cp_connections.nnz`, dropping updates silently (fixed via `_ensure_gate_capacity`,
`sim/bridge.py:770–791`). `cp_gabab_synapse_mask` is a per-synapse array, so it inherits this hazard.
**Mitigations (use both):** (i) the de-risk probe already sets `enable_structural_plasticity = False`
(`snc_stageb_critic_probe.py:97`), so the mask never needs to grow in the de-risk; (ii) for nav, either
slice the mask to `[:nnz]` at the use site (as block B4 already does: `self.cp_gabab_synapse_mask[:_nnz]`)
and/or add `cp_gabab_synapse_mask` to the `_ensure_gate_capacity` family. Slicing-at-use is the cheapest
correct mitigation and is already in the B4 snippet. (A bool mask grows with **False** padding — new
untagged synapses are correctly GABA_A by default.)

---

## 5. Cheap-first de-risk + anti-cheat

### 5.1 Extend the existing probe (CPU/numpy, no GPU, no nav build)
`research/runners/snc_stageb_critic_probe.py` already builds the minimal `cue → striosome_value → snc`
bridge, calibrates the dopamine threshold, runs the 4-gate falsifier, and has a `--lesion` anti-cheat — all
under `SIM_BACKEND=numpy`. The B′ scaffolding (`--bprime`, `--bprime-snr`) is already present. Add a
**`--gabab`** mode that:
1. Sets `cfg.enable_gabab = True` and tags the `striosome_value → snc` pathway `receptor="gaba_b"` (so its
   inhibition routes through the new slow K⁺ conductance with E_gabab = −90 mV) — **instead of** the weak
   GABA_A `str→snc` of the direct baseline. Keep the SNc's own GABA_A reversal at −55 mV (unchanged); the
   GABA_B current is a *separate*, parallel hyperpolarizing term on the same SNc neurons.
2. Reuses the four harness fixes the de-risk already found (do not regress): advance
   `runtime_state.current_time_ms` each step (STDP Δt); `stdp_w_max = 40` (soft-bound); STP off for this
   minimal mechanism; auto-calibrate the DA threshold to the measured tonic firing fraction
   (`snc_stageb_critic_probe.py:282–308` already do all four).
3. Keeps `cue → striosome_value` plastic (`plasticity_gate="value_input"`), trained by the SNc-derived
   `da_signal` exactly as now — the value-learning mechanism is unchanged; only the *subtraction conduit*
   becomes GABA_B/GIRK.

### 5.2 PASS / FAIL criteria (multi-seed, ≥3 seeds: 42/43/44)
- **(i) STATE-SPECIFIC SNc GAP — the unique discriminator (the gate the direct GABA_A projection FAILED
  0/3).** After training, learning frozen: **predicted** reward (CS+US) burst **<** **unpredicted** reward
  (US-alone) burst by a **robust margin**. Quantitative gate (mirror the probe's `state_specific`
  `:490`): `unpredicted_rate > 1.30 × predicted_rate`, **sign-consistent across ≥3 seeds**, with the gap
  surviving a modest sweep of `gabab_propagation_strength` / `gabab_tau_decay` (robustness, not a single
  lucky operating point). **This is the whole point of the edit** — a strong, sign-correct membrane
  subtraction the GABA_A projection could not deliver. A host global-EMA value cannot produce a
  cue-specific gap (the probe docstring `:23–26`), so a robust gap is positive proof the subtraction is
  both neural and state-dependent.
- **(ii) V STILL CUE-GATED (regression guard, was 3/3).** `predicted/omission` striosome rate ≫
  `unpredicted/baseline` striosome rate — the value must remain state-specific (`v_learned` + cue-gating,
  `:488`). The edit must not break the validated value-learning.
- **(iii) OMISSION DIP (regression guard, was 3/3).** CS with no reward → SNc dips below tonic
  (`omit_r < base_r`, `:491`). With a *strong* GABA_B subtraction this should be **cleaner/deeper** than
  the GABA_A baseline.
- **(iv) US-BURST SHRINK (R-W signature, was the value-learning consequence, `:489`).** Across training the
  reward burst shrinks as `V` cancels `r`. *(Honest scope from the de-risk: this is Rescorla-Wagner
  `δ = r − V`, not the full TD cue-shift burst-migration onto the CS — the TD bootstrap is a deeper, later
  increment, orthogonal to this conductance edit.)*

**Edit success = gate (i) now PASSES multi-seed while (ii)–(iv) are retained** — the precise delta over
the GABA_A direct-projection de-risk and the B′ circuit attempts.

### 5.3 Anti-cheat — proving the subtraction is the new conductance, not host arithmetic or GABA_A
1. **Conductance lesion (decisive).** Zero the GABA_B conductance — either set `cfg.enable_gabab = False`
   on a re-run with the same trained weights, or zero `cp_gabab_synapse_mask` (cut the GABA_B conduit) —
   and re-run the test phase. The state-specific gap (i) and the (deepened) omission dip (iii) must
   **vanish** (the SNc bursts to *every* reward regardless of prediction). If the subtraction survived,
   it would prove it was the residual weak GABA_A path or host arithmetic in disguise; with the GABA_B
   conduit cut it must not. (Extend the existing `_lesion_pathway`, `:326–342`, to zero the mask.)
2. **GABA_A-only control (the contrast that localizes the win to GABA_B).** Run the *same* circuit with
   `receptor="gaba_a"` (the de-risk baseline): gate (i) should still FAIL (reproducing the 0/3 wall),
   while `receptor="gaba_b"` PASSES. This is the clean A/B that attributes the fix to the new conductance,
   not to anything else that changed.
3. **Provenance assertion.** Under `--gabab`, assert the host `_V_scaffold` term is **removed** and the
   SNc current is `tonic/reward drive + (GABA_B synaptic hyperpolarization)` only — no host `V`/`reward_ema`
   reaches the SNc (mirror the de-risk's `current_reward_signal = 0.0` brain-based stance, `:99`).
4. **Coordinate-freedom assertion.** The critic's afferents are perceived-state regions only; combined
   with the perceived-reward work, the whole RPE loop references no coordinate.

### 5.4 Nav-score regression gate (necessary, not sufficient — only after the probe passes)
Flagship multi-goal-deterministic 6-seed (A+E+G v2.5) with `--spiking-snc --enable-neural-critic` +
GABA_B critic→SNc: summed reward **≥ Stage A** (which is ≥ the raw-reward baseline). An **honest negative
is a valid deliverable** (it maps a limit of the neural critic). The probe's gate (i) proves the
*mechanism* is the real biology; the nav score only proves nav did not break.

---

## 6. Ranked options + recommendation + open questions

**Recommendation: build the single-state slow GABA_B/GIRK conductance (`E_gabab = −90 mV`,
`tau_decay = 150 ms`) routed per-`RegionPathway` `receptor="gaba_b"`, default-off.** It is the principled
postsynaptic fix the three B′ circuit workarounds could not supply (the K⁺ reversal hyperpolarizes the
KCC2-lacking DA cell strongly and sign-correctly), it is the smallest faithful model of the established
biology (Eshel/Cohen subtraction via GABA_B→GIRK), and its protected surface is the NMDA pattern inverted
(additive, default-off, byte-identical-when-unused). Validate on the extended CPU probe (gate (i)
multi-seed + the conductance-lesion + the GABA_A-only A/B) **before** any nav build, then gate on the
flagship 6-seed.

| Option | Fidelity | Surface | Strength on DA cell | Verdict |
|---|---|---|---|---|
| **Single-state GABA_B/GIRK, per-pathway `receptor`** | good (phenomenological metabotropic) | small (NMDA-mirror + `receptor` field) | **strong** (E_K = −90 mV, non-chloride) | **RECOMMENDED build** |
| Single-state GABA_B/GIRK, per-region `enable_gabab_target` | good (receptor-expression framing) | **smallest** (no `RegionPathway` change; pure `cp.isin`) | strong | fallback if wiring-plan threading is fiddly |
| Dual-exponential GABA_B (`g_slow − g_rise`, NMDA-style) | better (finite rise) | +1 array, +1 decay cache | strong | optional refinement if the slow rise matters for the dip timing |
| Full cooperative Destexhe-Sejnowski (sigmoidal G-protein) | best (supralinear dose) | larger (extra state + Hill term in kernel) | strong | future; not needed for the value-subtraction de-risk |
| (Reject) circuit-only B′ workarounds | n/a | zero protected | **weak** (final GABA hop on depolarized SNc) | already failed 0/3 — this edit is the fix |

**What would change the recommendation.**
- If the per-pathway `receptor` wiring-plan threading proves awkward, switch to the **per-region
  `enable_gabab_target`** flag (smaller surface, arguably more faithful — GABA_B is post-receptor-expression).
- If the omission-dip *timing* matters (the slow rise delays the subtraction onset), add the
  **dual-exponential** rise (one more array + decay cache, the NMDA `g_slow − g_rise` shape).
- If the value-subtraction needs to be **dose-supralinear** (a real GABA_B property; matters if the critic's
  rate range is wide), add the **cooperative** Hill term on the increment — a later refinement.

**Open questions (for the owner to weigh before building).**
1. **Per-pathway `receptor` vs per-region `enable_gabab_target`.** Recommendation: per-pathway (isolates the
   value pathway; matches the gate idiom). Confirm the owner's preference — (b) is a smaller diff.
2. **τ value.** 150 ms is a defensible mid-range GIRK-IPSC decay; the measured range is ~150–500 ms.
   Start at 150 ms; the probe's gate (i) robustness sweep over `gabab_tau_decay` settles it.
3. **Checkpoint persistence.** Recommend **deferring** saving `cp_conductance_g_gabab` (it self-recovers in
   ~one τ, like the other conductances which are not checkpointed); the per-neuron reversal + mask rebuild
   from config + wiring on load. Confirm this is acceptable (keeps the surface smaller).
4. **Should the SNc's GABA_A `str_striosome_X → snc` (R3.11) and `gpi_X → snc` (R3.10) pathways be
   re-tagged GABA_B, or left GABA_A?** For the clean critic δ, the cleanest is to route the *value*
   pathway (`striosome_value → snc`) as GABA_B and leave the action-correlated GABA_A pathways as-is (or
   zero them under `--enable-neural-critic`, per the Stage-B scoping doc §6 Q5). A runner-side choice, not a
   protected-edit question.
5. **Cooperativity now or later?** The first version drops it (single-state). If the de-risk's gate (i)
   passes without it, cooperativity stays a future refinement; if the subtraction is too linear across the
   critic's rate range, add the Hill term.

---

## 7. Sources

### Project code (verified file:line this session)
- Inhibitory current kernel (`I_syn = g_e·(E_e−V) + g_i·(E_i−V)`, the GABA_A-only model): `sim/kernels.py:207–215`.
- NMDA kernel (the second-conductance template to mirror): `sim/kernels.py:217–239`.
- Membrane current assembly + GABA_A current + `E_inh_to_use` (per-neuron override): `sim/bridge.py:5316–5331`.
- Inhibitory-vs-excitatory routing from presynaptic trait (the single `g_i` matvec; **no per-pathway split today**): `sim/bridge.py:5333–5361`.
- NMDA current block (the additive-current + per-neuron-mask pattern GABA_B mirrors): `sim/bridge.py:5405–5425`.
- Conductance allocations (`g_e`, `g_i` `:1078–1079`; per-neuron `E_GABA` build `:1086–1101`; NMDA + per-neuron NMDA mask `:1124–1144`): `sim/bridge.py`.
- Decay caches (`_cached_decay_i` `:1500`, `_cached_decay_nmda` `:1501`; reconfigure path `:6743–6744`): `sim/bridge.py`.
- `cp_synapse_action_tag` per-synapse tag build (`cp.isin` on post indices — the GABA_B mask template): `sim/bridge.py:2242–2259`.
- `cp_d1_d2_sign` capacity-sizing + `_ensure_gate_capacity` (the per-synapse-array growth gotcha + its fix): `sim/bridge.py:770–791`, `:2208–2229`.
- `cp_conductance_g_gabab`/mask/reversal would declare beside: `sim/bridge.py:229–236, 290`.
- `RegionPathway` (where a `receptor` field goes; mirrors `transmission_gate`/`plasticity_gate`): `sim/regions.py:221–248`.
- `BrainRegion.enable_nmda` (the per-region opt-in flag a per-region `enable_gabab_target` would mirror) + `syn_reversal_potential_i_override`: `sim/regions.py:91–98, 112`.
- GABA_A / NMDA config block (where GABA_B fields go): `sim/config.py:128–139`.
- The de-risk probe (to extend with `--gabab`): `research/runners/snc_stageb_critic_probe.py` — SNc `E_GABA=−55` `:132`, harness fixes `:97, 105, 258, 282–308`, the 4 gates `:488–491`, lesion `:326–342`, B′ scaffolding `:141–198`.

### Project feature catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`)
- **B.15** SNc DA neurons lack KCC2 → depolarized `E_Cl ≈ −55 mV`; "DA cells remarkably resistant to direct striatal/pallidal GABA inhibition"; disynaptic disinhibition is the dominant route: `:351–358`.
- **B.14** MSN GABA_A reversal depolarized (`E_GABA = −60 mV`), shunting, KCC2 dependence: `:342–349`.
- **J.11** GABA neurotransmitter: "GABA-B (metabotropic, slow IPSP, presynaptic autoreceptor) is **missing** — would need an additional slower inhibitory channel pathway" (the exact gap this edit fills): `:678`.
- **J.13** GPCRs / metabotropic receivers (GABA-B is GPCR): `:3643–3645`. **J.02** ionotropic vs metabotropic: `:3533–3538`.
- **C.30** actor-critic mapping (SNc=δ / striosome=V(s) / matrix=actor); acceptance = cue-shift + omission dip. **C.28** TD error. (Stage-B context.)

### Peer-reviewed literature (verified via search this session)
- **Eshel N., Bukwich M., Rao V., Hemmelder V., Tian J., Uchida N. (2015)** "Arithmetic and local circuitry underlying dopamine prediction errors", *Nature* 527:398. Local VTA GABA neurons are a **source of subtraction** (linear/subtractive, not divisive shift); they carry the prediction and inhibit DA cells when reward is expected. https://www.nature.com/articles/nature14855
- **Cohen J.Y., Haesler S., Vong L., Lowell B.B., Uchida N. (2012)** "Neuron-type-specific signals for reward and punishment in the VTA", *Nature* 482:85. VTA GABA neurons encode **expected reward** (persistent delay activity, parametric, not modulated by delivery/omission); synapse onto DA dendrites (local). https://pmc.ncbi.nlm.nih.gov/articles/PMC6721851/
- **Tepper J.M. & Lee C.R. (2007)** "GABAergic control of substantia nigra dopaminergic neurons", *Prog. Brain Res.* 160 (PBR-160 ch 11). SNc DA lack KCC2 → depolarized E_GABA; **GABA_B → GIRK K⁺ channels provide the genuinely hyperpolarizing arm**; ≥70% of SN DA afferents GABAergic. https://pubmed.ncbi.nlm.nih.gov/17499115/
- **GABA_B IPSP reverses at −90 mV on DA neurons (baclofen-evoked).** Slice intracellular recordings: GABA_B IPSPs 10–20 mV; baclofen-induced slow hyperpolarization reverses at −90 mV (E_K). (Dopamine-modulates-GABA_B-transmission study) https://pmc.ncbi.nlm.nih.gov/articles/PMC2290228/
- **GABA_B and D2 activate the SAME GIRK conductance in DA neurons** (Beckstead & Williams; the GIRK-mediated D2-IPSC rises ~200 ms / decays ~500 ms — the slow GIRK time course GABA_B shares): https://www.jneurosci.org/content/29/42/13344
- **GABA_B/GIRK slow-IPSC kinetics regulated by Gβ5/RGS** (slow deactivation; the metabotropic slowness): https://elifesciences.org/articles/02053
- **Destexhe A. & Sejnowski T.J. (1995)** "G-protein activation kinetics and spill-over of GABA may account for differences between inhibitory responses in the hippocampus and thalamus", *PNAS* 92:9515 — the canonical **cooperative four-state** GABA_B→GIRK kinetic model (the supralinear dose-dependence dropped in v1). https://www.pnas.org/doi/10.1073/pnas.92.21.9515
- **Gulácsi A. et al. (2003)** "Cell type–specific differences in chloride-regulatory mechanisms… in the rat substantia nigra", *J. Neurosci.* 23:8237 — DA neurons lack KCC2 (the catalog B.15 source).
- **Kandel et al.** *Principles of Neural Science* 6e — Ch 43 (dopamine/reward), Ch 38 (basal ganglia); receptor pharmacology (GABA_A vs GABA_B), as cited in the catalog.

---

**Deliverable path:** `E:\Documents\Projects\sim\research\findings\2026-06-08-gabab-girk-conductance-design.md`
