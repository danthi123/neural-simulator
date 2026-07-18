# gap#5 emergent-DG gain-balance research gate (workflow synthesis, 2026-07-18)

# Synthesis: one buildable mechanism for a reliable, fixed-~10%, completable emergent CA3 assembly

## The three reports agree on the answer (rare, load-bearing consensus)

All three independently converge on the **same single mechanism**: replace the build-time coincidence threshold with a **live, activity-normalizing feedback-inhibition loop — the de Almeida–Idiart–Lisman 2009 E%-max k-WTA** (`10.1523/JNEUROSCI.6044-08.2009`, cited by all three). This is not three options to weigh; it is one mechanism corroborated from biology, from SNN/attractor theory, and — decisively — **already implemented in your own code** for CA1. The disagreements are only about *supporting* pieces, and I flag them below.

---

## (1) The ONE key insight

**Your sparsity is set by a static, build-time scalar (`coincidence_k_threshold`) that is decoupled from live population activity — so assembly size, amplification gain, binding gain, and inhibition are all coupled through one un-normalized loop, giving a knife-edge working point (0 ↔ runaway).** The fix is to make the *effective threshold a runtime function of the current total drive*: drive a fast PV-basket pool **feedforward from the mossy afferent (dg)**, so the CA3 firing threshold rises in proportion to total input. Then a ~**constant fraction** fires across a >10× drive range (divisive normalization; E%-max ≈ d/τ_m ≈ 5–15%, and your 10% target sits dead-center in the biological band). Size becomes an **inhibition-set set-point, not a tuned window** — which mechanically *decouples* size from the recurrent weight, so you can raise the recurrent gain freely for completion without runaway, and the coincidence threshold stops setting size (it only gates specificity).

**Honest caveat (from the snn-computational report, under-emphasized by the specific-fix report):** E%-max is invariant to *multiplicative* rescaling of drive, but the active *count* still depends on the *shape/spread* of the excitation distribution (Lisman's own words). So E%-max alone gives a *roughly* fixed fraction, not a hard k. To pin a stable count you likely need to pair it with a **homeostatic per-neuron adaptive threshold** (Diehl–Cook) that flattens the excitation distribution so the E% cut lands on a stable number. Budget for this — do not assume ff-inhib alone passes the CV<0.25 gate.

---

## (2) The concrete mechanism to build (reusing existing machinery)

**Port the `ca1_ff_inhib` block to CA3.** The specific-fix report found that `_riii_ca3_coincidence_completion_derisk.py::_build` already contains this exact fix for CA1 (lines 95–115), written on the CYCLE-1089 gate with a comment that describes *your CA3 problem verbatim* ("feedback inhibition is a knife-edge because its sparsity is the loop gain … the fix = feedforward inhibition … a ~CONSTANT FRACTION fires across a >10× drive range"). Your CA3 path only has the pure-**feedback** `ca3_fb_inhib` block — the knife-edge that block's own CA1 sibling was built to retire.

Add a `ca3_ff_inhib` branch mirroring `ca1_ff_inhib`, but driven by **dg** (the mossy volley), reusing `ca3_pv_basket` FS interneurons (`IZH2007_FS_CORTICAL_INTERNEURON`):
- **`dg → ca3_pv_basket`** (feedforward, ~0.4 density, weight `ca3_ff_basket_w`): sets HOW MANY CA3 fire (divisive-norm; the disynaptic path gives the ~2.5 ms lead so the basket clamps before the recurrent avalanche).
- **`ca3 → ca3_pv_basket`** (weak feedback, ~0.4 density, weight ~2.0): E%-max within-cycle competition selects WHICH fire, and — because DG is silent during CA3-direct recall — this arm also caps the *completed* assembly at ≈E%. One block normalizes sparsity at BOTH selection and completion.
- **`ca3_pv_basket → ca3`** (density 1.0, weight `ca3_ff_inhib`).

Then compose with what you already validated: **`encode_btsp=True`** (plateau-gated one-shot binds the within-assembly recurrent cluster) + **bistable dendrite** (`plateau_self_regen`, `apical_kir_g`) for completion + **`structural_sep=1`** (zero non-member→member, removes the permuted-cue leak). Keep `ca3_ff_inhib` and `ca3_fb_inhib` mutually exclusive; default `None` → byte-identical. **No `sim/` edit** — all runner-side region/pathway appends + existing cfg hooks.

**Add homeostatic per-neuron θ (Diehl–Cook)** as a support mechanism if the drive-invariance gate's CV is too high — it equalizes excitability so no cell wins structurally (also fixes the "seeds the wrong target" leg) and converts E%-max's fraction into a stable count.

---

## (3) The CHEAPEST single change to try FIRST

**Two cheap moves, in order:**

**Step 0 — decoupling probe (~5 lines, run first to BOUND the arc).** Before tuning any spiking sparsity, answer *"is a ~200-cell emergent-image assembly even completable?"* independent of the sparsity question. In `_gap5_emergent_dg_selection_derisk._select`, replace the θ-threshold read with a **fixed top-10% read** on the already-proven-reproducible CA3 rate vector (`A = argsort(-ca3_rate)[:int(0.10*n_ca3)]`), pipe each `A_m` straight into `_riii_ca3_synchronous_assembly_derisk.run(assemblies_ext=[A_m], encode_btsp=True, bistable=True, structural_sep=1, ca3_density≈0.08)`. **This is a host selection shortcut used ONLY as a probe** (it would launder the emergence claim if deployed — flag it as such). One 6-seed run tells you: if it completes → the 200-cell assembly is completable and FIX 1 is just the emergent realization; if it does NOT → the blocker is *completability* (density/BTSP strength), not sparsity, and you redirect before touching inhibition. Either outcome is decision-useful for ~5 lines.

**Step 1 — the deployment change most likely to convert 13/5/29 → ~200:** the **`ca3_ff_inhib` port** above. The 13/5/29 seeds are small because the isolated BTSP encode only potentiated the mossy image with *no amplification*; ff-inhib is precisely what lets you crank recurrent amplification to grow the seed toward 10% *without* runaway. This is the single change with the highest probability of turning the reliable-but-small separated seeds into completion-scale assemblies.

---

## (4) GO gate + anti-cheats (6-seed: 42,43,44,100,101,102; SIM_BACKEND=cupy; n_ca3=2000)

**Stage-1 — emergent fixed-sparsity selection (E%-max ON, NO top-k):**
- **Fixed sparsity:** every input `|A_m|/n_ca3 ∈ [0.08, 0.12]`.
- **★ Drive-invariance gate (the decisive one — separates E%-max from the old knife-edge):** sweep `mossy_weight ×0.5, ×1, ×2` → sparsity stays in-band, **and CV of `|A_m|` across inputs < 0.25**. If size tracks drive, the loop isn't clamping.
- **Reliable:** re-present same input → Jaccard ≥ 0.7.
- **Separated:** distinct inputs → cos < 0.3, Jaccard < 0.2.
- **Anti-cheats:** NO-INPUT → `|A| ≤ 0.1×mean` (moat); PERMUTE-INPUT → different assembly (input-driven, not hand-assigned); MOSSY-LESION (`mossy_weight=0`) → `|A|≈0` (provenance); **E%-MAX-LESION** (remove the `dg→basket` feedforward arm, revert to feedback-only) → CV explodes / 0-or-runaway returns (**proves the normalization is load-bearing, not the threshold**).

**Stage-2 — store + complete on the emergent `A_m` (your existing gate, unchanged):**
- **Completion:** `held_cue ≥ 0.20` AND `≥ 3× held_nocue` AND `≥ 3× held_perm`, with `held_nocue ≤ 0.10`.
- **Anti-cheats (these caught your retracted self-sustaining-attractor artifact — keep all):** NO-CUE → held silent (genuine bistable, not always-on); PERMUTED-RECALL → held silent (specificity); NO-ENCODE (`encode_btsp=False`) → no completion (learned attractor load-bearing); LINEAR (`coincidence=False`) → no completion (dendritic plateau load-bearing).

---

## (5) Ranked cheap-first build order

0. **Decoupling probe** (host top-k, ~5 lines) → is a 200-cell emergent-image assembly completable? *If NO, jump to step 3 before touching inhibition.*
1. **Port `ca1_ff_inhib` → `ca3_ff_inhib`** (mirror lines 95–115; new args `ca3_ff_inhib`, `ca3_ff_basket_w`, default `None`=byte-identical). Sweep the two weights to a settled 10% fraction; pass the Stage-1 gate **including the drive-invariance sweep + E%-max-lesion**.
2. **Add Diehl–Cook homeostatic per-neuron θ** *only if* Stage-1 CV > 0.25 (converts fraction→stable count; fixes reliability/wrong-target leg).
3. **Size/completability knobs** (only if seed short or won't complete): `ca3_density ≈ 0.06–0.10` (→ ~12 recurrent inputs/cell in a 200-cell assembly, the validated completion regime), *moderate* `mossy_weight` + `mossy_density ≈ 0.10–0.20` (high weight makes a few cells detonate and the rest get suppressed → small assembly; moderate spreads drive so E%-max picks the top-driven 10%), `encode_btsp` strength + `structural_sep=1`.
4. **Auto-calibrate recall threshold from the c_drive gap** (specificity refinement): wire the existing `_cdrive_for_cue` diagnostic → `recall_k_thresh = nonstored_cdrive + 0.5*(held_cdrive − nonstored_cdrive)`, so the plateau trigger tracks the *per-seed amplified* weight scale (the decoupling the runner comment at lines 242–247 anticipated).
5. **New composed runner** `_gap5_emergent_dg_fixed_sparsity_derisk.py`: Stage-1 (E%-max selection) → Stage-2 (`run(assemblies_ext=…)`), full gate, 6-seed.
6. **(Later refinement, likely unnecessary for v1) theta-phase gain separation.**

---

## Disagreements / nuances between the reports (flagged honestly)

- **Hard-count vs fraction (real, must design around):** snn-computational flags that E%-max sets a *fraction*, not a hard k, because it depends on the excitation-distribution shape; the specific-fix report treats ff-inhib as sufficient for a stable ~10%. **Resolution: budget for Diehl–Cook homeostatic θ (build step 2) as the count-stabilizer; don't assume the CV<0.25 gate passes on ff-inhib alone.**

- **Encode/recall gain separation — theta-SPEAR vs "ff-arm-silent-at-recall":** biology + snn both recommend **theta-phase multiplexing** (Hasselmo SPEAR: oscillate the `ca3→ca3` transmission gate + a plasticity gate in antiphase with `dg→ca3`) to bind at low recurrent gain and complete at high gain in one config. The specific-fix report achieves the same separation *more cheaply* via two existing facts: (a) your **isolated staged BTSP encode** already avoids runaway during binding (no soma co-drive → only mossy potentiates), and (b) the **feedforward arm is silent during CA3-direct recall** (no DG), so one loop caps both phases. **Resolution: theta-SPEAR is the biological ideal but is likely NOT needed for v1 — the existing staged encode + ff-inhib already separate the gains. Rank it a later refinement, not a v1 dependency.**

- **Mossy reliability — E%-max normalization vs facilitating detonator:** biology puts heavy weight on the **conditional-detonator STF** (Vyleta–Jonas 2016: a DG *burst* detonates via short-term facilitation, a tonic single spike does not — explaining your "weight 500 → only v_max −31") and a separate **feedforward-inhibition SDN front-end** (Pouille–Scanziani). The specific-fix report folds input-normalization into the *same* `dg→basket` ff arm. **Resolution: the ff-inhib arm IS the SDN front-end (cheaper, one mechanism). Keep the mossy-STF detonator as a fallback reliability lever ONLY if seeding is still unreliable after ff-inhib — it adds a new STP mechanism on the mossy synapse, so it is not v1.** Note a real open question: your DG code is *asynchronous sparse* — whether those DG cells actually burst determines whether the STF detonator even applies; verify before investing.

- **Recurrent architecture:** biology (Guzman–Jonas 2016) stresses that real completion comes from **sparse ~1% single-contact recurrence enriched in disynaptic motifs, NOT strong dense recurrence** — reinforcing that raising *weight* (not density) into a runaway regime was the wrong lever, and that `ca3_density ≈ 0.06–0.10` with the E%-max clamp is the right regime. All three are consistent here.

**Bottom line:** the one buildable mechanism is **the `ca3_ff_inhib` E%-max feedforward divisive-normalization loop** (a port of your existing CA1 block), composed with your validated BTSP-encode + bistable-dendrite completion, and paired with Diehl–Cook homeostatic θ to pin the count. Run the ~5-line host-top-k probe first to bound completability, then deploy the port. No `sim/` edit. Core refs: de Almeida–Idiart–Lisman 2009 (`10.1523/JNEUROSCI.6044-08.2009`), Guzman–Jonas 2016 (`10.1126/science.aaf1836`), Carandini–Heeger 2012 (`10.1038/nrn3136`); support: Diehl–Cook adaptive-θ, Vyleta–Jonas 2016 (`10.7554/eLife.17977`), Hasselmo–Bodelón–Wyble 2002 (`10.1162/089976602317318965`).

Key file: `/home/dant123/Projects/sim/research/runners/_riii_ca3_coincidence_completion_derisk.py` (`_build`, `ca1_ff_inhib` block ~lines 95–115 to mirror; `ca3_fb_inhib` block ~lines 60–78 to supersede).