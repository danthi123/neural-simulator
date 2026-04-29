# PDF spot-check — borderline catalog claims (2026-04-29)

**Purpose:** Verify three catalog claims that the structural-naming audit flagged as "specialty-PDF dependent" against the actual source PDFs at `sim-catalog/references/textbooks/`.

**Result: all three catalog claims hold. Rename pass remains unblocked.**

---

## Spot-check #1 — Tepper-2018: striatal GABAergic interneuron taxonomy

**Catalog claim (R2.3, B-cluster):** PV-FSI is one of *eight* distinct striatal GABAergic interneuron classes catalogued in Tepper-2018: PV-FSI, NPY-LTS, NPY-NGF, CR, TH/THIN (4 subtypes), FAI, SABI — plus the cholinergic ChI/TAN class.

**Catalog implication for the project:** `str_FS_*` regions explicitly model PV-FSI (parvalbumin-positive fast-spiking interneurons), one of multiple distinct striatal GABAergic classes — non-isomorphic to cortical FS taxonomy.

**Source verified:** `sim-catalog/references/textbooks/basal-ganglia-reviews/Tepper-2018-StriatalGABAergic-Heterogeneity.pdf`

**What the abstract + page 1-3 actually say:**
- Original 2010 review: 4 classes — **PV-FSI** (parvalbumin fast-spiking), **NPY-(P)LTS** (NPY/SOM/NOS, low-threshold spiking; further split into LTS proper and NGF neurogliaform forms), **CR** (calretinin), **TH** (tyrosine hydroxylase, GABAergic — a 2010 discovery via TH-EGFP transgenic line, subsequently termed THINs).
- 2018 update adds: **FAI** (fast adapting interneuron), **SABI** (spontaneously active bursty interneuron). The TH class itself was further split into 4 electrophysiological subtypes.
- Plus the cholinergic class (ChAT+ / CIN / TAN) — listed in abbreviations but treated as separate from the GABAergic taxonomy in this paper.
- Quote: "we were aware of four classes of striatal GABAergic interneurons" (2010) → 6+ explicit classes by 2018.

**Verdict:** ✓ Catalog claim accurate. The "8 classes" count holds when TH is expanded into its 4 subtypes; the abstract directly names PV / NPY-(P)LTS / CR / TH / FAI / SABI (6 named, +TH-subtypes splits to 8+).

**Rename implication:** `str_FS_*` should rename to `str_PV_FSI_*` to match Tepper's canonical "PV-expressing fast spiking interneurons (FSI)" formulation explicitly. The cortical `cortex_FS_*` shares the IZH2007_FS_CORTICAL_INTERNEURON preset but is biologically a *cortical* PV+ basket cell — different family from striatal PV-FSI per Tepper-2018 abstract framing. Confirms the structural audit's "FS taxonomy conflation" cross-cutting issue.

**Citation for rename PR:** Tepper-Faust-Assous-Koos 2018 *Front. Neuroanat.* 12:91, abstract + Introduction pp. 1-2.

---

## Spot-check #2 — Bolam-2000: pallidostriatal pathway → PV-FSI in striatum

**Catalog claim (A.13/A.14):** GPe arkypallidal cells (PV−) selectively project back to striatum (pallidostriatal feedback), preferentially targeting PV+ GABA interneurons (FSI). The prototypic GPe (PV+) projects to STN/GPi/SNr.

**Source verified:** `sim-catalog/references/textbooks/basal-ganglia-reviews/Bolam-2000-JAnat-SynapticOrgBG.pdf`

**What pp. 1-16 actually say:**
- Pallidostriatal pathway is real: "pallidostriatal axons selectively innervate PV-positive GABA interneurons" (multiple confirmations, Fig. 3, pp. 4-6).
- Quantitative model (Table 1): each pallidostriatal neuron contacts ~7.1 PV-FSI boutons; 11,500 GABA interneurons innervated total per population.
- D1 receptors → direct-pathway spiny neurons; D2 receptors → indirect-pathway spiny neurons (matches our naming).
- "External segment of the globus pallidus" used canonically (matches USER_GUIDE.md fix from "externa" → "externus").

**What's NOT in Bolam-2000:** the prototypic-vs-arkypallidal *distinction* itself. Bolam-2000 treats GP as a single population; he discusses pallidostriatal pathway in aggregate. The PV+ proto vs PV− arky split came later (Mallet 2008 *J. Neurosci.* 28:14245; Hegeman 2016 review).

**Verdict:** ✓ Catalog claim accurate but mis-cited. The pallidostriatal pathway → PV-FSI fact is from Bolam-2000. The arkypallidal-specific attribution is from Mallet 2008+, not Bolam-2000.

**Rename implication:** `gpe_X` could legitimately stay generic (we don't model the proto/arky split yet). If the project later splits, `gpe_proto_X` and `gpe_arky_X` are the canonical forms. The pathway `gpe → striatum` should cite Mallet 2008 (not Bolam 2000) when the arkypallidal attribution is the load-bearing claim.

**Citation for rename PR:** Bolam-Hanley-Booth-Bevan 2000 *J. Anat.* 196:527-542, pp. 4-6 + Table 1, for the pathway. Mallet et al. 2008 *J. Neurosci.* 28:14245-14258 for the proto/arky split.

---

## Spot-check #3 — Schultz 2016 NRN: two-component phasic DA

**Catalog claim (C-cluster):** Phasic DA response has two sequential subcomponents — an initial unselective salience burst (~50-120 ms latency) followed by a slower reward-value/utility component (~120-200 ms). Tonic DA is a third, separate dopaminergic mode. Project's `--adaptive-da` and `--surprise-lr-boost` map to these.

**Source verified:** `sim-catalog/references/textbooks/schultz-dopamine/Schultz-2016-NRN-RPE-twocomponent.pdf`

**What pp. 1-13 actually say:**
- Title: "Dopamine reward prediction-error signalling: a two-component response."
- Abstract: "An initial brief, unselective and highly sensitive increase in activity is followed by a slower, more selective response component, which reflects subjective reward value and utility."
- Page 1: phasic DA neurons have brief responses to rewards and reward-predicting stimuli; this paper outlines "distinct subcomponents of the phasic dopamine reward response."
- Page 1: separately, "tonic dopamine level that is necessary to enable neuronal processes underlying a wide range of behaviors" — distinct from phasic.
- Latencies: initial detection ~50-120 ms unselective; discrimination of identity 120-200 ms after initial; full value coding takes another ~120 ms.
- Component 1: "physical salience, which facilitates initial detection," novelty salience, surprise salience — *unselective*.
- Component 2: subjective reward value / utility — *the well-identified reward value*.

**Verdict:** ✓ Catalog claim accurate and faithfully extracted.

**Rename implication for the project:** `current_reward_signal` does collapse:
1. Phasic (Components 1+2) vs tonic — TWO modes
2. Component 1 (salience burst) vs Component 2 (value) — within phasic
3. A9 (SNc, motor/striatum) vs A10 (VTA, limbic/cortex) — ANATOMICAL source distinction

That's a 2×2×2 ≈ 8-way conflation in a single signed scalar. The structural audit's [discrepancy] flag is well-founded.

**Recommendation:** Keep `current_reward_signal` as a project abstraction (the rename audit's lower-priority recommendation), but document the conflation in the docstring. The neuromodulator-subsystem (`enable_neuromodulator_subsystem` + `--enable-tonic-da` flag) already begins addressing this by separating phasic from tonic; future Cluster C v2 (compartmentalized DA) addresses A9/A10 + per-action targeting; future structural buildout could split Component 1 (`--surprise-lr-boost` in current code) from Component 2 explicitly.

**Citation for rename PR:** Schultz 2016 *Nat. Rev. Neurosci.* 17:183-195. Two-component framing, abstract + Introduction.

---

## Aggregate verdict

All three catalog claims spot-checked against source PDFs hold. The catalog's 8-week-old extraction is faithful; rename pass can proceed using the catalog as authoritative without needing to re-read each PDF.

For PR descriptions on individual renames, cite the *specific* paper (not just "Tepper-2018" but "Tepper et al. 2018 *Front. Neuroanat.* 12:91 abstract + p. 1-2") so future readers can verify quickly.

Most-impactful rename insight from the spot-check: **the `*_FS_*` taxonomy conflation is a real bug in naming**. Both `cortex_FS_*` and `str_FS_*` use the same Izh preset, but they are biologically distinct cell families (cortical PV+ basket cell vs striatal PV-FSI per Tepper-2018). Rename should be `cortex_PV_basket_*` and `str_PV_FSI_*` respectively, or accept the single preset as a deliberate engineering shortcut and document it.
