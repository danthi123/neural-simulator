---
type: biology
id: apical-nmda-conductance
mechanism: The synapses that carried a word into a tuft leave a voltage-dependent NMDA conductance that decays over tens of ms after the input stops (glutamate stays bound), Mg2+-blocked at rest and unblocked by the depolarization it helps hold
status: specified
last_verified: 2026-10-28
current_finding: research/coordination/cs-race-f1j-PACKET.md
current_status: "ENGINE SUPPORT BUILT, NO RESULT. The default-OFF sim/ edit (cfg.enable_apical_nmda; per-cell gating cp_apical_nmda_s on the BDSP apical compartment; Jahr & Stevens Mg block with the engine's own constants; reversal 0 mV; bounded exponential-Euler step) is pinned by tests/test_apical_nmda.py: flag-off byte identity, inert at rest, APV (g0 0) term exactly 0, the per-step bound, and a match to the f1j open-loop replay reference. The cs-race f1j runner flags (--nmda-g0, --nmda-tau, --tau-gi) are pinned by tests/test_cs_race_f1j_flags.py. Whether this conductance rescues the self-terminating C6 hold without breaking the eraser (cs-race f1j step 0: rescue / eraser / F1-F4) is unmeasured; no step-0 line has run."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "extracellular Mg2+ binds tightly to"
    note: "Kandel 6e Ch. 13 (NMDA receptors have unique biophysical and pharmacological properties; Fig. 13-8): at the resting potential extracellular Mg2+ blocks the NMDA pore and depolarization expels it. The block is voltage-dependent and extrinsic (a plug), not an intrinsic voltage sensor."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "rises and decays with a much slower time course than"
    note: "Kandel 6e Ch. 13: the NMDA receptor current rises and decays far more slowly than the AMPA current, so it carries the late, slow phase of the EPSP. This slow decay is the conductance memory the engine lacked."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "of dendritic spike called a plateau potential, which"
    note: "Kandel 6e Fig. 13-17C (after Larkum et al. 1999): the plateau's ionic basis is voltage-gated Ca2+ channels plus NMDA receptors. Already bound in research/biology/dendritic-plateau-coincidence-burst.md."
  - path: doi:10.1073/pnas.0804318105
    anchor: "A specialized NMDA receptor function in layer 5 recurrent microcircuitry of the adult rat prefrontal cortex"
    note: "Wang, Stradtman, Wang & Gao 2008 PNAS 105:16791 (PMID 18922773, PMC2575498). FULL TEXT read by the f1j packet author (packet section 'Biology'): adult rat L5-L5 paired recordings at 35-36 C, NMDA-EPSC decay tau 40.0 +/- 2.1 ms (V1) and 79.2 +/- 6.8 ms (mPFC); ifenprodil (NR2B block) cut them to 32.3 and 38.9 ms; adult per-synapse NMDA:AMPA peak ratio derived about 1.7. The abstract (checked via PubMed by the build session, 2026-10-28) states the twofold longer prefrontal decay in adult but not young rats. Anchors the 40-80 ms adult range and the grid's tauN 40 (V1) and 75 (near mPFC 79.2)."
  - path: doi:10.1113/jphysiol.2004.080028
    anchor: "Subunit-specific gating controls rat NR1/NR2A and NR1/NR2B NMDA channel kinetics and synaptic signalling profiles"
    note: "Erreger et al. 2005 J Physiol 563:345 (PMID 15649985, PMC1665591). FULL TEXT read by the packet author: recombinant receptors near 23 C, weighted decay about 42 ms (GluN2A) and about 249 ms (GluN2B). Why a decay of 100 ms or more is a GluN2B-rich, room-temperature choice."
  - path: doi:10.1113/jphysiol.2003.058842
    anchor: "Kinetics of Mg2+ unblock of NMDA receptors: implications for spike-timing dependent synaptic plasticity"
    note: "Kampa, Clements, Jonas & Stuart 2004 J Physiol 556:337 (PMID 14754998, PMC1664940). FULL TEXT read by the packet author: native L5 nucleated patches, P14-28 rats near 23 C, weighted decay about 103 ms (juvenile, room temperature). The abstract (checked via PubMed) states that Mg2+ unblock is NOT instantaneous: a dominant fast component (about 100 us) plus slower components (4 ms and about 300 ms). The engine's instantaneous block is an approximation of this."
  - path: doi:10.1523/JNEUROSCI.3791-05.2006
    anchor: "The properties and implications of NMDA spikes in neocortical pyramidal cells"
    note: "Rhodes 2006 J Neurosci 26:6704 (PMID 16793878, PMC6673826). FULL TEXT read by the packet author: L5 compartment model, NMDA tau2 75.2 ms; prints the block as exp(-0.080 V) (a modeller's steepening, not used here); argues NMDA spikes up to P35 need NMDA:AMPA of about 5:1 or more. Abstract (PubMed): a targeted dendritic (not somatic) inhibitory input vetoes an NMDA spike within a 30 ms window."
  - path: doi:10.1016/j.celrep.2017.10.035
    anchor: "Timed Synaptic Inhibition Shapes NMDA Spikes, Influencing Local Dendritic Processing and Global I/O Properties of Cortical Neurons"
    note: "Doron, Chindemi, Muller, Markram & Segev 2017 Cell Rep 21:1550 (PMID 29117560; no PMC copy). ABSTRACT ONLY, per the packet: early inhibition transiently dampens an NMDA spike, later inhibition terminates it. The packet's further details (tuft tau 75 ms, NMDA:AMPA 1:1, the Mg slope moved from 0.062 to 0.08) are recorded there as read through the paper; this entry's builder did not read the full text. An analogy for the older/newer asymmetry (decay phase), not a binding for the eraser: in f1j the eraser ends the memoryless BDSP regen latch, not a late NMDA spike."
  - path: doi:10.1371/journal.pcbi.1003590
    anchor: "Glutamate-bound NMDARs arising from in vivo-like network activity extend spatio-temporal integration in a L5 cortical pyramidal cell model"
    note: "Farinella, Ruedt, Gleeson, Lanore & Silver 2014 PLoS Comput Biol 10:e1003590 (PMID 24763087, PMC3998913). Per the packet: tuft model with NMDA decay 70 ms (35 ms tested, shorter spikes), NMDA:AMPA 2:1. Abstract (PubMed): glutamate-bound NMDA receptors lower the NMDA-spike threshold and extend spike duration in the L5 tuft. Access level of the full text is the packet author's; not re-read here."
  - path: doi:10.1371/journal.pone.0140254
    anchor: "Effects of Neural Morphology and Input Distribution on Synaptic Processing by Global and Focal NMDA-Spikes"
    note: "Poleg-Polsky 2015 PLoS One 10:e0140254 (PMID 26460829, PMC4604166). Per the packet: model NMDA decay 50 ms, the grid's centre value and the replay window's centre. Abstract (PubMed): L5b cells favour focal NMDA spikes. Full-text access is the packet author's; not re-read here."
  - path: doi:10.1073/pnas.0305337101
    anchor: "Division of labor among distinct subtypes of inhibitory neurons in a cortical microcircuit of working memory"
    note: "Wang, Tegner, Constantinidis & Goldman-Rakic 2004 PNAS 101:1368 (PMID 14742867, PMC337059). FULL TEXT read by the packet author: a working-memory model using a 100 ms NMDA decay, with dendrite-targeting (calbindin) interneurons controlling distractor resistance. A model choice, not an adult measurement; 100 ms is outside the adult range used here."
  - path: doi:10.1038/346565a0
    anchor: "Channel kinetics determine the time course of NMDA receptor-mediated synaptic currents"
    note: "Lester, Clements, Westbrook & Jahr 1990 Nature 346:565 (PMID 1974037). ABSTRACT ONLY: in cultured hippocampal neurons the NMDA EPSC decays over several hundred ms because glutamate stays bound (APV applied after the synaptic activation has no effect). The mechanism of the conductance memory (bound glutamate, not rebinding); the decay value is an upper bound from culture, not adult cortex."
  - path: doi:10.1523/JNEUROSCI.10-09-03178.1990
    anchor: "Voltage dependence of NMDA-activated macroscopic conductances predicted by single-channel kinetics"
    note: "Jahr & Stevens 1990 J Neurosci 10:3178 (PMID 1697902, PMC6570236). ABSTRACT ONLY (packet and this build): extracellular Mg2+ blocks the pore at hyperpolarized potentials; whole-cell conductance-voltage data in 1 uM-10 mM Mg2+ fit a gating function. The constants of the form used (3.57 mM, 0.062/mV) were NOT read in this paper; they were checked through full texts that cite it (Rhodes 2006 prints exp(-0.080 V); Doron 2017 reports moving the slope from 0.062 to 0.08), and they are the engine's existing expression (sim/kernels.py:1372)."
  - path: doi:10.1002/jnr.22444
    anchor: "The decade of the dendritic NMDA spike"
    note: "Antic, Zhou, Moore, Short & Ikonomu 2010 J Neurosci Res 88:2991 (PMID 20544831, PMC5643072). The packet records its access level as NOT RECORDED. Abstract (checked via PubMed by the build session): synchronous activation of 10-50 neighbouring synapses triggers an NMDA spike/plateau of 40-50 mV lasting up to several hundred ms in thin (basal, oblique, tuft) dendrites. It supports a GLUTAMATE-DEPENDENT plateau; it does not support the engine's glutamate-independent 650 ms regen-latch hold (see Caveats)."
  - path: doi:10.1152/jn.00070.2003
    anchor: "The NMDA-to-AMPA ratio at synapses onto layer 2/3 pyramidal neurons is conserved across prefrontal and visual cortices"
    note: "Myme, Sugino, Turrigiano & Nelson 2003 J Neurophysiol 90:771 (PMID 12672778). ABSTRACT ONLY: L2/3, rats p16-21, nominally 0 Mg2+ mEPSCs; NMDA/AMPA 27 +/- 1% (PFC) and 28 +/- 3% (VC); NMDAR decay 69 +/- 8 and 67 +/- 6 ms. A PEAK ratio from juvenile mEPSCs; it does not anchor gN0 here."
constants:
  E_nmda_mV: 0.0
  mg_block_K_mM: 3.57
  mg_block_slope_per_mV: 0.062
  mg_o_mM: 1.0
  tau_n_adult_cortex_ms_min: 40.0
  tau_n_adult_cortex_ms_max: 80.0
  tau_n_grid_ms: [40, 50, 75]
  g_n0_grid_gL: [4, 8, 16]
  rise_per_ms: 0.6931471805599453
implemented_by:
  - sim/bridge.py
  - sim/config.py
  - tests/test_apical_nmda.py
  - research/runners/cs_race_f1.py
  - tests/test_cs_race_f1j_flags.py
findings:
  - research/coordination/cs-race-f1i-0-SCORE.md
  - research/findings/2026-08-20-d5-dendritic-latch-self-terminates-into-discrete-apical-plateau-BTSP-window-6seed-GO.md
---

# A heard word leaves a decaying NMDA conductance in its own tuft

**The claim the code must respect.** Glutamate released by a word's synapses stays bound to NMDA receptors after the
input stops, so the NMDA conductance outlives the input and decays over tens of ms (Lester et al. 1990; Kandel Ch. 13:
the NMDA current "rises and decays" far more slowly than the AMPA current). The conductance is blocked by extracellular
Mg2+ at rest and unblocked by depolarization (Kandel Fig. 13-8; Jahr & Stevens 1990), so it is strongest exactly while
the plateau it helps hold is up. It belongs to the synapses that carried the input. A cell that heard nothing has none.

**The rule used for the decay (tauN).** 40-80 ms is the adult-cortex range: Wang et al. 2008, adult rat L5-L5 pairs at
35-36 C, 40.0 ms in V1 and 79.2 ms in mPFC. A tauN of 100 ms or more is a juvenile, room-temperature or GluN2B-rich
choice and is labelled as such: Kampa 2004 (P14-28, about 23 C, about 103 ms), Erreger 2005 (recombinant GluN2B, about
249 ms), Wang et al. 2004 (a 100 ms model value), Lester 1990 (culture, several hundred ms, an upper bound). The f1j grid
uses 40, 50 and 75 ms (Wang 2008 V1; Poleg-Polsky 2015 and the replay's centre; Rhodes 2006 / Doron 2017, near Wang 2008
mPFC).

**The Mg block.** B(V) = 1 / (1 + ([Mg]o / 3.57 mM) exp(-0.062 V)), V in mV, [Mg]o = 1 mM (`nmda_mg_concentration`).
This is the engine's existing expression (`sim/kernels.py:1372`), reused with the same constants
(`sim/bridge.py:apical_nmda_mg_block`, asserted equal to the kernel in `tests/test_apical_nmda.py`). J&S was read at
abstract level only; the 0.08/mV steepening in Rhodes 2006 and Doron 2017 is a modellers' choice and is not used.

**What the engine does (default OFF).** With `cfg.enable_apical_nmda` (BDSP on, the two-compartment DAP and the BDSP
microcircuit off), each cell keeps a gating fraction `cp_apical_nmda_s` on the BDSP apical compartment. Each step it
decays by exp(-dt/tauN) and rises toward 1 by (1 - s)(1 - exp(-rise u dt)), where u = max(drive, 0) / drive_ref is the
cell's own apical drive (the runner's existing write, the world's input). The conductance is g_N = g0 s / s_full (s_full
the discrete steady state at u = 1, so g_N equals g0 at the end of a sustained input). The term drives v_apical toward
the excitatory reversal (0 mV, `syn_reversal_potential_e`) as a bounded exponential-Euler step, computed from the same
pre-step v_apical as the update and the apical GABA term, so v_apical never passes 0 mV. A cell with no drive keeps s at
exactly 0.0 (inert at rest). `apical_nmda_g0 = 0` with the flag on is the APV arm: the gating runs and the term is 0.

**Caveats (what is NOT claimed).**
- **The drive is a current source.** The runner's apical drive is a box-car current, not a synaptic conductance, so there is
  no AMPA conductance to scale from: gN0 is UNANCHORED in this model. {4, 8, 16} g_L is the memo's set; NMDA:AMPA peak
  ratios (Wang 2008 about 1.7; Myme 2003 27-28% in juvenile L2/3; Rhodes 2006 5:1 or more up to P35) do not map onto it.
  Whether the rescue needs the largest value is read as a finding (f1j F4), never judged against a literature range.
- **The hold is the regen latch.** Once g_N has decayed (at Y_on, 150 ms after a word's input ended, g_N is 2.4%, 5.0% and
  13.5% of gN0 at tauN 40, 50 and 75), the rest of a 650 ms hold is carried by the BDSP regen latch
  (`coincidence_plateau_self_regen`, `v_hold` -35 mV): a voltage-gated inward current with no glutamate dependence and no
  memory. The NMDA-spike biology (Antic 2010) supports a glutamate-dependent plateau, not that hold. f1j does not change
  the latch; it is its own faithfulness item (board task #275).
- **The block is instantaneous.** Kampa 2004 measured non-instantaneous unblock (about 100 us dominant, plus 4 ms and
  about 300 ms components). The engine evaluates B(V) at the pre-step voltage each ms.
- **The gating is per cell, not per synapse.** s is one fraction per neuron, driven by the cell's total apical drive.
  Receptor saturation, desensitization and the GluN2A/2B mix are not modelled; the rise rate (ln 2 per ms) is set so the
  per-step rise is 0.5 at full input, and the replay found the rise moved no metric.
- **Doron 2017 is an analogy.** f1j's older/newer asymmetry is the decay phase (time since the slot's last input). The
  eraser reaches the older word 158-287 ms after its input ended, when its g_N is a few % of gN0, so the eraser ends the
  latch, not an NMDA spike.
- **Nothing is measured yet.** No cs-race f1j step-0 line has run. The open-loop replay that informed tauN and the grid is
  unrecorded (research/derisk/cs_race_f1j_replay/README.md): predictions, not results.
