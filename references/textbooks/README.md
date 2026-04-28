# Reference textbooks

Local-only library of textbook and review PDFs supporting the [feature catalog](../feature-catalog.md) and [biology buildout roadmap](../biology-buildout-roadmap.md). The PDFs themselves are gitignored (`references/textbooks/**/*.pdf`); only this README is checked in to track what's locally available and what still needs to be sourced.

## Contents (all freely available online)

### Primary textbook
- **`kandel-pns-6e/full-book.pdf`** — Kandel et al., *Principles of Neural Science*, 6th ed. (McGraw-Hill, 2021). 1,500 pp. Entire feature catalog is keyed to this volume.

### Cerebellum (Cluster F supporting material)
- **`cerebellum-marr/Marr-1969-cerebellar-cortex.pdf`** — Marr, D. (1969). "A theory of cerebellar cortex." *J. Physiol.* 202:437–470. [Caltech mirror](https://www.its.caltech.edu/~jkenny/nb250c/papers/Marr-1969.pdf). Foundational PF→Purkinje LTD theory.
- **`cerebellum-albus/Albus-1971-cerebellar-function.pdf`** — Albus, J. S. (1971). "A theory of cerebellar function." *Math. Biosci.* 10:25–61. [robotictechnologyinc.com mirror](https://robotictechnologyinc.com/images/upload/file/Albus%20Theory%20Of%20Cerebellar%20Function.pdf). Companion to Marr — adds the supervised-learning interpretation.
- **`cerebellum-marr/Hesslow-2013-classical-conditioning-motor.pdf`** — Hesslow, G. (2013). "Classical conditioning of motor responses: What is the learning mechanism?" *Neural Networks* 47:81–87. [CMU mirror](https://www.cs.cmu.edu/afs/cs/academic/class/15883-f17/readings/hesslow-2013.pdf). Updated version of the 2002 chapter with current eyeblink-conditioning evidence.
- **`cerebellum-marr/Moore-ed-2002-NeuroscientistsGuide-ClassicalConditioning.pdf`** — Moore, J. W. (ed.) (2002). *A Neuroscientist's Guide to Classical Conditioning*. Springer. Manually sourced. **Contains Hesslow & Yeo Ch 4 "The Functional Anatomy of Skeletal Conditioning" (pp. 86–146)** — the original full-length cerebellar-eyeblink review that the Hesslow 2013 paper is the update of.

### Basal ganglia (Cluster A + B supporting material)
- **`basal-ganglia-reviews/Bolam-2000-JAnat-SynapticOrgBG.pdf`** — Bolam, Hanley, Booth, Bevan (2000). "Synaptic organisation of the basal ganglia." *J. Anat.* 196:527–542. [Oxford MRC BNDU mirror](https://www.mrcbndu.ox.ac.uk/sites/default/files/pdfs/bolam2000janat.pdf). The canonical anatomy reference for direct/indirect pathway wiring.
- **`basal-ganglia-reviews/Tepper-Koos-2017-StriatalGABAergicInterneurons.pdf`** — Tepper, J. M. & Koos, T. (2017). "Chapter 8: GABAergic Interneurons of the Striatum." In *Handbook of Behavioral Neuroscience* (Elsevier). [Rutgers Garcia Lab mirror](https://www.garcia.rutgers.edu/data/ewExternalFiles/Chapter-8-GABAergic-Interneurons-of-the-Striatum_2017_Handbook-of-Behavioral-Neuroscience.pdf). PV-FSI / NPY-LTS / TH / CR taxonomy.
- **`basal-ganglia-reviews/Tepper-2018-StriatalGABAergic-Heterogeneity.pdf`** — Tepper, J. M., Koós, T., et al. (2018). "Heterogeneity and Diversity of Striatal GABAergic Interneurons: Update 2018." *Front. Neuroanat.* 12:91. [Rutgers Garcia Lab mirror](https://www.garcia.rutgers.edu/ewExternalFiles/Tepper%20et%20al%202018.pdf). Latest update on interneuron classes.
- **`basal-ganglia-reviews/TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.pdf`** — Tepper, J. M., Abercrombie, E. D., & Bolam, J. P. (eds.) (2007). *GABA and the Basal Ganglia: From Molecules to Systems*. *Progress in Brain Research* Vol. 160. Elsevier. Manually sourced. The complete edited volume — covers GABA-A pharmacology, MSN intrinsic properties, GP/STN circuit physiology, striatal compartmentation in addition to the interneuron material already covered by the 2017+2018 sources.

### Hippocampus (Cluster D supporting material)
- **`okeefe-nadel-cognitive-map/OKeefe-Nadel-1978-HippocampusCognitiveMap.pdf`** — O'Keefe, J. & Nadel, L. (1978). *The Hippocampus as a Cognitive Map*. Oxford University Press. ~570 pp. [University of Arizona Repository](https://repository.arizona.edu/handle/10150/620894). Authors regained copyright from OUP and made it freely available. Foundational cognitive-map theory + place-cell discovery.

### Brain rhythms / oscillations (cross-cutting, Cluster N + D + G)
- **`buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.pdf`** — Buzsáki, G. (2006). *Rhythms of the Brain*. Oxford University Press. ~450 pp. [UCSD course mirror](https://neurophysics.ucsd.edu/courses/physics_171/Buzsaki%20G.%20Rhythms%20of%20the%20brain.pdf). Theta, gamma, ripples, slow oscillations.

### Reward / reinforcement learning (Cluster C + O)
- **`sutton-barto/SuttonBarto-RL-2nd-ed.pdf`** — Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. [Stanford course mirror](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). Authors make this freely available; also at [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html). The algorithmic side of RPE.
- **`schultz-dopamine/Schultz-1998-JNeurophysiol-PredictiveReward.pdf`** — Schultz, W. (1998). "Predictive reward signal of dopamine neurons." *J. Neurophysiol.* 80:1–27. [University of Oklahoma course mirror](https://www.cs.ou.edu/~fagg/umass/classes/691c/papers/Schultz_98.pdf). The foundational RPE review.
- **`schultz-dopamine/Hollerman-Schultz-1998-NatNeuro.pdf`** — Hollerman, J. R. & Schultz, W. (1998). "Dopamine neurons report an error in the temporal prediction of reward during learning." *Nat. Neurosci.* 1:304–309. [Harvard Bornlab course mirror](https://www.hms.harvard.edu/bss/neuro/bornlab/nb204/papers/Hollerman_Schultz_NatNeuro_1998.pdf). The temporal-RPE evidence.
- **`schultz-dopamine/Schultz-2016-NRN-RPE-twocomponent.pdf`** — Schultz, W. (2016). "Dopamine reward prediction-error signalling: a two-component response." *Nat. Rev. Neurosci.* 17:183–195. Manually sourced. The two-component DA response framing (initial salience burst + reward-value component) — directly mapped to the project's `--adaptive-da` asymmetric ramp and `--surprise-lr-boost`.
- **`schultz-dopamine/Schultz-2016-JNeuralTransm-RewardFunctionsBG.pdf`** — Schultz, W. (2016). "Reward functions of the basal ganglia." *J. Neural Transm.* 123:679–693. Manually sourced. Updates earlier Schultz reviews with BG-specific framing — DA, PPN, striatum reward processing in detail. Direct citation for the project's BG cascade ↔ RPE framework alignment.

## All previously-needed texts now acquired

The four texts originally flagged as "still needed — manual sourcing required" have all been provided by the project owner (2026-04-28). The list above is the complete reference library.

## How to use this directory

Skim `feature-catalog.md` for citation strings like "Kandel 6e Ch 38 p 932" — those map to `kandel-pns-6e/full-book.pdf`. Citations to other texts (e.g., "Marr 1969") map to the specialty PDFs in this directory.

If you encounter a citation in the catalog you can't trace back to a local PDF, check this README's "Still needed" table.

## Re-sourcing if needed

All download URLs are documented above. To re-acquire any missing PDF after a clean checkout, simply `curl -fsSL <URL> -o <path>` from the URLs listed.

---

*Last updated: 2026-04-28. Maintained alongside the catalog/roadmap on the `catalog-build` branch.*
