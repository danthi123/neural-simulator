# Scaffolds still to replace (each is host code or a hand-set constant standing in for a brain process)

| Scaffold | Where | Replace with | Added |
|---|---|---|---|
| Neutral "birth" text that sets the conversational register | `brain.py` BIRTH | a consolidated baseline state learned in sleep | 11-02 |
| Which turns get stored (questions/requests skipped by a word list) | `brain.py` `_is_telling` | surprise/salience-gated encoding (`modulators.py`) | 11-02 |
| Reinstatement gain fixed at 4.0; familiarity bars 0.12 / 0.45 | `hippocampus.py` | gain and bars set by neuromodulators; learned readout | 11-02 |
| Pattern completion = centred cosine on the layer-0 associative trace (lexical; paraphrases miss) | `hippocampus.py` | a spiking CA3 attractor over a semantic key (port from the old code if it wins the weekly test) | 11-02 |
| Self-knowledge seeded as three Q/A exchanges at birth | `brain.py` `_grow_self` | identity learned in early conversations and consolidated | 11-02 |
| Sleep learning = gradient descent on low-rank cortical adapters (declared trade) | `plastic_cortex.py`, `sleep.py` | a local plasticity rule on the same synapses (e.g. replay-driven Hebbian/BTSP-style) | 11-02 |
| Sleep replays the stored episode text (the hippocampus keeps text only for inspection otherwise) | `sleep.py` | replay generated from the reinstated cortical state, no stored text | 11-02 |
| Teacher-written fact bank with rewordings and confusables (test set; also the training pairs of the `learned_teacher` reference arm, on facts disjoint from the tested ones) | `experiments/facts_bank.py` | rewordings the brain meets in real conversation | 11-03 |
| Prompt asking the cortex, while replaying an episode in sleep, to pose a question about it (pairs for the learned key) | `experiments/selection_probe.py` SELFQ_PROMPT; `hippocampus_keys.py` LearnedKey | questions the brain actually heard and answered from that episode, or spontaneous replay-time prediction | 11-03 |
| Learned key trained by gradient descent (InfoNCE) on a low-rank correction | `hippocampus_keys.py` LearnedKey | a local plasticity rule on entorhinal-to-hippocampal synapses | 11-03 |
