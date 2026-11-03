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
| Innate primaries (13 bodily + social events per sign, a seed draws 8) define the valence axis | `affect.py` | a body/interoception model whose states are the unconditioned stimuli; valence learned by conditioning on outcomes | 11-03 |
| Appraisal zero = neutral statements; mood TAU 4 events; modulation GAIN 16 (dev-calibrated) | `affect.py` | adaptive reference (expectation-relative appraisal) and gain set by arousal | 11-03 |
