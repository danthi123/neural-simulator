# Scaffolds still to replace (each is host code or a hand-set constant standing in for a brain process)

| Scaffold | Where | Replace with | Added |
|---|---|---|---|
| Neutral "birth" text that sets the conversational register | `brain.py` BIRTH | a consolidated baseline state learned in sleep | 11-02 |
| Which turns get stored (questions/requests skipped by a word list) | `brain.py` `_is_telling` | surprise/salience-gated encoding (`modulators.py`) | 11-02 |
| Reinstatement gain fixed at 4.0; familiarity bars 0.12 / 0.45 | `hippocampus.py` | gain and bars set by neuromodulators; learned readout | 11-02 |
| Pattern completion = centred cosine on the layer-0 associative trace (lexical; paraphrases miss) | `hippocampus.py` | a spiking CA3 attractor over a semantic key (port from the old code if it wins the weekly test) | 11-02 |
| Self-knowledge seeded as three Q/A exchanges at birth | `brain.py` `_grow_self` | identity learned in early conversations and consolidated | 11-02 |
