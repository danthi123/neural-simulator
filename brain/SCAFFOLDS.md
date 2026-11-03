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
| Ordinary text interleaved in every sleep (KL guard) = three fixed chat snippets | `brain.py` ORDINARY | the day's own lived non-episode experience | 11-03 |
| Sleep timing = once per restart/shutdown if anything new was stored | `brain.py` `Brain.__init__`/`close` | a sleep pressure signal (time awake, episodes pending) | 11-03 |
| Feeling-of-knowing bars (hippocampal familiarity, cortical cue response) fitted on a labelled 36-question set | `brain.py` `KNOW_BAR_*` | bars set by neuromodulatory state / learned from the brain's own outcomes (being corrected) | 11-03 |
| Below the feeling-of-knowing bar the reply is spoken with ALL consolidated synapses off (all-or-none retrieval gate) | `brain.py` `hear` | graded, content-specific suppression of retrieval (prefrontal monitoring of what is being recalled) | 11-03 |
| Method 2 (comparison only, not wired by default): declared "I don't know" Q/A pairs replayed in sleep | `confidence_probe.py --dont-know`, `Brain.sleep_extra` | the brain's own abstentions, once made, replayed as episodes | 11-03 |
