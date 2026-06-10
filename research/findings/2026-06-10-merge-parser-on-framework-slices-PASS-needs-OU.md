# Merge STEP 2a — the parser ports cleanly onto framework slices (risk 4.1 retired); the train pass needs OU (2026-06-10)

**Roadmap step 2 (nav+conv merge), the cheapest-first FIRST build step** per the implementation design
(`docs/plans/2026-06-10-nav-conv-merge-implementation-design.md` §7 step 1, §4.1). The highest implementation
risk was that the conversational parser (`BridgeParser`) cannot be a drop-in `shared_bridge=` (its merge path
re-injects and would clobber navigation; it assumes a contiguous parser block) and so must be re-expressed as
framework regions with its drive/train/read PORTED to raw framework slice indices. This micro-check tests
exactly that port.

## Setup

`research/runners/nav_conv_merged_bridge.py --microcheck`: a brain-region-framework bridge with
`[nav_stub(50), parse_conj(6), parse_role(3*40)]` — the `nav_stub` forces a NON-ZERO offset
(`conj_base=50`, `role_base=56`) so the parser's slice arithmetic is exercised under the real merged
condition. The `parse_conj → parse_role` pathway is all-to-all (720 synapses), init weight 0.5,
`plasticity_gate="parser_fixed"`. The config is the merged target: `stdp_w_max=400`, `hebbian_max_weight=400`
(the 5a clip mitigation), Hebbian ON for the train pass, STDP/reward OFF, homeostasis OFF. The parser's
`_train`/`role_of`/`parse` are ported to the framework slices; after the train pass the gate is frozen
(`set_plasticity_gate("parser_fixed", 0.0)`).

## Result

| OU during the train pass | active parse | passive parse | verdict |
|---|---|---|---|
| **OFF** (merged default) | `{agent: north}` (collapsed — most conjunctions read "agent") | `{agent: dog, action: go}` | **FAIL** (degenerate readout) |
| **20 pA** (standalone parser's setting) | `{agent: dog, action: go, patient: north}` | `{patient: north, action: go, agent: dog}` | **PASS** |

**With OU=20: the parser learns the voice-invariant role map on framework slices.** Active "dog go north" →
agent=dog, action=go, patient=north; the passive frame "north go dog" → agent=dog (voice-invariant). The
framework slice arithmetic, the plasticity gate, and the merged clip bounds all work for the parser. **Risk
4.1 is retired** — the parser is a clean framework-region port (~40 lines, no `sim/` edit).

## The honest condition: the parser train pass needs OU noise

With OU OFF (the navigation default), the readout is degenerate — the conjunctions collapse to "agent". OU
jitter breaks the symmetry at the winner-take-all role readout: without it, the deterministic Izhikevich role
ensembles fire too uniformly (the driven conjunction reaches all roles via the residual 0.5 weights and the
selective Hebbian strengthening does not dominate the readout). This is the design §2.3 anticipated fallback,
now confirmed: **the merge's parser train pass must temporarily enable OU at 20 pA, then restore OU OFF for
navigation** (navigation runs OU off, `g11_bg_runner.py:4094`). Biologically benign — functional spontaneous
activity breaking ties in a WTA readout, consistent with the project's motor-exploration-noise findings.

## Implication for the STEP 2a build

The merged builder's parser train pass: temporarily set `enable_ou_process=True`, `ou_std_current_pA=20.0`
(plus Hebbian on, STDP/reward off) for the pass; restore OU off + Hebbian off + STDP/reward on afterward;
then `set_plasticity_gate("parser_fixed", 0.0)`. The reusable port helpers (`parser_regions_pathways`,
`train_parser_on_slices`, `role_of_on_slices`, `parse_on_slices`) are in
`research/runners/nav_conv_merged_bridge.py`. Next: the full STEP 2a construction (nav + parser + dlPFC, the
combined injection per design §2.5, the dlPFC port) and the acceptance gates.
