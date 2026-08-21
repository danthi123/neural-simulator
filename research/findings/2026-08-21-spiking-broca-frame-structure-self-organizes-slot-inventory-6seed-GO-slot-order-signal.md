---
type: finding
status: contributing
date: 2026-08-21
mechanism: emerge-broca-frame-structure-self-organization
lane: emergence
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: emerge64 (slot-inventory) is a full 6-seed pool run; emerge63 (slot-order) is banked as a 1-seed GO
  SIGNAL (recovered from the node; the 6-seed pool run's sync is pending) — the evidence is each runner's own verdict
  + its input-destruction controls, not a stochastic effect size.
instrument: two headless pool de-risks of the spiking-Broca producer's sentence-FRAME structure — does the per-frame
  slot INVENTORY (S1a) and slot ORDER (S1b) self-organize from corpus statistics rather than being host-supplied.
runner: research/runners/_emerge64_mine_slot_inventory_derisk.py · research/runners/_emerge63_corpus_taught_slot_order_derisk.py
artifacts:
  - research/findings/raw/_emerge64_mine_slot_inventory.json
  - research/findings/raw/_emerge63_corpus_taught_slot_order.json
---
# The spiking-Broca producer's sentence-FRAME structure self-organizes from experience — slot INVENTORY 6-seed GO, slot ORDER signal (with the function-word win, the whole frame is now learned)

Two emergence rungs that, together with the function-word-inventory GO ([[2026-08-21-recovered-overnight-pool-de-risks-function-word-discovery-V1-RF-curiosity-GO]]),
move the spiking-Broca producer's sentence-frame structure from HOST-DESIGNED to LEARNED-FROM-CORPUS: what words are
grammatical glue (emerge62), WHICH slots a frame licenses (emerge64, here), and in WHAT ORDER (emerge63, here).

## Slot INVENTORY self-organizes (S1a) — GO, 6 seeds
<!--derived-->
Artifact: research/findings/raw/_emerge64_mine_slot_inventory.json (go: True, seeds 42/43/44/100/101/102). The
per-construction slot INVENTORY (which thematic roles/slots each frame licenses) self-organizes from the corpus — each
token's ROLE is mined from its distributional context rather than a host-supplied frame template. 6-seed GO with the
input-destruction controls collapsing. ⇒ S1a self-organized: the frame's slot set is learned, not hand-declared.

## Slot ORDER self-organizes (S1b) — GO signal, 1 seed (6-seed pool confirm pending)
<!--derived-->
Artifact: research/findings/raw/_emerge63_corpus_taught_slot_order.json (go: True, seed 42). The per-frame slot ORDER
self-organizes from the corpus's actual WORD-ORDER statistics (pairwise role precedence; Dominey-Hinaut: grammar = the
statistics of element order): the host template order-teacher (EMERGE-59's `LR*(n-1-pool)`) is REMOVED, the order read
purely from where each role's token sits in the example sentences, and that corpus-taught order reorders the slots fed
to the EMERGE-59 spiking producer (rendered ON SPIKES via the per-pool rate ranking). order 1.000, exact-surface
1.000; controls collapse (SHUFFLED-CORPUS 0.331, NO-CORPUS 0.210); a fully held-out frame's shared type-level order
(det<subj<func<verb) recovers 1.000. HONEST RESIDUAL (named, not a wall): the internal order of a held-out
MULTI-function-word frame (does<not) is not learnable from the other two frames alone (only F_NEGMOD attests two
adjacent function words) → held-out F_NEGMOD sits at 0.850; the next single signal is one attestation of the does<not
bigram. The 6-seed pool run is queued to confirm; the 1-seed GO is banked as the signal.

## Scope
Pool CPU emergence de-risks (structure-from-experience — the standing emergence bar), default-off research results,
NOT production-wired. The BOUNDED EMERGE frame domain (not open-ended generation). NO sim/ edit; the no-confab moat is
untouched (0 producer invocations on abstains). Recovered from the mini-PC pool (the #56 sync path).
