# Research gate — novel-referent (open) discourse tracking: ~80% already GO; the residual is the tracker's SUPERVISED read; the emergence-aligned fix is a content-agnostic HEBBIAN FAST-WEIGHT bind (not the FHRR scaffold)

**Date:** 2026-07-13
**Gate:** deep-research scout (a-1 own record + external literature, read-only) + controller source-verification + the emergence-bar decision. Scopes the next build after RUNG6b (`2026-07-13-RUNG6b-...`) resolved unbounded tracking for BOUNDED referents.

## The gate result (verified against our own source)
Decomposing "novel-referent tracking" against our record:
| sub-capability | status | evidence |
|---|---|---|
| mint a fresh DISTINCT code for a novel entity | **GO (6/6)** | `entity_instance_layer.py` — `z(e#i)=normalize((1-α)·z_type+α·z_barcode_i)` + overlap-rejection (REJECT_COS 0.12) = DG pattern-separation/neurogenesis; `2026-06-27-tier1-entity-instances-GO.md`. **Controller-verified by reading the source** (the barcode is a developmental-random wiring primitive — emergence-defensible, not a hand-designed scaffold). |
| bind facts to the novel token | GO | fixed FHRR bind/unbind, composer concept-agnostic |
| keep distinct / no-confab moat | GO (load-bearing) | merge-lesion (α=0 → identical → collapse) + 0-FA |
| **sequentially TRACK the novel referent** (who-holds-it) | **NOT for novel** | D3/RUNG6b use a **fixed-K SUPERVISED read** → a novel entity lands at chance |

⇒ **~80% is already GO.** The genuine residual is ONE thing: the sequential tracker's slot assignment is a per-entity **supervised** read (RUNG6b's ridge; D3's K attractor pools), so a never-seen entity can't be placed. A "referent slot" is a role and a novel entity is a filler — and binding tolerates overlapping fillers (`2026-05-31-composition-REVISION-...`) — so the gap is PURELY that the read is content-SPECIFIC, not content-AGNOSTIC.

## External mechanisms (cited; the field/biology uses INDIRECTION, not a per-entity classifier)
- **LLM binding-ID vectors** (Feng-Steinhardt arXiv:2310.17191): a fresh ~distinct binding-ID from a continuous subspace per entity = our barcode.
- **Rebind-at-retrieval** (arXiv:2606.08644): don't re-encode state on a swap; apply the transformation to the referred ID AT READ — the VSA-native "unbind at query."
- **PFC/BG indirection** (Kriete-Noelle-Cohen-O'Reilly 2013 PNAS 110(41):16390): fillers-in-roles as POINTERS to filler stripes → **generalizes to NOVEL role-filler pairs**. The neuroscience answer.
- **Flexible WM from random connections** (Bouchacourt-Buschman 2019 Neuron 103(1):147): a high-D **random untuned** recurrent layer maintains ANY input — hold a novel item with no pre-tuned attractor.
- **Fast weights / Hebbian on-the-fly bind** (Ba et al. 2016; Schlag-Irie-Schmidhuber 2021): a Hebbian outer-product fast-weight stores transient key→value bindings with NO slow-weight training = STP/Mongillo synaptic-WM.
- **Object files / DRT** (Kahneman-Treisman-Gibbs 1992; Kamp 1981): a FILE is opened for a novel individual.
Convergent theme: **indirection** — a content-free pointer/slot dynamically bound to whatever entity arrives, dereferenced at read; none learns a per-entity classifier.

## Controller decision (the emergence bar)
The scout ranked #1 = a VSA register (mint barcode + track via the fixed FHRR bind/unbind). It is cheap AND ~already-done — but it is the **FHRR exact-inverse ALGEBRA = the scaffold the emergence bar (2026-07-10) says to REPLACE, not build on.** So #1 is a scaffold-confirmation, low new value. **The mission-aligned build is candidate #3: a content-agnostic HEBBIAN FAST-WEIGHT bind** (Ba-2016 / Bouchacourt-Buschman flexible random WM / STP-Mongillo) — a one-shot Hebbian outer-product binds a novel entity's code to a fresh slot on first mention (NO supervised read, NO fixed algebra), retrieved by the same fast weight later. It is emergent (Hebbian/random, developmental), content-agnostic (a novel entity binds identically to a known one BY CONSTRUCTION), and directly replaces RUNG6b's supervised ridge. Novel-entity CODES come from the developmental-random barcode (a defensible primitive).

## THE FIRST DE-RISK (recommended, building now)
`_novel_referent_hebbian_fastweight_derisk.py` (numpy-CPU, reuse-by-import: the barcode mint + the D3 possession-narrative + the discrete attractor; NO `sim/` edit). Entities are barcodes; a Hebbian fast-weight `W += onehot(fresh_slot) ⊗ code(e)` binds each entity to a fresh slot on FIRST mention; read `slot = argmax(W @ code(e))`; the discrete attractor tracks the holder over the fast-weight-assigned slots. **THE KEY TEST: the tracked entities are MINTED AT TEST, disjoint from any setup set (held-out NOVEL).**
Anti-cheats (all load-bearing): held-out-novel + held-out-deeper (generalization-not-memorization); **merge-lesion** (α=0 barcodes → identical → collapse); **no-bind lesion** (sever the fast weight → chance); moat (never-holder → abstain); non-commutativity preserved; retention/last-mention floors at chance.
GO bar: novel-entity holder-track at held-out-deeper >> reservoir/floors, merge+no-bind collapse, ≥3→6 seeds. Predicted GO (novel binds identically to known by construction); if so → the fully-spiking STP realization of the fast weight is the next rung, and wire into the D3 event-agent path (opt-in, byte-identical off).

Sources to (re)read before the spiking rung: Kriete-O'Reilly 2013 PNAS; Bouchacourt-Buschman 2019; Ba et al. 2016; our `entity_instance_layer.py` + `biased_competition_buffer.py`. NO `sim/` edit.
