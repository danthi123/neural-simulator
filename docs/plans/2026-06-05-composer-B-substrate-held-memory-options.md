# Composer full-clear (B) — substrate-held memory: design OPTIONS (pre-A-completion prep) — 2026-06-05

Forward prep while (A) the cleanup validates. (B) is the deeper shortcut (the owner's "full clear A+B"). This frames
the mechanism options + the de-risk-first plan for the owner to steer after (A) is cleared. NOT yet a committed design.

## The (B) target (what's still numpy in the composer after A)
`research/runners/core_sim_composition.py`:
- `bind_fact(fact)` (lines 275-286): for each role, the SPIKING bind gives `(o, f)`; then **numpy superposition**
  `bon += o; boff += f` (line 285) + **numpy ON/OFF opponency** `onoff(bon - boff)` (line 286).
- `store(...)` (line 349): `self.kb.append((fact, bind_fact(fact)))` — the bound fact is a **numpy (ON,OFF) vector
  held in a Python list**; each `unbind` re-drives that numpy vector into the substrate. **The memory is not in the
  substrate.**

So (B) has three sub-pieces: (i) in-network superposition (the role binds' rates SUM on a shared bank), (ii)
in-network opponency (ON/OFF lateral inhibition), (iii) **substrate-held storage** of the bound fact (the big one).

## Mechanism options for substrate-held storage
1. **Engram-tag per fact (catalog D.14).** The project HAS the API (`start_engram_recording` / `commit_engram_tag` /
   `stimulate_tag`, `sim/bridge.py`) + validated multitag retrieval (90% FULL multi-seed, 2026-05-14). Store: drive
   the fact's role binds, capture the co-firing bind-bank pattern as an engram tag. Recall: `stimulate_tag` → the
   bound pattern reactivates → unbind. **KEY OPEN RISK:** an engram tags NEURON INDICES (+ spike counts), so does
   stimulation reproduce the GRADED ON/OFF superposition magnitudes the unbind needs, or only a binarized pattern
   (degrading recall)? This is the load-bearing de-risk.
2. **Dedicated persistent bank per fact (NMDA working-memory latch).** Each fact's bound pattern held in a small
   recurrent bank (the Wang-2002 / dlPFC NMDA latch, which the project supports). Holds the GRADED pattern; capacity =
   number of facts (fine for ~30-fact KB). More neurons/wiring per fact.
3. **CA3 autoassociator (synthesis §6).** Store bound patterns as attractors in a recurrent CA3-like net; partial cue
   completes (Treves-Rolls capacity ~36k at sparse coding). BUT the bound patterns are DENSE superpositions, not
   sparse → less fit (the synthesis flagged dense-vs-sparse); a settling attractor (slower). Reserve for partial-cue.
4. **One-shot fast-weight Hebbian imprint.** The bound pattern → a synaptic weight vector onto a per-fact "memory"
   neuron; recall reconstructs. Like the project's structural memory; needs a fast-weight path.

## De-risk-first plan (mirroring the A arc that just worked)
The A cleanup arc's lesson: a cheap-first de-risk on the REAL composer state + grounding in the literature beats
parameter-guessing. For (B):
1. **De-risk the graded-pattern fidelity** (the crux): store ONE bound fact via option 1 (engram) AND option 2
   (bank); recall it; unbind every role; compare to the numpy-held bound vector's unbind (parity per role, multi-seed).
   GATE: substrate-held recall must support unbind at numpy parity. If the engram binarizes too much → option 2 (bank)
   or a research pass.
2. **A focused literature pass** (like A's) on spiking associative memory for HOLDING superposed bound vectors (not
   just cleanup): Nengo SPA memory, the engram/Tonegawa fidelity, sparse-vs-dense storage — to pick the faithful
   mechanism before building. (A's research pass found the exact mechanism; (B) likely warrants the same.)
3. **Then** in-network superposition + opponency (the two LINEAR pieces — superposition = rate summation on the
   shared bank; opponency = ON/OFF lateral inhibition) once the storage mechanism is fixed.
4. Build into the composer as an opt-in flag (like `enable_spiking_cleanup`), no-regression on the capability matrix.

## Honest scope note
(B) is materially bigger than (A): (A) replaced a READOUT (a single argmax) with a feed-forward circuit; (B)
re-architects how facts are STORED + recalled. The graded-pattern-fidelity de-risk is the make-or-break, and (B) may
need its own deep-research pass. Recommend the owner steer (B) after (A) is confirmed cleared, starting with the
graded-fidelity de-risk + a literature pass.
