# Composer shortcut audit — is the numpy cleanup the LAST non-core-biological shortcut? — 2026-06-04

**Short answer: NO.** The owner chose item 2 (build the real cortical cleanup circuit, migrate the numpy argmax
cleanup off the composer) *conditional on it being the last non-core-biological shortcut*. An honest line-by-line
audit of the production path in `research/runners/core_sim_composition.py` shows the cleanup is **one of three**
per-query numpy steps. The bind/unbind COMPUTE is genuinely spiking; the cleanup, the superposition, and the ON/OFF
opponency — and the fact STORAGE itself — are still numpy. So "just the cleanup circuit" would NOT clear the last
shortcut.

## The spiking-vs-numpy boundary in the production path (store → query)

GENUINELY SPIKING (validated, pillar n=111):
- **The ±1 Hadamard bind/unbind** — `hadamard_spiking()` (lines 151–175) drives `cp_external_input_current` for the
  role (±1 → ON/OFF) and filler (graded ON/OFF) populations, runs `bridge._run_one_simulation_step()` over the
  readout window, and accumulates `cp_firing_states` from the four coincidence banks A/B/C/D. `_op` (per-role bind)
  and `_unbind_onoff` (unbind) both call it. This is the load-bearing compute and it is real spiking.

STILL NUMPY (the shortcuts), per query:
| # | shortcut | where | what it stands in for in-substrate |
|---|---|---|---|
| 1 | **Cleanup** (nearest-concept readout) | `np.argmax([concepts[w] @ est])` — `unbind` line 263, `_render_filler` line 277 | attractor/competitive dynamics that settle a noisy estimate onto the nearest stored concept (decorrelation + temporal integration + divisive normalization) — **item 2's named target** |
| 2 | **Superposition** (combine role-fillers) | `bon += o; boff += f` in `bind_fact` line 250 | multiple bound assemblies co-active so their rates sum on a shared memory bank |
| 3 | **ON/OFF opponency** (common-mode removal) | `onoff(bon - boff)` `bind_fact` line 251 | lateral inhibition between the substrate's ON and OFF channels |
| 4 | **Storage** (the fact memory) | `self.kb.append((fact, bound))` line 301; `bound` is a numpy (ON,OFF) vector held in a Python list, re-driven into the substrate on each query | persistent substrate-held memory (engram/attractor state), not a host-side vector list |

(Setup-time numpy — NOT per-query compute, not counted as a runtime shortcut: the ZCA decorrelation of the codebook
lines 207–211, and the random projection of the raw concept codes line 67. These prepare the codes once.)

## Two categories of remaining shortcut

- **(A) The READOUT shortcut** — the cleanup (#1). This is what item 2 names ("the numpy argmax cleanup is a
  DISCLOSED high-precision readout"). The real build is a spiking cleanup region: decorrelation (already a setup
  step; would move in-line) + temporal integration + divisive normalization. BOUNDED.
- **(B) The MEMORY shortcut** — superposition + opponency + storage (#2–#4). The bound fact is *produced* by a
  numpy superposition+opponency and *held* as a numpy vector in `self.kb`; the substrate re-computes unbind on
  demand by driving that vector back in. The memory is not in the substrate. Clearing this is the substrate-held
  fact-memory direction (engram/attractor storage — the project already has the engram-tagging API, catalog D.14,
  as a substrate-held-memory primitive). LARGER.

## Implication for the owner's conditional

"Item 2 handles the last non-core-biological shortcut" is **false** as scoped (cleanup only). The honest options:
1. **Cleanup circuit only (A):** removes the readout shortcut (#1). The memory shortcut (#2–#4) remains numpy. Does
   NOT satisfy "the last shortcut," but is the bounded, named piece and is a clean prerequisite for a fully-spiking
   readout.
2. **Full clear (A)+(B):** cleanup circuit + move superposition/opponency in-substrate + substrate-held fact memory
   (engram/attractor). This genuinely clears the last non-core-biological shortcut in the composer, but it is a
   materially larger arc (it re-architects how facts are stored, not just how they are read out).
3. **Re-order:** since (B) is the bigger and more load-bearing shortcut (the *memory* being numpy is a deeper
   departure than the *readout* being numpy), the owner may prefer to scope item 2 as the full memory+readout clear,
   or to do the fully-grounded run first on the honestly-disclosed current substrate and treat the full clear as its
   own arc.

This audit is the prerequisite the owner's conditional required; the scope decision is theirs.
