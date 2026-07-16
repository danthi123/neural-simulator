# EDGE 5 rung 1 (plan #1, the Gap-A "one-brain-that-LEARNS discourse memory" MECHANISM unification, 6-seed GO): the content-addressable delta-STORE + the clean-barcode-key BINDER compose on one memory — and the honest crux is settled precisely (the store alone content-addresses by barcode; the binder is load-bearing ONLY under a role/recency cue + for novel referents)

**Date:** 2026-07-15 · **Runner:** `research/runners/_edge5_binder_store_role_recall_derisk.py` (reuse-by-import `HebbianBinder`/`_mint_codes` from RUNG 6c; a self-contained delta-store + fading-recency baseline; numpy-CPU; NO `sim/` edit). Scoped by the 2026-07-15 design gate (subagent); follows the store #1 GO + RUNG 6c GO.

## The task + the crux
`[ e_0 v_0 … e_{P-1} v_{P-1}  (fillers ×T=30 ≫ the fading window)  PROBE ] -> v_j`. Entities e_i = sparse BARCODES (some NOVEL at test); values v_i = disjoint symbols. Two PROBE cues: ROLE (the ordinal j, pronoun-like) vs BARCODE (re-present e_j). The scoping's crux (from our own record): a barcode is a CLEAN key, so a store keyed by the raw barcode already content-addresses — **is the binder load-bearing at all, or does the store alone suffice?** The de-risk runs BOTH cues, with the store-barcode+BARCODE-cue arm as the decisive falsification.

## Result (6-seed 42/43/44/100/101/102; chance 0.125)
| arm | acc (mean) | reading |
|---|---|---|
| fading-recency baseline (the store's foil) | 0.26 | FADES past T — can't recover a specific role past the window |
| **★ STORE-barcode + BARCODE-cue** (falsification) | **1.00** | the store ALONE content-addresses by barcode → the binder is NOT needed for retrieve-by-content |
| **BINDER-slot + ROLE-cue** (the genuine binder+store) | **1.00** | binder maps entity→ordinal slot, store holds slot→value, the role cue dereferences |
| store-barcode + ROLE-cue | 0.11 | FAILS — a role cue has no barcode to index a barcode-keyed store (**the binder IS load-bearing here**) |
| no-bind lesion (random slots → collisions) | 0.26 | collapses (binder load-bearing) |
| keyshuffle (barcode keys ↔ values shuffled) | 0.25 | content-addressing breaks |
| permuted (label scramble) | 0.24 | chance |
| **BINDER-slot + ROLE-cue, HELD-OUT NOVEL entities** | **1.00** | novel referents (minted at test) bind + recall identically — the binder's content-agnostic value |
- **6/6 GO, unanimous.** binder-load-bearing-under-role-cue 6/6; store-alone-suffices-for-content 6/6.

## ⇒ The honest Gap-A read (the crux settled, no inflation)
The two fast-weight mechanisms UNIFY on one memory and each is load-bearing FOR ITS OWN JOB — but NOT interchangeably, and the finding says so plainly:
- **The delta-STORE** does content-addressable retrieve-by-barcode past the horizon (barcode-cue 1.00; the fading baseline 0.26) — for *retrieve-by-content*, the store ALONE suffices (the binder adds nothing there; the scoping's honest falsification).
- **The BINDER** buys exactly two things the barcode-keyed store structurally cannot: (i) **role/recency-cued (pronoun-like) retrieval** — bounded-slot INDIRECTION so "the j-th referent / the current holder" dereferences to a value (binder+role 1.00 vs store-barcode+role 0.11); (ii) **novel-referent handling** — content-agnostic slots so an entity invented at test binds + recalls identically (NOVEL 1.00).
- ⇒ the unified binder+store is the Gap-A "one-brain-that-LEARNS discourse memory": role-cued, novel-referent, past-horizon value recall — which neither a fading baseline (no horizon) nor a barcode-keyed store alone (no role cue, no ordinal indirection) delivers.

## ⇒ Next
This is the MECHANISM unification (numpy, both fast weights validated). The fully-spiking on-ONE-`SimulationBridge` realization (the Mongillo STP store + the STP-facilitated binder co-resident — BOTH already source-read + scoped: `cp_stp_u`/`tau_f`, `sim/kernels.py:333`) is the named next rung. RUNG 2 = compose in the D3 discrete-attractor for holder-TRACKING across clauses (possession transfer; RUNG 6c already composed binder+D3). NO `sim/` edit. Runner: `_edge5_binder_store_role_recall_derisk.py`.
