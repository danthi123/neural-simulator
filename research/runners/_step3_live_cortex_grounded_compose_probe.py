"""Step-3 cheap-first de-risk — do LIVE cortex_it RATE-derived grounded codes COMPOSE (not just recall)?

The step-3 scoping (`research/findings/2026-06-16-step3-compose-perceived-content-scoping.md`, controller-verified)
recommends SHARED GROUNDED CODES: a fixed map sends the navigation perception's `cortex_it` RATE code onto a
composer-ready phasor code, so a PERCEIVED object enters the EXISTING (validated) bundling algebra and can be
COMPOSED into a NOVEL fact — dissolving the rate-vs-phasor wall that the (B) perception->memory milestone maps as
its honest boundary (it RECALLS "I saw the apple" but cannot bind a perceived apple into a new role-filler fact).

The precedent `_visual_grounded_composition_probe.py` showed grounded (V1-matrix-derived) phasor codes compose
(clean / corrupted) — BUT from a numpy V1 matrix, not the live spiking substrate. THIS de-risk closes that one
gap: source the grounded code from a LIVE `cortex_it` RATE forward pass on a real SimulationBridge (drive the
object's percept, read the spiking firing-rate pattern), project it to a phasor, and run the composer
bind/bundle/unbind/cleanup on a 2-role fact of two PERCEIVED objects.

THE LOAD-BEARING QUESTION: does the noisy `cortex_it` spiking RATE code -> phasor -> compose recover the perceived
objects (clean + from a CORRUPTED percept), AND generalize to HELD-OUT novel (object, role) combinations far above
a MEMORIZATION FLOOR? The held-out-vs-memorization control is the anti-cheat that separates COMPOSE (an algebraic
bind that generalizes to never-composed pairings) from RECALL (a lookup that only retrieves stored facts).

GATE (multi-seed 42/43/44, CPU numpy, no sim/ edit):
  GO       : clean compose >= 0.90 AND corrupted-percept compose >= 0.80 AND held-out-novel compose >> chance AND
             held-out compose >> the memorization floor (the algebra generalizes; a recall baseline does not).
  PARTIAL  : clean composes but corrupted/held-out is weak (the rate code grounds but noisily).
  NEGATIVE : the rate code is too noisy to ground a composer code (compose ~ chance) -> localize (more cortex_it
             neurons / a LEARNED vs fixed projection / population averaging). An honest negative maps the limit.

HONEST SCOPE: this de-risks COMPOSITION over PERCEIVED OBJECTS bound into facts (the tasking's exact example),
sourced from a live spiking rate code. It is NOT the learn-the-whole-bind-including-multi-attribute-bundling
version — point-neuron bundling needs a fixed self-inverse / dendritic primitive (the 2026-06-16 capability map);
the dendritic rewrite stays the deferred owner call. The composer's bind/unbind algebra here is the validated
fixed VSA primitive (the same one the production conversational system uses).

Reuse-by-import: the (B) probe's `cortex_it` bridge + percept render (the live rate source); the precedent's phasor
projection + compose. CPU `SIM_BACKEND=numpy`. No `sim/` edit.
  SIM_BACKEND=numpy python -m research.runners._step3_live_cortex_grounded_compose_probe
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.backend import get_backend, to_host
from sim.text_embeddings import orthogonal_drive_pattern

# the (B) probe's live cortex_it bridge + percept render (the rate source) + the object vocabulary/constants.
from research.runners.funcint_perception_to_memory_probe import (
    OBJECT_WORDS, N_OBJECTS, N_CORTEX_IT, PERCEPT_SPARSITY, PERCEPT_DRIVE_PA,
    build_probe_bridge,
)

D = 2048                      # phasor (FHRR) code dimension (matches the precedent _visual_grounded_composition_probe).
RATE_READ_STEPS = 80          # the cortex_it rate-read window: accumulate firing under the percept drive -> rate code.


def _projection(n_in, seed=42):
    """Fixed random complex projection cortex_it-rate(n_in) -> D phases. Deterministic -> the phasor code is a fixed
    function of the RATE features (grounded), not a free random code (the precedent's `_projection`, n_in=cortex_it)."""
    rng = np.random.default_rng(seed * 5077 + 11)
    return (rng.standard_normal((D, n_in)) + 1j * rng.standard_normal((D, n_in))).astype(np.complex128)


def _to_phasor(rate_vec, proj):
    z = proj @ rate_vec.astype(np.complex128)
    return np.exp(1j * np.angle(z))               # unit-magnitude phasor code (FHRR)


def _role(seed):
    rng = np.random.default_rng(seed * 13 + 3)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, D))


def _cleanup(query, names, codebook):
    sims = [float(np.abs(np.vdot(codebook[nm], query)) / D) for nm in names]
    i = int(np.argmax(sims)); s = np.sort(sims)[::-1]
    return names[i], float(s[0] - (s[1] if len(s) > 1 else 0.0))


def read_cortex_it_rate(bridge, handles, obj_idx, corrupt_rng=None):
    """LIVE rate forward pass: drive object `obj_idx`'s percept into cortex_it (optionally CORRUPTED) and read the
    cortex_it spiking FIRING-RATE pattern over the read window = the grounded RATE code (the navigation perception
    of the object, a real spiking rate code on the substrate). Returns a host float vector (N_CORTEX_IT,)."""
    xp, _ = get_backend()
    it_indices = handles["it_indices"]
    n_it = int(it_indices.size)
    drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS, n_neurons=n_it,
                                     drive_max_pA=PERCEPT_DRIVE_PA, sparsity=PERCEPT_SPARSITY).astype(np.float64)
    if corrupt_rng is not None:
        # CORRUPTED percept: additive noise + a sparsity-level dropout (a degraded sensory render of the object).
        drive = drive + corrupt_rng.normal(0.0, 0.35 * PERCEPT_DRIVE_PA, size=drive.shape)
        drop = corrupt_rng.random(drive.shape) < 0.15
        drive[drop] = 0.0
        drive = np.maximum(drive, 0.0)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):                                          # settle (clear prior state)
        bridge._run_one_simulation_step()
    counts = np.zeros(n_it, dtype=np.float64)
    for _ in range(RATE_READ_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[it_indices] = xp.asarray(drive, dtype=xp.float32)
        bridge._run_one_simulation_step()
        counts += np.asarray(to_host(bridge.cp_firing_states[it_indices])).astype(np.float64)
    bridge.cp_external_input_current[:] = 0.0
    return counts / RATE_READ_STEPS                              # per-neuron firing rate = the grounded rate code


def run_seed(seed):
    bridge, handles = build_probe_bridge(seed)
    proj = _projection(N_CORTEX_IT, seed)
    names = list(OBJECT_WORDS)
    # GROUNDED phasor codebook: each object's LIVE cortex_it rate code -> phasor (the percept IS the concept code).
    codebook = {nm: _to_phasor(read_cortex_it_rate(bridge, handles, i), proj) for i, nm in enumerate(names)}
    R_AGENT, R_PATIENT = _role(seed + 1), _role(seed + 2)
    crng = np.random.default_rng(seed * 99 + 7)

    # all (agent, patient) ordered pairs of DISTINCT objects -> facts. Split into MEMORIZED (a recall baseline sees
    # them) vs HELD-OUT (never composed before) -> the held-out-vs-memorization anti-cheat (compose != recall).
    pairs = [(a, b) for a in range(N_OBJECTS) for b in range(N_OBJECTS) if a != b]
    crng.shuffle(pairs)
    n_mem = len(pairs) // 2
    memorized, held_out = pairs[:n_mem], pairs[n_mem:]

    # the MEMORIZATION FLOOR (a recall-only baseline): it stores each MEMORIZED fact's bound vector + its (agent,
    # patient); at query it returns the stored answer for the nearest stored fact -> on a HELD-OUT (novel) fact it
    # has no matching stored fact, so it cannot recover the right fillers (it scores at the floor).
    mem_store = []
    for (ai, bi) in memorized:
        fact = R_AGENT * codebook[names[ai]] + R_PATIENT * codebook[names[bi]]
        mem_store.append((fact, names[ai], names[bi]))

    def _mem_recall(fact, role):
        # nearest stored fact by cosine; return its remembered filler for `role` (recall, NOT compose).
        best, bi = -1.0, 0
        for k, (f, a, b) in enumerate(mem_store):
            c = float(np.abs(np.vdot(f, fact)) / D)
            if c > best:
                best, bi = c, k
        return mem_store[bi][1] if role == "agent" else mem_store[bi][2]

    def _compose_eval(pair_set):
        """For each fact in pair_set: ALGEBRAIC compose (bind+bundle), unbind each role, cleanup -> recovered
        concept. Returns (clean_acc, corrupt_acc, mem_floor_acc)."""
        clean_ok = corrupt_ok = mem_ok = tot = 0
        for (ai, bi) in pair_set:
            a, b = names[ai], names[bi]
            fact = R_AGENT * codebook[a] + R_PATIENT * codebook[b]            # bind + bundle (the validated VSA algebra)
            ra, _ = _cleanup(fact * np.conj(R_AGENT), names, codebook)
            rb, _ = _cleanup(fact * np.conj(R_PATIENT), names, codebook)
            clean_ok += int(ra == a) + int(rb == b)
            # CORRUPTED: rebuild the agent slot from a NOISY live percept of object a, then unbind agent.
            a_noisy = _to_phasor(read_cortex_it_rate(bridge, handles, ai, corrupt_rng=crng), proj)
            fact_c = R_AGENT * a_noisy + R_PATIENT * codebook[b]
            rac, _ = _cleanup(fact_c * np.conj(R_AGENT), names, codebook)
            corrupt_ok += int(rac == a)
            # MEMORIZATION FLOOR on the SAME fact (recall baseline): does the stored-fact lookup recover the fillers?
            mem_ok += int(_mem_recall(fact, "agent") == a) + int(_mem_recall(fact, "patient") == b)
            tot += 1
        return clean_ok / (2 * tot), corrupt_ok / tot, mem_ok / (2 * tot)

    ho_clean, ho_corrupt, ho_mem = _compose_eval(held_out)       # HELD-OUT (novel) — the load-bearing generalization
    mm_clean, _, mm_mem = _compose_eval(memorized)               # memorized (sanity: the recall baseline CAN do these)

    chance = 1.0 / N_OBJECTS
    print(f"  [seed {seed}] HELD-OUT novel: compose clean {ho_clean:.3f} corrupt {ho_corrupt:.3f} | "
          f"memorization-floor {ho_mem:.3f} (chance {chance:.3f}) || memorized: compose {mm_clean:.3f} "
          f"mem-floor {mm_mem:.3f}", flush=True)
    return {"seed": seed, "held_out_compose_clean": ho_clean, "held_out_compose_corrupt": ho_corrupt,
            "held_out_mem_floor": ho_mem, "memorized_compose_clean": mm_clean, "memorized_mem_floor": mm_mem,
            "chance": chance, "n_held_out": len(held_out), "n_memorized": len(memorized)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default="research/findings/raw/_step3_live_cortex_grounded_compose.json")
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    print("[step3-live-compose] do LIVE cortex_it RATE-derived grounded codes COMPOSE (held-out novel >> "
          "memorization floor)?", flush=True)
    rows = [run_seed(s) for s in args.seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    clean, corrupt, mem = m("held_out_compose_clean"), m("held_out_compose_corrupt"), m("held_out_mem_floor")
    chance = rows[0]["chance"]
    clean_ok = all(r["held_out_compose_clean"] >= 0.90 for r in rows)
    corrupt_ok = all(r["held_out_compose_corrupt"] >= 0.80 for r in rows)
    beats_floor = all(r["held_out_compose_clean"] >= r["held_out_mem_floor"] + 0.30 for r in rows)
    if clean_ok and corrupt_ok and beats_floor:
        v = "GO"
    elif clean >= 0.90 and clean >= mem + 0.30:
        v = "PARTIAL"
    else:
        v = "NEGATIVE"
    print(f"\n{'='*98}\n  MEAN ({len(rows)} seeds): HELD-OUT compose clean {clean:.3f} corrupt {corrupt:.3f} | "
          f"memorization-floor {mem:.3f} (chance {chance:.3f})", flush=True)
    print(f"{'='*98}", flush=True)
    if v == "GO":
        print(f"  GO: LIVE cortex_it RATE-derived grounded codes COMPOSE — held-out novel (object,role) combos "
              f"recover at {clean:.3f} clean / {corrupt:.3f} corrupted, FAR above the memorization floor "
              f"({mem:.3f}) and chance ({chance:.3f}). The percept enters the validated bundling algebra: a "
              f"PERCEIVED object can be bound into a NOVEL fact (not just recalled). ==> the rate-vs-phasor wall is "
              f"dissolved for perceived-object facts via shared grounded codes (sourced from the live spiking "
              f"substrate). Present to owner; the full integration + the dendritic multi-attribute arc remain the "
              f"owner call.", flush=True)
    elif v == "PARTIAL":
        print(f"  PARTIAL: held-out compose clean {clean:.3f} >> mem-floor {mem:.3f} but corrupted {corrupt:.3f} "
              f"< 0.80 — the rate code grounds but noisily; population-average / more cortex_it neurons / a learned "
              f"projection should lift the corrupted case.", flush=True)
    else:
        print(f"  NEGATIVE: live rate-derived codes don't compose above the floor ({clean:.3f} vs {mem:.3f}) — the "
              f"cortex_it rate code is too noisy to ground a clean composer code; localize (learned vs fixed "
              f"projection / population code / the dendritic path).", flush=True)
    import json as _j
    with open(args.out, "w") as f:
        _j.dump({"mean_held_out_clean": clean, "mean_held_out_corrupt": corrupt, "mean_mem_floor": mem,
                 "chance": chance, "verdict": v, "per_seed": rows}, f, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    raise SystemExit(0 if v == "GO" else (2 if v == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
