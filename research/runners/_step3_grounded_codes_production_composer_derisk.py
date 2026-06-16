"""Step-3 integration-readiness — do LIVE cortex_it-rate-derived grounded codes drop into the PRODUCTION composer?

The cheap-first + scaled de-risks showed live `cortex_it` spiking-rate -> phasor codes COMPOSE in a mini-algebra
(my own roles + cleanup). THIS closes the last cheap gap before any gated integration build: feed those SAME
live-perception-grounded codes into the ACTUAL production `RFPhasorComposer` and run its real `store` /
`query_patient` / `query_agent` / no-confab moat — on the real 3-way SVO bundle (agent+action+patient), not a
2-role toy.

WHY this is the exact right test: `RFPhasorComposer.__init__` ALREADY exposes `grounded_codes={word: phases[D]}`
(the "cheat-A conversion" sensory-grounding INTERFACE), and its docstring states the interface is "validated ==
random at parity" but that "producing MEANINGFUL grounded codes ... is the open problem -- the embodied-cognition
limit." The step-3 arc PRODUCED meaningful grounded codes (from the live spiking perception). So this de-risk =
(validated interface) x (meaningful live-perception codes) -> does the production conversational composer COMPOSE +
ABSTAIN correctly when its concept codes ARE the navigation perception's live spiking responses?

GATE (multi-seed 42/43/44): GO if the grounded composer (a) RECALLS every stored fact's patient AND agent, (b)
ABSTAINS (returns None) on every absent (agent, action) query (the no-confab moat intact), AND (c) MATCHES the
random-code baseline composer behavior on the same facts/queries (parity -> grounding doesn't degrade the pipeline).
A miss on (a) or (c) localizes a representation gap; a moat breach on (b) is a hard stop (never weaken the moat).

CPU/GPU: the composer runs D-dim RF bridges per op; D=2048 (production tier) -> GPU (`SIM_BACKEND=cupy`).
  SIM_BACKEND=cupy python -m research.runners._step3_grounded_codes_production_composer_derisk --D 2048
No sim/ edit; reuse-by-import (the (B) probe's live cortex_it bridge + the production RFPhasorComposer).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host
from sim.text_embeddings import orthogonal_drive_pattern

from research.runners.funcint_perception_to_memory_probe import (
    OBJECT_WORDS, N_OBJECTS, N_CORTEX_IT, PERCEPT_SPARSITY, PERCEPT_DRIVE_PA, build_probe_bridge,
)
from research.runners.rf_phasor_composer import RFPhasorComposer

RATE_READ_STEPS = 80
SETTLE_STEPS = 20

# perceived OBJECTS = grounded (agent/patient fillers); ACTIONS = native composer codes (verbs aren't perceived).
ACTIONS = ["chase", "near", "see"]
# facts over PERCEIVED objects: every agent + patient is a live-perception-grounded code.
FACTS = [("dog", "chase", "cat"), ("apple", "near", "river"), ("cat", "see", "dog")]
# absent (agent, action) queries -> the no-confab moat must ABSTAIN (return None). None of these is a stored pair.
ABSENT = [("river", "chase"), ("apple", "see"), ("dog", "near")]


def _projection(D, n_in, seed):
    rng = np.random.default_rng(seed * 5077 + 11)
    return (rng.standard_normal((D, n_in)) + 1j * rng.standard_normal((D, n_in))).astype(np.complex128)


def read_cortex_it_rate(bridge, it_indices, obj_idx):
    """LIVE cortex_it spiking firing-rate vector for object obj_idx (the grounded rate code)."""
    xp, _ = get_backend()
    n_it = int(it_indices.size)
    drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS, n_neurons=n_it,
                                     drive_max_pA=PERCEPT_DRIVE_PA, sparsity=PERCEPT_SPARSITY).astype(np.float64)
    drive_dev = xp.asarray(drive, dtype=xp.float32)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    counts = np.zeros(n_it, dtype=np.float64)
    for _ in range(RATE_READ_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[it_indices] = drive_dev
        bridge._run_one_simulation_step()
        counts += np.asarray(to_host(bridge.cp_firing_states[it_indices])).astype(np.float64)
    bridge.cp_external_input_current[:] = 0.0
    return counts / RATE_READ_STEPS


def grounded_phases(rate_vec, proj):
    """Live rate code -> composer phases[D] in [0,1): the composer's _to_phasor(phases)=exp(2pi i phases) then equals
    exp(i angle(proj@rate)) = the SAME grounded phasor the step-3 probe composed."""
    z = proj @ rate_vec.astype(np.complex128)
    return (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)


def _eval_composer(comp):
    """Run store + the conversational queries; return (recall_ok, recall_tot, moat_ok, moat_tot, answers)."""
    for (a, v, p) in FACTS:
        comp.store(a, v, p)
    recall_ok = recall_tot = 0
    answers = {}
    for (a, v, p) in FACTS:
        qp = comp.query_patient(a, v)
        qa = comp.query_agent(v, p)
        answers[f"{a}-{v}->patient"] = qp
        answers[f"{v}-{p}->agent"] = qa
        recall_ok += int(qp == p) + int(qa == a)
        recall_tot += 2
    moat_ok = moat_tot = 0
    for (a, v) in ABSENT:
        qp = comp.query_patient(a, v)
        answers[f"ABSENT {a}-{v}->patient"] = qp
        moat_ok += int(qp is None)            # the no-confab moat: abstain on an unstored (agent, action)
        moat_tot += 1
    return recall_ok, recall_tot, moat_ok, moat_tot, answers


def run_seed(seed, D):
    bridge, handles = build_probe_bridge(seed)
    it_indices = handles["it_indices"]
    proj = _projection(D, N_CORTEX_IT, seed)
    grounded = {OBJECT_WORDS[i]: grounded_phases(read_cortex_it_rate(bridge, it_indices, i), proj)
                for i in range(N_OBJECTS)}
    vocab = list(OBJECT_WORDS) + ACTIONS

    g = RFPhasorComposer(seed=seed, D=D, vocab=vocab, grounded_codes=grounded)
    r_ok, r_tot, m_ok, m_tot, g_ans = _eval_composer(g)
    # parity baseline: identical composer with the composer's OWN random codes (no grounding).
    b = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    br_ok, br_tot, bm_ok, bm_tot, b_ans = _eval_composer(b)
    parity = (g_ans == b_ans)

    go = (r_ok == r_tot and m_ok == m_tot and parity)
    print(f"  [seed {seed} D={D}] grounded recall {r_ok}/{r_tot}  moat-abstain {m_ok}/{m_tot}  || "
          f"baseline recall {br_ok}/{br_tot} moat {bm_ok}/{bm_tot}  parity={parity}  [{'GO' if go else 'MISS'}]",
          flush=True)
    if not go:
        for k in g_ans:
            if g_ans[k] != b_ans.get(k):
                print(f"      DIFF {k}: grounded={g_ans[k]!r} baseline={b_ans.get(k)!r}", flush=True)
    return {"seed": seed, "D": D, "recall_ok": r_ok, "recall_tot": r_tot, "moat_ok": m_ok, "moat_tot": m_tot,
            "baseline_recall_ok": br_ok, "parity": parity, "go": go, "grounded_answers": g_ans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_step3_grounded_codes_production_composer.json")
    args = ap.parse_args()
    _, backend = get_backend()
    print(f"[step3-prod-composer] backend={backend} D={args.D} — do LIVE cortex_it-rate grounded codes drop into "
          f"the PRODUCTION RFPhasorComposer (store/query/moat, 3-way SVO bundle)?", flush=True)
    rows = [run_seed(s, args.D) for s in args.seeds]
    all_recall = all(r["recall_ok"] == r["recall_tot"] for r in rows)
    all_moat = all(r["moat_ok"] == r["moat_tot"] for r in rows)
    all_parity = all(r["parity"] for r in rows)
    verdict = "GO" if (all_recall and all_moat and all_parity) else (
        "MOAT_BREACH" if not all_moat else "NEGATIVE")
    print(f"\n{'='*96}", flush=True)
    print(f"  {len(rows)} seeds: recall {'PASS' if all_recall else 'FAIL'}  moat {'INTACT' if all_moat else 'BREACH'}"
          f"  parity-vs-random {'PASS' if all_parity else 'FAIL'}  ==> {verdict}", flush=True)
    if verdict == "GO":
        print("  GO: the navigation perception's LIVE spiking-rate codes drop into the PRODUCTION conversational "
              "composer as grounded concept codes — it composes the 3-way SVO fact, recalls patient + agent, and "
              "ABSTAINS on unstored queries (no-confab moat intact), == the random-code baseline. The composer's "
              "documented open boundary ('producing meaningful grounded codes') is closed for PERCEIVED objects: "
              "the grounding interface x meaningful live-perception codes = a production composer composing what "
              "the agent SAW. The remaining build (wiring this onto the merged nav+conv bridge) is owner-gated.",
              flush=True)
    elif verdict == "MOAT_BREACH":
        print("  MOAT_BREACH (HARD STOP): a grounded composer accepted an unstored query — investigate before any "
              "build; never weaken the moat.", flush=True)
    else:
        print("  NEGATIVE: grounded codes don't reach random-code parity in the production composer — localize "
              "(phase conversion / cleanup competition with action codes / D).", flush=True)
    print(f"{'='*96}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict, "backend": backend, "D": args.D, "per_seed": rows}, f, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    raise SystemExit(0 if verdict == "GO" else 1)


if __name__ == "__main__":
    main()
