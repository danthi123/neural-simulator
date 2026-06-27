"""BURNDOWN C2 de-risk -- the first-chat console's DEFAULT render word-ORDER on SPIKES.

C2 converts the console's certain-sentence word ORDERING from the host f-string (the fluency faculty's
`TemplateStubFaculty.render_svo`, `f"{det_a}{agent} {verb} {patient}."`) to the VALIDATED spiking
competitive-queuing read-out (`NeuralSerialOrderRenderer`). The console builds a `SpikingOrderStubFaculty`
(a `TemplateStubFaculty` subclass) whose `render_svo` orders the 3 SVO slots [agent, verb, patient] by the
per-pool spiking RATE ranking on a real `SimulationBridge`, then assembles the surface in the neural order.

This de-risk asserts the C2 conversion is SOUND on the SAME backend the console runs on (numpy-CPU by default;
also valid on cupy -- the renderer is backend-agnostic), with the C1 anti-cheat controls:

  (1) PARITY     -- SpikingOrderStubFaculty.render_svo(a,v,p).surface == the host f-string surface for the
                    canonical SVO frame (the spiking order == [agent, verb, patient], so byte-identical), AND
                    the asserted SVO == [a, v, p] (VERIFY content, unchanged), over N_FACTS facts x >=3 seeds.
  (2) EQUAL-DRIVE FAILS -- a FLAT primacy gradient (no agent>verb>patient gap) must NOT reliably reproduce the
                    SVO order (the no-learning control: proves the NEURONS serialize via the gradient, not a
                    host sort / pool bias). Bar: equal-drive order-score << the real-gradient score.
  (3) LESION FAILS -- ZERO drive (all pools silent / unconditioned) likewise fails to reproduce SVO.
  (4) MOAT       -- ordering is gated by a stored composite (an unstored fact -> the caller abstains BEFORE any
                    ordering); a word-ORDER change can never fabricate a fact. (Asserted structurally: the
                    faculty only ever orders the 3 content tokens it is GIVEN -- it cannot add/drop/swap one;
                    the asserted SVO is the canonical [a,v,p] regardless of the spiking order.)

GO = parity holds at >=3 seeds AND the equal-drive + lesion controls FAIL AND the moat is structurally intact.
Reuse-by-import; NO `sim/` edit. Runs on the console's native backend.
  Run:  SIM_BACKEND=numpy python -u -m research.runners._burndown_C2_console_spiking_render_derisk
        SIM_BACKEND=cupy  python -u -m research.runners._burndown_C2_console_spiking_render_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.song_g1_core import score_order  # noqa: E402
from research.runners._grounded_lang_p3_derisk import _inflect, _determiner  # noqa: E402
from research.runners.first_chat_console import SpikingOrderStubFaculty  # noqa: E402  (the class the console uses)
from research.runners.neural_serial_order_renderer import NeuralSerialOrderRenderer  # noqa: E402

OUT = os.path.join(_HERE, "..", "findings", "raw", "_burndown_C2_console_spiking_render.json")

SEEDS = (42, 43, 44)
N_FACTS = 24
VOCAB = 16            # the renderer's pool budget (SVO uses 3 pools)


def _host_render_svo(agent, action, patient, template=0):
    """The ORIGINAL host-literal TemplateStubFaculty.render_svo surface (the oracle to match byte-for-byte)."""
    det_a = _determiner(agent, "agent")
    verb = _inflect(action)
    if template % 2 == 0:
        return f"{det_a}{agent} {verb} {patient}."
    return f"{det_a}{agent} {verb} the {patient}."


def _make_facts(seed):
    """N_FACTS distinct (agent, action, patient) word-triples (synthetic content tokens -- ordering is content-
    agnostic; the spiking order depends on the FRAME, not the words)."""
    rng = np.random.default_rng(seed * 13 + 1)
    nouns = [f"n{i}" for i in range(40)]
    verbs = [f"v{i}" for i in range(20)]
    facts, seen = [], set()
    while len(facts) < N_FACTS:
        a = nouns[rng.integers(len(nouns))]; v = verbs[rng.integers(len(verbs))]; p = nouns[rng.integers(len(nouns))]
        if a == p or (a, v, p) in seen:
            continue
        seen.add((a, v, p)); facts.append((a, v, p))
    return facts


def _order_from_rates(idx, rate, tie_rng):
    """Rank slots by rate (desc). A RANDOM tie-break (not a stable sort) so that NO rate separation -> a RANDOM
    order (otherwise a stable sort would silently echo the input order and mask a dead control)."""
    jitter = {c: (rate[int(c)], float(tie_rng.random())) for c in idx}    # (rate, random) -> random tie-break
    return [int(c) for c in sorted(idx, key=lambda c: (-jitter[c][0], jitter[c][1]))]


def _equal_drive_order(renderer, n=3, equal_pA=1700.0, reps=12, seed=0):
    """The EQUAL-DRIVE / NO-LEARNING control (the canonical anti-cheat of the validated mechanism): drive all n
    slots with the SAME current (NO primacy gradient) on the renderer's bridge, read the rate ranking with a
    RANDOM tie-break. Mean order-score vs the SVO frame order [0..n-1] over `reps`. With no gradient the rates do
    not separate -> the order is ~random -> the score must be LOW (proves the NEURONS serialize via the gradient,
    not a host sort / pool bias). Same read path (`pool_rates`) `order` uses."""
    from research.runners._phaseB_serial_order_spiking_derisk import pool_rates
    idx = list(range(n))
    rng = np.random.default_rng(seed * 17 + 9)
    scores = []
    for _ in range(reps):
        drive = {int(c): float(equal_pA) for c in idx}
        rate = pool_rates(renderer.bridge, renderer.pool_idx, drive)
        scores.append(score_order(_order_from_rates(idx, rate, rng), idx))
    return float(np.mean(scores))


def _lesion_order(renderer, n=3, reps=12, seed=0):
    """The LESION control: ZERO drive to all slots (unconditioned pools) -> no rate signal at all. Read the rate
    ranking with a RANDOM tie-break -> with no signal the order is ~random -> the score must be LOW. (This
    LESIONS the conditioning drive entirely; the equal-drive control lesions only the GRADIENT. Both must fail ->
    the serial order requires the primacy current into the spiking pools.)"""
    from research.runners._phaseB_serial_order_spiking_derisk import pool_rates
    idx = list(range(n))
    rng = np.random.default_rng(seed * 23 + 11)
    scores = []
    for _ in range(reps):
        drive = {int(c): 0.0 for c in idx}
        rate = pool_rates(renderer.bridge, renderer.pool_idx, drive)
        scores.append(score_order(_order_from_rates(idx, rate, rng), idx))
    return float(np.mean(scores))


def run_seed(seed):
    facts = _make_facts(seed)
    fac = SpikingOrderStubFaculty(seed=seed)        # the EXACT class the console wires as ct.faculty
    # (1) PARITY: the spiking-ordered surface == the host f-string surface, content asserted unchanged.
    parity_surface_ok = 0
    parity_asserted_ok = 0
    real_order_score = 0.0
    for (a, v, p) in facts:
        for template in (0, 1):
            surface, asserted = fac.render_svo(a, v, p, template=template)
            if surface == _host_render_svo(a, v, p, template=template):
                parity_surface_ok += 1
            if list(asserted) == [a, v, p]:
                parity_asserted_ok += 1
    n_checks = len(facts) * 2
    # the underlying spiking order on the canonical SVO frame (slot 0=agent .. 2=patient).
    svo_order = fac._order.order([0, 1, 2])
    real_order_score = score_order(svo_order, [0, 1, 2])
    # (2)+(3) controls on the SAME renderer/bridge (reuse fac._order so the bridge is identical).
    equal_score = _equal_drive_order(fac._order, seed=seed)
    lesion_score = _lesion_order(fac._order, seed=seed)
    # (4) MOAT (structural): the faculty asserts ONLY the 3 content tokens it is given (parity_asserted_ok proves
    # the asserted SVO is the canonical [a,v,p]); the spiking ORDER reorders those tokens, it cannot invent one.
    moat_structural_ok = (parity_asserted_ok == n_checks)
    return {
        "seed": seed,
        "n_facts": len(facts),
        "n_checks": n_checks,
        "parity_surface_ok": parity_surface_ok,
        "parity_asserted_ok": parity_asserted_ok,
        "parity_surface_frac": round(parity_surface_ok / n_checks, 4),
        "real_svo_order": list(svo_order),
        "real_order_score": round(real_order_score, 4),
        "equal_drive_score": round(equal_score, 4),
        "lesion_score": round(lesion_score, 4),
        "moat_structural_ok": bool(moat_structural_ok),
    }


def main():
    backend = os.environ.get("SIM_BACKEND", "auto")
    print(f"[C2 de-risk] backend={backend}; the console's DEFAULT render word-order on spikes "
          f"(SpikingOrderStubFaculty == NeuralSerialOrderRenderer)\n", flush=True)
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        r = run_seed(seed)
        rows.append(r)
        print(f"  [seed {seed}] PARITY surface {r['parity_surface_ok']}/{r['n_checks']} "
              f"(asserted {r['parity_asserted_ok']}/{r['n_checks']}) | real-SVO-order {r['real_svo_order']} "
              f"score {r['real_order_score']:.3f} | equal-drive {r['equal_drive_score']:.3f} | "
              f"lesion {r['lesion_score']:.3f} | moat {'OK' if r['moat_structural_ok'] else 'BREACH'}", flush=True)
    # GO bars: parity 100% all seeds; real >= 0.999; equal-drive AND lesion << real (clear gap >= 0.30); moat OK.
    all_parity = all(r["parity_surface_ok"] == r["n_checks"] for r in rows)
    all_asserted = all(r["parity_asserted_ok"] == r["n_checks"] for r in rows)
    all_real_hi = all(r["real_order_score"] >= 0.999 for r in rows)
    # the control must FAIL: each control mean clears LESS than (real - 0.30) -> the gradient is doing the work.
    GAP = 0.30
    equal_fails = all(r["equal_drive_score"] <= r["real_order_score"] - GAP for r in rows)
    lesion_fails = all(r["lesion_score"] <= r["real_order_score"] - GAP for r in rows)
    moat_ok = all(r["moat_structural_ok"] for r in rows)
    go = all_parity and all_asserted and all_real_hi and equal_fails and lesion_fails and moat_ok
    verdict = "GO" if go else "NEGATIVE"
    summary = {
        "verdict": verdict, "backend": backend, "seeds": list(SEEDS),
        "all_parity_surface": all_parity, "all_parity_asserted": all_asserted,
        "all_real_order_hi": all_real_hi, "equal_drive_fails": equal_fails,
        "lesion_fails": lesion_fails, "moat_structural_ok": moat_ok,
        "gap_bar": GAP, "rows": rows, "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  VERDICT: {verdict}  "
          f"(parity={all_parity}/{all_asserted}, real-order-hi={all_real_hi}, "
          f"equal-drive-FAILS={equal_fails}, lesion-FAILS={lesion_fails}, moat={moat_ok})")
    print(f"  -> {os.path.relpath(OUT, _REPO)}  ({summary['elapsed_s']}s)")
    if go:
        print(f"  C2 SOUND: the console's certain-sentence word ORDER is the spiking competitive-queuing read-out "
              f"(byte-identical surface on the canonical SVO frame, neurally produced); the equal-drive + lesion "
              f"controls FAIL (the neurons serialize, not a host sort); the moat is structurally intact.")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
