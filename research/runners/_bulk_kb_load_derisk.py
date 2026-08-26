"""BULK KNOWLEDGE-BASE LOADING de-risk (board #65; serves the #66 knowledge-SCALE crux).

THE QUESTION. #66's RETRIEVAL half is solved -- the DG-CA3 sparse index makes lookup sublinear at V up to 200k
(`research/biology/dg-ca3-sparse-index.md`, 6-seed GO, wired into `one_brain_composer.py`). This is the LOADING
half: can a LARGE structured knowledge base (Wikidata/ConceptNet-style agent-action-patient triples) be bulk-loaded
into the FHRR fact store so that (a) load throughput is practical (facts/sec), (b) recall accuracy holds at scale,
(c) the no-confab moat holds (out-of-store cues abstain)?

THE ENCODING IS THE COMPOSER'S. Each triple is encoded as the composer's FHRR phasor binding: a fact = a BUNDLE of
role-filler binds, composite = sum_r exp(2*pi*i*(role_phase_r + filler_phase_r)); recall = unbind (subtract the role
phase) + matched-filter cleanup (argmax_w Re<rec, conj(code_w)>). This runner pulls the EXACT concept + role phase
codes from a real `RFPhasorComposer` (`comp.concepts`, `comp.roles`), so the bulk path encodes the identical FHRR
codes the spiking composer would.

TWO THROUGHPUT REGIMES, both reported honestly:
  - FAITHFUL spiking encode: `comp.store()` runs the per-op resonate (bind+bundle over ~208 RF steps each). This is
    the biologically-faithful cost and is SLOW on CPU (~1 fact/s); production batches it on GPU. Measured on a
    subsample.
  - PRACTICAL bulk encode: the closed-form FHRR algebra (phase-add bind + phasor-sum bundle) the resonate CONVERGES
    to, evaluated as one vectorized pass. This is the practical bulk loader (maps to the GPU-batched resonate).
    CROSS-CHECKED against the spiking `comp.store()`+`comp.query_patient()` on a subsample: the bulk path must
    reproduce the spiking composer's recall answers (this is what makes the fast path faithful, not a shortcut).

WHY the store scales at all: each fact is its OWN composite (a 3-role bundle), NOT superposed into one shared vector,
so per-fact decode integrity does not degrade with N -- the loading risk is throughput + storage + that the loaded
bindings stay recallable. (Sublinear RETRIEVAL over the N composites is the DG-index's job, #66; here recall is
measured as per-block decode integrity, which is what LOADING must preserve.)

GO (per seed): patient top-1 >= 0.95 AND agent top-1 >= 0.95 on the recalled sample; moat 0 new confab (out-of-store
(agent,action) cues abstain); bulk-load practical (>= 1000 facts/s); the bulk path reproduces the spiking composer's
recall on the cross-check subsample (agreement == 1.0). Anti-cheats: (1) SHUFFLED triples -- encode with the patient
column permuted, recall vs the TRUE mapping collapses to ~chance; (2) OUT-OF-STORE -- cues never stored abstain.

Reuse-by-import (RFPhasorComposer for the exact codes + the faithful cross-check); NO sim/ edit; numpy or cupy.
Run (real 6-seed sweep, mini-PC pool / CPU):
    python -u -m research.runners._bulk_kb_load_derisk --seeds 42,43,44,100,101,102 --n-facts 100000 \
        --out research/findings/raw/_bulk_kb_load_6seed.json
Smoke (1 seed, small N, numpy):
    SIM_BACKEND=numpy python -u -m research.runners._bulk_kb_load_derisk --seeds 42 --n-facts 5000 --spiking-n 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.rf_phasor_composer import RFPhasorComposer, ROLES  # noqa: E402

TRIPLE_ROLES = ("agent", "action", "patient")


def synth_kb(n_facts, na, nr, npat, seed):
    """Synthesize N structured agent-action-patient triples with UNIQUE (agent, action) keys (so query_patient is
    well-defined). Wikidata/ConceptNet-style: a bounded vocabulary of entities + relations, each (entity, relation)
    asserting one patient. Returns (triples, vocab)."""
    rng = np.random.default_rng(seed)
    agents = [f"ent{i}" for i in range(na)]
    actions = [f"rel{i}" for i in range(nr)]
    patients = [f"val{i}" for i in range(npat)]
    if na * nr < n_facts:
        raise ValueError(f"na*nr={na*nr} < n_facts={n_facts}: not enough unique (agent,action) keys")
    seen = set()
    triples = []
    # draw unique (agent, action) pairs
    while len(triples) < n_facts:
        a = agents[int(rng.integers(na))]
        r = actions[int(rng.integers(nr))]
        if (a, r) in seen:
            continue
        seen.add((a, r))
        p = patients[int(rng.integers(npat))]
        triples.append((a, r, p))
    vocab = agents + actions + patients
    return triples, vocab


def _codes_matrix(comp, words):
    """(len(words), D) phasor-PHASE matrix of the composer's exact concept codes."""
    return np.stack([comp.concepts[w] for w in words])


def bulk_encode(comp, triples):
    """VECTORIZED closed-form FHRR encode of all triples -> (N, D) composite PHASES, using the composer's exact
    concept + role codes. bind = (role_phase + filler_phase) mod 1 (phasor product = phase add); bundle = angle of
    the phasor sum. This is the fixed point the composer's `_encode` resonate converges to."""
    D = comp.D
    N = len(triples)
    Z = np.zeros((N, D), dtype=np.complex128)
    for col, role in enumerate(TRIPLE_ROLES):
        rp = comp.roles[role]  # (D,)
        fillers = np.stack([comp.concepts[t[col]] for t in triples])  # (N, D) phases
        Z += np.exp(2j * np.pi * ((rp[None, :] + fillers) % 1.0))
    return (np.angle(Z) / (2.0 * np.pi)) % 1.0


def decode_role(comp, comps, role, cb, chunk=20000):
    """VECTORIZED unbind + matched-filter cleanup: for each composite row, recover `role` (subtract the role phase)
    and return the argmax concept index over the codebook `cb` ((V,D) phasors). Chunked over rows to bound memory.
    This is exactly `RFPhasorComposer._cleanup_all` (sims = Re(rec_phasor @ conj(code)^T)), vectorized over the
    whole store."""
    rp = comp.roles[role]
    N = comps.shape[0]
    out = np.empty(N, dtype=np.int64)
    cbH = np.conj(cb).T  # (D, V)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        rec = np.exp(2j * np.pi * ((comps[s:e] - rp[None, :]) % 1.0))  # (c, D)
        sims = (rec @ cbH).real  # (c, V)
        out[s:e] = np.argmax(sims, axis=1)
    return out


def run_seed(seed, n_facts, na, nr, npat, D, sample, moat_sample, spiking_n, cross_n):
    t0 = time.time()
    triples, vocab = synth_kb(n_facts, na, nr, npat, seed)
    comp = RFPhasorComposer(seed=seed, D=D, vocab=sorted(vocab))
    words = comp.words
    vidx = {w: i for i, w in enumerate(words)}
    cb = np.exp(2j * np.pi * _codes_matrix(comp, words))  # (V, D) phasors

    # --- (a) THROUGHPUT: practical bulk encode (vectorized closed-form FHRR) ---
    t = time.time()
    comps = bulk_encode(comp, triples)  # (N, D)
    dt_bulk = time.time() - t
    bulk_fps = n_facts / dt_bulk if dt_bulk > 0 else float("inf")

    # --- FAITHFUL spiking encode throughput (subsample) + CROSS-CHECK the bulk path reproduces its recall ---
    m = min(spiking_n, n_facts)
    t = time.time()
    for a, r, p in triples[:m]:
        comp.store(a, r, p)  # per-op resonate bind+bundle -> numpy kb
    dt_spk = time.time() - t
    spk_fps = m / dt_spk if dt_spk > 0 else float("inf")
    # cross-check: on cross_n of the spiking-stored facts, the spiking composer's query_patient == the bulk decode
    cn = min(cross_n, m)
    ga = np.array([vidx[t[0]] for t in triples])
    gr = np.array([vidx[t[1]] for t in triples])
    gp = np.array([vidx[t[2]] for t in triples])
    p_hat_all = decode_role(comp, comps, "patient", cb)
    agree = 0
    for i in range(cn):
        a, r, p = triples[i]
        spk_ans = comp.query_patient(a, r)  # spiking scan+unbind+cleanup (numpy kb)
        bulk_ans = words[int(p_hat_all[i])]
        if spk_ans == bulk_ans:
            agree += 1
    cross_agree = agree / cn if cn else 0.0

    # --- (b) RECALL at scale (decode integrity over a sample) ---
    rng = np.random.default_rng(seed + 1)
    samp = rng.choice(n_facts, size=min(sample, n_facts), replace=False)
    a_hat = decode_role(comp, comps[samp], "agent", cb)
    r_hat = decode_role(comp, comps[samp], "action", cb)
    p_hat = p_hat_all[samp]
    patient_top1 = float((p_hat == gp[samp]).mean())
    agent_top1 = float((a_hat == ga[samp]).mean())
    action_top1 = float((r_hat == gr[samp]).mean())
    all_three = float(((p_hat == gp[samp]) & (a_hat == ga[samp]) & (r_hat == gr[samp])).mean())

    # --- (c) MOAT: out-of-store (agent, action) cues must abstain (no confab) ---
    # the store's decoded (agent, action) pair set == what a scan would match on (recall=1 => == stored pairs)
    a_hat_all = decode_role(comp, comps, "agent", cb)
    r_hat_all = decode_role(comp, comps, "action", cb)
    stored_pairs = set(zip((words[i] for i in a_hat_all), (words[i] for i in r_hat_all)))
    true_pairs = set((t[0], t[1]) for t in triples)
    agents = [f"ent{i}" for i in range(na)]
    actions = [f"rel{i}" for i in range(nr)]
    confab = 0
    tested = 0
    tries = 0
    while tested < moat_sample and tries < moat_sample * 50:
        tries += 1
        a = agents[int(rng.integers(na))]
        r = actions[int(rng.integers(nr))]
        if (a, r) in true_pairs:
            continue  # genuinely absent cue only
        tested += 1
        if (a, r) in stored_pairs:  # a scan WOULD match it -> a confabulation
            confab += 1
    moat_confab = confab
    moat_abstain_rate = (tested - confab) / tested if tested else 0.0

    # --- ANTI-CHEAT 1: SHUFFLED triples -> recall collapses ---
    perm = rng.permutation(n_facts)
    shuf = [(triples[i][0], triples[i][1], triples[perm[i]][2]) for i in range(n_facts)]
    comps_shuf = bulk_encode(comp, shuf)
    p_hat_shuf = decode_role(comp, comps_shuf[samp], "patient", cb)
    # decode recovers the SHUFFLED patient; recall vs the TRUE patient must collapse to ~chance (1/npat)
    shuf_recall_vs_true = float((p_hat_shuf == gp[samp]).mean())

    row = {
        "seed": seed, "n_facts": n_facts, "D": D, "V": len(words),
        "na": na, "nr": nr, "npat": npat,
        "bulk_fps": bulk_fps, "dt_bulk_s": dt_bulk,
        "spiking_fps": spk_fps, "spiking_n": m, "dt_spiking_s": dt_spk,
        "cross_check_agree": cross_agree, "cross_n": cn,
        "patient_top1": patient_top1, "agent_top1": agent_top1,
        "action_top1": action_top1, "all_three_top1": all_three, "recall_sample": int(len(samp)),
        "moat_confab": moat_confab, "moat_tested": tested, "moat_abstain_rate": moat_abstain_rate,
        "shuffle_recall_vs_true": shuf_recall_vs_true, "chance_recall": 1.0 / npat,
        "wall_s": time.time() - t0,
    }
    row["go"] = bool(
        patient_top1 >= 0.95 and agent_top1 >= 0.95 and moat_confab == 0
        and cross_agree == 1.0 and bulk_fps >= 1000.0
        and shuf_recall_vs_true < max(0.05, 5.0 / npat)
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-facts", type=int, default=100000)
    ap.add_argument("--na", type=int, default=2000, help="num agents/entities")
    ap.add_argument("--nr", type=int, default=100, help="num actions/relations")
    ap.add_argument("--npat", type=int, default=2000, help="num patients/values")
    ap.add_argument("--D", type=int, default=512, help="FHRR dimension (production capacity-probe scale)")
    ap.add_argument("--sample", type=int, default=3000, help="recall-integrity sample size")
    ap.add_argument("--moat-sample", type=int, default=2000, help="out-of-store cues tested for abstain")
    ap.add_argument("--spiking-n", type=int, default=60, help="facts stored via the faithful spiking encode (timing)")
    ap.add_argument("--cross-n", type=int, default=40, help="facts cross-checked spiking-vs-bulk recall")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_bulk_kb_load_derisk.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for s in seeds:
        r = run_seed(s, args.n_facts, args.na, args.nr, args.npat, args.D,
                     args.sample, args.moat_sample, args.spiking_n, args.cross_n)
        rows.append(r)
        print(f"[seed {s}] bulk={r['bulk_fps']:.0f} f/s  spk={r['spiking_fps']:.2f} f/s  "
              f"xcheck={r['cross_check_agree']:.3f}  patient@1={r['patient_top1']:.4f}  "
              f"agent@1={r['agent_top1']:.4f}  moat_confab={r['moat_confab']}/{r['moat_tested']}  "
              f"shuffle={r['shuffle_recall_vs_true']:.4f}  -> {'GO' if r['go'] else 'NO-GO'}", flush=True)

    n_go = sum(1 for r in rows if r["go"])
    all_go = n_go == len(rows)
    min_patient = min(r["patient_top1"] for r in rows)
    min_agent = min(r["agent_top1"] for r in rows)
    max_confab = max(r["moat_confab"] for r in rows)
    min_xcheck = min(r["cross_check_agree"] for r in rows)
    min_bulk = min(r["bulk_fps"] for r in rows)
    max_shuffle = max(r["shuffle_recall_vs_true"] for r in rows)
    chance = 1.0 / args.npat

    # A verdict must travel with what earned it (tools/gates/verdict_preconditions).
    from tools.verdict import Verdict  # noqa: E402
    v = Verdict("bulk-KB loading (LOADING half of the knowledge-scale crux)", chance=chance)
    v.require("patient top-1 >= 0.95 (worst seed)", min_patient, expect=lambda x: x >= 0.95)
    v.require("agent top-1 >= 0.95 (worst seed)", min_agent, expect=lambda x: x >= 0.95)
    v.require("moat: 0 new confab on out-of-store cues (worst seed)", max_confab, expect=lambda x: x == 0)
    v.require("bulk path reproduces spiking composer recall (xcheck == 1.0)", min_xcheck, expect=lambda x: x >= 1.0)
    v.require("bulk load practical (>= 1000 f/s, worst seed)", min_bulk, expect=lambda x: x >= 1000.0)
    v.require("anti-cheat: shuffled-triples recall collapses (< 0.05, near chance %.4f)" % chance,
              max_shuffle, expect=lambda x: x < 0.05)
    verdict_block = v.decide(go=all_go)

    agg = {
        "verdict": verdict_block["status"],
        "n_go": n_go, "n_seeds": len(rows),
        "min_patient_top1": min_patient,
        "min_agent_top1": min_agent,
        "max_moat_confab": max_confab,
        "min_cross_check_agree": min_xcheck,
        "min_bulk_fps": min_bulk,
        "median_spiking_fps": float(np.median([r["spiking_fps"] for r in rows])),
        "max_shuffle_recall": max_shuffle,
        "chance": chance,
        "seeds": seeds, "n_facts": args.n_facts, "rows": rows,
        **{k: verdict_block[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(agg, fh, indent=2, default=str)
    print(f"\n{'='*70}\nVERDICT: {agg['verdict']}  ({n_go}/{len(rows)} seeds GO)")
    print(f"  min patient@1={agg['min_patient_top1']:.4f}  min agent@1={agg['min_agent_top1']:.4f}  "
          f"max moat confab={agg['max_moat_confab']}  min xcheck={agg['min_cross_check_agree']:.3f}")
    print(f"  bulk >= {agg['min_bulk_fps']:.0f} f/s  faithful spiking ~{agg['median_spiking_fps']:.2f} f/s  "
          f"(shuffle recall <= {agg['max_shuffle_recall']:.4f}, chance {1.0/args.npat:.4f})")
    print(f"  wrote {args.out}")
    return 0 if all_go else 1


if __name__ == "__main__":
    sys.exit(main())
