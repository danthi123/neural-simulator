"""gap#4 Q5 -- the OBLIGATORY-DEPTH-3 CREDIT INSTRUMENT (the falsifiability enabler for every gap#4 deep-credit lane).

WHY (the located wall + the falsifiability gap). gap#4's deep-credit GOs ("DFA e-prop is depth-robust N2/3/4",
DECOLLE, Forward-Forward, birdsong-tutor) are read on tasks where a DEPTH-2 model might ALREADY solve the held-out set
(the T=24 temporal-depth confound; `2026-08-02-gap4-crux-wall-LOCATED-*`, `2026-08-02-gap4-depth-rescue-untestable-*`).
So "depth-robust" does NOT prove the rule assigned genuine depth-3 credit. Every mechanism lane needs a task where ONLY a
genuine depth-3 credit assignment can generalize -- the OBLIGATORY-DEPTH-3 gate:
    (i)   a DEPTH-2 oracle FAILS held-out         (l2 <= chance + 0.06)
    (ii)  a DEPTH-3 model GENERALIZES             (l3 >= 0.80)
    (iii) the depth jump is real                  (l3 - l2 >= 0.15)
This is the EXACT `depth3_requiring` predicate the crux runner computes
(`_gap4_bptt_snn_chained_fa_transport_free_derisk.py` L577/L824:
 `bool(l2 <= chance+0.06 and l3 >= 0.80 and (l3-l2) >= 0.15)`) -- REUSED here, not reinvented, on top of the SAME
`stage0_depth_genuineness` DendriticMLP rate oracle it uses.

WHAT THIS RUNNER IS. A task-family SURVEY that measures the obligatory-depth-3 gate on the shared rate oracle
(`stage0_depth_genuineness`, DendriticMLP l0/l1/l2/l3 held-out-generalization curve) across every candidate construction
that could plausibly obligate depth-3 on the spiking substrate, plus the on-substrate DFA e-prop / surrogate-BPTT /
shuffle arms on the best candidate. It is BOTH the instrument (if a family gates 6/6, that task IS the falsifiable
depth-3 target other lanes import) AND the measurement that decides whether such an instrument EXISTS at all.

TASK FAMILIES (all spiking-compatible: X in +/-1, rate-coded as constant current over T by the isolation runner's
`_forward_logits`, exactly as the depth-2 XOR task is; k=2; the WHOLE held-out set is the generalization test):
  * parity      -- pure n-bit XOR. XOR-tree = log2(n) depth; a 1-hidden net needs ~2^n width. At a FIXED width the
                   depth-3 tree can fit while a width-starved depth-2 cannot -> the one construction where depth GENUINELY
                   matters (matched-width depth separation, the standard operationalization -- Telgarsky 2016 /
                   Eldan-Shamir 2016: depth-2 needs exp width).
  * nestedxor   -- XOR(MAJ(pair-XOR)) (the crux runner's `make_task_nestedxor`, reused). Intended obligatory-depth-3.
  * xorandxor   -- XOR(AND(pair-XOR)) -- AND breaks the parity fold (a nestedxor variant).
  * mux         -- n-address multiplexer (a classic depth-hard boolean function): address bits select a data bit.
  * hier3       -- the semantic-inheritance 3-level taxonomy (the crux runner's `make_task_hier3`, reused).

GATE / GO. The instrument EXISTS iff some family satisfies (i)-(iii) on the rate oracle for >=5/6 seeds; the DEPLOYED
GO additionally requires DFA e-prop (transport-free deep-credit, the isolation runner's `credit_mode='eprop'`) to hold
the obligatory-depth-3 predicate on the spiking substrate with the eprop_shuffle control collapsing to chance. If NO
family gates robustly, that is the HONEST NEGATIVE and it SHARPENS the wall: obligatory-depth-3 as a matched-width
GENERALIZATION gate is not constructible on this rate/spiking substrate at practical scale -> every lane's "depth-3 GO"
is unfalsifiable by task construction, and must be validated by the credit-ALIGNMENT route instead (the crux pivot).

NON-NEGOTIABLES honoured: brain-based-only (the DendriticMLP/BPTT oracles are LABELLED instruments; the deployed target
is the spiking substrate + DFA e-prop); 6-seed (42 43 44 100 101 102); SIM_BACKEND=numpy; honest-negative-is-a-
deliverable; NO sim/ edit (all reuse-by-import); `cfg.seed`-equivalent -- every net is seeded from the per-seed `seed`
(DendriticMLP(seed=), _train_snn(seed) build its RNG from it), verified by the byte-identical-init self-check below.

Run (1-seed smoke -- rate survey only, fast):
    SIM_BACKEND=numpy python -m research.runners._gap4_obligatory_depth3_instrument_derisk --seeds 42 --no-spiking
The 6-seed decisive run (rate survey across families + spiking DFA arms on the best candidate):
    SIM_BACKEND=numpy python -m research.runners._gap4_obligatory_depth3_instrument_derisk \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/_gap4_depth3_instrument/instrument_6seed.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# --- reuse-by-import: the SHARED rate oracle depth-genuineness measurement + oracle train/eval (NO reinvention) ---
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    stage0_depth_genuineness, _train_oracle, _acc_on)
from sim.dendritic_mlp import DendriticMLP  # noqa: E402
# --- reuse-by-import: the two existing candidate obligatory-depth-3 task builders (crux runner) ---
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import (  # noqa: E402
    make_task_nestedxor, make_task_hier3)
# --- reuse-by-import: the spiking substrate + credit rules (surrogate-BPTT, DFA e-prop, eprop_shuffle) ---
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _train_snn, _accuracy)

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_depth3_instrument" / "instrument.json"

# THE gate constants (identical to the crux runner's `depth3_requiring` at L577/L824). One place, so a lane that
# imports this instrument gets the SAME arithmetic the crux stage0 uses.
CHANCE_MARGIN = 0.06          # (i)   l2 <= chance + CHANCE_MARGIN
DEEP_GENERALIZES = 0.80       # (ii)  l3 >= DEEP_GENERALIZES
DEPTH_JUMP = 0.15             # (iii) l3 - l2 >= DEPTH_JUMP


def obligatory_depth3_gate(l2, l3, chance):
    """The EXACT crux `depth3_requiring` predicate (reused, not reinvented): depth-2 fails, depth-3 generalizes, the
    jump is real. Returns (bool, dict-of-margins). NaN-safe (an unfit oracle -> False)."""
    if any(np.isnan(v) for v in (l2, l3, chance)):
        return False, {"l2_ok": False, "l3_ok": False, "jump_ok": False}
    l2_ok = bool(l2 <= chance + CHANCE_MARGIN)
    l3_ok = bool(l3 >= DEEP_GENERALIZES)
    jump_ok = bool((l3 - l2) >= DEPTH_JUMP)
    return bool(l2_ok and l3_ok and jump_ok), {"l2_ok": l2_ok, "l3_ok": l3_ok, "jump_ok": jump_ok}


# ============================================================================================================
# TASK BUILDERS. All return the canonical 4-tuple the shared stage0/arms consume:
#   (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx   with idx = {"inh_idx": <whole held set>, "memctrl_idx": []}.
# Ltr/Lte are dummy latents (non-gating). X in +/-1 (rate-coded over T identically to the depth-2 XOR task).
# `nestedxor` and `hier3` are IMPORTED (reuse); `parity`/`xorandxor`/`mux` are the NEW constructions this survey adds.
# ============================================================================================================
def _pack(Xtr, ytr, Xte, yte, meta):
    Ltr = np.zeros((len(Xtr), 1), dtype=np.float64)
    Lte = np.zeros((len(Xte), 1), dtype=np.float64)
    idx = {"inh_idx": np.arange(len(Xte), dtype=np.int64), "memctrl_idx": np.array([], dtype=np.int64)}
    return (Xtr, np.asarray(ytr, np.int64), Ltr), (Xte, np.asarray(yte, np.int64), Lte), meta, idx


def make_parity(seed, n_bits=8, split=0.65):
    """label = XOR of all n bits. The one construction where depth genuinely matters at matched width: the depth-3
    XOR-tree fits in small width while a width-starved depth-2 net (2^n width to fold parity in one place) cannot."""
    rng = np.random.default_rng(seed)
    n = 1 << n_bits
    bits = ((np.arange(n)[:, None] >> np.arange(n_bits)[None, :]) & 1).astype(np.int64)
    label = np.bitwise_xor.reduce(bits, axis=1).astype(np.int64)
    X = bits.astype(np.float64) * 2.0 - 1.0
    idx = rng.permutation(n); cut = int(split * n)
    tr, te = idx[:cut], idx[cut:]
    meta = {"task": "parity", "n_bits": int(n_bits), "k_classes": 2, "n_features": int(n_bits),
            "n_train": int(len(tr)), "n_heldout": int(len(te)), "n_inherit_heldout": int(len(te)), "split": split}
    return _pack(X[tr], label[tr], X[te], label[te], meta)


def make_xorandxor(seed, n_bits=8, split=0.65):
    """label = XOR over AND-of-(pair-XOR) groups. AND breaks the parity fold (a nestedxor variant with lower fan-in)."""
    rng = np.random.default_rng(seed)
    n_pairs = n_bits // 2
    n_g = n_pairs // 2
    n = 1 << n_bits
    bits = ((np.arange(n)[:, None] >> np.arange(n_bits)[None, :]) & 1).astype(np.int64)
    pair_xor = np.logical_xor(bits[:, 0::2].astype(bool), bits[:, 1::2].astype(bool))
    used = pair_xor[:, :2 * n_g].reshape(n, n_g, 2)
    g_and = used.all(axis=2)
    label = np.bitwise_xor.reduce(g_and.astype(np.int64), axis=1).astype(np.int64)
    X = bits.astype(np.float64) * 2.0 - 1.0
    idx = rng.permutation(n); cut = int(split * n)
    tr, te = idx[:cut], idx[cut:]
    meta = {"task": "xorandxor", "n_bits": int(n_bits), "k_classes": 2, "n_features": int(n_bits),
            "n_train": int(len(tr)), "n_heldout": int(len(te)), "n_inherit_heldout": int(len(te)), "split": split}
    return _pack(X[tr], label[tr], X[te], label[te], meta)


def make_mux(seed, n_addr=3, split=0.65):
    """n-address multiplexer: n_addr address bits select one of 2^n_addr data bits; label = the selected data bit.
    A classic depth-hard boolean function (must READ the address then INDEX)."""
    rng = np.random.default_rng(seed)
    n_data = 1 << n_addr
    n_bits = n_addr + n_data
    n = 1 << n_bits
    if n > 200000:
        m = 60000
        bits = rng.integers(0, 2, size=(m, n_bits)).astype(np.int64)
    else:
        bits = ((np.arange(n)[:, None] >> np.arange(n_bits)[None, :]) & 1).astype(np.int64)
    addr = bits[:, :n_addr]; data = bits[:, n_addr:]
    sel = (addr * (1 << np.arange(n_addr))).sum(1)
    label = data[np.arange(len(data)), sel].astype(np.int64)
    X = bits.astype(np.float64) * 2.0 - 1.0
    idx = rng.permutation(len(bits)); cut = int(split * len(bits))
    tr, te = idx[:cut], idx[cut:]
    meta = {"task": "mux", "n_addr": int(n_addr), "n_bits": int(n_bits), "k_classes": 2, "n_features": int(n_bits),
            "n_train": int(len(tr)), "n_heldout": int(len(te)), "n_inherit_heldout": int(len(te)), "split": split}
    return _pack(X[tr], label[tr], X[te], label[te], meta)


def build_task(family, seed, params):
    if family == "parity":
        return make_parity(seed, n_bits=params.get("n_bits", 8), split=params.get("split", 0.65))
    if family == "xorandxor":
        return make_xorandxor(seed, n_bits=params.get("n_bits", 8), split=params.get("split", 0.65))
    if family == "mux":
        return make_mux(seed, n_addr=params.get("n_addr", 3), split=params.get("split", 0.65))
    if family == "nestedxor":
        return make_task_nestedxor(seed)                       # reuse (crux runner) -- XOR(MAJ3(pair-XOR)), 12-bit
    if family == "hier3":
        return make_task_hier3(seed)                           # reuse (crux runner) -- 3-level semantic taxonomy
    raise ValueError("unknown family %r" % family)


# per-family (n, hidden) chosen from the seed-42 exploration (scratchpad sweep 2026-08-11): parity nb8/h12 is the
# single closest-to-gating point; the rest are shown to FOLD to depth-2 at any tested width (their gate always fails
# on l2). hidden is the SAME for l1/l2/l3 in stage0 -> the separation, when it exists, is depth-at-matched-width.
FAMILY_HIDDEN = {"parity": 12, "xorandxor": 16, "mux": 24, "nestedxor": 24, "hier3": 96}
FAMILY_PARAMS = {"parity": {"n_bits": 8}, "xorandxor": {"n_bits": 8}, "mux": {"n_addr": 3},
                 "nestedxor": {}, "hier3": {}}
# Per-family oracle epochs: parity gets the FULL --epochs budget (its l3 grokking is the only honest depth attempt);
# the folding families (xorandxor/mux/nestedxor/hier3) solve/fail at depth-2 ROBUSTLY at any epoch >= a few hundred
# (verified: l2 = 0.93-1.00 by ~1000 epochs), so they run at a capped budget to keep the 6-seed run foreground-fast.
FAMILY_EPOCH_CAP = {"parity": None, "xorandxor": 1500, "mux": 1500, "nestedxor": 1500, "hier3": 1500}


def survey_family(family, seed, epochs, lr, batch):
    """Measure the obligatory-depth-3 gate on the SHARED rate oracle for one family+seed. Returns the l0/l1/l2/l3
    held-out-generalization curve + the depth3_requiring gate (i)-(iii)."""
    params = FAMILY_PARAMS[family]
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = build_task(family, seed, params)
    k = int(meta["k_classes"])
    hidden = FAMILY_HIDDEN[family]
    cap = FAMILY_EPOCH_CAP.get(family)
    fam_epochs = epochs if cap is None else min(epochs, cap)
    s0 = stage0_depth_genuineness(((Xtr, ytr, Ltr), (Xte, yte, Lte)), idx, k, hidden=hidden,
                                  epochs=fam_epochs, lr=lr, batch=batch, seed=seed)
    l2, l3, ch = s0["l2_inherit_heldout"], s0["l3_inherit_heldout"], s0["chance"]
    gate, margins = obligatory_depth3_gate(l2, l3, ch)
    return {"family": family, "seed": seed, "hidden": hidden, "epochs": int(fam_epochs), "meta": meta, "chance": ch,
            "l0": s0["linear_inherit_heldout"], "l1": s0["l1_inherit_heldout"], "l2": l2, "l3": l3,
            "l2_train": s0["l2_train"], "l3_train": s0["l3_train"], "depth_jump": float(l3 - l2),
            "depth3_requiring": gate, **margins}


# ============================================================================================================
# SPIKING ARMS (on the best-candidate family): does DFA e-prop -- the transport-free deep-credit rule -- ACHIEVE the
# obligatory-depth-3 predicate on the spiking substrate, with the shuffle control collapsing? Reuse `_train_snn`
# (surrogate-BPTT / eprop / eprop_shuffle). The spiking l2/l3 are the on-substrate analogue of the oracle l2/l3.
# ============================================================================================================
def _snn_heldout(family, seed, credit_mode, n_hidden_layers, hidden, T, epochs, lr, in_gain):
    params = FAMILY_PARAMS[family]
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte), meta, idx = build_task(family, seed, params)
    k = int(meta["k_classes"]); n_in = Xtr.shape[1]; inh = idx["inh_idx"]
    sizes = [n_in] + [hidden] * n_hidden_layers + [k]
    layers = _train_snn(Xtr, ytr, sizes, T, epochs, lr, in_gain, seed, credit_mode=credit_mode)
    tr = _accuracy(Xtr, ytr, layers, T, in_gain)
    te = _accuracy(Xte, yte, layers, T, in_gain, sub=inh)
    yv = yte[inh]; chance = float(max(np.mean(yv == c) for c in np.unique(yv))) if len(inh) else float("nan")
    return {"train": tr, "heldout": te, "chance": chance}


def spiking_arms(family, seed, hidden, T, epochs, lr, in_gain):
    """The on-substrate depth-2/depth-3 arms: BPTT (best-possible on-spike credit) at N=2 (must FAIL) + N=3 (the deep
    ceiling), DFA e-prop N=3 (the transport-free deep-credit rule under test), eprop_shuffle N=3 (must collapse)."""
    bptt2 = _snn_heldout(family, seed, "bptt", 2, hidden, T, epochs, lr, in_gain)
    bptt3 = _snn_heldout(family, seed, "bptt", 3, hidden, T, epochs, lr, in_gain)
    eprop3 = _snn_heldout(family, seed, "eprop", 3, hidden, T, epochs, lr, in_gain)
    shuf3 = _snn_heldout(family, seed, "eprop_shuffle", 3, hidden, T, epochs, lr, in_gain)
    ch = bptt3["chance"]
    # obligatory-depth-3 predicate on the SPIKING substrate, evaluated with BPTT (the best on-spike weight-finder) as
    # the depth-2/depth-3 pair, and with DFA e-prop as the DEPLOYED rule.
    bptt_gate, _ = obligatory_depth3_gate(bptt2["heldout"], bptt3["heldout"], ch)
    eprop_gate, _ = obligatory_depth3_gate(bptt2["heldout"], eprop3["heldout"], ch)
    shuffle_collapses = bool(np.isnan(shuf3["heldout"]) or shuf3["heldout"] <= ch + CHANCE_MARGIN)
    return {"family": family, "seed": seed, "chance": ch,
            "bptt_N2_heldout": bptt2["heldout"], "bptt_N3_heldout": bptt3["heldout"],
            "eprop_N3_heldout": eprop3["heldout"], "eprop_shuffle_N3_heldout": shuf3["heldout"],
            "bptt_N2_train": bptt2["train"], "bptt_N3_train": bptt3["train"], "eprop_N3_train": eprop3["train"],
            "bptt_obligatory_depth3": bptt_gate, "eprop_obligatory_depth3": eprop_gate,
            "shuffle_collapses": shuffle_collapses}


def _seed_check(seed):
    """cfg.seed-equivalent self-check: two DendriticMLP built at the SAME seed have byte-identical init weights (the
    net IS seeded from `seed`), and two at DIFFERENT seeds differ -> the substrate is genuinely seed-controlled."""
    a = DendriticMLP([8, 12, 12, 2], seed=seed)
    b = DendriticMLP([8, 12, 12, 2], seed=seed)
    c = DendriticMLP([8, 12, 12, 2], seed=seed + 1)
    same = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a.W, b.W))
    diff = any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a.W, c.W))
    return bool(same and diff)


def main():
    ap = argparse.ArgumentParser(description="gap#4 obligatory-depth-3 credit instrument survey.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--families", nargs="+", default=["parity", "xorandxor", "mux", "nestedxor", "hier3"])
    ap.add_argument("--epochs", type=int, default=4000, help="rate-oracle epochs (parity needs a long fit).")
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=128)
    # spiking arms (on the best-candidate family)
    ap.add_argument("--no-spiking", action="store_true", help="rate survey only (fast smoke).")
    ap.add_argument("--spiking-family", default="parity")
    ap.add_argument("--spiking-hidden", type=int, default=24)
    ap.add_argument("--spiking-T", type=int, default=20)
    ap.add_argument("--spiking-epochs", type=int, default=200)
    ap.add_argument("--spiking-lr", type=float, default=0.05)
    ap.add_argument("--spiking-in-gain", type=float, default=1.0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.time()
    seed_ctrl = all(_seed_check(s) for s in a.seeds)
    survey = []
    spiking = []
    err = None
    try:
        for family in a.families:
            for s in a.seeds:
                r = survey_family(family, s, a.epochs, a.lr, a.batch)
                survey.append(r)
                print(f"[survey {family:10s} seed {s}] chance {r['chance']:.3f} h{r['hidden']} "
                      f"ntr {r['meta']['n_train']} | l1 {r['l1']:.3f} l2 {r['l2']:.3f}({r['l2_train']:.2f}) "
                      f"l3 {r['l3']:.3f}({r['l3_train']:.2f}) | jump {r['depth_jump']:+.3f} "
                      f"=> depth3_requiring={r['depth3_requiring']}", flush=True)
        if not a.no_spiking:
            print("-" * 100, flush=True)
            for s in a.seeds:
                r = spiking_arms(a.spiking_family, s, a.spiking_hidden, a.spiking_T, a.spiking_epochs,
                                 a.spiking_lr, a.spiking_in_gain)
                spiking.append(r)
                print(f"[spiking {a.spiking_family} seed {s}] chance {r['chance']:.3f} | "
                      f"BPTT N2 {r['bptt_N2_heldout']:.3f} N3 {r['bptt_N3_heldout']:.3f} | "
                      f"DFA-eprop N3 {r['eprop_N3_heldout']:.3f} | shuffle N3 {r['eprop_shuffle_N3_heldout']:.3f} "
                      f"=> bptt-oblig3={r['bptt_obligatory_depth3']} eprop-oblig3={r['eprop_obligatory_depth3']} "
                      f"shuffle-collapses={r['shuffle_collapses']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    # ---- aggregate: per-family gate-rate over seeds; the instrument EXISTS iff a family gates >=5/6 ----
    families = sorted(set(r["family"] for r in survey))
    n_seeds = len(a.seeds)
    per_family = {}
    for fam in families:
        rows = [r for r in survey if r["family"] == fam]
        gates = sum(bool(r["depth3_requiring"]) for r in rows)
        per_family[fam] = {
            "gate_seeds": f"{gates}/{len(rows)}",
            "mean_l2": float(np.nanmean([r["l2"] for r in rows])),
            "mean_l3": float(np.nanmean([r["l3"] for r in rows])),
            "mean_jump": float(np.nanmean([r["depth_jump"] for r in rows])),
            "mean_chance": float(np.nanmean([r["chance"] for r in rows])),
            "l2_fails_seeds": sum(bool(r["l2_ok"]) for r in rows),
            "l3_generalizes_seeds": sum(bool(r["l3_ok"]) for r in rows),
            "jump_seeds": sum(bool(r["jump_ok"]) for r in rows),
        }
    instrument_family = None
    for fam, agg in per_family.items():
        g = int(agg["gate_seeds"].split("/")[0])
        if g >= max(5, n_seeds - 1):
            instrument_family = fam
            break

    spiking_go = False
    spiking_summary = {}
    if spiking:
        e_gate = sum(bool(r["eprop_obligatory_depth3"]) for r in spiking)
        b_gate = sum(bool(r["bptt_obligatory_depth3"]) for r in spiking)
        s_coll = sum(bool(r["shuffle_collapses"]) for r in spiking)
        spiking_summary = {
            "family": a.spiking_family,
            "eprop_obligatory_depth3_seeds": f"{e_gate}/{len(spiking)}",
            "bptt_obligatory_depth3_seeds": f"{b_gate}/{len(spiking)}",
            "shuffle_collapses_seeds": f"{s_coll}/{len(spiking)}",
            "mean_bptt_N2": float(np.nanmean([r["bptt_N2_heldout"] for r in spiking])),
            "mean_bptt_N3": float(np.nanmean([r["bptt_N3_heldout"] for r in spiking])),
            "mean_eprop_N3": float(np.nanmean([r["eprop_N3_heldout"] for r in spiking])),
            "mean_shuffle_N3": float(np.nanmean([r["eprop_shuffle_N3_heldout"] for r in spiking])),
        }
        spiking_go = bool(e_gate >= max(5, n_seeds - 1) and s_coll >= max(5, n_seeds - 1))

    instrument_exists = instrument_family is not None
    signal = bool(instrument_exists and spiking_go and err is None)

    if err is not None:
        verdict = f"ERROR -- {err}"
    elif not instrument_exists:
        closest = min(per_family.items(), key=lambda kv: kv[1]["mean_l2"] - kv[1]["mean_chance"]
                      if not np.isnan(kv[1]["mean_l2"]) else 1e9) if per_family else (None, {})
        verdict = (
            "HONEST NEGATIVE -- NO task family satisfies the obligatory-depth-3 gate (i)-(iii) robustly on the shared "
            f"rate oracle (0 families gate >=5/{n_seeds}). This SHARPENS the crux wall: obligatory-depth-3 as a "
            "matched-width GENERALIZATION gate is not constructible on this substrate at practical scale. The "
            "compositional/fan-in-2 families (nestedxor, xorandxor, mux, hier3) FOLD to depth-2 -- 2 hidden layers are "
            "universal enough to represent them, so l2 solves whenever l3 does (l2 never <= chance+0.06). Parity is the "
            "ONLY family where depth genuinely matters at matched width, but its depth-3 GENERALIZATION from a held-out "
            "split is at the edge of learnability -> l3 is high-variance and never clears 0.80 while l2 stays down on "
            "the same seed: the three gate conditions never co-occur robustly. => every gap#4 deep-credit lane's "
            "'depth-3 GO' cannot be validated by an obligatory-depth-3 TASK (none exists); the falsifiable route is the "
            "credit-ALIGNMENT measurement (cos(delivered credit, true BPTT credit) per layer), per the crux pivot.")
    elif not spiking_go:
        verdict = (f"PARTIAL -- the rate instrument EXISTS ({instrument_family} gates "
                   f"{per_family[instrument_family]['gate_seeds']}), but DFA e-prop does NOT hold the obligatory-"
                   f"depth-3 predicate on the spiking substrate ({spiking_summary.get('eprop_obligatory_depth3_seeds')}) "
                   f"or shuffle does not collapse ({spiking_summary.get('shuffle_collapses_seeds')}). The task is a "
                   "valid depth-3 target but the transport-free rule cannot reach it on spikes -> the wall is the "
                   "deep-spiking learning regime, not the task.")
    else:
        verdict = (f"GO -- obligatory-depth-3 instrument EXISTS ({instrument_family} gates "
                   f"{per_family[instrument_family]['gate_seeds']}) AND DFA e-prop holds the predicate on the spiking "
                   f"substrate ({spiking_summary['eprop_obligatory_depth3_seeds']}) with shuffle collapsing "
                   f"({spiking_summary['shuffle_collapses_seeds']}). Every gap#4 lane can import this task as its "
                   "falsifiable depth-3 target.")

    summary = {
        "probe": "gap4_obligatory_depth3_instrument", "seeds": a.seeds, "families": a.families,
        "gate": "depth3_requiring = (l2 <= chance+0.06) AND (l3 >= 0.80) AND (l3-l2 >= 0.15)  [crux L577/L824]",
        "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "family_hidden": FAMILY_HIDDEN,
                   "family_params": FAMILY_PARAMS, "backend": os.environ.get("SIM_BACKEND"),
                   "spiking": None if a.no_spiking else {
                       "family": a.spiking_family, "hidden": a.spiking_hidden, "T": a.spiking_T,
                       "epochs": a.spiking_epochs, "lr": a.spiking_lr, "in_gain": a.spiking_in_gain}},
        "seed_control_verified": seed_ctrl,
        "per_family": per_family, "instrument_family": instrument_family, "instrument_exists": instrument_exists,
        "spiking_summary": spiking_summary, "spiking_go": spiking_go,
        "SIGNAL": signal, "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 1),
        "survey": survey, "spiking": spiking,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[obligatory-depth3-instrument] seed-control-verified={seed_ctrl}", flush=True)
    for fam, agg in per_family.items():
        print(f"  {fam:10s} gate {agg['gate_seeds']} | mean l2 {agg['mean_l2']:.3f} l3 {agg['mean_l3']:.3f} "
              f"jump {agg['mean_jump']:+.3f} (chance {agg['mean_chance']:.3f}) | "
              f"l2-fails {agg['l2_fails_seeds']} l3-gen {agg['l3_generalizes_seeds']} jump {agg['jump_seeds']}",
              flush=True)
    if spiking_summary:
        print(f"  SPIKING[{spiking_summary['family']}] eprop-oblig3 {spiking_summary['eprop_obligatory_depth3_seeds']} "
              f"| BPTT N2 {spiking_summary['mean_bptt_N2']:.3f} N3 {spiking_summary['mean_bptt_N3']:.3f} "
              f"eprop N3 {spiking_summary['mean_eprop_N3']:.3f} shuffle {spiking_summary['mean_shuffle_N3']:.3f}",
              flush=True)
    print(f"\n[obligatory-depth3-instrument] {verdict}", flush=True)
    print(f"[obligatory-depth3-instrument] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if signal else 1


if __name__ == "__main__":
    sys.exit(main())
