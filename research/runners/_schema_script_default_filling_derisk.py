"""SCHEMA / SCRIPT DEFAULT-FILLING on the emergent spiking HTM Temporal-Memory cortex (the 2026-07-08 open-domain
frontier gate's item (b): §3(b) + §4 verdict row (b)). Toward OPEN-WORLD inference: run the EMERGE-14 on-bridge HTM
next-state predictor over a LEARNED event-SCRIPT corpus to infer the UNSTATED-BUT-TYPICAL continuation of a partially
described situation ("at a restaurant -> [fill the typical] enter -> wait -> ... -> eat"). Successor-representation-style
multi-hop reachability (Stachenfeld-Botvinick-Gershman 2017) is LATENT in the substrate's one-step next-state predictor
(Schank-Abelson 1977 scripts; the SR reframe = a next-state predictor over a relational graph IS an implicit schema
engine). Reuse-by-import of the rung-4 on-bridge learner (`_emerge14`); NO `sim/` edit; CPU numpy-backend.

THE CORPUS (high-order, situation-cue-dependent). Each SCRIPT = a MARKER (the situation cue: @restaurant / @shop / ...)
+ a SHARED generic middle (enter, wait, stand -- the SAME event tokens across ALL scripts) + a script-TYPICAL
continuation (eat / buy / consult / ...) + a shared ending (pay, leave|exit). The typical continuation depends on the
MARKER *several events back*, through the shared middle -> any FIXED-ORDER n-gram up to len(middle) sees an IDENTICAL
context before the continuation and is at chance (1/n_scripts) there (the marker is out of its window). Only a mechanism
that carries the situation cue through the shared middle can fill the script-typical event. Two ending variants per
script = "several instances with minor variation" so the branch is the STABLE typical across instances.

THE SCHEMA-FILLING TASK (not memorization). Present a HELD-OUT PARTIAL: just the MARKER (the situation cue alone, a
1-token prefix that is NOT a trained sequence) and ROLL OUT the next-state predictor autoregressively (SR-style
multi-hop) to fill the whole typical continuation. The gated metric = the rollout reaches the script-TYPICAL branch
event at its script-position (>=len(middle) steps ahead of the cue) -- an event a 1-step / fixed-order memorizer of the
short prefix CANNOT reach. The continuation is UNSTATED in the partial (the schema fills it), the typical event is
INFERRED from the learned script structure (the marker carried through the shared middle), and the tested partial is
never a verbatim training sequence.

ANTI-CHEATS (6-seed 42/43/44/100/101/102):
  (1) MARKOV FLOOR: bigram/trigram/4-gram next-event at the branch position (order-blind through the shared middle) --
      the HTM must BEAT it (mirrors EMERGE-15's n-gram floor).
  (2) PERMUTED-SCRIPT: shuffle each training script's post-marker event ORDER -> the typical-ordering structure is
      destroyed -> the rollout can't reach the script-typical branch -> collapses to chance/floor (the learned ORDER is
      load-bearing).
  (3) dAP/COINCIDENCE LESION: coincidence off / no dendritic prediction -> the rollout abstains -> collapse.
  (4) HELD-OUT DISJOINT: the tested partial (the bare marker) is asserted to be NOT any verbatim training sequence
      (0 exact-sequence overlap); the continuation is GENERATED, not looked up.
  (5) MOAT: a partial from a NEVER-SEEN script (disjoint marker + disjoint events) -> the predictor must NOT confabulate
      a typical continuation -> it ABSTAINS (no known event driven above the apical floor), preserving the no-confab moat.
  Reported boundary probe (NOT gated): OMITTED-MIDDLE -- a partial that skips a middle event tests whether the schema is
  robust to an omission (a next-state chain vs a true omission-robust SR). Honest map of what the HTM predictor can/can't.

GO = rollout-from-cue schema-fill accuracy >= 0.90 AND >= Markov floor + 0.30 AND >= dAP-lesion + 0.30 AND permuted
collapses (<= chance + 0.15) AND moat-abstain >= 0.90, 6-seed. HONEST NEGATIVE (a first-class deliverable) if the HTM
predictor does NOT reach the unstated-typical continuation above the Markov floor -- it maps that schema-filling needs
more than a next-state predictor (a true SR / multi-hop reachability mechanism).

Run: SIM_BACKEND=numpy python -m research.runners._schema_script_default_filling_derisk \
        --seeds 42 43 44 100 101 102 --out research/findings/raw/_schema_script_default_filling.json
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._emerge14_stageC_onbridge_learning_derisk import (
    build_pool_bridge, OnBridgeLearner, coincidence_predict, _host)
from research.runners._emerge15_word_sequence_lm_derisk import ngram_nextword_acc

OUT = Path("research/findings/raw/_schema_script_default_filling.json")

# ---- readable script vocabulary (extends synthetically past 8 scripts; the tokens are just distinct columns) ----
_SCRIPT = ["restaurant", "shop", "clinic", "bank", "library", "gym", "cafe", "salon"]
_TYP = ["eat", "buy", "consult", "deposit", "read", "workout", "sip", "trim"]
SHARED_MID = ["enter", "wait", "stand"]          # generic events shared across ALL scripts (defeats the n-gram)
END_A = ["pay", "leave"]
END_B = ["pay", "exit"]                            # a second instance-variant ending ("minor variation")


def build_corpus(n_scripts=4, ending_free=False):
    """Ordered event-script corpus. Returns (train_seqs[col lists], vocab, word2col, meta).
    meta carries markers/typs (as cols), branch position, and the never-seen NOVEL script (moat)."""
    scr = (_SCRIPT + [f"scr{i}" for i in range(max(0, n_scripts - len(_SCRIPT)))])[:n_scripts]
    typ = (_TYP + [f"typ{i}" for i in range(max(0, n_scripts - len(_TYP)))])[:n_scripts]
    markers = [f"@{s}" for s in scr]
    typs = list(typ)
    novel = {"marker": "@NOVELSCR", "mid": ["nov_a", "nov_b", "nov_c"], "typ": "nov_typ"}
    # deterministic vocab ordering: markers, shared middle, typicals, endings, novel-script tokens
    ordered = markers + SHARED_MID + typs + ["pay", "leave", "exit"] \
        + [novel["marker"]] + novel["mid"] + [novel["typ"]]
    seen = set()
    vocab = [w for w in ordered if not (w in seen or seen.add(w))]
    w2c = {w: i for i, w in enumerate(vocab)}
    # full script (two ending variants) = [marker] + shared middle + [typical] + ending
    train_words = []
    for i in range(n_scripts):
        base = [markers[i]] + SHARED_MID + [typs[i]]
        if ending_free:
            # EMERGE-15-exact: the shared ENDING (pay/leave/exit) after the typical creates SHARED post-typical
            # high-order contexts on the typical columns that compete with the marker->typical prediction and starve
            # the shared-mid segment budget -> pos-ctrl collapses to chance. Training the schema WITHOUT the ending
            # (the typical is the sequence terminus) isolates the marker->typical high-order chain (EMERGE-15-exact).
            train_words.append(base)
            train_words.append(base)               # 2 identical instances (minor-variation slot, ending removed)
        else:
            train_words.append(base + END_A)
            train_words.append(base + END_B)
    train_seqs = [[w2c[w] for w in s] for s in train_words]
    branch_pos = 1 + len(SHARED_MID)                  # index of the typical continuation in the full script
    meta = {"n_scripts": n_scripts, "markers": [w2c[m] for m in markers], "typs": [w2c[t] for t in typs],
            "branch_pos": branch_pos, "shared_mid": [w2c[m] for m in SHARED_MID],
            "novel_marker": w2c[novel["marker"]], "novel_mid": [w2c[m] for m in novel["mid"]],
            "typ_words": typs, "marker_words": markers, "train_words": train_words}
    return train_seqs, vocab, w2c, meta


def _permute_postmarker(seqs, seed):
    """PERMUTED-SCRIPT anti-cheat: shuffle the ORDER of every training instance's post-marker events (keep the marker at
    the front). Destroys the typical-ordering structure -> the cue-driven rollout can no longer reach the script-typical
    branch. A per-instance random permutation (seed-derived)."""
    rng = np.random.RandomState(seed * 7 + 3)
    out = []
    for s in seqs:
        tail = list(s[1:])
        rng.shuffle(tail)
        out.append([s[0]] + tail)
    return out


# ---------------- PROVEN EMERGE-15 next-state readout (dAP-predicted column vote) + autoregressive rollout ----------

def _step(lr, c, predictive):
    """One context step: present column c (primed cells if predicted else burst), prime the coincidence pathway,
    return the new dAP-PREDICTED cell set. Mirrors OnBridgeLearner.predict_branch's per-symbol logic exactly."""
    col = lr._col(c)
    primed = [i for i in col if i in predictive] if not lr.lesion else []
    active = set(primed[:lr.k_win]) if primed else set(col)
    return coincidence_predict(lr.b, lr.cells_idx, active, lr.N, lr.nE)


def _vote_column(lr, predicted_set):
    """EMERGE-15's PROVEN readout: the winning next-event COLUMN is the one with the MOST dAP-predicted cells
    (column vote over coincidence_predict's thresholded set = the high-order dAP prediction, NOT raw graded drive).
    None (ABSTAIN) when nothing is predicted -> the no-confab moat."""
    if not predicted_set:
        return None
    cnt = Counter(int(i) // lr.nE for i in predicted_set)
    return cnt.most_common(1)[0][0]


def rollout(lr, start_cols, n_steps):
    """Autoregressive SR-style rollout using the proven column-vote readout: seed with start_cols, then repeatedly
    take the dAP-predicted winning column, feed it back, continue. ABSTAIN (stop) when nothing is predicted (moat)."""
    predictive = set()
    for c in start_cols:
        predictive = _step(lr, c, predictive)
    out = []
    for _ in range(n_steps):
        c_next = _vote_column(lr, predictive)
        if c_next is None:                           # nothing dAP-predicted -> abstain
            break
        out.append(c_next)
        predictive = _step(lr, c_next, predictive)
    return out


def predict_next(lr, prefix):
    """Single next-event prediction after a full prefix (positive control): the dAP-predicted column vote (EMERGE-15
    branch readout), or None (ABSTAIN)."""
    predictive = set()
    for c in prefix:
        predictive = _step(lr, c, predictive)
    return _vote_column(lr, predictive)


# ------------------------------------------- arms -------------------------------------------

def _build_learner(seqs, vocab_n, nE, seed, epochs, lesion=False):
    b, cells_idx, row, col = build_pool_bridge(vocab_n, nE, seed, act_th=3, coincidence=(not lesion))
    lr = OnBridgeLearner(b, row, col, cells_idx, vocab_n, nE, k_win=4, act_th=3, lesion=lesion)
    for _ in range(epochs):
        for s in seqs:
            lr.train_sequence(s)
    return lr


def _schema_fill_acc(lr, meta):
    """Rollout-from-cue schema fill: from the bare MARKER, roll out len(shared_mid)+1 events; PASS if the branch event
    (position len(shared_mid) in the rollout) == this script's typical continuation. Returns (acc, per-script rollouts)."""
    nmid = len(meta["shared_mid"])
    ok = 0
    rolls = []
    for i in range(meta["n_scripts"]):
        gen = rollout(lr, [meta["markers"][i]], nmid + 1)
        hit = len(gen) > nmid and gen[nmid] == meta["typs"][i]
        ok += int(hit)
        rolls.append({"script": meta["marker_words"][i], "rollout": gen, "want_typ": meta["typs"][i], "hit": bool(hit)})
    return ok / meta["n_scripts"], rolls


def _positive_control_acc(lr, meta):
    """Full-partial next-event (chain intact): [marker]+shared_mid -> predict the typical. (Not gated; a sanity ceiling.)"""
    ok = 0
    for i in range(meta["n_scripts"]):
        prefix = [meta["markers"][i]] + meta["shared_mid"]
        ok += int(predict_next(lr, prefix) == meta["typs"][i])
    return ok / meta["n_scripts"]


def _omitted_middle_acc(lr, meta):
    """BOUNDARY probe (reported, NOT gated): a partial that OMITS the middle event -> can the schema still fill the
    typical? Tests next-state-chain vs omission-robust SR. Rollout from [marker]+shared_mid[without the last mid]."""
    nmid = len(meta["shared_mid"])
    if nmid < 2:
        return None
    ok = 0
    for i in range(meta["n_scripts"]):
        start = [meta["markers"][i]] + meta["shared_mid"][:-1]   # drop the last shared middle event ("stand")
        gen = rollout(lr, start, 2)                              # expect the typical as the next confident event
        ok += int(len(gen) > 0 and meta["typs"][i] in gen[:2])
    return ok / meta["n_scripts"]


def _moat_abstain(lr, meta):
    """MOAT: rollout from a NEVER-SEEN script marker (disjoint columns) -> must ABSTAIN (empty rollout / no typ)."""
    gen = rollout(lr, [meta["novel_marker"]], len(meta["shared_mid"]) + 1)
    known_typ = set(meta["typs"])
    return float(len(gen) == 0 or not (known_typ & set(gen)))


def run_seed(seed, n_scripts, epochs, ending_free=False):
    seqs, vocab, w2c, meta = build_corpus(n_scripts, ending_free=ending_free)
    nE = 4 * n_scripts + 8                             # each shared column holds n_scripts disjoint high-order SDRs + slack
    vocab_n = len(vocab)
    # held-out-disjoint assertion: the tested partial (bare marker) is NOT any verbatim training sequence
    train_set = set(tuple(s) for s in seqs)
    for m in meta["markers"]:
        assert (m,) not in train_set, "held-out partial must not be a verbatim training sequence"
    assert (meta["novel_marker"],) not in train_set

    # n-gram floors at the branch position (structure is seed-independent; identical across seeds)
    bp = meta["branch_pos"]
    bigram = ngram_nextword_acc(seqs, 1, bp)
    trigram = ngram_nextword_acc(seqs, 2, bp)
    fourgram = ngram_nextword_acc(seqs, 3, bp)
    ngram_floor = max(bigram, trigram, fourgram)
    chance = 1.0 / n_scripts

    lr = _build_learner(seqs, vocab_n, nE, seed, epochs, lesion=False)
    schema, rolls = _schema_fill_acc(lr, meta)
    pos_ctrl = _positive_control_acc(lr, meta)
    omitted = _omitted_middle_acc(lr, meta)
    moat = _moat_abstain(lr, meta)

    lr_perm = _build_learner(_permute_postmarker(seqs, seed), vocab_n, nE, seed, epochs, lesion=False)
    permuted, _ = _schema_fill_acc(lr_perm, meta)

    lr_les = _build_learner(seqs, vocab_n, nE, seed, epochs, lesion=True)
    lesion, _ = _schema_fill_acc(lr_les, meta)

    out = {"seed": seed, "schema_fill": schema, "positive_control": pos_ctrl, "omitted_middle": omitted,
           "moat_abstain": moat, "permuted": permuted, "lesion": lesion,
           "bigram": bigram, "trigram": trigram, "fourgram": fourgram, "ngram_floor": ngram_floor, "chance": chance,
           "example_rollouts": rolls}
    print(f"[schema seed={seed}] schema-fill {schema:.3f} (pos-ctrl {pos_ctrl:.3f}) | moat-abstain {moat:.3f} | "
          f"permuted {permuted:.3f} | lesion {lesion:.3f} | omitted-mid {omitted} || "
          f"n-gram floor {ngram_floor:.3f} (bi {bigram:.2f}/tri {trigram:.2f}/4 {fourgram:.2f}) chance {chance:.3f}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-scripts", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--ending-free", action="store_true", help="EMERGE-15-exact: train scripts without the shared ending (isolates the marker->typical high-order chain; fixes the pos-ctrl)")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds")
        return 2
    t0 = time.time()
    err = None
    res = []
    seed_errors = {}
    for s in a.seeds:
        try:
            res.append(run_seed(s, a.n_scripts, a.epochs, ending_free=a.ending_free))
        except Exception as e:
            seed_errors[s] = repr(e)
            traceback.print_exc()
            print(f"[schema] seed {s} CRASHED: {e!r}", flush=True)
    if seed_errors and not res:
        err = f"all seeds crashed: {seed_errors}"

    if res:
        def mean(k):
            return float(np.mean([r[k] for r in res]))
        schema = mean("schema_fill")
        moat = mean("moat_abstain")
        permuted = mean("permuted")
        lesion = mean("lesion")
        pos_ctrl = mean("positive_control")
        ngram_floor = mean("ngram_floor")
        chance = mean("chance")
        omitted_vals = [r["omitted_middle"] for r in res if r["omitted_middle"] is not None]
        omitted = float(np.mean(omitted_vals)) if omitted_vals else None
        go = bool(schema >= 0.90 and schema >= ngram_floor + 0.30 and schema >= lesion + 0.30
                  and permuted <= chance + 0.15 and moat >= 0.90
                  and not seed_errors and len(res) == len(a.seeds))
        if go:
            verdict = (
                f"GO -- SCHEMA/SCRIPT DEFAULT-FILLING runs on the emergent spiking HTM cortex: from the SITUATION CUE "
                f"alone (the bare marker, a non-trained 1-token partial) the on-bridge next-state predictor ROLLS OUT the "
                f"unstated-but-typical script continuation and reaches the script-typical branch event {schema:.3f} "
                f">> the best fixed-order n-gram Markov floor {ngram_floor:.3f} (bi {mean('bigram'):.2f}/tri "
                f"{mean('trigram'):.2f}/4 {mean('fourgram'):.2f}) -- an event a 1-step/fixed-order memorizer of the short "
                f"prefix cannot reach (SR-style multi-hop reachability latent in the one-step predictor). dAP-LESION "
                f"collapses it to {lesion:.3f} (the learned dendritic prediction is load-bearing); PERMUTED-SCRIPT "
                f"collapses to {permuted:.3f} (the learned event ORDER carries it); a NEVER-SEEN script ABSTAINS "
                f"{moat:.3f} (the no-confab moat holds -- no fabricated typical continuation). Full-partial positive "
                f"control {pos_ctrl:.3f}. Multi-seed; NO sim/ edit. => open-world schema inference (Schank-Abelson scripts "
                f"/ Stachenfeld successor-representation) on the substrate: the brain fills in what a script IMPLIES but "
                f"was never told, moat-preserving.")
        else:
            miss = []
            if schema < 0.90:
                miss.append(f"schema-fill {schema:.3f} < 0.90")
            if schema < ngram_floor + 0.30:
                miss.append(f"didn't beat the n-gram floor ({schema:.3f} vs {ngram_floor:.3f})")
            if schema < lesion + 0.30:
                miss.append(f"dAP-lesion didn't collapse ({schema:.3f} vs {lesion:.3f})")
            if permuted > chance + 0.15:
                miss.append(f"permuted didn't collapse ({permuted:.3f} vs chance {chance:.3f})")
            if moat < 0.90:
                miss.append(f"moat didn't abstain ({moat:.3f})")
            verdict = (
                "NEGATIVE / BOUNDARY (build-informative) -- " + "; ".join(miss) + f". positive-control {pos_ctrl:.3f}, "
                f"omitted-middle {omitted}. If the full-partial positive control is high but the rollout-from-cue schema-"
                f"fill is not, the HTM next-state predictor MEMORIZES exact continuations but does not perform the SR-"
                f"style multi-hop cue->typical fill unaided -> schema default-filling needs more than the one-step "
                f"predictor (a true successor-representation / multi-hop reachability read). An honest map of the residual.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {
        "probe": "schema_script_default_filling",
        "verdict": verdict,
        "mechanism": "the emergent on-bridge HTM Temporal-Memory next-state predictor (EMERGE-14, self-organizing high-"
                     "order context + on-substrate three-term learning) rolled out autoregressively over a learned event-"
                     "script corpus to fill the unstated-but-typical continuation from the situation cue; graded apical "
                     "read gives an abstain (no-confab moat) on never-seen scripts",
        "task": "event-script corpus (marker + shared middle + script-typical continuation); rollout-from-cue schema fill "
                "vs bigram/trigram/4gram Markov floor + permuted-script + dAP-lesion + held-out-disjoint + moat abstain, 6-seed",
        "citations": "Schank & Abelson 1977 (scripts); Stachenfeld-Botvinick-Gershman 2017 (successor representation -> "
                     "inference); gate 2026-07-08 §3(b)/§4 row (b)",
        "seeds": a.seeds, "config": {"n_scripts": a.n_scripts, "epochs": a.epochs},
        "seed_errors": seed_errors,
        "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if not res else {
            "schema_fill": mean("schema_fill"), "positive_control": mean("positive_control"),
            "omitted_middle": omitted, "moat_abstain": mean("moat_abstain"), "permuted": mean("permuted"),
            "lesion": mean("lesion"), "ngram_floor": mean("ngram_floor"), "chance": mean("chance")},
        "per_seed": res,
        "HONEST_NOTE": "reuse-by-import of the rung-4 EMERGE-14 on-bridge learner; NO sim/ edit. The corpus is a small "
                       "high-order script structure isolating the cue->typical dependency the n-gram cannot capture. The "
                       "OMITTED-MIDDLE probe is reported (not gated): it maps whether the next-state predictor is robust "
                       "to a skipped event (a true SR) or only carries the cue through an unbroken chain.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    ng = sum(1 for r in res if r["schema_fill"] >= 0.90 and r["moat_abstain"] >= 0.90)
    print("\n" + "=" * 108, flush=True)
    if seed_errors:
        print(f"[schema] SEED ERRORS: {seed_errors}", flush=True)
    print(f"[schema] per-seed schema&moat GO: {ng}/{len(a.seeds)}", flush=True)
    print(f"[schema] VERDICT: {verdict}", flush=True)
    print(f"[schema] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
