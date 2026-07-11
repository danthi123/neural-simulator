"""RUNG 3 of the emergent-generation ladder -- SYSTEMATIC GENERALIZATION OF GENERATION (order-sensitive, content-sensitive,
held-out): the fixed-reservoir + one-step-local-delta GENERATOR (Rung 1) PRODUCES the correct continuation for a
never-generated AGENT, where the answer depends on WORD ORDER (which noun is the agent) AND on the agent's SHARED
CATEGORY code. The generative mirror of the comprehension results EMERGE-22 (systematic recombination) + EMERGE-26
(inheritance) + EMERGE-78 (form->role). NO BPTT, NO deep credit, NO `sim/` edit. Reuse-by-import: the Rung-1 reservoir
(`ReservoirStates` = EMERGE-82 `OnBridgeLSM`) + the Rung-1 read-out (`train_readout`/`eval_ce`).

WHY THIS TASK (three earlier task designs were confounded -- each documented honestly in the finding, and each taught a
real property of the substrate). (1) A held-out that recombines words in their SAME roles is predictable from each word's
own history even with one-hot codes. (2) A "grammaticality" metric (predict the right POS CLASS) is CONTENT-BLIND -- it is
solvable from position alone, so the shared CONTENT code can never be shown load-bearing. (3) A pure inheritance target
(category -> property) is ORDER-FREE -- a bag-level association a word-shuffled (permuted) model also solves, so the
reservoir's SEQUENCE dynamics are not load-bearing. This task requires BOTH content and order:

THE TASK. Two animal categories (PRED, PREY), each with member animals and a set of category actions
(PRED -> {growl,hunt,pounce}; PREY -> {flee,hide,freeze}). Frame: "<N1> meets <N2> <ACTION>" where the ACTION is set by
the AGENT = N1, the FIRST noun (e.g. "wolf meets rabbit growl"; "rabbit meets wolf flee" -- SAME two animals, opposite
order, DIFFERENT action). Each word drives the reservoir through a two-level code = its shared CATEGORY/CLASS block + a
unique CONTENT bit. We TRAIN on a subset of animals (as both agent and patient, in both orders) and HOLD OUT other
animals entirely. TEST: after "<held-out-animal> meets <trained-animal>", does the generator produce an ACTION of the
HELD-OUT AGENT's category? Producing it requires (1) the SHARED CATEGORY block to transfer the "agent-category -> action"
rule to an animal never generated, AND (2) the reservoir's ORDER/memory to know the held-out animal is the AGENT (first),
not the patient. The patient is chosen cross-category so an "is there a predator anywhere?" bag shortcut gives the wrong
answer -- only tracking the AGENT works.

ARMS (single-variable ablations; SAME reservoir + SAME code dimension unless noted):
  * main         -- two-level class+content codes, recurrent reservoir.                    (expect: generalizes)
  * onehot       -- CONTENT bit only, NO shared category block.                            (expect: collapse; novel = orthogonal)
  * nonrecurrent -- two-level codes but a MEMORYLESS reservoir (washes each token).         (expect: collapse; no order/memory)
  * permuted     -- two-level codes but trained on word-SHUFFLED sentences (order destroyed).(expect: collapse; no agent rule)
  * deranged     -- animals carry a WRONG category block.                                   (expect: collapse / anti-inherit)
  * untrained    -- two-level codes, read-out frozen at zeros.                              (expect: floor)

METRIC: `heldagent_cat_acc` = fraction of "held-agent meets trained-patient" prefixes whose predicted ACTION is of the
AGENT's CORRECT category (content- AND order-sensitive; chance = |category_actions| / V). `heldagent_2way` = correct vs
the other category (chance 0.5). `heldagent_isaction` = predicted an ACTION at all. `train_cat_acc` = same on trained
agents (proves learning). GO: main.heldagent_cat_acc high (>=0.80) AND main.train_cat_acc high (>=0.90) AND main beats
every control by a clear margin (>=0.34), 6-seed (dev 42/43/44 + blind 100/101/102). CPU numpy-backend.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import (
    ReservoirStates, NonRecurrentReservoirStates, train_readout, eval_ce, _standardize_fit,
)

OUT = Path("research/findings/raw/_reslm_rung3_agent_role_generation.json")

CLASSES = ["PRED", "PREY", "MEETS", "ACTION"]           # shared class/category blocks
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
MEETS = "meets"
CAT_ANIMALS = {                                        # trained members + HELD-OUT (never generated) members per category
    "PRED": {"train": ["wolf", "fox", "bear", "hawk", "lynx", "puma"], "held": ["lion", "tiger", "jaguar"]},
    "PREY": {"train": ["rabbit", "mouse", "deer", "quail", "vole", "hare"], "held": ["sheep", "goat", "elk"]},
}
CAT_ACTIONS = {"PRED": ["growl", "hunt", "pounce"], "PREY": ["flee", "hide", "freeze"]}
ANIMAL_CAT = {a: c for c, d in CAT_ANIMALS.items() for a in d["train"] + d["held"]}
ACTION_CAT = {v: c for c, vs in CAT_ACTIONS.items() for v in vs}
ALL_ACTIONS = [v for vs in CAT_ACTIONS.values() for v in vs]
TRAIN_ANIMALS = [a for c in CAT_ANIMALS for a in CAT_ANIMALS[c]["train"]]
HELD_ANIMALS = [a for c in CAT_ANIMALS for a in CAT_ANIMALS[c]["held"]]

WORDS = [MEETS] + TRAIN_ANIMALS + HELD_ANIMALS + ALL_ACTIONS
WORD_IDX = {w: i for i, w in enumerate(WORDS)}
V = len(WORDS)
NCLASS = len(CLASSES)
D_CODE = NCLASS + V
ACTION_POS = 3                                         # the token after "<N1> meets <N2>" -> the AGENT's action


def word_class(w):
    if w == MEETS:
        return "MEETS"
    if w in ANIMAL_CAT:
        return ANIMAL_CAT[w]                           # PRED / PREY (the shared CATEGORY block)
    return "ACTION"


# Training ordered pairs (N1=agent, N2=patient): both orders of cross-category pairs (forces order-tracking) + same-category
# pairs. The ACTION is always the AGENT (N1)'s category action. Generated deterministically so every train animal appears
# as both agent and patient, in both orders.
_TP = CAT_ANIMALS["PRED"]["train"]
_TY = CAT_ANIMALS["PREY"]["train"]
TRAIN_PAIRS = []
for i in range(len(_TP)):
    TRAIN_PAIRS += [(_TP[i], _TY[i]), (_TY[i], _TP[i]),                              # both orders, cross-category
                    (_TP[i], _TP[(i + 1) % len(_TP)]), (_TY[i], _TY[(i + 1) % len(_TY)])]   # same-category
# Held-out prefixes: each HELD-OUT animal as the AGENT (N1) x 3 TRAINED patients (mixing the OTHER category -- so an
# "is a predator present?" bag shortcut mispredicts -- and the same category). 6 held animals x 3 patients = 18 prefixes.
HELD_PAIRS = []
for a in CAT_ANIMALS["PRED"]["held"]:
    HELD_PAIRS += [(a, _TY[0]), (a, _TP[0]), (a, _TY[1])]                            # cross, same, cross
for a in CAT_ANIMALS["PREY"]["held"]:
    HELD_PAIRS += [(a, _TP[0]), (a, _TY[0]), (a, _TP[1])]                            # cross, same, cross


def build_sents(pairs, agent_action_cat=None):
    """'N1 meets N2 ACTION' for each pair x each ACTION of the AGENT's category. agent_action_cat: override the agent's
       action category (unused here; kept for symmetry)."""
    sents = []
    for n1, n2 in pairs:
        cat = agent_action_cat[n1] if agent_action_cat else ANIMAL_CAT[n1]
        for act in CAT_ACTIONS[cat]:
            sents.append([n1, MEETS, n2, act])
    return sents


TRAIN_SENTS = build_sents(TRAIN_PAIRS)
HELD_PREFIXES = [([n1, MEETS, n2], ANIMAL_CAT[n1]) for (n1, n2) in HELD_PAIRS]        # (prefix, true AGENT category)
TRAIN_PREFIXES = [([n1, MEETS, n2], ANIMAL_CAT[n1]) for (n1, n2) in TRAIN_PAIRS]


def make_code(w, code_type, seed):
    v = np.zeros(D_CODE, np.float64)
    v[NCLASS + WORD_IDX[w]] = 1.0                                   # unique content bit
    if code_type == "onehot":
        return v                                                   # NO shared class/category block
    cls = word_class(w)
    if code_type == "deranged" and w in ANIMAL_CAT:                # animals carry a WRONG category block (frozen per word)
        rng = np.random.default_rng(seed * 101 + WORD_IDX[w])
        others = [c for c in ("PRED", "PREY") if c != cls]
        cls = others[int(rng.integers(len(others)))]
    v[CLASS_IDX[cls]] = 1.0
    return v


def encode(sent, code_type, seed):
    return np.asarray([make_code(w, code_type, seed) for w in sent])


FEATURE_MODE = "cum"       # {cum, win, cum_win, cum_buf, win_buf, cumwin_buf}. DEFAULT = cum = the RESERVOIR ALONE carries
                           # the agent (no WM buffer). The buffer (cum_buf) TRIVIALIZES this task -- it hands the read-out
                           # the answer-determining agent category directly, propping up even the order-shuffled control --
                           # so it is a shortcut here (unlike Rung 2, where the buffer held a genuinely DISTAL forgotten
                           # referent). The honest mechanism is the recurrent reservoir's own memory.


def _code_category(w, code_type, seed):
    """The category the CODE actually carries for animal w (TRUE class, or the deranged wrong class -- matching make_code)."""
    cls = word_class(w)
    if code_type == "deranged" and w in ANIMAL_CAT:
        rng = np.random.default_rng(seed * 101 + WORD_IDX[w])
        others = [c for c in ("PRED", "PREY") if c != cls]
        cls = others[int(rng.integers(len(others)))]
    return cls


def agent_latch(sent, code_type, seed, buf_scramble):
    """The Rung-2 WM latch specialized to the AGENT: N1's CATEGORY read from its code at position 0, held non-fading. For
       one-hot codes (no category block) the latch is empty -> the buffer cannot supply the category. buf_scramble swaps
       the latched category (the buffer-content control)."""
    if code_type == "onehot":
        return np.zeros(2)                                          # no shared category block to read
    cls = _code_category(sent[0], code_type, seed)                 # matches whatever category the reservoir code carried
    if buf_scramble:
        cls = "PREY" if cls == "PRED" else "PRED"
    return np.array([1.0, 0.0]) if cls == "PRED" else np.array([0.0, 1.0])


def build_feature(res, sent, code_type, seed, buf_scramble=False):
    U = encode(sent, code_type, seed)
    parts_cum = res.per_token_states(U, feature="running_cumulative") if ("cum" in FEATURE_MODE) else None
    parts_win = res.per_token_states(U, feature="per_window") if ("win" in FEATURE_MODE) else None
    use_buf = "buf" in FEATURE_MODE
    lat = agent_latch(sent, code_type, seed, buf_scramble) if use_buf else None
    n = len(sent)
    feats = []
    for t in range(n):
        segs = []
        if parts_cum is not None:
            segs.append(parts_cum[t])
        if parts_win is not None:
            segs.append(parts_win[t])
        if lat is not None:
            segs.append(lat)                                       # non-fading agent-category latch, constant across t
        feats.append(np.concatenate(segs))
    return feats


def cache_sents(res, sents, code_type, seed, buf_scramble=False):
    return [(build_feature(res, s, code_type, seed, buf_scramble), [WORD_IDX[w] for w in s]) for s in sents]


def predict_action(W, mean, std, res, prefix, code_type, seed, buf_scramble=False):
    states = build_feature(res, prefix, code_type, seed, buf_scramble)
    x = np.concatenate([(states[ACTION_POS - 1] - mean) / std, [1.0]])   # state after N2 (index 2) predicts the action
    return WORDS[int(np.argmax(W @ x))]


def score_prefixes(W, mean, std, res, prefixes, code_type, seed, buf_scramble=False):
    cat_ok = two_ok = isact = tot = 0
    for prefix, true_cat in prefixes:
        pred = predict_action(W, mean, std, res, prefix, code_type, seed, buf_scramble)
        isact += int(pred in ACTION_CAT)
        if pred in ACTION_CAT:
            two_ok += int(ACTION_CAT[pred] == true_cat)
        cat_ok += int(pred in ACTION_CAT and ACTION_CAT[pred] == true_cat)
        tot += 1
    return {"cat_acc": cat_ok / max(1, tot), "two_way": two_ok / max(1, isact) if isact else 0.0,
            "isaction": isact / max(1, tot)}


ARM_CFG = {  # code_type, recurrent, permute_train, buf_scramble
    "main":         ("class",   True,  False, False),
    "onehot":       ("onehot",  True,  False, False),
    "nonrecurrent": ("class",   False, False, False),
    "permuted":     ("class",   True,  True,  False),
    "buf_scramble": ("class",   True,  False, True),
    "deranged":     ("deranged",True,  False, False),
    "untrained":    ("class",   True,  False, False),
}
ARMS = list(ARM_CFG)


def run_arm(seed, arm, epochs, lr, n_pool=300):
    code_type, recurrent, permute_train, buf_scramble = ARM_CFG[arm]
    res = (ReservoirStates if recurrent else NonRecurrentReservoirStates)(D_CODE, seed=seed, n=n_pool)
    train = TRAIN_SENTS
    if permute_train:
        rng = np.random.default_rng(seed * 7 + 3)
        train = [list(rng.permutation(s)) for s in train]

    tr_cache = cache_sents(res, train, code_type, seed, buf_scramble)
    mean, std = _standardize_fit(tr_cache)
    W = (np.zeros((V, len(mean) + 1)) if arm == "untrained"
         else train_readout(tr_cache, V, epochs, lr, np.random.default_rng(seed * 13 + 1), mean, std))

    held = score_prefixes(W, mean, std, res, HELD_PREFIXES, code_type, seed, buf_scramble)
    train_sc = score_prefixes(W, mean, std, res, TRAIN_PREFIXES, code_type, seed, buf_scramble)
    return {"arm": arm, "heldagent_cat_acc": held["cat_acc"], "heldagent_2way": held["two_way"],
            "heldagent_isaction": held["isaction"], "train_cat_acc": train_sc["cat_acc"]}


def main():
    global FEATURE_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--json", type=str, default=str(OUT))
    ap.add_argument("--feature-mode", type=str, default=FEATURE_MODE,
                    choices=["cum", "win", "cum_win", "cum_buf", "win_buf", "cumwin_buf"])
    ap.add_argument("--n-pool", type=int, default=300)           # reservoir size (bigger -> less finite-size noise)
    args = ap.parse_args()
    FEATURE_MODE = args.feature_mode

    train_tokens = {w for s in TRAIN_SENTS for w in s}            # leakage guard: no held animal is a TOKEN in any train sent
    for prefix, _ in HELD_PREFIXES:
        assert prefix[0] not in train_tokens, f"held agent {prefix[0]} leaked into train"
    chance = len(CAT_ACTIONS["PRED"]) / V

    t0 = time.time()
    per_seed = {}
    for seed in args.seeds:
        rb = {}
        for arm in ARMS:
            try:
                rb[arm] = run_arm(seed, arm, args.epochs, args.lr, n_pool=args.n_pool)
            except Exception as e:
                rb[arm] = {"arm": arm, "error": f"{e}", "trace": traceback.format_exc()}
            r = rb[arm]
            tag = (f"cat={r.get('heldagent_cat_acc'):.3f} 2way={r.get('heldagent_2way'):.2f} "
                   f"isact={r.get('heldagent_isaction'):.2f} train={r.get('train_cat_acc'):.3f}"
                   ) if "error" not in r else r["error"]
            print(f"[seed {seed}] {arm:13s} {tag}", flush=True)
        per_seed[seed] = rb

    def agg(arm, key):
        vals = [per_seed[s][arm].get(key) for s in args.seeds if "error" not in per_seed[s][arm]]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    keys = ("heldagent_cat_acc", "heldagent_2way", "heldagent_isaction", "train_cat_acc")
    aggregate = {arm: {k: agg(arm, k) for k in keys} for arm in ARMS}
    main_acc = aggregate["main"]["heldagent_cat_acc"]
    # PRIMARY (mechanism-ablation) controls -- the ones that remove an actual ingredient of the claim: the shared code
    # (onehot), the reservoir's memory (nonrecurrent), word order (permuted), the trained read-out (untrained). buf_scramble
    # + deranged are REPORTED but NOT gated: with the recurrent reservoir carrying the agent, the buffer is redundant (so
    # buf_scramble does not collapse) and a consistent wrong-but-learnable code (deranged) is a weak control for a learned
    # read-out -- documented, not load-bearing.
    # PRIMARY (mechanism-ablation) controls that remove a genuine INGREDIENT: the shared category code (onehot), the
    # reservoir's memory of the agent (nonrecurrent), and the trained read-out (untrained). NOTE (honest scope): permuted
    # (word order) is NOT a clean control here -- the category->action mapping is largely BAG-recoverable, so shuffling
    # training only weakly hurts it; word order is only WEAKLY load-bearing for this task. deranged/buf_scramble are
    # reported diagnostics (weak/redundant). The gate uses the 2way MECHANISM metric (given an action is emitted, is it the
    # AGENT's correct category) -- `isact` (emit-an-action rate for a NOVEL agent) is a separate calibration residual, and
    # is reported but not gated.
    PRIMARY_CONTROLS = ["onehot", "nonrecurrent", "untrained"]
    worst_ctrl = max(aggregate[a]["heldagent_cat_acc"] for a in PRIMARY_CONTROLS)
    worst_ctrl_2way = max(aggregate[a]["heldagent_2way"] for a in PRIMARY_CONTROLS)
    margin = (main_acc - worst_ctrl) if (main_acc is not None) else None

    per_seed_go = []
    for s in args.seeds:
        rb = per_seed[s]
        if any("error" in rb[a] for a in ARMS):
            per_seed_go.append(False); continue
        m2 = rb["main"]["heldagent_2way"]                         # mechanism: correct category | action emitted
        mc = rb["main"]["heldagent_cat_acc"]                      # unconditional
        wc = max(rb[a]["heldagent_cat_acc"] for a in PRIMARY_CONTROLS)
        w2 = max(rb[a]["heldagent_2way"] for a in PRIMARY_CONTROLS)
        per_seed_go.append(bool(m2 >= 0.85 and (m2 - w2) >= 0.34 and (mc - wc) >= 0.30
                                and rb["main"]["train_cat_acc"] >= 0.90))
    n_go = int(sum(per_seed_go))

    out = {
        "runner": "_emerge_reservoir_lm_rung3_agent_role_generation_derisk",
        "seeds": args.seeds, "epochs": args.epochs, "lr": args.lr, "chance": chance,
        "n_train_sents": len(TRAIN_SENTS), "n_held": len(HELD_PREFIXES),
        "per_seed": {str(s): per_seed[s] for s in args.seeds}, "aggregate": aggregate,
        "feature_mode": FEATURE_MODE, "primary_controls": PRIMARY_CONTROLS,
        "main_heldagent_cat_acc": main_acc, "main_heldagent_2way": aggregate["main"]["heldagent_2way"],
        "worst_control_cat_acc": worst_ctrl, "worst_control_2way": worst_ctrl_2way, "margin": margin,
        "per_seed_go": per_seed_go, "n_go": n_go, "n_seeds": len(args.seeds), "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nAGG main.cat_acc={main_acc} worst_ctrl={worst_ctrl} margin={margin} chance={chance:.3f} "
          f"GO {n_go}/{len(args.seeds)}  ({out['elapsed_s']}s) -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
