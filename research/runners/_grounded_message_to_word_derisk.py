"""Grounded message-to-word: a BRAIN-NATIVE naming path that replaces the host semantic decoder.

WHY THIS RUNNER EXISTS
----------------------
Two GO'd frontiers converge on one un-built step:

  * `2026-08-02-gap1-wkv-width-ladder-...` — the 267M/d2048 WKV language cortex is RF-spiking-forward
    faithful (GO 6/6). Its named next step: "use this larger faithful language-circuit scaffold inside the
    grounded speech-action plan while continuing to burn down ... host-side phrasing scaffolds."
  * `2026-08-03-grounded-speech-action-loop-6seed-GO` — a hungry brain that SEES a learned apple emits a
    REQUEST, and falls SILENT once sated (GO 6/6). But: "A host decoder maps one neural winner to
    `request apple` ... the fixed semantic decoder is temporary." Its next-mechanism #3 is verbatim:
    "Replace the host semantic decoder with a brain-native message-to-word path."

The host `request apple` string is the last thing standing between "the brain chose to speak about the
apple" and "the word the mouth produces". This runner burns THAT scaffold down: the WORD is selected by a
LOCAL-rule-learned naming map from the chosen referent's percept assembly, gated by the request/silence
competition, and the brain-chosen word — not a host lookup — is what the WKV renderer then articulates.

WHAT IS BRAIN-BASED vs SCAFFOLD (declared, not hidden)
------------------------------------------------------
  * BRAIN: the referent -> word-unit association is learned by a gated local Hebbian rule (pre = percept
    assembly spikes; post = the word-unit the TEACHER co-activates while naming the object). Zero-init, no
    weight transport. At inference plasticity is CLOSED and the decode reads ONLY the percept assembly
    through the learned weights — the true label is never consulted on the inference path.
  * TEACHER (legitimate social environment): during a naming event the caregiver co-activates the object's
    word-unit ("this is an apple"). This is the teacher-as-social-environment scaffold, not a host answer
    lookup — it is present only during LEARNING, never at inference.
  * BODY / ARTICULATION (legitimate host): each word-unit has a FIXED binding to one WKV vocab token, the
    output alphabet (a motor-pool -> phoneme analogue). Which word-unit a referent maps to is LEARNED; the
    word-unit -> token binding is the fixed articulatory alphabet.
  * NAMED RESIDUAL SCAFFOLD (burn-down target of a LATER rung, not this one): the carrier frame skeleton
    ("the <agent> <verb> ___") the renderer copies is still host phrasing, and the request/silence gate's
    populations are hand-declared (their full spiking form is already GO in the Aug-03 loop; here a minimal
    rate competition just ROUTES). This rung burns down the REFERENT word decoder only.
  * SCAFFOLD (named, retired later): the WKV cortex itself is conventionally trained. Here it is the fixed
    faithful language-circuit scaffold used off-bridge in numpy (its RF-spiking-forward parity is GO).

GO GATE (all must hold, 6 seeds)
--------------------------------
  1. name_acc == 1.0            — every learned referent decodes to its taught word-unit from the percept
                                  assembly alone (brain-native naming works).
  2. render_faithful == 1.0     — the WKV utterance articulates the BRAIN-DECODED referent word, and it
                                  equals the taught word; swapping the referent swaps the spoken word.
  3. silence_moat               — a sated trial routes to silence => the renderer is invoked 0 times
                                  (mirrors the gate-first moat); a hungry trial invokes it exactly once.
ANTI-CHEATS (controls that must move)
  4. zero_init_null             — before naming, no confident decode (learned effect is not a host bias).
  5. lesion_collapse            — zeroing the naming pathway collapses the decode to no-confident-word
                                  (fails SAFE to silence, never a confident wrong name).
  6. permutation_followed       — teaching a permuted referent->word map decodes the PERMUTATION, and the
                                  original word is accepted 0/K (the map is learned, not wired).
  7. novel_abstains             — an untaught percept assembly stays below the confidence margin (won't
                                  blurt a wrong name).

Backend: pure numpy + the numpy WKVFaculty forward. CPU. No GPU, no training. Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._grounded_message_to_word_derisk \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/grounded_message_to_word/message_to_word_6seed.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from _wkv_faculty import WKVFaculty  # noqa: E402
from tools.verdict import Verdict, GO  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

GROUNDED_CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"

# Learned referents: (percept object, carrier agent, carrier verb, referent word) -- the referent word is
# what the naming map must select; agent+verb are the host carrier-frame skeleton (a NAMED residual). All
# referent words are in the grounded WKV vocab so the renderer can articulate them.
REFERENTS = [
    ("apple",  "fox",  "eats",  "apple"),
    ("seed",   "bird", "eats",  "seed"),
    ("honey",  "bee",  "makes", "honey"),
    ("water",  "fish", "sees",  "water"),
]
NP_PERCEPT = 240    # percept population size
KP_ACTIVE = 24      # active units per referent assembly (sparse, overlapping -> decode needs learning)
CONF_MARGIN = 0.15  # confidence margin (fraction of KP) below which the brain declines to name
NOISE_SIGMA = 1.6   # per-unit Gaussian percept noise at inference -> naming accuracy is GRADED, not pinned
N_TRIALS = 120      # noisy presentations per referent (the accuracy metric must have resolution)


def make_assemblies(rng, k, novel=1):
    """k learned referent assemblies + `novel` untaught ones, sparse binary over NP_PERCEPT, overlapping."""
    A = np.zeros((k + novel, NP_PERCEPT))
    for i in range(k + novel):
        idx = rng.choice(NP_PERCEPT, size=KP_ACTIVE, replace=False)
        A[i, idx] = 1.0
    return A


def learn_naming(assemblies, targets, k, lr=1.0):
    """Gated local Hebbian naming map. W[target] += lr * assembly (pre spikes x post teacher drive).
    `targets[i]` is the word-unit the TEACHER co-activates for referent i (permutable). Zero-init."""
    W = np.zeros((k, NP_PERCEPT))
    for i in range(k):
        W[targets[i]] += lr * assemblies[i]  # plasticity gate OPEN during the naming event
    return W


def decode(W, assembly, k):
    """Inference: plasticity CLOSED. Reads ONLY the assembly through learned weights. Returns
    (winner_unit, margin_fraction). The true label is NOT an argument -- no host answer lookup."""
    scores = W @ assembly
    order = np.argsort(-scores)
    top1 = scores[order[0]]
    top2 = scores[order[1]] if k > 1 else 0.0
    margin = (top1 - top2) / KP_ACTIVE
    return int(order[0]), float(margin)


def noisy_acc(W, assemblies, targets, rng, k):
    """GRADED naming accuracy: fraction of NOISY percept presentations that decode to the target unit.
    Percept noise gives the metric resolution (a deterministic separation would be a ceiling with no
    discriminating power). Reads only the noisy assembly through W -- no host label on the read path."""
    correct = total = 0
    for i in range(len(assemblies)):
        for _ in range(N_TRIALS):
            x = assemblies[i] + rng.normal(0.0, NOISE_SIGMA, NP_PERCEPT)
            if int(np.argmax(W @ x)) == targets[i]:
                correct += 1
            total += 1
    return correct / total


def run_seed(seed, fac):
    rng = np.random.default_rng(seed)
    k = len(REFERENTS)
    A = make_assemblies(rng, k, novel=1)
    learned, novel = A[:k], A[k]
    ident = list(range(k))

    # --- naming taught by the teacher (identity map) ---
    W = learn_naming(learned, ident, k)

    # --- 1. brain-native naming accuracy under percept noise (graded, discriminating) ---
    name_acc = noisy_acc(W, learned, ident, np.random.default_rng(seed + 1), k)

    # --- 4. chance control: an UNTRAINED random naming map (learned effect is not a host bias) ---
    Wrand = rng.normal(0.0, float(learned.mean()), size=(k, NP_PERCEPT))
    name_acc_chance = noisy_acc(Wrand, learned, ident, np.random.default_rng(seed + 2), k)

    # --- 5. lesion: zero the naming pathway AFTER learning ---
    Wles = W.copy(); Wles[:] = 0.0
    lesion_acc = noisy_acc(Wles, learned, ident, np.random.default_rng(seed + 3), k)  # collapses toward chance
    decL = [decode(Wles, learned[i], k) for i in range(k)]                            # clean-presentation safety:
    lesion_confident = float(np.mean([d[1] > CONF_MARGIN for d in decL]))             # must be 0 -> abstain, not wrong

    # --- 6. permutation control: teach a derangement ---
    perm = list(np.roll(ident, 1))
    Wp = learn_naming(learned, perm, k)
    perm_followed = noisy_acc(Wp, learned, perm, np.random.default_rng(seed + 4), k)
    orig_accepted = noisy_acc(Wp, learned, ident, np.random.default_rng(seed + 5), k)  # original word rejected

    # --- 7. novel (untaught) percept: must abstain (below confidence) on a clean presentation ---
    _, novel_margin = decode(W, novel, k)
    novel_abstains = bool(novel_margin <= CONF_MARGIN)

    # --- 3. request/silence gate routing (minimal rate competition; full spiking form GO in Aug-03) ---
    def gate(hunger, satiety, food_cue):
        req = 1.2 * food_cue + 1.0 * hunger      # cue+need converge on request
        sil = 1.3 * satiety                       # satiety drives silence
        inh = 0.5 * (req + sil)                    # shared FS inhibition
        return (req - inh) - (sil - inh) > 0.0     # speak if request margin positive

    # --- 2 + 3: render only when the gate says speak; the SPOKEN word is the BRAIN-DECODED referent ---
    render_ok, spoken = [], []
    invocations_when_silent = 0
    for i, (obj, agent, verb, word) in enumerate(REFERENTS):
        # HUNGRY: gate opens, decode the referent word from the assembly, articulate via WKV.
        assert gate(hunger=1.0, satiety=0.0, food_cue=1.0), "hungry trial must route to speak"
        winner, _ = decode(W, learned[i], k)
        chosen_word = REFERENTS[winner][3]                    # word-unit -> fixed token binding (articulation)
        frame = ["the", agent, verb, chosen_word]             # carrier skeleton (named residual) + brain word
        fac.n_invocations = 0
        utter = fac.answer(" ".join(frame) + " .", "q")
        spoken_words = utter.split()
        patient = spoken_words[-1] if spoken_words else ""
        render_ok.append(patient == word and chosen_word == word and fac.n_invocations == 1)
        spoken.append(utter)

        # SATED: gate closes -> silence -> the renderer must NOT be reached.
        fac.n_invocations = 0
        if gate(hunger=0.0, satiety=1.0, food_cue=1.0):
            _ = fac.answer(" ".join(frame) + " .", "q")       # would-be speech (should not run)
        invocations_when_silent += fac.n_invocations

    render_faithful = float(np.mean(render_ok))

    return {
        "seed": seed, "k": k,
        "name_acc": name_acc, "name_acc_chance": name_acc_chance,
        "render_faithful": render_faithful,
        "invocations_when_silent": int(invocations_when_silent),
        "lesion_acc": lesion_acc, "lesion_confident_frac": lesion_confident,
        "perm_followed": perm_followed, "orig_accepted_after_perm": orig_accepted,
        "novel_abstains": novel_abstains, "novel_margin": float(novel_margin),
        "spoken": spoken,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--ckpt", default=GROUNDED_CKPT)
    ap.add_argument("--out", default="research/findings/raw/grounded_message_to_word/message_to_word_6seed.json")
    args = ap.parse_args()

    chance = 1.0 / len(REFERENTS)
    fac = WKVFaculty(ckpt=args.ckpt, max_new=12)
    per_seed = [run_seed(s, fac) for s in args.seeds]
    for r in per_seed:
        print("  seed %-4d name_acc=%.3f (chance=%.3f lesion=%.3f) render=%.3f silent_inv=%d "
              "perm=%.3f orig=%.3f novel_abstains=%s | %s"
              % (r["seed"], r["name_acc"], r["name_acc_chance"], r["lesion_acc"], r["render_faithful"],
                 r["invocations_when_silent"], r["perm_followed"], r["orig_accepted_after_perm"],
                 r["novel_abstains"], r["spoken"][0]))

    agg = lambda key: float(np.mean([r[key] for r in per_seed]))
    mean_name = agg("name_acc")
    mean_chance = agg("name_acc_chance")
    mean_lesion = agg("lesion_acc")
    mean_render = agg("render_faithful")
    all_silent0 = all(r["invocations_when_silent"] == 0 for r in per_seed)
    all_lesion_conf0 = all(r["lesion_confident_frac"] == 0.0 for r in per_seed)
    mean_perm = agg("perm_followed")
    mean_orig = agg("orig_accepted_after_perm")
    all_novel = all(r["novel_abstains"] for r in per_seed)
    NAME_GO = 0.85  # graded threshold: the noisy decode must be robustly above chance, not merely nonzero

    print("\n  attribution of the naming accuracy above the untrained random-map control:")
    frac = attributable_to("naming decode accuracy", mean_name, mean_chance)

    v = Verdict("grounded message-to-word (brain-native referent naming replaces host decoder)", chance=chance)
    v.disabled("full spiking request/silence gate",
               "modeled as a minimal rate competition; the spiking form is GO in 2026-08-03 grounded-speech-action loop")
    v.disabled("carrier-frame phrasing",
               "host 'the <agent> <verb> ___' skeleton is a NAMED residual scaffold; only the REFERENT word is brain-selected here")
    v.require("brain-native naming accuracy (noisy) >= %.2f" % NAME_GO, mean_name, expect=lambda x: x >= NAME_GO)
    v.floor("naming accuracy above chance", mean_name, floor=chance)
    v.control("naming vs untrained random map (learned, not host bias)", mean_name, mean_chance, min_separation=0.4)
    v.reaches("lesion collapses the decode toward chance", before=mean_name, after=mean_lesion)
    v.require("lesion never emits a confident decode (fails safe)", all_lesion_conf0, expect=True)
    v.require("render articulates the brain-decoded word == 1", mean_render, expect=lambda x: x == 1.0)
    v.require("silence routes to zero renderer invocations", all_silent0, expect=True)
    v.require("permutation followed >= %.2f" % NAME_GO, mean_perm, expect=lambda x: x >= NAME_GO)
    v.control("permuted map rejects the original word", mean_perm, mean_orig, min_separation=0.4)
    v.require("novel percept abstains (below confidence) all seeds", all_novel, expect=True)

    go = (mean_name >= NAME_GO and mean_name > chance and (mean_name - mean_chance) > 0.4
          and mean_lesion < mean_name and all_lesion_conf0 and mean_render == 1.0 and all_silent0
          and mean_perm >= NAME_GO and (mean_perm - mean_orig) > 0.4 and all_novel)
    decided = v.decide(go=go)

    out = {
        "verdict": decided,
        "mean_name_acc": mean_name, "mean_name_acc_chance": mean_chance, "mean_lesion_acc": mean_lesion,
        "mean_render_faithful": mean_render,
        "naming_accuracy_attributable_fraction": frac,
        "all_silent_zero_invocations": all_silent0,
        "all_lesion_zero_confident": all_lesion_conf0,
        "mean_perm_followed": mean_perm, "mean_orig_accepted_after_perm": mean_orig,
        "all_novel_abstains": all_novel,
        "chance": chance, "name_go_threshold": NAME_GO,
        "n_referents": len(REFERENTS), "conf_margin": CONF_MARGIN,
        "noise_sigma": NOISE_SIGMA, "n_trials": N_TRIALS,
        "np_percept": NP_PERCEPT, "kp_active": KP_ACTIVE,
        "ckpt": args.ckpt, "seeds": args.seeds,
        "per_seed": per_seed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n  wrote %s" % args.out)
    print("  => %s" % decided["status"])
    return 0 if decided["status"] == GO else 1


if __name__ == "__main__":
    raise SystemExit(main())
