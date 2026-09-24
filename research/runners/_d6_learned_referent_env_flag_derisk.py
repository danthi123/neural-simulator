"""D6 LEARNED-REFERENT ROUTE, THE ENV-FLAG WIRING (language lane E, next rung, 2026-09-24).

WHAT THIS ADDS ON TOP OF THE ALREADY-BANKED v2 RESULT. `_lexicon_spiking_referent_derisk.py` (6-seed GO,
`2026-09-23-cpu-lane-harvest-language-lexicon-referent-v2-S2-scored-6seed-GO.md`) proved the MECHANISM (a coupled
spiking WTA, Hebbian frame->category synapses, load-bearing learned edge) by injecting the trained lexicon directly
onto a fresh organ's `.referent_lexicon` attribute (`_organ_arm`, `capability_arm`). That is not the PRODUCTION route:
production never sets `.referent_lexicon` -- it reads `BRAIN_LEARNED_REFERENT_LEXICON` from the process environment
through `d6_multiref_wm_production_organ._flag_learned_referent_lexicon()` -> `lexicon_spiking_frame_category.get_lexicon()`
(a MODULE-LEVEL SINGLETON, always seed=42, built once per process). That indirection was never exercised end-to-end
before this file: a bug in the env-var parse, in the singleton's lazy-build guard, or in `get_organ()`'s per-session
construction leaving a stale `.referent_lexicon` set from an earlier test would be invisible to the direct-injection
gate above and would still ship broken.

THE ONE NEW THING UNDER TEST: the WIRING (env var -> singleton -> organ), not the detector (already proven). This is
the "next rung" the 2026-09-23 harvest named: "route the referent extraction ... through the LEARNED lexicon behind a
default-off flag, so the wm-binding-advanced faculty can be exercised on nouns the hand list lacks (the allfixes2
battery missed 'owl')".

PRE-REGISTERED (this file, before any evaluation run). EVIDENCE gates, each stating the outcome that FAILS it:
  R1 ROUTE-ON, held-out capability. `BRAIN_LEARNED_REFERENT_LEXICON=1` in the process env, a FRESH
     `MultiReferentWMOrgan()` (never touched `.referent_lexicon` -- must resolve the flag itself), turn
     "the wolf watches the owl" ("owl" is in NEITHER `_REFERENT_NOUNS` nor `HAND_NOUN_SEEDS`/`NONNOUN_SEEDS` --
     genuinely held out of every hand list). FAILS if `judge()` returns None (out of scope), or `n_referents != 2`,
     or "owl" is absent from `input_order` -- i.e. if the env var does not actually reach the organ's decision.
  R2 ROUTE-OFF, byte-identical contrast. The SAME process, SAME singleton already built, env var deleted (unset,
     the production default) before the call. FAILS if the SAME turn is STILL in-scope with "owl" recovered -- that
     would mean "default off" does not actually gate anything (the contrast gate the checklist requires: an OFF arm
     credited with the ON arm's effect is not a control).
  R3 POPULATION RECOVERY, routed through the flag (not attribute injection). Over >=12 held-out noun PAIRS (POS
     ground-truth fixture, excluded from every hand list and from the training curriculum), with the flag ON, a
     fresh organ's "the A and the B walked in" -> "who are we talking about?" recovers BOTH held-out nouns.
     GATED (matches `score()` exactly): the MINIMUM per-seed recovered-both rate across all 6 seeds must be
     >= 0.50 (a 10-point tolerance band under the already-banked S7 direct-injection threshold of 0.60, for the
     extra indirection hop -- this file's job is to catch a BROKEN WIRE, not re-litigate the detector). FAILS if
     any seed's routed rate drops below 0.50, i.e. the indirection itself is losing the capability on that seed.
  R4 LESION, ROUTED. Flag ON AND `BRAIN_LEARNED_REFERENT_LESION=1`, same population battery. GATED (matches
     `score()` exactly): the MAXIMUM per-seed recovered-both rate across all 6 seeds must be <= 0.20 (S7's lesion
     ceiling). FAILS if the lesion does not reduce recovery on every seed when reached through the env-var route
     (the single-word HELD-phrase lesion outcome is explicitly NOT gated here either -- S3b/S8 already documented
     it as a ~30%-residual coin flip at n=1; this is the 12-trial population arm instead). This gate ALSO fails
     (see `run_seed`'s `r4_lever_moved`) if the intact and lesioned rates are numerically equal for a seed: an
     unmoved lever means the lesion did not reach the route at all (a broken wire or a lesion the flag ignores),
     which must record as a FAILED gate, not crash the process before a JSON is ever written.
  R5 SINGLETON REUSE. `get_lexicon()` is called exactly ONCE across the whole R1-R4 sequence in-process (its
     `_LEXICON is None` branch fires once) -- reports the observed call count. REPORT-ONLY, NOT GATED (matches
     `score()`: `R5_singleton_built_all_seeds` is reported alongside the evidence table but excluded from the
     set `passed` is computed over, because R1 passing already implies R5 -- a check that cannot independently
     fail must never sit inside the verdict). A count of zero is impossible given R1 passed, so a human reading
     the report can still use it to catch an unreachable-import regression; it just cannot flip GO/NO-GO itself.
INTEGRITY SMOKE (must hold, not evidence): the organ's own `all_recovered`/`recovered` bookkeeping and the
detector's own weight-hash check (`decide()` raises on a lesion that does not hold at measurement) are reused
unmodified -- no new pass-by-construction check is added here.

HONEST RESIDUALS (unchanged from the mechanism gate, restated because this file exercises production code, not a
private harness): `get_lexicon()` always trains at seed=42 regardless of which conversational seed the calling
organ was built with -- **all 6 organ seeds in R3/R4 therefore share the SAME ONE lexicon/detector** (the
module-level singleton is force-reset once per process at the top of `run_seed`, then reused for every
condition/trial within that seed's process, and separately for each of the 6 seed-processes -- but every one of
those 6 processes trains the identical detector at seed=42). The R3/R4 6-seed result is 1 lexicon x 6 organ-seed
population/pair draws, NOT 6 independently-trained substrates -- do not write it up as 6-seed replication of the
detector itself (that claim is already banked separately by the direct-injection gate's own 6-seed run). NOUN-hood
not REFERENT-hood; teacher-supervised curriculum, not self-organized category discovery. See
`lexicon_spiking_frame_category`'s own docstring for the full list -- not repeated here.

Run (per seed; numpy backend, real TinyStories corpus + fixture required -- `corpus_fetch.fetch_corpus("tinystories")`
or `_lexicon_build_pos_gt_fixture.py` first, if either cache is missing):
    SIM_BACKEND=numpy python -u -m research.runners._d6_learned_referent_env_flag_derisk --seed 42 \
        --json research/findings/raw/_d6_learned_referent_env_flag/s42.json
    python -m research.runners._d6_learned_referent_env_flag_derisk --score research/findings/raw/_d6_learned_referent_env_flag
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._comprehension_learned_animacy_cue_derisk import load_tokens, build_vocab  # noqa: E402
from research.runners._lexicon_learned_referent_derisk import _held_out, FIXTURE  # noqa: E402
from research.runners import lexicon_spiking_frame_category as L  # noqa: E402
from research.runners import d6_multiref_wm_production_organ as D6  # noqa: E402
from tools.lab import lever  # noqa: E402

SEEDS6 = [42, 43, 44, 100, 101, 102]
N_PAIRS = 12          # held-out noun-PAIR population trials (matches S7)
GATE = dict(r3_recover_min=0.50, r4_lesion_max=0.20)
HELD_WORD_PHRASE = "the wolf watches the owl"   # "owl" is in no hand list; the banked capability-arm example


def _clean_env():
    os.environ.pop("BRAIN_LEARNED_REFERENT_LEXICON", None)
    os.environ.pop("BRAIN_LEARNED_REFERENT_LESION", None)


def _file_sha256(path):
    """Content hash of an input file, so seeds run on different inputs cannot be pooled (2026-09-24 review: seed 42
    ran on a 7.99 MB prefix of the 19.97 MB tinystories.txt; the per-seed JSON recorded neither, and R3 moved
    0.9167 -> 0.8333 between the two)."""
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest(), os.path.getsize(path)
    except OSError:
        return None, None


def _env(corpus, max_chars=8_000_000, top_v=2000):
    tokens = load_tokens(corpus, max_chars)
    vocab, _ = build_vocab(tokens, top_v)
    env = L.FrameEnvironment(tokens, vocab + [w for w in L.HAND_NOUN_SEEDS + L.NONNOUN_SEEDS if w not in vocab])
    return vocab, env


def _r4_gate(intact_rate, lesion_rate, hand_rate):
    """The R4 lesion gate, factored out so it is unit-testable without a corpus/organ build.

    MUST NEVER RAISE: a broken wire (intact=lesion=0.0) or a lesion the route ignores (intact=lesion>0) are
    exactly the two failure modes R4 exists to catch, and both make intact_rate == lesion_rate. `lever(...,
    required=False)` records that as `r4_lever_moved=False` instead of raising LeverError before the caller's
    JSON is written -- a crash here used to turn a would-be NO-GO into an INCOMPLETE at score() time, because no
    artifact ever landed on disk for score() to read. Returns (r4_lever_moved, r4_pass).
    """
    r4_lever_moved = lever(
        "BRAIN_LEARNED_REFERENT_LESION via the env-var route (intact vs lesioned population recovery)",
        intact_rate, lesion_rate, required=False)
    # ATTRIBUTION: is the recovery drop actually owned by the learned-edge lesion (BRAIN_LEARNED_REFERENT_LESION),
    # routed through the SAME env-var wire as the intact arm, or by something else the flag toggle also touched
    # (e.g. the route falling back to the hand table regardless of the lesion bit)? Compare the routed lesion arm
    # against the routed hand-only arm (flag OFF): if the lesion arm sits at the HAND baseline, the lesion is
    # reverting the WIRE to the hand path (the documented outcome); if it sits above hand, the lesion is not
    # load-bearing through this route and R4 should already have failed.
    lever("routed lesion arm vs the flag-OFF hand-only arm (does the lesion revert to the hand path?)",
          hand_rate, lesion_rate, required=False)
    # An unmoved lever is itself a failed gate: the lesion must be OBSERVED to change the routed outcome, not
    # merely happen to land at or below the ceiling by coincidence (e.g. both arms sitting at 0.0).
    r4_pass = bool(lesion_rate <= GATE["r4_lesion_max"] and r4_lever_moved)
    return r4_lever_moved, r4_pass


def run_seed(seed, corpus, pos_gt, n_pairs=N_PAIRS):
    t0 = time.time()
    _clean_env()
    L._LEXICON = None   # force a fresh singleton for THIS process's route test (production builds it once per process)
    vocab, env = _env(corpus)
    held = _held_out(vocab, pos_gt)
    noun_pool = sorted(w for w in held if held[w] and w in env.pos)
    out = {"seed": seed, "held_word_phrase": HELD_WORD_PHRASE}
    # INPUT PROVENANCE: the frame environment reads `corpus`; the learned lexicon (get_lexicon) always reads
    # L._DEFAULT_CORPUS and ignores --corpus. Record both; score() refuses to pool seeds whose inputs differ.
    out["corpus_env_path"], out["corpus_lexicon_path"] = corpus, L._DEFAULT_CORPUS
    out["corpus_env_sha256"], out["corpus_env_bytes"] = _file_sha256(corpus)
    out["corpus_lexicon_sha256"], out["corpus_lexicon_bytes"] = _file_sha256(L._DEFAULT_CORPUS)

    # R1: route ON via env var, fresh organ, never touches .referent_lexicon.
    os.environ["BRAIN_LEARNED_REFERENT_LEXICON"] = "1"
    org_on = D6.MultiReferentWMOrgan(seed=seed)
    j = org_on.judge(HELD_WORD_PHRASE)
    out["r1_in_scope"] = j is not None
    out["r1_n_referents"] = (j or {}).get("n_referents")
    out["r1_input_order"] = (j or {}).get("input_order")
    out["r1_pass"] = bool(j is not None and j.get("n_referents") == 2 and "owl" in (j.get("input_order") or []))

    # R2: route OFF (env var unset, the production default), same process, SAME already-built singleton.
    os.environ.pop("BRAIN_LEARNED_REFERENT_LEXICON", None)
    org_off = D6.MultiReferentWMOrgan(seed=seed)
    j2 = org_off.judge(HELD_WORD_PHRASE)
    out["r2_in_scope"] = j2 is not None
    out["r2_pass"] = not (j2 is not None and "owl" in (j2.get("input_order") or []))

    # R3 / R4: population battery over held-out noun PAIRS, routed through the env var (not attribute injection).
    if len(noun_pool) < 2 * n_pairs:
        n_pairs = max(2, len(noun_pool) // 2)
    trials = []
    q = "who are we talking about?"
    for cond, flag_on, lesion_on in (("intact", True, False), ("lesion", True, True), ("hand", False, False)):
        os.environ["BRAIN_LEARNED_REFERENT_LEXICON"] = "1" if flag_on else "0"
        if lesion_on:
            os.environ["BRAIN_LEARNED_REFERENT_LESION"] = "1"
        else:
            os.environ.pop("BRAIN_LEARNED_REFERENT_LESION", None)
        rec_rate = []
        cond_rng = np.random.default_rng(seed + 991)
        for _ in range(n_pairs):
            a, b = [str(x) for x in cond_rng.choice(noun_pool, 2, replace=False)]
            org = D6.MultiReferentWMOrgan(seed=seed)   # fresh organ each trial -- no cross-trial state leak
            j1 = org.judge(f"the {a} and the {b} walked in")
            j2 = org.judge(q)
            rec = set(((j2 or {}).get("recovered") or {}).values())
            rec_rate.append(bool(j2 is not None and {a, b} <= rec))
        trials.append({"cond": cond, "recovered_both_rate": float(np.mean(rec_rate)), "n_pairs": n_pairs})
    _clean_env()
    out["population"] = trials
    intact_rate = next(t["recovered_both_rate"] for t in trials if t["cond"] == "intact")
    lesion_rate = next(t["recovered_both_rate"] for t in trials if t["cond"] == "lesion")
    hand_rate = next(t["recovered_both_rate"] for t in trials if t["cond"] == "hand")
    out["r3_recovered_both_rate"] = intact_rate
    out["r3_pass"] = intact_rate >= GATE["r3_recover_min"]
    out["r4_lesion_recovered_both_rate"] = lesion_rate
    out["hand_baseline_recovered_both_rate"] = hand_rate   # report-only: the pre-existing (flag-off) route
    out["r4_lever_moved"], out["r4_pass"] = _r4_gate(intact_rate, lesion_rate, hand_rate)
    out["r5_lexicon_built"] = L._LEXICON is not None
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def score(src):
    per = {}
    for p in sorted(glob.glob(os.path.join(src, "*.json"))):
        if p.endswith(".prov.json") or os.path.basename(p) == "verdict.json":
            continue
        d = json.load(open(p))
        per[d["seed"]] = d
    M = [per[s] for s in SEEDS6 if s in per]
    complete = len(M) == 6
    ev = {}
    if M:
        ev["R1_route_on_all_seeds"] = (int(sum(r["r1_pass"] for r in M)), all(r["r1_pass"] for r in M))
        ev["R2_route_off_all_seeds"] = (int(sum(r["r2_pass"] for r in M)), all(r["r2_pass"] for r in M))
        rates = [r["r3_recovered_both_rate"] for r in M]
        ev["R3_population_recover_mean"] = (float(np.mean(rates)), bool(rates) and min(rates) >= GATE["r3_recover_min"])
        lrates = [r["r4_lesion_recovered_both_rate"] for r in M]
        lever_ok = all(r.get("r4_lever_moved") for r in M)
        ev["R4_lesion_recover_mean"] = (float(np.mean(lrates)),
                                         bool(lrates) and max(lrates) <= GATE["r4_lesion_max"] and lever_ok)
    # R5 is PRE-REGISTERED as report-only: R1 passing already implies a built lexicon, so R5 can never
    # independently fail and must never sit inside `passed` (it belongs beside `ev`, not inside it).
    report_only = {}
    if M:
        report_only["R5_singleton_built_all_seeds"] = (int(sum(r["r5_lexicon_built"] for r in M)),
                                                        all(r["r5_lexicon_built"] for r in M))
    passed = bool(M) and all(bool(v[1]) for v in ev.values())
    # One input for all seeds, or no verdict: a seed without recorded input hashes, or two distinct inputs, is
    # MIXED-INPUT (never GO), whatever the gates read.
    inputs = sorted({(r.get("corpus_env_sha256"), r.get("corpus_lexicon_sha256")) for r in M}, key=str)
    one_input = len(inputs) == 1 and None not in inputs[0]
    verdict = ("INCOMPLETE" if not complete else "MIXED-INPUT" if not one_input
               else ("GO" if passed else "NO-GO"))
    return {"verdict": verdict, "complete_6seed": complete, "gate": GATE, "evidence": ev,
            "inputs": [list(i) for i in inputs], "one_input": one_input,
            "report_only": report_only,
            "seeds": sorted(per), "per_seed": {s: {k: per[s].get(k) for k in
                ("r1_pass", "r2_pass", "r3_recovered_both_rate", "r4_lesion_recovered_both_rate", "r4_lever_moved",
                 "hand_baseline_recovered_both_rate")} for s in sorted(per)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--json", default="")
    ap.add_argument("--score", default="")
    a = ap.parse_args()
    if a.score:
        out = score(a.score)
        print(json.dumps(out, indent=1, default=str))
        os.makedirs(a.score, exist_ok=True)
        json.dump(out, open(os.path.join(a.score, "verdict.json"), "w"), indent=1, default=str)
        return
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    fx = json.load(open(FIXTURE))
    r = run_seed(a.seed, corpus, fx["pos"], a.n_pairs)
    print(f"[seed {r['seed']}] R1(on)={r['r1_pass']} R2(off)={r['r2_pass']} "
          f"R3(recover)={r['r3_recovered_both_rate']:.2f} R4(lesion)={r['r4_lesion_recovered_both_rate']:.2f} "
          f"hand={r['hand_baseline_recovered_both_rate']:.2f} ({r['elapsed_s']}s)", flush=True)
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(r, open(dst, "w"), indent=1, default=str)
        print("wrote", dst)


if __name__ == "__main__":
    main()
