"""Semantic-recall SCAFFOLD RETIREMENT de-risk: wire the GO'd SPIKING unbind + SPIKING cleanup into the
production bulk-knowledge (LTM) recall path AT PRODUCTION SCALE, and measure whether it holds.

WHY (the live residual). `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `semantic-recall` row (re-verified
2026-09-16) is `scaffold_retired: NO`: for any recall that falls through the small conversation buffer to the
cortical LTM (`ShardedPhasorStore` -> `RFPhasorComposer` shards -- i.e. essentially all bulk-knowledge semantic
recall over the shipped `wikidata_100k` bundle, 78,857 facts / D=128 / 395 shards / vocab 23,914), BOTH the
UNBIND arithmetic (`local_reciprocal_unbind=False` default -> host `np.conj(role)`) and the CLEANUP SELECTION
(`enable_spiking_cleanup=False` default -> host `np.argmax` over the matched-filter membrane) still run on the
HOST. Spiking replacements for both EXIST and are individually GO -- but only at TOY scale:
  * `local_reciprocal_unbind` -- research/findings/2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md
    (4 seeds x 3 dims, 5-10 facts) + tests/test_rf_phasor_composer.py (3 seeds, D=96). Documented BIT-FOR-BIT
    identical to the host-conj path (a local per-synapse quadrature flip == conj for a unit phasor), so this flag
    cannot change an answer -- it is the neuromorphic-port STRUCTURE property. This runner checks that
    byte-identity survives at prodscale (the UNBIND-ONLY arm) but the load-bearing anti-cheats target the cleanup.
  * `enable_spiking_cleanup` -- research/findings/2026-06-18-production-fully-brain-based-flag-flips-GO.md
    (6-seed 42/43/44/100/101/102 GO, 16-word vocab / 5 facts, UNSHARDED) +
    research/findings/2026-06-20-burndown-1-onebrain-spiking-cleanup.md (3 toy facts). The Izhikevich-WTA
    matched-filter SELECT that replaces host argmax. THIS is the answer-affecting mechanism and the crosstalk
    risk: its competition is over the FULL vocab (23,914 concepts at D=128), a regime never tested.

NEITHER flag has ever been threaded through a `ShardedPhasorStore` (grep of every construction site: zero hits),
and neither's validation covers this row's own production scale. This runner closes exactly that gap: it BUILDS
a `ShardedPhasorStore` from the REAL wikidata_100k facts with `composer_kwargs={local_reciprocal_unbind:True,
enable_spiking_cleanup:True}` (the wiring under test -- ZERO source change; `ShardedPhasorStore.__init__` already
threads `**composer_kwargs` to every shard, verified by an assertion below) and measures 6-seed ANSWER PARITY
vs the shipped host path on `query_patient`/`query_agent`/`ask_yes_no`, plus no-confab MOAT-abstain preservation.

BRAIN-BASED-ONLY NOTE (docs/... standing standard): flags-ON routes recall through the on-bridge matched filter
(complex-synapse matvec) + the Izhikevich spiking WTA (argmax-over-FIRING). The bulk teacher-LOAD uses the
closed-form `encode_fast` (a declared bulk-load shortcut, recall-identical to the neural resonate bind; the
COGNITION -- unbind + cleanup -- stays neural), so this de-risks the RECALL op the ledger row is about.

ARMS (all on ONE built store; storage is flag-independent so composites are byte-identical across arms):
  HOST         local_reciprocal_unbind=False enable_spiking_cleanup=False  (the shipped default = the baseline)
  UNBIND_ONLY  local_reciprocal_unbind=True  enable_spiking_cleanup=False  (must be BYTE-IDENTICAL to HOST)
  SPIKING      local_reciprocal_unbind=True  enable_spiking_cleanup=True   (the wired production-scale path)
  LESION       SPIKING + `_spiking_cleanup` OUTPUT corrupted (returns a deterministically-wrong concept). If the
               query answer follows this, the spiking selection is LOAD-BEARING, not a passthrough hiding a host
               argmax. Anti-cheat (a).
  SCRAMBLE     SPIKING + the recovered phasor fed to `_spiking_cleanup` is dimension-permuted, so the matched
               filter correlates a scrambled cue -> the membrane carries no coherent match. A scrambled
               matched-filter membrane must NOT recover the right answer. Anti-cheat (b).

PRE-REGISTERED GO GATE (a NO-GO -- spiking-cleanup crosstalk/false-positives that appear only at prodscale
D=128/vocab=23,914 -- is an ACCEPTABLE, BANKABLE outcome; this measures it honestly, it does not force a GO):
  GO  <=>  ALL of:
    (1) PARITY:   spiking_recall >= host_recall - 0.02  on >= 5/6 seeds, AND
                  combined spiking-vs-host answer agreement (patient+agent+yesno) >= 0.98 on those seeds.
    (2) UNBIND:   UNBIND_ONLY answers are byte-identical to HOST on ALL 6 seeds (the documented bit-identity).
    (3) MOAT:     spiking moat-confab count == host moat-confab count on ALL 6 seeds (no NEW confabulation;
                  the no-confab moat still abstains where it should).
    (4) ANTI-CHEATS (ALL 6 seeds): LESION drops spiking-vs-host agreement by >= 0.5 (load-bearing), AND
                  SCRAMBLE drops recall by >= 0.5 vs the clean spiking arm (membrane load-bearing).

⚠️ DO NOT run the decisive 6-seed prodscale sweep here casually: flags-ON cleanup is O(vocab)=~24k spiking ops
per unbind and is GPU-bound. Queue it on the serialized gpu_queue (command printed at the end). The `--smoke`
mode (tiny compact-vocab CPU run) only proves it imports / constructs / both arms run / fields populate.

Run (TINY CPU smoke -- plumbing only, NOT decisive):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._semantic_recall_spiking_unbind_cleanup_prodscale_derisk \
      --n-facts 50 --seeds 42 --smoke

Run (DECISIVE prodscale 6-seed -- QUEUE ON GPU, do not run inline):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._semantic_recall_spiking_unbind_cleanup_prodscale_derisk \
      --full-vocab --n-facts 0 --n-shards 395 --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_semantic_recall_spiking_prodscale_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

BUNDLE_DEFAULT = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k"
SPIKING_KWARGS = {"local_reciprocal_unbind": True, "enable_spiking_cleanup": True}

# GO-gate constants (pre-registered above).
RECALL_MARGIN = 0.02      # spiking_recall >= host_recall - this
AGREEMENT_FLOOR = 0.98    # combined spiking-vs-host answer agreement
ANTICHEAT_DROP = 0.5      # min drop a lesion/scramble must cause to prove load-bearing
GO_SEEDS_MIN = 5          # >= 5/6 seeds must pass parity


# --------------------------------------------------------------------------------------------------------------
# bundle / ground truth
# --------------------------------------------------------------------------------------------------------------
def _load_bundle(bundle):
    """manifest + facts.json (unwrapped to a flat fact-dict list). facts.json shape is [{shard, fact}, ...]."""
    with open(os.path.join(bundle, "manifest.json")) as f:
        mani = json.load(f)
    with open(os.path.join(bundle, "facts.json")) as f:
        raw = json.load(f)
    facts = [r["fact"] if isinstance(r, dict) and "fact" in r else r for r in raw]
    return mani, facts


def _str_facts(facts):
    """Only plain str-agent/action/patient facts (the fast closed-form encode path; excludes clause/attributed
    patients so agreement/recall compare a single decoded word)."""
    out = []
    for f in facts:
        a, act, p = f.get("agent"), f.get("action"), f.get("patient")
        if isinstance(a, str) and isinstance(act, str) and isinstance(p, str):
            out.append({"agent": a, "action": act, "patient": p, "polarity": f.get("polarity") or "AFFIRM"})
    return out


def _first_match_patient(facts):
    """(agent, action) -> first AFFIRM patient -- the flat first-match convention query_patient itself uses."""
    fm = {}
    for f in facts:
        k = (f["agent"], f["action"])
        if k not in fm and f["polarity"] == "AFFIRM":
            fm[k] = f["patient"]
    return fm


def _first_match_agent(facts):
    """(action, patient) -> first agent (informational gt for query_agent; parity vs HOST is the primary metric,
    since both arms fan out the same store in the same order)."""
    fm = {}
    for f in facts:
        k = (f["action"], f["patient"])
        if k not in fm and f["polarity"] == "AFFIRM":
            fm[k] = f["agent"]
    return fm


# --------------------------------------------------------------------------------------------------------------
# arms: flag toggling + anti-cheat wrappers (runner-local; NO source change)
# --------------------------------------------------------------------------------------------------------------
def _set_flags(store, unbind, cleanup):
    for sh in store.shards:
        sh.local_reciprocal_unbind = bool(unbind)
        sh.enable_spiking_cleanup = bool(cleanup)


def _assert_wired(store):
    """The composer_kwargs wiring must have reached EVERY shard (this IS the wiring under test)."""
    bad = [i for i, sh in enumerate(store.shards)
           if not (sh.local_reciprocal_unbind and sh.enable_spiking_cleanup)]
    if bad:
        raise AssertionError(
            "composer_kwargs did NOT thread the spiking flags to %d/%d shards (e.g. shard %s): the wiring is "
            "broken." % (len(bad), len(store.shards), bad[:5]))
    return True


def _make_lesion(orig):
    """Anti-cheat (a) LESION: return a deterministically-WRONG concept instead of the real spiking winner. If the
    query answer follows this corruption, the answer is genuinely produced by `_spiking_cleanup` (load-bearing),
    not by a host argmax silently running underneath (a passthrough). `words` is always non-None here (`_cleanup`
    passes `self.words`/`self.pol_words`)."""
    def wrapped(rec_phases, words):
        true = orig(rec_phases, words)
        for w in words:
            if w != true:
                return w        # a wrong concept -> matching unbinds fail / the answer word diverges from HOST
        return true             # V==1 (nothing else to pick) -- cannot corrupt; harmless
    return wrapped


def _make_scramble(orig, rng):
    """Anti-cheat (b) SCRAMBLE: dimension-permute the recovered phasor before the matched filter, so the membrane
    correlates a scrambled cue and carries no coherent match. A scrambled matched-filter membrane must not
    recover the right answer."""
    def wrapped(rec_phases, words):
        rp = np.asarray(rec_phases)
        return orig(rp[rng.permutation(rp.shape[0])], words)
    return wrapped


def _patch(store, factory):
    """Shadow every shard's `_spiking_cleanup` with `factory(orig_bound_method)`; return a restore token."""
    saved = []
    for sh in store.shards:
        orig = sh._spiking_cleanup            # the real bound method (bound to this shard)
        sh._spiking_cleanup = factory(orig)   # instance attr shadows the class method (plain fn, no self)
        saved.append(sh)
    return saved


def _unpatch(saved):
    for sh in saved:
        sh.__dict__.pop("_spiking_cleanup", None)   # remove the instance shadow -> class method restored


# --------------------------------------------------------------------------------------------------------------
# probe battery
# --------------------------------------------------------------------------------------------------------------
def _sample_probes(str_facts, fm_p, fm_a, vocab, counts, seed):
    rng = np.random.default_rng(seed + 7)
    # query_patient probes: (agent, action, gt_patient)
    pk = list(fm_p.keys())
    pi = rng.choice(len(pk), size=min(counts["patient"], len(pk)), replace=False)
    patient_probes = [(pk[i][0], pk[i][1], fm_p[pk[i]]) for i in pi]
    # query_agent probes: (action, patient, gt_agent) drawn from UNIQUE (action,patient) pairs where possible
    ak = list(fm_a.keys())
    ai = rng.choice(len(ak), size=min(counts["agent"], len(ak)), replace=False)
    agent_probes = [(ak[i][0], ak[i][1], fm_a[ak[i]]) for i in ai]
    # ask_yes_no probes: a real fact -> expect 'yes'; same cue + a DIFFERENT real patient -> expect 'no'
    yesno_probes = []
    vocab_list = list(vocab)
    for i in pi[: counts["yesno"]]:
        a, act = pk[i]
        gt = fm_p[(a, act)]
        yesno_probes.append((a, act, gt, "yes"))
        wrong = None
        for _ in range(8):
            w = vocab_list[int(rng.integers(0, len(vocab_list)))]
            if w != gt:
                wrong = w
                break
        if wrong is not None:
            yesno_probes.append((a, act, wrong, "no"))
    # moat cues: unknown agents (synthetic, guaranteed not in vocab) + a known agent with an unknown relation
    vocab_set = set(vocab)
    n_moat = counts["moat"]
    unknown_agents = [f"zzz_unknown_entity_{j}_xq" for j in range(n_moat // 2)]
    assert not (set(unknown_agents) & vocab_set), "unknown-agent cue collides with real vocab"
    real_actions = sorted({f["action"] for f in str_facts})
    moat_patient = [(ua, real_actions[int(rng.integers(0, len(real_actions)))]) for ua in unknown_agents]
    known_agent = pk[int(rng.integers(0, len(pk)))][0]
    moat_patient += [(known_agent, "zzz_unknown_relation_never_taught")
                     for _ in range(n_moat - len(moat_patient))]
    # moat for query_agent: unknown patient
    unknown_pats = [f"zzz_unknown_patient_{j}_xq" for j in range(n_moat // 2)]
    moat_agent = [(real_actions[int(rng.integers(0, len(real_actions)))], up) for up in unknown_pats]
    return patient_probes, agent_probes, yesno_probes, moat_patient, moat_agent


def _run_arm(store, patient_probes, agent_probes, yesno_probes, moat_patient, moat_agent, warm=True):
    """Return every arm's raw answers + timing. Deterministic given the store state."""
    if warm:  # one warm pass builds the bridge/izh caches so latency is steady-state
        for (a, act, _g) in patient_probes[:5]:
            store.query_patient(a, act)
    t0 = time.perf_counter()
    pat = [store.query_patient(a, act) for (a, act, _g) in patient_probes]
    ag = [store.query_agent(act, p) for (act, p, _g) in agent_probes]
    yn = [store.ask_yes_no(a, act, p) for (a, act, p, _e) in yesno_probes]
    elapsed = time.perf_counter() - t0
    moat_p = [store.query_patient(a, act) for (a, act) in moat_patient]
    moat_a = [store.query_agent(act, p) for (act, p) in moat_agent]
    n_q = len(pat) + len(ag) + len(yn)
    return {
        "patient": pat, "agent": ag, "yesno": yn, "moat_p": moat_p, "moat_a": moat_a,
        "lat_ms_per_query": round(1000.0 * elapsed / n_q, 3) if n_q else None,
    }


def _agree(x, y):
    n = min(len(x), len(y))
    return (sum(1 for i in range(n) if x[i] == y[i]) / n) if n else None


def _recall(answers, gts):
    n = min(len(answers), len(gts))
    return (sum(1 for i in range(n) if answers[i] == gts[i]) / n) if n else None


def _moat_confab(moat_p, moat_a):
    """Confabulation = a non-abstain answer on a cue that MUST abstain (unknown entity/relation)."""
    return sum(1 for a in moat_p if a is not None) + sum(1 for a in moat_a if a is not None)


def _combined_agreement(host, spk):
    n = len(host["patient"]) + len(host["agent"]) + len(host["yesno"])
    ok = (sum(1 for i in range(len(host["patient"])) if host["patient"][i] == spk["patient"][i])
          + sum(1 for i in range(len(host["agent"])) if host["agent"][i] == spk["agent"][i])
          + sum(1 for i in range(len(host["yesno"])) if host["yesno"][i] == spk["yesno"][i]))
    return (ok / n) if n else None


# --------------------------------------------------------------------------------------------------------------
# per-seed
# --------------------------------------------------------------------------------------------------------------
def run_seed(str_facts, vocab, n_shards, seed, D, counts):
    from research.runners.tiered_fact_store import build_ltm_from_facts
    from tools.lab import attributable_to, void_if

    res = {"seed": seed}
    t_build = time.time()
    store = build_ltm_from_facts(str_facts, vocab=list(vocab), n_shards=n_shards, seed=seed, D=D,
                                 composer_kwargs=dict(SPIKING_KWARGS), fast=True)
    res["build_s"] = round(time.time() - t_build, 2)
    res["total_facts"] = store.total_facts()
    res["n_shards"] = store.n_shards
    res["wired_all_shards"] = _assert_wired(store)   # the flags reached every shard (the wiring under test)

    fm_p = _first_match_patient(str_facts)
    fm_a = _first_match_agent(str_facts)
    patient_probes, agent_probes, yesno_probes, moat_p, moat_a = _sample_probes(
        str_facts, fm_p, fm_a, vocab, counts, seed)
    res["n_probes"] = {"patient": len(patient_probes), "agent": len(agent_probes),
                       "yesno": len(yesno_probes), "moat_p": len(moat_p), "moat_a": len(moat_a)}

    gt_pat = [g for (_a, _v, g) in patient_probes]
    gt_ag = [g for (_a, _v, g) in agent_probes]
    exp_yn = [e for (_a, _v, _p, e) in yesno_probes]

    # HOST (baseline)
    _set_flags(store, unbind=False, cleanup=False)
    host = _run_arm(store, patient_probes, agent_probes, yesno_probes, moat_p, moat_a)
    # UNBIND_ONLY (documented byte-identical)
    _set_flags(store, unbind=True, cleanup=False)
    unb = _run_arm(store, patient_probes, agent_probes, yesno_probes, moat_p, moat_a)
    # SPIKING (the wired path)
    _set_flags(store, unbind=True, cleanup=True)
    spk = _run_arm(store, patient_probes, agent_probes, yesno_probes, moat_p, moat_a)
    # LESION (spiking output corrupted)
    saved = _patch(store, _make_lesion)
    les = _run_arm(store, patient_probes, agent_probes, yesno_probes, moat_p, moat_a, warm=False)
    _unpatch(saved)
    # SCRAMBLE (matched-filter input scrambled)
    rng = np.random.default_rng(seed + 4242)
    saved = _patch(store, lambda orig: _make_scramble(orig, rng))
    scr = _run_arm(store, patient_probes, agent_probes, yesno_probes, moat_p, moat_a, warm=False)
    _unpatch(saved)

    # --- metrics ---
    host_recall = _recall(host["patient"], gt_pat)
    spk_recall = _recall(spk["patient"], gt_pat)
    les_recall = _recall(les["patient"], gt_pat)
    scr_recall = _recall(scr["patient"], gt_pat)
    combined_agree = _combined_agreement(host, spk)
    unbind_identical = (unb["patient"] == host["patient"] and unb["agent"] == host["agent"]
                        and unb["yesno"] == host["yesno"])
    host_moat = _moat_confab(host["moat_p"], host["moat_a"])
    spk_moat = _moat_confab(spk["moat_p"], spk["moat_a"])
    # anti-cheat agreements (spiking-vs-host is ~1.0 when clean; lesion should collapse it)
    spk_vs_host = _combined_agreement(host, spk)
    les_vs_host = _combined_agreement(host, les)

    print("  [seed %s] host_recall=%s spiking_recall=%s combined_agree=%s unbind_identical=%s "
          "host_moat=%s spiking_moat=%s | lesion_recall=%s lesion_agree=%s scramble_recall=%s"
          % (seed, host_recall, spk_recall, combined_agree, unbind_identical, host_moat, spk_moat,
             les_recall, les_vs_host, scr_recall))

    # anti-cheat attributions (informational: fraction of the effect that is genuinely the spiking mechanism)
    attributable_to("lesion (spiking recall vs lesioned)", spk_recall or 0.0, les_recall or 0.0)
    attributable_to("scramble (spiking recall vs scrambled)", spk_recall or 0.0, scr_recall or 0.0)
    void_if(spk_recall is not None and spk_recall <= 0.0,
            "spiking arm recalled NOTHING at this scale -- parity is UNDEFINED, not a clean 0")

    # per-seed gate components
    recall_ok = (spk_recall is not None and host_recall is not None
                 and spk_recall >= host_recall - RECALL_MARGIN)
    agree_ok = (combined_agree is not None and combined_agree >= AGREEMENT_FLOOR)
    moat_ok = (spk_moat == host_moat)
    lesion_ok = (spk_vs_host is not None and les_vs_host is not None
                 and les_vs_host <= spk_vs_host - ANTICHEAT_DROP)
    scramble_ok = (spk_recall is not None and scr_recall is not None
                   and scr_recall <= spk_recall - ANTICHEAT_DROP)

    res.update({
        "host_recall": host_recall, "spiking_recall": spk_recall,
        "combined_agreement": combined_agree, "unbind_identical": bool(unbind_identical),
        "host_moat_confab": host_moat, "spiking_moat_confab": spk_moat,
        "lesion_recall": les_recall, "lesion_vs_host_agreement": les_vs_host,
        "spiking_vs_host_agreement": spk_vs_host, "scramble_recall": scr_recall,
        "yesno_accuracy_host": _recall(host["yesno"], exp_yn),
        "yesno_accuracy_spiking": _recall(spk["yesno"], exp_yn),
        "agent_agreement": _agree(host["agent"], spk["agent"]),
        "gt_agent_recall_spiking": _recall(spk["agent"], gt_ag),
        "lat_ms_per_query": {"host": host["lat_ms_per_query"], "spiking": spk["lat_ms_per_query"]},
        "gate": {"recall_ok": recall_ok, "agree_ok": agree_ok, "moat_ok": moat_ok,
                 "unbind_identical": bool(unbind_identical),
                 "lesion_ok": lesion_ok, "scramble_ok": scramble_ok},
        "seed_parity_go": bool(recall_ok and agree_ok and moat_ok and unbind_identical),
    })
    return res


# --------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=BUNDLE_DEFAULT, help="real wikidata_100k bundle dir (facts + manifest)")
    ap.add_argument("--n-facts", type=int, default=0,
                    help="use the first N str-patient bundle facts; 0 = ALL (78,857). Scale this up on GPU.")
    ap.add_argument("--n-shards", type=int, default=-1,
                    help="shards; -1 = the bundle manifest's n_shards (395); --smoke auto-picks a small value")
    ap.add_argument("--full-vocab", action="store_true",
                    help="use the bundle's FULL manifest vocab (23,914 -> the real cleanup-competition scale). "
                         "Off (smoke default) = compact vocab from the selected facts (fast).")
    ap.add_argument("--seeds", default="42", help="comma-separated seeds (decisive set: 42,43,44,100,101,102)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU plumbing check: compact vocab, small shards, few probes. NOT decisive.")
    ap.add_argument("--out", default=None, help="write the full result JSON here")
    a = ap.parse_args()

    t_start = time.time()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out = {"bundle": a.bundle, "seeds": seeds, "smoke": a.smoke, "full_vocab": a.full_vocab,
           "spiking_kwargs": SPIKING_KWARGS}

    if not os.path.isdir(a.bundle):
        out["error"] = f"bundle not found: {a.bundle}"
        print("ERROR:", out["error"])
        if a.out:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(out, open(a.out, "w"), indent=2, default=str)
        return 1

    mani, facts = _load_bundle(a.bundle)
    D = int(mani["D"])
    str_facts = _str_facts(facts)
    if a.n_facts and a.n_facts > 0:
        str_facts = str_facts[: a.n_facts]
    counts = ({"patient": 10, "agent": 6, "yesno": 6, "moat": 6} if a.smoke
              else {"patient": 120, "agent": 60, "yesno": 40, "moat": 40})

    if a.full_vocab:
        vocab = list(mani["vocab"])
    else:
        vs = set()
        for f in str_facts:
            vs.update((f["agent"], f["action"], f["patient"]))
        vocab = sorted(vs)

    from research.runners.tiered_fact_store import auto_n_shards
    if a.n_shards > 0:
        n_shards = a.n_shards
    elif a.smoke:
        n_shards = auto_n_shards(len(str_facts))
    else:
        n_shards = int(mani["n_shards"])

    out["config"] = {"D": D, "n_facts_used": len(str_facts), "vocab_size": len(vocab),
                     "n_shards": n_shards, "bundle_n_facts": mani["n_facts"],
                     "bundle_vocab": len(mani["vocab"])}
    print(f"[config] D={D} n_facts={len(str_facts)} vocab={len(vocab)} n_shards={n_shards} "
          f"seeds={seeds} smoke={a.smoke} full_vocab={a.full_vocab}", flush=True)

    error = None
    seed_results = []
    try:
        for s in seeds:
            print(f"\n=== seed {s} ===", flush=True)
            seed_results.append(run_seed(str_facts, vocab, n_shards, s, D, counts))
    except Exception as e:
        import traceback
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        out["error"] = error
        print("ERROR:", error)

    out["seed_results"] = seed_results

    # --- verdict ---
    from tools.verdict import Verdict
    v = Verdict("semantic-recall spiking unbind+cleanup prodscale de-risk (ledger: semantic-recall)")
    if error is None and seed_results:
        n_parity = sum(1 for r in seed_results if r["seed_parity_go"])
        all_moat = all(r["gate"]["moat_ok"] for r in seed_results)
        all_unbind = all(r["unbind_identical"] for r in seed_results)
        all_lesion = all(r["gate"]["lesion_ok"] for r in seed_results)
        all_scramble = all(r["gate"]["scramble_ok"] for r in seed_results)
        n_seeds = len(seed_results)
        parity_min = GO_SEEDS_MIN if n_seeds >= 6 else n_seeds   # smoke (1 seed) requires that seed to pass

        v.require(f"PARITY: >= {parity_min}/{n_seeds} seeds meet spiking_recall>=host_recall-{RECALL_MARGIN} "
                  f"AND combined agreement>={AGREEMENT_FLOOR}", n_parity, expect=lambda x: x >= parity_min)
        v.require("UNBIND byte-identity: UNBIND_ONLY == HOST on all seeds (documented bit-identity)",
                  all_unbind, expect=True)
        v.require("MOAT preserved: spiking moat-confab == host moat-confab on all seeds", all_moat, expect=True)
        v.require("ANTI-CHEAT lesion: corrupting the spiking output collapses agreement (load-bearing) all seeds",
                  all_lesion, expect=True)
        v.require("ANTI-CHEAT scramble: a scrambled matched-filter membrane fails to recall, all seeds",
                  all_scramble, expect=True)
        go = bool(n_parity >= parity_min and all_moat and all_unbind and all_lesion and all_scramble)
        out["aggregate"] = {"n_parity_go": n_parity, "n_seeds": n_seeds, "all_moat_ok": all_moat,
                            "all_unbind_identical": all_unbind, "all_lesion_ok": all_lesion,
                            "all_scramble_ok": all_scramble}
    else:
        go = False
        v.require("run completed with >=1 seed result", bool(seed_results and not error), expect=True)

    decided = v.decide(go=go)
    out.update(decided)
    out["elapsed_s"] = round(time.time() - t_start, 2)

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print("\nwrote", a.out)
    print(f"\n===== VERDICT: {out.get('status')} (go={out.get('go')}) "
          f"smoke={a.smoke} elapsed={out['elapsed_s']}s =====")
    if a.smoke:
        print("NOTE: --smoke is a plumbing check ONLY (tiny compact-vocab CPU). The verdict is NOT decisive; "
              "queue the prodscale 6-seed on the GPU (command in the module docstring).")
    return 0 if out.get("go") else 1


if __name__ == "__main__":
    raise SystemExit(main())
