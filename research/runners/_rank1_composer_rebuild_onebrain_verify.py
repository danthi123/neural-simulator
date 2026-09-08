"""RANK-1 scaffold-retirement de-risk: rebuild the DEPLOYED bundle's REAL facts OFF the host closed-form 'rf'
composer ONTO the spiking 'onebrain' composer, and verify GO (parity + recall + no-confab moat) against the 'rf'
baseline -- ON THE REAL DEPLOYED BUNDLE, not synthetic facts.

WHY (backlog rank-1, the owner's #1 arc): the live LTM/recall composer is pinned to `composer_kind='rf'` in the
deployed bundle's manifest (`bridges/developed/scale787/day_33/brain.json` -> "rf"), so every live recall/store/
abstain runs on the HOST closed-form FHRR algebra -- the spiking unbind + NEF/Izhikevich WTA cleanup that
`OneBrainComposer` enables BY DEFAULT never touch a live answer. Brain-based-only: the composing must be the
spiking substrate's job. The retrieval-latency sub-blocker is resolved (DG-CA3 sharded spiking retrieval,
`BRAIN_FACT_SHARD_RETRIEVAL`, flipped default-on 2026-09-05, GO 6/6), so the literal rebuild is now unblocked.

WHAT'S NEW vs the wire-in verify (2026-09-05-onebrain-fact-shard-wirein-production-composer.md): that runner drove
SYNTHETIC, deliberately UNIQUE-(agent, action) facts. THIS runner rebuilds the ACTUAL 404-fact deployed bundle,
which is heavily AMBIGUOUS (only ~190 distinct (agent, action) cues; 113 cues carry >1 patient) -- exactly the
"degenerate same-(agent, action) different-patient" regime the OneBrainComposer.ask_yes_no docstring flags as
"outside the production regime". So this is the first parity check of the spiking composer vs the host 'rf' on the
real ambiguous store.

METHOD (one seed = the bundle's own seed 42; cost-routing: numpy/CPU, matching the wire-in de-risk):
  * rf BASELINE: `load_developed_brain(BUNDLE)` -- the EXACT deployed brain (manifest composer_kind='rf',
    kb_composites fast-path = byte-identical to production).
  * onebrain REBUILD: build `BrainConversationalAgent(composer_kind='onebrain', onebrain_k_max=n_facts+16)` over
    the SAME grounded codes + seed, then re-store every real fact on-substrate (the spiking store). Fact-shard
    retrieval ON (the production default) so the O(k_max) scan is tractable.
  * Compare, over the distinct cues, for query_patient / query_agent / ask_yes_no:
      - STRICT parity: onebrain answer == rf answer.
      - VALID recall: the answer is a member of the cue's real answer-set (the honest metric under ambiguity --
        "returns A stored fact for the cue", since many cues have several valid answers). Measured for BOTH
        composers; GO needs onebrain's valid recall >= rf's (the spiking store recalls at least as well).
      - MOAT: out-of-store cues abstain (query_* -> None, ask_yes_no -> 'unknown'); 0 new confabulation, and
        onebrain abstains wherever rf abstains.
  * The ambiguous vs unambiguous cues are reported separately, so any onebrain under-recall is attributed to the
    ambiguous regime (a moat-SAFE abstain, never a confabulation) rather than papered over.

ANTI-CHEATS: parity-vs-rf is a HARD gate (content-addressable -- the routing key is the cue WORD's code, the
answer is read off the spiking decode, never an answer id); a SCRAMBLE control (shuffle the stored blocks' codes ->
recall must collapse) proves the recall is attributable to the learned binding, not the harness; the moat is an
explicit out-of-store probe.

Determinism: cfg.seed discipline (BrainConversationalAgent seeds the substrate from `seed`). SIM_BACKEND defaults
to numpy. A rebuilt onebrain bundle is written to --out-bundle so the literal artifact exists for review.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
# fact-shard sublinear retrieval = the production default since 2026-09-05; ON here so the 404-block onebrain scan
# is tractable (answer-identical to the full scan -- proven parity in the wire-in finding).
os.environ.setdefault("BRAIN_FACT_SHARD_RETRIEVAL", "1")
# the DECISIVE parity is on the BARE OneBrainComposer (its local_reciprocal_unbind + spiking-cleanup defaults are
# the brain-based reclaim); pool1-merge is a separately-verified follow-on (wire-in pool1_reachability, N=60).
os.environ.setdefault("BRAIN_COMPOSER_MERGE", "0")

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.developed_brain_io import (  # noqa: E402
    _read_manifest, _load_codes_npz, _load_facts_json, load_developed_brain,
)
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

try:
    from tools.verdict import Verdict
except Exception:
    Verdict = None
from tools.lab import attributable_to  # scramble-control attribution (anti-cheat: whose is the recall?)

DEFAULT_BUNDLE = "/home/dant123/Projects/sim/bridges/developed/scale787/day_33"


def _flat_facts(facts):
    """Keep the flat-SVO facts (agent, action, patient all strings). The deployed bundle is 100% flat SVO; any
    non-flat fact (clause/attributed) is reported + skipped so the parity is like-for-like."""
    flat, skipped = [], 0
    for f in facts:
        a, v, p = f.get("agent"), f.get("action"), f.get("patient")
        if isinstance(a, str) and isinstance(v, str) and isinstance(p, str):
            flat.append((a, v, p, f.get("polarity")))
        else:
            skipped += 1
    return flat, skipped


def build_answer_maps(flat):
    patient_set, agent_set, svo = {}, {}, set()
    for a, v, p, _pol in flat:
        patient_set.setdefault((a, v), set()).add(p)
        agent_set.setdefault((v, p), set()).add(a)
        svo.add((a, v, p))
    return patient_set, agent_set, svo


def build_onebrain(seed, vocab, codes, flat, k_max):
    """Build the production onebrain agent and re-store the real facts on-substrate (the spiking store)."""
    concepts = {w: None for w in vocab}
    agent = BrainConversationalAgent(seed=seed, concepts=concepts,
                                     grounded_codes=codes if codes else None,
                                     composer_kind="onebrain", onebrain_k_max=k_max,
                                     enable_neural_render=False, defer_parser=True)
    t0 = time.time()
    for a, v, p, pol in flat:
        agent.composer.store(a, v, p, polarity=pol)
    store_s = time.time() - t0
    return agent, store_s


def _scramble_patients(flat, seed):
    """The SCRAMBLE control (anti-cheat): permute the PATIENT assignments across facts, breaking the (cue ->
    answer) binding while leaving the codebook (concept codes) intact. A composer that stores THIS and is then
    queried against the TRUE answer-set must collapse to ~chance -- proving recall is the LEARNED binding, not the
    harness. (A global relabel of ALL codes is an isomorphism and preserves recall, so it is NOT a valid control.)"""
    rng = np.random.default_rng(seed * 131 + 7)
    pats = [p for _a, _v, p, _pol in flat]
    perm = rng.permutation(len(pats))
    return [(a, v, pats[perm[i]], pol) for i, (a, v, p, pol) in enumerate(flat)]


def write_onebrain_bundle(out_bundle, src_bundle, manifest, flat):
    """Write the LITERAL rf->onebrain rebuilt bundle: the SAME grounded codes + facts as the source, a manifest
    with composer_kind='onebrain', and NO kb_composites.npz (onebrain composites live on-substrate, so a reload
    re-stores every fact on the spiking composer). This IS the file-level rebuild the backlog's rank-1 names."""
    import shutil
    os.makedirs(out_bundle, exist_ok=True)
    # copy the codes + facts verbatim (byte-for-byte the developed brain's own learned codes + knowledge)
    for fn in ("grounded_codes.npz", "facts.json"):
        src = os.path.join(src_bundle, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_bundle, fn))
    rb = dict(manifest)
    rb["composer_kind"] = "onebrain"
    rb.pop("n_kb_composites", None)            # onebrain: composites on-substrate, not persisted
    files = dict(rb.get("files") or {})
    files.pop("kb_composites", None)
    rb["files"] = files
    rb["metadata"] = dict(rb.get("metadata") or {})
    rb["metadata"]["rebuilt_from"] = src_bundle
    rb["metadata"]["rebuild_note"] = ("rank-1 rf->onebrain literal rebuild (de-risk in isolation): same codes+facts, "
                                      "manifest flipped to onebrain, kb_composites dropped; reload re-stores on the "
                                      "spiking substrate. NOTE: load_developed_brain must thread onebrain_k_max>=n_facts "
                                      "for a >32-fact onebrain bundle to reload without truncation (default k_max=32).")
    with open(os.path.join(out_bundle, "brain.json"), "w", encoding="utf-8") as fh:
        json.dump(rb, fh, indent=2, ensure_ascii=False)
    return {"path": out_bundle, "composer_kind": "onebrain", "n_facts": rb.get("n_facts"),
            "files": sorted(os.listdir(out_bundle))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--out", default="research/findings/raw/_rank1_composer_rebuild/verify.json")
    ap.add_argument("--out-bundle", default=None,
                    help="write the rebuilt onebrain bundle here (default: alongside --out)")
    ap.add_argument("--moat-n", type=int, default=80, help="out-of-store cue probes")
    ap.add_argument("--limit-facts", type=int, default=0, help="cap facts (smoke); 0 = all")
    args = ap.parse_args()

    manifest = _read_manifest(args.bundle)
    if manifest is None:
        raise SystemExit(f"no manifest at {args.bundle}")
    seed = int(manifest.get("seed", 42))
    D = int(manifest.get("D", 128))
    codes = _load_codes_npz(args.bundle)
    facts_json = _load_facts_json(args.bundle)
    flat, skipped = _flat_facts(facts_json)
    if args.limit_facts:
        flat = flat[: args.limit_facts]
    manifest_ck = manifest.get("composer_kind")

    # vocab = union of grounded codes + every fact word (so both composers can encode every cue)
    vocab_set = set(codes.keys())
    for a, v, p, _ in flat:
        vocab_set.update((a, v, p))
    vocab = sorted(vocab_set)

    patient_set, agent_set, svo = build_answer_maps(flat)
    distinct_qp = sorted(patient_set.keys())
    distinct_qa = sorted(agent_set.keys())
    distinct_yn = sorted(svo)
    ambiguous_qp = {cue for cue, ps in patient_set.items() if len(ps) > 1}

    print(f"[rank1] bundle={args.bundle} manifest_composer_kind={manifest_ck!r} "
          f"n_facts={len(flat)} (skipped_non_flat={skipped}) vocab={len(vocab)} D={D} seed={seed}", flush=True)
    print(f"[rank1] distinct cues: qp={len(distinct_qp)} (ambiguous={len(ambiguous_qp)}) "
          f"qa={len(distinct_qa)} yesno_svo={len(distinct_yn)}", flush=True)

    # ---- rf BASELINE (the exact deployed brain) ----
    t0 = time.time()
    agent_rf, _m = load_developed_brain(args.bundle)
    rf_load_s = time.time() - t0
    rf_ck = getattr(type(agent_rf.composer), "__name__", "?")
    print(f"[rank1] rf baseline loaded in {rf_load_s:.1f}s ({rf_ck})", flush=True)

    # ---- onebrain REBUILD (the spiking composer) ----
    t0 = time.time()
    agent_ob, ob_store_s = build_onebrain(seed, vocab, codes, flat, k_max=len(flat) + 16)
    ob_build_s = time.time() - t0
    ob_ck = getattr(type(agent_ob.composer), "__name__", "?")
    print(f"[rank1] onebrain rebuilt in {ob_build_s:.1f}s (store={ob_store_s:.1f}s) ({ob_ck}); "
          f"fact_shard={os.environ.get('BRAIN_FACT_SHARD_RETRIEVAL')}", flush=True)

    rf, ob = agent_rf.composer, agent_ob.composer

    # ================= PARITY + RECALL =================
    def cmp_query_patient():
        strict = valid_rf = valid_ob = 0
        amb_strict = amb_valid_ob = 0
        mism = []
        for (a, v) in distinct_qp:
            r = rf.query_patient(a, v)
            o = ob.query_patient(a, v)
            ans = patient_set[(a, v)]
            vr = r in ans
            vo = o in ans
            valid_rf += vr
            valid_ob += vo
            st = (r == o)
            strict += st
            if (a, v) in ambiguous_qp:
                amb_strict += st
                amb_valid_ob += vo
            if not st and len(mism) < 40:
                mism.append({"cue": [a, v], "rf": r, "ob": o, "answers": sorted(ans),
                             "ambiguous": (a, v) in ambiguous_qp,
                             "rf_valid": vr, "ob_valid": vo})
        n = len(distinct_qp)
        na = len(ambiguous_qp)
        return {
            "n_cues": n, "n_ambiguous": na,
            "strict_parity": strict / n, "valid_recall_rf": valid_rf / n, "valid_recall_ob": valid_ob / n,
            "ambiguous_strict_parity": (amb_strict / na) if na else None,
            "ambiguous_valid_recall_ob": (amb_valid_ob / na) if na else None,
            "unambiguous_strict_parity": ((strict - amb_strict) / (n - na)) if (n - na) else None,
            "mismatches_sample": mism,
        }

    def cmp_query_agent():
        strict = valid_rf = valid_ob = 0
        mism = []
        for (v, p) in distinct_qa:
            r = rf.query_agent(v, p)
            o = ob.query_agent(v, p)
            ans = agent_set[(v, p)]
            valid_rf += r in ans
            valid_ob += o in ans
            st = (r == o)
            strict += st
            if not st and len(mism) < 40:
                mism.append({"cue": [v, p], "rf": r, "ob": o, "answers": sorted(ans)})
        n = len(distinct_qa)
        return {"n_cues": n, "strict_parity": strict / n,
                "valid_recall_rf": valid_rf / n, "valid_recall_ob": valid_ob / n,
                "mismatches_sample": mism}

    def cmp_ask_yes_no():
        # every stored SVO should answer 'yes' on rf; onebrain selects the first (a,v) block then checks patient,
        # so on an ambiguous cue whose first block has a different patient it abstains ('unknown') -- moat-SAFE.
        rf_yes = ob_yes = strict = 0
        ob_unknown_on_stored = ob_no_on_stored = 0
        unamb_ob_yes = unamb_n = 0
        mism = []
        for (a, v, p) in distinct_yn:
            r = rf.ask_yes_no(a, v, p)
            o = ob.ask_yes_no(a, v, p)
            rf_yes += (r == "yes")
            ob_yes += (o == "yes")
            strict += (r == o)
            if o == "unknown":
                ob_unknown_on_stored += 1
            elif o == "no":
                ob_no_on_stored += 1
            if (a, v) not in ambiguous_qp:
                unamb_n += 1
                unamb_ob_yes += (o == "yes")
            if r != o and len(mism) < 40:
                mism.append({"svo": [a, v, p], "rf": r, "ob": o, "ambiguous_av": (a, v) in ambiguous_qp})
        n = len(distinct_yn)
        return {"n_svo": n, "rf_yes_rate": rf_yes / n, "ob_yes_rate": ob_yes / n,
                "strict_parity": strict / n,
                "ob_unknown_on_stored": ob_unknown_on_stored, "ob_no_on_stored (CONFAB if>0)": ob_no_on_stored,
                "unambiguous_ob_yes_rate": (unamb_ob_yes / unamb_n) if unamb_n else None,
                "unambiguous_n": unamb_n, "mismatches_sample": mism}

    print("[rank1] comparing query_patient ...", flush=True)
    qp = cmp_query_patient()
    print(f"        qp strict_parity={qp['strict_parity']:.4f} "
          f"valid_recall rf={qp['valid_recall_rf']:.4f} ob={qp['valid_recall_ob']:.4f}", flush=True)
    print("[rank1] comparing query_agent ...", flush=True)
    qa = cmp_query_agent()
    print(f"        qa strict_parity={qa['strict_parity']:.4f} "
          f"valid_recall rf={qa['valid_recall_rf']:.4f} ob={qa['valid_recall_ob']:.4f}", flush=True)
    print("[rank1] comparing ask_yes_no ...", flush=True)
    yn = cmp_ask_yes_no()
    print(f"        yesno rf_yes={yn['rf_yes_rate']:.4f} ob_yes={yn['ob_yes_rate']:.4f} "
          f"strict={yn['strict_parity']:.4f} ob_no_on_stored={yn['ob_no_on_stored (CONFAB if>0)']}", flush=True)

    # ================= MOAT (out-of-store abstention) =================
    rng = np.random.default_rng(seed * 977 + 3)
    words = list(vocab)
    rf_confab = ob_confab = probes = ob_abstain = rf_abstain = 0
    for _ in range(args.moat_n * 20):
        if probes >= args.moat_n:
            break
        a = words[int(rng.integers(len(words)))]
        v = words[int(rng.integers(len(words)))]
        if (a, v) in patient_set:
            continue
        probes += 1
        r = rf.query_patient(a, v)
        o = ob.query_patient(a, v)
        rf_abstain += (r is None)
        ob_abstain += (o is None)
        rf_confab += (r is not None)
        ob_confab += (o is not None)
    moat = {"probes": probes, "rf_abstain": rf_abstain, "ob_abstain": ob_abstain,
            "rf_confab": rf_confab, "ob_confab": ob_confab}
    print(f"[rank1] moat: probes={probes} rf_abstain={rf_abstain} ob_abstain={ob_abstain} "
          f"ob_confab={ob_confab}", flush=True)

    # ================= SCRAMBLE control (anti-cheat) =================
    print("[rank1] scramble control (shuffled cue->answer binding) ...", flush=True)
    flat_scr = _scramble_patients(flat, seed)
    agent_scr, _ = build_onebrain(seed, vocab, codes, flat_scr, k_max=len(flat) + 16)
    scr = agent_scr.composer
    scr_valid = 0
    scr_n = min(120, len(distinct_qp))
    for (a, v) in distinct_qp[:scr_n]:
        o = scr.query_patient(a, v)
        scr_valid += o in patient_set[(a, v)]   # measured against the TRUE answer-set -> must collapse
    ob_same = sum(ob.query_patient(a, v) in patient_set[(a, v)] for (a, v) in distinct_qp[:scr_n]) / scr_n
    scr_recall = scr_valid / scr_n if scr_n else None
    # attribution: what fraction of the onebrain recall is NOT present in the code-scrambled control (i.e. is
    # attributable to the LEARNED binding, not the harness). ~1.0 = the recall is the substrate's, not a leak.
    recall_attribution = attributable_to("onebrain recall vs code-scramble", ob_same, scr_recall)
    scramble = {"n": scr_n, "valid_recall_scrambled": scr_recall,
                "valid_recall_ob_same_cues": ob_same, "recall_attribution": recall_attribution}
    print(f"        scramble valid_recall={scr_recall:.4f} vs ob={ob_same:.4f} "
          f"attribution={recall_attribution}", flush=True)

    # ================= write the rebuilt onebrain bundle (the literal artifact) =================
    out_bundle = args.out_bundle or os.path.join(os.path.dirname(args.out), "rebuilt_onebrain_bundle")
    os.makedirs(os.path.dirname(out_bundle) or ".", exist_ok=True)
    try:
        rebuilt = write_onebrain_bundle(out_bundle, args.bundle, manifest, flat)
    except Exception as e:
        rebuilt = {"path": out_bundle, "error": repr(e)}
    print(f"[rank1] rebuilt bundle: {rebuilt}", flush=True)

    # ================= VERDICT =================
    # GO conditions (the same gate the 'rf' bundle passed, applied to the spiking rebuild):
    #   (1) query_patient: onebrain valid recall >= rf valid recall (recalls a valid stored fact for every cue rf
    #       does), AND strict parity on the UNAMBIGUOUS cues is 1.0 (where the answer is unique, the spiking
    #       composer returns the SAME answer as the host closed-form).
    #   (2) query_agent: same (onebrain valid recall >= rf; unambiguous strict parity high).
    #   (3) MOAT: onebrain 0 new confabulation on out-of-store cues (the no-confab moat holds).
    #   (4) SCRAMBLE: recall is attributable (scrambled recall << real recall).
    #   ask_yes_no under-recall on AMBIGUOUS cues is a moat-SAFE abstain (characterized, not a failure) as long as
    #   ob_no_on_stored == 0 (it never asserts 'no' for a stored fact).
    go_qp_recall = qp["valid_recall_ob"] >= qp["valid_recall_rf"] - 1e-9
    go_qp_unamb = (qp["unambiguous_strict_parity"] is None) or (qp["unambiguous_strict_parity"] >= 0.999)
    go_qa_recall = qa["valid_recall_ob"] >= qa["valid_recall_rf"] - 1e-9
    go_moat = (moat["ob_confab"] == 0)
    go_scramble = scramble["valid_recall_scrambled"] < 0.5 * scramble["valid_recall_ob_same_cues"]
    go_yn_safe = (yn["ob_no_on_stored (CONFAB if>0)"] == 0)

    go = bool(go_qp_recall and go_qp_unamb and go_qa_recall and go_moat and go_scramble and go_yn_safe)

    verdict_block = None
    if Verdict is not None:
        v = Verdict("rank-1 rf->onebrain composer rebuild parity+recall on the real deployed bundle")
        v.require("query_patient valid recall onebrain >= rf", go_qp_recall, expect=True)
        v.require("query_patient unambiguous strict parity ~1.0", go_qp_unamb, expect=True)
        v.require("query_agent valid recall onebrain >= rf", go_qa_recall, expect=True)
        v.require("moat: onebrain 0 out-of-store confab", go_moat, expect=True)
        v.control("scramble collapses recall", treatment=scramble["valid_recall_ob_same_cues"],
                  control=scramble["valid_recall_scrambled"])
        v.require("ask_yes_no never asserts 'no' for a stored fact (moat-safe)", go_yn_safe, expect=True)
        try:
            _ = v.decide(go=go)
            verdict_block = v.to_dict()
        except Exception as e:
            verdict_block = {"error": repr(e)}

    result = {
        "runner": "_rank1_composer_rebuild_onebrain_verify",
        "bundle": args.bundle, "manifest_composer_kind": manifest_ck,
        "sim_backend": os.environ.get("SIM_BACKEND"),
        "fact_shard_retrieval": os.environ.get("BRAIN_FACT_SHARD_RETRIEVAL"),
        "composer_merge": os.environ.get("BRAIN_COMPOSER_MERGE"),
        "seed": seed, "D": D, "n_facts": len(flat), "skipped_non_flat": skipped, "vocab": len(vocab),
        "rf_composer": rf_ck, "ob_composer": ob_ck,
        "timings_s": {"rf_load": rf_load_s, "ob_build": ob_build_s, "ob_store": ob_store_s},
        "query_patient": qp, "query_agent": qa, "ask_yes_no": yn,
        "moat": moat, "scramble_control": scramble, "rebuilt_bundle": rebuilt,
        "go_flags": {"qp_recall": go_qp_recall, "qp_unambiguous_parity": go_qp_unamb,
                     "qa_recall": go_qa_recall, "moat": go_moat, "scramble": go_scramble,
                     "yesno_moat_safe": go_yn_safe},
        "verdict": "GO" if go else "NO-GO",
        "verdict_block": verdict_block,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\n[rank1] VERDICT = {result['verdict']}  (flags={result['go_flags']})", flush=True)
    print(f"[rank1] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
