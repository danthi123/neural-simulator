"""Composer-path reply records for a stratum: per turn, did the production reply ANSWER or ABSTAIN with BRAIN_MOAT_SLUG_NORM off (stock)
and on, and which turns does the flag HURT (stock answered, flag-on abstained) or HELP (stock abstained, flag-on answered).

Sources (all recorded by drive_memo.py, i.e. produced by the production brain + RichAnswerComposer, none emulated):
  STOCK    the answer/abstained of the LIVE generation turn (flag unset, real Qwen) stored in the stratum's corpus.json / s3_corpus.json, and
           (validation) a replay of the same turns with the flag unset when replay-off files are given: a replay is a valid stand-in
           for the live turn iff its answer text equals the live turn's.
  FLAG-ON  drive_memo.py --renderer replay with DRIVE_ENV {"BRAIN_MOAT_SLUG_NORM":"1"}: the production brain and rich path, the Qwen
           renderer replaced by the recorded sentences of the live turn, so flag-on decisions are made by the production composer on the
           very sentences the live Qwen produced.  Greedy decoding makes render_svo / render_svo_regen a function of the triple, so
           the recorded sentence is the one a live flag-on run would receive; a request with no recorded sentence is a "replay miss".
Per candidate (RichAnswerComposer._verify_rendered) the replay records the number of gathered facts, which sub-verifier decided (claim gate
vs the single-triple chat._verify fallback when the claim verifier is unbuildable) and the verdict.

usage (cwd = worktree, as a module):
  python -m research.findings.raw._moat_slug_norm_heldout.composer_records <out.json> <stratum name> <corpus.json> <on1.json,on2.json> [<off1.json,...>] [--off-is-stock]
  --off-is-stock: the baseline is the recorded flag-off run itself (used for the stub-mouth arms, whose baseline is not the live Qwen turn).
"""
import collections, hashlib, json, os, socket, sys

import research.runners  # noqa: F401  (provenance door)


def load_many(arg):
    resp, trace, misses, meta = {}, [], [], []
    for p in [x for x in (arg or "").split(",") if x]:
        raw = open(p, "rb").read()
        d = json.loads(raw)
        resp.update(d["responses"]); trace += d["trace"]; misses += d.get("replay_misses", [])
        meta.append({"file": os.path.basename(p), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "host": d.get("host"), "device": d.get("device"), "drive_env": d.get("drive_env"),
                     "renderer": d.get("renderer")})
    return resp, trace, misses, meta


def cand_by_turn(trace):
    out = collections.defaultdict(list)
    for t in trace:
        if t.get("fn") == "_verify_rendered":
            out[t["turn"].split("|")[0]].append(t)
    return out


def main(out_path, name, corpus_path, on_arg, off_arg=None, off_is_stock=False):
    corpus = json.load(open(corpus_path))
    live = {t["label"]: t for t in corpus["turns"]}
    on_resp, on_trace, on_miss, on_meta = load_many(on_arg)
    off_resp, off_trace, off_miss, off_meta = load_many(off_arg)
    on_c = cand_by_turn(on_trace); off_c = cand_by_turn(off_trace)
    rows, hurts, helps = [], [], []
    for lab, t in live.items():
        if not t.get("calls") or lab not in on_resp:
            continue
        r = on_resp[lab]
        if off_is_stock:          # the stock baseline is itself a recorded flag-off run with the same renderer (stub arms)
            if lab not in off_resp:
                continue
            t = dict(t, abstained=off_resp[lab].get("abstained"), answer=off_resp[lab].get("answer"))
        stock_abs = bool(t["abstained"]); on_abs = bool(r.get("abstained"))
        cands = on_c.get(lab, [])
        row = {"label": lab, "entity": t["entity"], "n_gathered": len(t["gated"]),
               "stock_live": {"abstained": stock_abs, "answer": t["answer"]},
               "flag_on_replay": {"abstained": on_abs, "answer": r.get("answer"), "n_candidates": len(cands),
                                  "n_accepted": sum(1 for c in cands if c["ok"]),
                                  "claim_gate_decided": sum(1 for c in cands if c["claim"] not in ("not_called", None)),
                                  "fell_back_to_single_triple": sum(1 for c in cands if c["single_triple"] != "not_called"),
                                  "accepted_by_fallback": sum(1 for c in cands if c["single_triple"] is True and c["ok"]),
                                  "candidates": [{"svo": c["svo"], "surface": c["surface"], "claim": c["claim"], "claim_reject": c.get("claim_reject"),
                                                  "single_triple": c["single_triple"], "ok": c["ok"]} for c in cands]},
               "status": ("HURTS" if (not stock_abs and on_abs) else "HELPS" if (stock_abs and not on_abs)
                          else "both_answer" if not stock_abs else "both_abstain")}
        if lab in off_resp and not off_is_stock:
            ro = off_resp[lab]
            row["flag_off_replay"] = {"abstained": bool(ro.get("abstained")), "answer": ro.get("answer"),
                                      "equals_live": ro.get("answer") == t["answer"] and bool(ro.get("abstained")) == stock_abs}
        rows.append(row)
        if row["status"] == "HURTS":
            hurts.append(row)
        if row["status"] == "HELPS":
            helps.append(row)
    st = collections.Counter(r["status"] for r in rows)
    by_n = collections.defaultdict(collections.Counter)
    for r in rows:
        by_n["1" if r["n_gathered"] == 1 else "2-3" if r["n_gathered"] <= 3 else "4+"][r["status"]] += 1
    validated = [r for r in rows if "flag_off_replay" in r]
    res = {"_doc": "composer-path reply records for stratum %s (rule: module docstring)" % name,
           "backend": "numpy", "device": "cpu (replay) / live generation device per corpus raw_files",
           "host": socket.gethostname(), "script": "research/findings/raw/_moat_slug_norm_heldout/composer_records.py",
           "argv": sys.argv, "stratum": name, "corpus": corpus_path, "flag_on_files": on_meta, "flag_off_files": off_meta,
           "n_turns": len(rows), "status_counts": dict(st), "status_by_gathered_size": {k: dict(v) for k, v in by_n.items()},
           "stock_answered": sum(1 for r in rows if not r["stock_live"]["abstained"]),
           "flag_on_answered": sum(1 for r in rows if not r["flag_on_replay"]["abstained"]),
           "HURTS_turns": hurts, "HELPS_turns": helps,
           "replay_misses_flag_on": on_miss, "replay_misses_flag_off": off_miss,
           "replay_validation": {"turns_replayed_flag_off": len(validated),
                                 "answer_and_abstain_equal_to_live": sum(1 for r in validated if r["flag_off_replay"]["equals_live"]),
                                 "mismatches": [r["label"] for r in validated if not r["flag_off_replay"]["equals_live"]]},
           "turns": rows}
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    print(name, "turns", len(rows), dict(st), "stock answered", res["stock_answered"], "flag-on answered", res["flag_on_answered"],
          "| HURTS", [r["label"] for r in hurts], "| validation", res["replay_validation"], "| misses", len(on_miss), len(off_miss))


if __name__ == "__main__":
    main(*sys.argv[1:6], off_is_stock=("--off-is-stock" in sys.argv))
