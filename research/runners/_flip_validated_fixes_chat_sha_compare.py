"""Compare two `_flip_validated_fixes_chat_sha` outputs turn by turn (exact sha256 equality) -> a verdict JSON.

  python -m research.runners._flip_validated_fixes_chat_sha_compare --ref BASE.json --test BRANCH.json \
      --ref-label base_033e385b8 --out VERDICT.json [--extra NAME=PATH ...]

`identical` is True only if both files have the same number of turns, every turn returned HTTP 200, and every turn's
`sha_canon` matches exactly. The raw-byte hashes are compared too and reported (`raw_identical`); a raw mismatch with a
canonical match means only a volatile (timing) key differed, and the dropped key paths are listed. `--extra` files
(e.g. the branch DEFAULT arm) are summarized against the test file for information; they do not enter the verdict.
"""
from __future__ import annotations

import argparse
import json
import os


def _cmp(a, b):
    ta, tb = a["turns"], b["turns"]
    rows = []
    for i in range(max(len(ta), len(tb))):
        ra = ta[i] if i < len(ta) else {}
        rb = tb[i] if i < len(tb) else {}
        rows.append({"turn": i, "message": ra.get("message") or rb.get("message"),
                     "status": [ra.get("status"), rb.get("status")],
                     "canon_equal": bool(ra) and bool(rb) and ra.get("sha_canon") == rb.get("sha_canon"),
                     "raw_equal": bool(ra) and bool(rb) and ra.get("sha_raw") == rb.get("sha_raw"),
                     "answers": [ra.get("answer"), rb.get("answer")],
                     "dropped_volatile_keys": sorted(set(ra.get("dropped_volatile_keys", []))
                                                     | set(rb.get("dropped_volatile_keys", [])))})
    all_200 = all(r["status"] == [200, 200] for r in rows)
    return {"n_turns": [len(ta), len(tb)], "all_http_200": all_200,
            "identical": len(ta) == len(tb) and len(ta) > 0 and all_200 and all(r["canon_equal"] for r in rows),
            "raw_identical": len(ta) == len(tb) and all(r["raw_equal"] for r in rows),
            "n_turns_canon_differ": sum(1 for r in rows if not r["canon_equal"]), "turns": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--ref-label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    ref, test = json.load(open(a.ref)), json.load(open(a.test))
    main_cmp = _cmp(ref, test)
    out = {"question": "With all three flipped flags explicitly =0, is the scripted /api/brain-chat session "
                       "byte-identical (per-turn sha256 of the canonicalized response body) to the pinned pre-flip "
                       "commit?",
           "ref": {"label": a.ref_label, "arm": ref.get("arm"), "flags_env": ref.get("flags_env"),
                   "session_sha_canon": ref.get("session_sha_canon"), "wall_s": ref.get("wall_s")},
           "test": {"path": a.test, "arm": test.get("arm"), "flags_env": test.get("flags_env"),
                    "session_sha_canon": test.get("session_sha_canon"), "wall_s": test.get("wall_s")},
           "sim_backend": [ref.get("sim_backend"), test.get("sim_backend")],
           "verdict": "IDENTICAL" if main_cmp["identical"] else "DIFFERS",
           **main_cmp, "informational": {}}
    for kv in a.extra:
        name, _, path = kv.partition("=")
        ex = json.load(open(path))
        c = _cmp(test, ex)
        out["informational"][name] = {"path": path, "arm": ex.get("arm"), "flags_env": ex.get("flags_env"),
                                      "wall_s": ex.get("wall_s"), "identical_to_test": c["identical"],
                                      "n_turns_canon_differ": c["n_turns_canon_differ"],
                                      "differing_turns": [{"turn": r["turn"], "message": r["message"],
                                                           "answers": r["answers"]}
                                                          for r in c["turns"] if not r["canon_equal"]]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("[chat-sha-compare] verdict=%s n_differ=%d raw_identical=%s -> %s"
          % (out["verdict"], out["n_turns_canon_differ"], out["raw_identical"], a.out))


if __name__ == "__main__":
    main()
