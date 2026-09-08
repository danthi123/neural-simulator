"""Byte-identical check for the 'rf' bundle reload across the onebrain_k_max load-path thread change
(docs/TERMS.md: 'byte-identical' must be asserted IN THE DATA -- a hash/exact compare -- never inferred from
reading the code). Loads the PRODUCTION bundle (scale787/day_33, composer_kind='rf', untouched by this change)
via `load_developed_brain` and writes a deterministic hash of the resulting composer state (every stored fact's
dict + its composite array bytes) plus the manifest. Run once at HEAD~1 (before the onebrain_k_max thread) and
once at HEAD (after), diff the two JSON outputs -- an identical hash proves the rf path is unaffected, not just
plausibly unaffected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.developed_brain_io import load_developed_brain

DEFAULT_BUNDLE = "/home/dant123/Projects/sim/bridges/developed/scale787/day_33"


def _hash_composer_kb(comp) -> str:
    h = hashlib.sha256()
    kb = getattr(comp, "kb", None)
    if kb is None:
        h.update(b"NO_KB")
        return h.hexdigest()
    for fact_dict, arr in kb:
        h.update(json.dumps(fact_dict, sort_keys=True, default=str).encode("utf-8"))
        if arr is not None:
            h.update(arr.tobytes())
        else:
            h.update(b"NONE")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--out", default="research/findings/raw/_onebrain_kmax_loadpath_thread/rf_byteident.json")
    ap.add_argument("--tag", default="unlabeled")
    args = ap.parse_args()

    agent, manifest = load_developed_brain(args.bundle, use_multiturn=False, enable_neural_render=False)
    kb_hash = _hash_composer_kb(agent.composer)
    result = {
        "runner": "_onebrain_kmax_rf_byteident_check",
        "tag": args.tag,
        "bundle": args.bundle,
        "composer_class": type(agent.composer).__name__,
        "manifest_composer_kind": manifest.get("composer_kind"),
        "manifest_n_facts": manifest.get("n_facts"),
        "n_recalled": len(getattr(agent.composer, "kb", []) or []),
        "kb_sha256": kb_hash,
    }
    print(f"[byteident:{args.tag}] composer={result['composer_class']} n_recalled={result['n_recalled']} "
          f"kb_sha256={kb_hash}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[byteident:{args.tag}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
