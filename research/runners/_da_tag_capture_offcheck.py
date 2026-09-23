"""Byte-identical-OFF check for the DA tag-and-capture companion, asserted IN DATA against a PINNED pre-change tree.

Runs the production DA-encoding store path (the same `install_encoding_gain` coupling, production env, flag
`BRAIN_DA_TAG_CAPTURE` UNSET) on a code tree given by --root, stores the four natural-drive facts under a fixed DA
trace, runs the idle-tick homeostasis pass, recalls, and prints a sha256 over (store_conns, recall replies).

  # the pinned pre-change tree (git archive of the pinned SHA) and the branch tree, same script:
  .venv/bin/python research/runners/_da_tag_capture_offcheck.py --root <pinned-tree> --out a.json
  .venv/bin/python research/runners/_da_tag_capture_offcheck.py --root .              --out b.json
  .venv/bin/python research/runners/_da_tag_capture_offcheck.py --compare a.json b.json --out offcheck.json
"""
import argparse
import hashlib
import json
import os
import sys


def run(root):
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    os.chdir(root)
    os.environ["SIM_BACKEND"] = "numpy"
    os.environ.pop("BRAIN_DA_TAG_CAPTURE", None)
    os.environ.pop("BRAIN_DA_CAPTURE_LESION", None)
    import logging
    logging.disable(logging.CRITICAL)
    import random

    import numpy as np
    from research.runners.one_brain_composer import OneBrainComposer
    from webapp import da_encoding_drives_chat as DAE
    facts = [("zebra", "swallow", "violin"), ("otter", "steal", "lantern"), ("goat", "eat", "passport"),
             ("parrot", "hide", "key")]
    vocab = sorted({w for f in facts for w in f} | {"dog", "cat", "see"})

    class _Inner:
        def __init__(self, c):
            self.composer = c

    class _Chat:
        def __init__(self, c):
            self.inner = _Inner(c)
            self._last_da_drives = {"da_level": 0.5}

    def _one(ledger=None):
        comp = OneBrainComposer(seed=7, D=128, vocab=vocab, k_max=16)
        chat = _Chat(comp)
        DAE.install_encoding_gain(chat)
        for f, da in zip(facts, (0.55, 0.9, 0.62, 0.8)):
            chat._last_da_drives = {"da_level": da}
            comp.store(*f)
            if ledger is not None:
                ledger.on_store(comp, 0.0)
        scales = DAE.apply_substrate_homeostasis(chat)
        replies = [comp.query_patient(a, act) for (a, act, _p) in facts]
        h = hashlib.sha256()
        for (p, q, w) in comp.store_conns:
            c = complex(w)
            h.update(repr((int(p), int(q), c.real.hex(), c.imag.hex())).encode())
        h.update(repr(replies).encode())
        return h.hexdigest(), replies, scales, len(comp.store_conns)

    np.random.seed(7)
    random.seed(7)
    sha, replies, scales, n = _one()
    tag_module_present = os.path.exists(os.path.join(root, "webapp", "da_tag_capture.py"))
    ledger_none = sha_on = None
    if tag_module_present:
        from webapp.da_tag_capture import SynapticTagCaptureLedger, maybe_ledger
        ledger_none = maybe_ledger(7) is None
        # sensitivity control: the SAME hash with the companion ARMED must differ (else the compare cannot fail)
        np.random.seed(7)
        random.seed(7)
        sha_on = _one(SynapticTagCaptureLedger(7, gamma=10.0, beta=1.0))[0]
    return {"root": root, "store_sha256": sha, "n_store_conns": n, "replies": replies,
            "homeostasis_scales": None if scales is None else [float(s) for s in scales],
            "tag_module_present": tag_module_present, "maybe_ledger_is_none_flag_unset": ledger_none,
            "store_sha256_companion_armed_sensitivity_control": sha_on}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--pinned-sha", default=None)
    ap.add_argument("--branch-sha", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if a.compare:
        x, y = (json.load(open(p)) for p in a.compare)
        r = {"pinned": x, "branch": y, "pinned_sha": a.pinned_sha, "branch_sha": a.branch_sha,
             "byte_identical_store_and_replies": x["store_sha256"] == y["store_sha256"] and x["replies"] == y["replies"]
             and x["homeostasis_scales"] == y["homeostasis_scales"],
             "branch_flag_unset_constructs_no_ledger": y["maybe_ledger_is_none_flag_unset"] is True,
             "sensitivity_control_hash_differs_when_armed":
                 y["store_sha256_companion_armed_sensitivity_control"] not in (None, y["store_sha256"])}
        r["pass"] = bool(r["byte_identical_store_and_replies"] and r["branch_flag_unset_constructs_no_ledger"]
                         and r["sensitivity_control_hash_differs_when_armed"])
    else:
        r = run(a.root)
    json.dump(r, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in r.items() if k not in ("pinned", "branch")}, indent=2))


if __name__ == "__main__":
    main()
