#!/usr/bin/env python3
"""b2c_smoke_check.py -- score ONE pulled-back b2c smoke line against the declared checks (prereg "Integrity
smoke"; automated in this fix round -- the checks used to be a comment, not code).

Reads the local directory b2c_smoke.sh rsync'd the remote out_dir into: <local_dir>/lb.json plus its
intact_a_*.json (and intact_b_*.json when the row has a second arm) arm files, and lb.json.prov.json for the
LTM tier. Prints one JSON line of {"fails": [...], "n_managed_blocks": ..., "n_epochs": ..., "ltm_mode": ...}
and exits 1 if `fails` is non-empty (or nothing could be read), 0 otherwise. b2c_smoke.sh is the caller; this
script has no side effects of its own.

Checks (mirroring the "Validity of a cell" rule-5 fix, so the smoke and the eventual tools/b2c_score.py agree):
  - at least one turn carries `da_tag_capture` at all (the pair ran);
  - no `error` key anywhere under any turn's `da_tag_capture`, at ANY depth -- top-level (`da_tag_capture.error`,
    `after_store_chat` raising) and nested (`da_tag_capture.observe.error`, `observe_chat_turn` raising:
    webapp/server.py's two try/excepts around da_tag_capture_chat calls);
  - for the sleep-replay (NIGHT) row: `n_managed_blocks` > 0 at some turn (slp_teach1/slp_teach3 are expected to
    parse) and `sleep_replay_capture.n_epochs` >= 1 at the `slp_recall` turn once a block is managed (its absence
    with blocks managed means the sleep route did not run -- webapp/continuous_engine.py's idle tick only LOGS a
    tick failure, so this JSON-level check is what catches the case the log grep in b2c_smoke.sh cannot: a tick
    that silently never fires vs. one that fires and raises);
  - the LTM tier the arm actually built against (`env.BRAIN_DATA_ROOT` presence in the sidecar, same rule
    tools/lb_shard.py's aggregate uses for `ltm_mode`), reported, not gated.

Usage:  python3 b2c_smoke_check.py <local_dir> <row>
Exit:   0 = every check passed; 1 = a check failed or nothing could be read; 2 = usage.
"""
import glob
import json
import os
import sys


def find_errors(node, path=""):
    """Every 'error' key anywhere under a dict, at any depth (list elements walked too)."""
    out = []
    if isinstance(node, dict):
        if "error" in node:
            out.append(path + ".error" if path else "error")
        for k, v in node.items():
            out.extend(find_errors(v, (path + "." + k) if path else k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(find_errors(v, "%s[%d]" % (path, i)))
    return out


def main(argv):
    if len(argv) != 3:
        print("usage: b2c_smoke_check.py <local_dir> <row>", file=sys.stderr)
        return 2
    local_dir, row = argv[1], argv[2]
    fails = []
    lb_path = os.path.join(local_dir, "lb.json")
    if not os.path.exists(lb_path):
        print(json.dumps({"fails": ["no lb.json at %s" % lb_path], "n_managed_blocks": None,
                           "n_epochs": None, "ltm_mode": "unrecorded"}))
        return 1

    arm_files = sorted(glob.glob(os.path.join(local_dir, "intact_a_*.json"))
                        + glob.glob(os.path.join(local_dir, "intact_b_*.json")))
    if not arm_files:
        fails.append("no intact arm file landed alongside lb.json")

    any_dtc = False
    n_managed_blocks = None       # max over turns seen
    n_epochs_at_recall = None
    for af in arm_files:
        try:
            turns = json.load(open(af))
        except Exception as e:
            fails.append("%s unreadable: %s" % (os.path.basename(af), e))
            continue
        if not isinstance(turns, dict):
            fails.append("%s is not a turn-name -> response dict" % os.path.basename(af))
            continue
        for turn_name, resp in turns.items():
            if not isinstance(resp, dict):
                continue
            dtc = resp.get("da_tag_capture")
            if dtc is None:
                continue
            any_dtc = True
            errs = find_errors(dtc)
            if errs:
                fails.append("%s turn %r: error key(s) under da_tag_capture: %s"
                              % (os.path.basename(af), turn_name, errs))
            nmb = dtc.get("n_managed_blocks")
            if isinstance(nmb, int):
                n_managed_blocks = nmb if n_managed_blocks is None else max(n_managed_blocks, nmb)
            if turn_name == "slp_recall":
                src = dtc.get("sleep_replay_capture")
                if isinstance(src, dict):
                    n_epochs_at_recall = src.get("n_epochs")

    if not any_dtc:
        fails.append("no turn in any arm file carried da_tag_capture -- the pair never ran")

    if row == "sleep-replay":
        if not n_managed_blocks:
            fails.append("sleep-replay row managed 0 blocks (expected >0: slp_teach1/slp_teach3 parse)")
        elif n_epochs_at_recall is None or n_epochs_at_recall < 1:
            fails.append("sleep-replay row managed %r block(s) but sleep_replay_capture.n_epochs at slp_recall "
                          "is %r (want >=1 -- the sleep route may not have run)" % (n_managed_blocks, n_epochs_at_recall))

    ltm_mode = "unrecorded"
    prov_path = lb_path + ".prov.json"
    if os.path.exists(prov_path):
        try:
            penv = (json.load(open(prov_path)).get("env") or {})
            ltm_mode = "on" if penv.get("BRAIN_DATA_ROOT") else "off"
        except Exception:
            pass

    print(json.dumps({"fails": fails, "n_managed_blocks": n_managed_blocks,
                       "n_epochs": n_epochs_at_recall, "ltm_mode": ltm_mode}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
