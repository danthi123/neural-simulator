#!/usr/bin/env python3
"""Re-run the leak audit on a finished batch with the current rules (run.py's audit), rewriting each result.json's
audit_flags / audit_warnings / disqualified. For a batch that ran before an audit rule was corrected; the old flags
are kept as audit_flags_before so the change is visible in the record.

  python3 reaudit.py --batch screen1
"""
import argparse
import glob
import json
import os

from run import RUNS, audit, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    a = ap.parse_args()
    for res_path in sorted(glob.glob(os.path.join(RUNS, a.batch, "*", "*", "result.json"))):
        res = load_json(res_path)
        run_dir = os.path.dirname(res_path)
        flags, warns = audit(os.path.join(run_dir, "transcript.jsonl"), run_dir)
        if flags != res.get("audit_flags"):
            res.setdefault("audit_flags_before", res.get("audit_flags"))
            print("%-34s %-44s DQ %s -> %s" % (res["model"][:34], res["task"][:44], res.get("disqualified"), bool(flags)))
        res["audit_flags"], res["audit_warnings"], res["disqualified"] = flags, warns, bool(flags)
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
