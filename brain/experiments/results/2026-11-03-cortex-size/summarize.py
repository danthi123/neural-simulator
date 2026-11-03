# Capability: Sleep consolidation
# Record: brain/research/2026-11-02-record-brief-consolidation-and-selection.md
"""Print one row per sleep_probe json: recall/paraphrase/false/generic_kl for no_sleep, sleep_true, sleep_shuffled.
  python summarize.py <json>...
"""
import json
import sys

print("file | arm | recall | para | false | gKL | elapsed_s")
for f in sys.argv[1:]:
    d = json.load(open(f))
    for a, r in d["arms"].items():
        print(f"{f.split('/')[-1]} | {a} | {r['recall']} | {r['paraphrase']} | {r['false_recall']} | "
              f"{r['generic_kl']:.4f} | {d['elapsed_s']:.0f}")
