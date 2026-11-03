# Capability: Affect shaping speech
# Record: brain/research/2026-11-03-mood-state-design.md
"""Six-seed table for the mood probe.  python -m brain.experiments.mood_table <dir with mood_0.1b_s*.json>

Echo is recomputed from the saved replies with the current (revised) instrument.
"""
import glob
import json
import sys

from brain.experiments.sentiment_instrument import primary_echo, primary_echo_registered


def main(d):
    print("| seed | appr acc +/- | m_pos | m_neg | lex pos-neg [CI] | emb pos-neg [CI] | lex pos-off | lex neg-off | "
          "lesion identical (by construction) | rand lex diff | rand emb diff | echo pos/neg/off (revised, post hoc) | "
          "echo mood-off REGISTERED (bar 0.10) | NLL pos/neg/off |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(f"{d}/mood_0.1b_s*.json"), key=lambda x: int(x.split("_s")[-1][:-5])):
        j = json.load(open(f))
        for arm in j["rows"]:
            j["summary"][arm]["echo_rate"] = sum(primary_echo(r["reply"]) for r in j["rows"][arm]) / len(j["rows"][arm])
        reg = {a: sum(primary_echo_registered(r["reply"]) for r in j["rows"][a]) / len(j["rows"][a]) for a in j["rows"]}
        reg_d = (reg["pos"] + reg["neg"]) / 2 - (reg["lesion_pos"] + reg["lesion_neg"]) / 2
        c, s, ind, acc = j["contrasts"], j["summary"], j["induction"], j["appraisal_sign_acc"]
        m, r = c["mood"], c["random"]
        print(f"| {j['seed']} | {acc['pos']:.2f}/{acc['neg']:.2f} | {ind['m_pos']:+.3f} | {ind['m_neg']:+.3f} | "
              f"{m['lex_diff']:+.2f} [{m['lex_ci95'][0]:+.2f},{m['lex_ci95'][1]:+.2f}] | "
              f"{m['emb_diff']:+.3f} [{m['emb_ci95'][0]:+.3f},{m['emb_ci95'][1]:+.3f}] | "
              f"{c['pos_vs_off']['lex_diff']:+.2f} | {c['neg_vs_off']['lex_diff']:+.2f} | "
              f"{c['lesion']['identical_replies']:.2f} | {r['lex_diff']:+.2f} | {r['emb_diff']:+.3f} | "
              f"{s['pos']['echo_rate']:.2f}/{s['neg']['echo_rate']:.2f}/{s['lesion_pos']['echo_rate']:.2f} | "
              f"{reg_d:+.3f} {'FAIL' if reg_d > 0.10 else 'ok'} | "
              f"{s['pos']['nll']:.2f}/{s['neg']['nll']:.2f}/{s['lesion_pos']['nll']:.2f} |")


if __name__ == "__main__":
    main(sys.argv[1])
