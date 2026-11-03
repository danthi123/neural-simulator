# Capability: Affect shaping speech
# Record: brain/research/2026-11-03-mood-state-design.md (section 9: pre-registered criteria, frozen before the run)
"""Iteration-2 six-seed table + the pre-registered PASS criteria.  python -m brain.experiments.mood_table_v2 <dir>"""
import glob
import json
import sys


def criteria(j):
    acc, ind, c, s = j["appraisal_sign_acc"], j["induction"], j["contrasts"], j["summary"]
    echo_d = (s["pos"]["echo_rate"] + s["neg"]["echo_rate"]) / 2 - s["off"]["echo_rate"]
    return {
        "1_heldout_appraisal": acc["heldout_pos"] >= 0.75 and acc["heldout_neg"] >= 0.75,
        "2_mood_signs": ind["m_pos"] > 0 > ind["m_neg"],
        "3_pos_vs_off_ci_above_0": c["pos_off"]["lex_ci95"][0] > 0,
        "4_neg_vs_off_ci_below_0": c["neg_off"]["lex_ci95"][1] < 0,
        "5_embedding_agrees": c["pos_off"]["emb_diff"] > 0 > c["neg_off"]["emb_diff"],
        "6_specificity_ci_above_0": c["specificity"]["lex_ci95"][0] > 0,
        "7_echo_le_0.10": echo_d <= 0.10,
    }, echo_d


def main(d):
    files = sorted(glob.glob(f"{d}/mood2_0.1b_s*.json"), key=lambda x: int(x.split("_s")[-1][:-5]))
    print("| seed | held-out acc +/- | dev acc +/- | m_pos | m_neg | lex pos-off [CI] | lex neg-off [CI] | "
          "emb pos-off / neg-off | lex rand_pos-off / rand_neg-off | specificity lex [CI] | changed vs off pos/neg | "
          "echo mood-off (rev / reg) | NLL pos/neg/off | criteria met |")
    print("|" + "---|" * 14)
    tally = {}
    for f in files:
        j = json.load(open(f))
        acc, ind, c, s = j["appraisal_sign_acc"], j["induction"], j["contrasts"], j["summary"]
        cr, echo_d = criteria(j)
        for k, v in cr.items():
            tally[k] = tally.get(k, 0) + int(v)
        reg_d = (s["pos"]["echo_reg_rate"] + s["neg"]["echo_reg_rate"]) / 2 - s["off"]["echo_reg_rate"]
        po, no, sp = c["pos_off"], c["neg_off"], c["specificity"]
        print(f"| {j['seed']} | {acc['heldout_pos']:.2f}/{acc['heldout_neg']:.2f} | {acc['pos']:.2f}/{acc['neg']:.2f} | "
              f"{ind['m_pos']:+.3f} | {ind['m_neg']:+.3f} | "
              f"{po['lex_diff']:+.2f} [{po['lex_ci95'][0]:+.2f},{po['lex_ci95'][1]:+.2f}] | "
              f"{no['lex_diff']:+.2f} [{no['lex_ci95'][0]:+.2f},{no['lex_ci95'][1]:+.2f}] | "
              f"{po['emb_diff']:+.3f} / {no['emb_diff']:+.3f} | "
              f"{c['rand_pos_off']['lex_diff']:+.2f} / {c['rand_neg_off']['lex_diff']:+.2f} | "
              f"{sp['lex_diff']:+.2f} [{sp['lex_ci95'][0]:+.2f},{sp['lex_ci95'][1]:+.2f}] | "
              f"{1 - po['identical_replies']:.2f}/{1 - no['identical_replies']:.2f} | {echo_d:+.3f} / {reg_d:+.3f} | "
              f"{s['pos']['nll']:.2f}/{s['neg']['nll']:.2f}/{s['off']['nll']:.2f} | "
              f"{sum(cr.values())}/7 (fail: {','.join(k[0] for k, v in cr.items() if not v) or '-'}) |")
    need = {"2_mood_signs": 6, "7_echo_le_0.10": 6}
    n = len(files)
    print(f"\nCriteria over {n} seeds (bar: 5/6, criteria 2 and 7: 6/6):")
    allpass = True
    for k, v in tally.items():
        ok = v >= need.get(k, 5)
        allpass &= ok
        print(f"- {k}: {v}/{n} {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if allpass and n == 6 else 'NOT PASS'}")


if __name__ == "__main__":
    main(sys.argv[1])
