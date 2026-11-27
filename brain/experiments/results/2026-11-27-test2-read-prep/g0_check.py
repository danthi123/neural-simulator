# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md
# Expected-fail: none (read helper; changes no brain behaviour)
"""G0 engagement read (prereg section 5 table, row G0). Read-only; NOT a frozen scorer (the frozen scorers do not compute
G0; this restates kp_bars.py 1a/1b and d2pr2_bars.py 3.1 with the prereg thresholds). Record its sha256 beside the read.
  python g0_check.py kp <life.json> <keep_bank.json>      one K-P life + its pulled bank (work/lifeKP_sS/trunk/keep_bank.json)
  python g0_check.py pr <life.json>                       one D2-PR2 life
Prints one line per check and TWO verdict lines: 'G0 seed (prereg-literal): ...' and 'G0 seed (helper-stricter): ...'.
Missing bank or missing record = UNDEFINED (prereg). The report quotes BOTH; the prereg-literal one is the registered rule.
K-P prereg-literal (prereg section 5, row G0): a keep_bank record on every one of the 15 trunk nights, bank_size 512, bank file
present with 512 texts, >= 90% distinct, mean_tokens >= 40, < 10% told-key hits.
K-P HELPER'S OWN CHOICES, not in the prereg (taken from the unfrozen DEV script kp_bars.py 1a): one subset_n, 15 distinct
subset_hash, mean_keep_kl > 0 from night 2. Also the helper's own: "told key" is operationalised as a whole-word case-insensitive
regex over the keys of ledger entries with a telling (the prereg does not define it). A seed that passes literal and fails stricter
is reported as such; the prereg-literal verdict is the one that enters the decision (section 6).
Report-only (G0 note): share of bank texts whose non-User turn matches the G8 refusal pattern, and 'my name is'.
D2-PR2: nights 5 and 10: permanent_share < strength_permanent_share AND top6_share < strength_top6_share."""
import json, re, sys
sys.path.insert(0, ".")
from brain.eval import life_costs as C

mode, life = sys.argv[1], json.load(open(sys.argv[2]))
trunk = [n for n in life["nights"] if n.get("line", "trunk") == "trunk"]
if mode == "pr":
    ok = []
    for n in (5, 10):
        r = next((x for x in trunk if x["night"] == n), None)
        rp = (r or {}).get("replay_priority")
        if not rp:
            print(f"night {n}: no replay_priority record -> UNDEFINED"); ok.append(None); continue
        a, b = rp["permanent_share"] < rp["strength_permanent_share"], rp["top6_share"] < rp["strength_top6_share"]
        print(f"night {n}: perm {rp['permanent_share']:.4f} vs ref {rp['strength_permanent_share']:.4f} ({a}); top6 {rp['top6_share']:.4f} vs ref {rp['strength_top6_share']:.4f} ({b})")
        ok.append(a and b)
    res = None if None in ok else all(ok)
else:
    kb = [(n["night"], n.get("keep_bank")) for n in trunk]
    have = [x for _, x in kb if x]
    hashes = [x["subset_hash"] for x in have]; sizes = {x["bank_size"] for x in have}; subn = {x["subset_n"] for x in have}
    kls = [x["mean_keep_kl"] for n, x in kb if x and n >= 2]
    rec_lit = len(have) == len(trunk) == 15 and sizes == {512}
    extra = len(subn) == 1 and len(set(hashes)) == 15 and bool(kls) and all(v > 0 for v in kls)
    print(f"record (prereg-literal: 15/15 nights, bank_size 512): {len(have)}/{len(trunk)} nights, sizes {sorted(sizes)} -> {rec_lit}")
    print(f"record (HELPER extras, not prereg: one subset_n, 15 distinct hashes, keep_kl > 0 n2+): subset_n {sorted(subn)}, distinct hashes {len(set(hashes))}, min keep_kl n2+ {min(kls) if kls else None} -> {extra}")
    try:
        b = json.load(open(sys.argv[3])); tx = b["texts"]
    except Exception as e:
        print(f"bank file missing/unreadable ({e}) -> UNDEFINED")
        print("G0 seed (prereg-literal): UNDEFINED"); print("G0 seed (helper-stricter): UNDEFINED"); sys.exit(0)
    told = sorted({k for v in life["ledger"].values() if v.get("telling") for k in v["keys"]})
    rx = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in told) + r")\b", re.I)
    ht = sum(1 for t in tx if rx.search(t)); nd = len(set(tx))
    def nonuser(t):   # non-User turns: split on 'User:'/'AI:'/'Bot:' frame labels
        parts = re.split(r"(?m)^(User|AI|Bot|Assistant):", t)
        return [parts[i + 1] for i in range(1, len(parts) - 1, 2) if parts[i] != "User"]
    def refusal(t):
        for p in nonuser(t):
            first = re.split(C.SENT_SPLIT, p.strip())[0].lower()
            if C.REFUSAL_RX.search(first) or C.REFUSAL_SUBSTR in p.lower(): return True
        return False
    nm = sum(1 for t in tx if any(re.search(r"my name is|i'?m an assistant|i am an assistant", p, re.I) for p in nonuser(t)))
    bank_ok = len(tx) == 512 and nd / len(tx) >= 0.90 and b["mean_tokens"] >= 40 and ht / len(tx) < 0.10
    print(f"bank: {len(tx)} texts, distinct {nd} ({nd/len(tx):.3f}), mean_tokens {b['mean_tokens']:.1f}, told-key hits {ht} ({ht/len(tx):.3f}) -> {bank_ok}")
    print(f"report-only: refusal-pattern share {sum(refusal(t) for t in tx)}/{len(tx)}, 'my name is'/assistant-self-name share {nm}/{len(tx)}")
    res = rec_lit and bank_ok
    res_strict = res and extra
V = {True: "PASS", False: "FAIL", None: "UNDEFINED"}
if mode == "pr":
    print("G0 seed (prereg-literal):", V[res]); print("G0 seed (helper-stricter):", V[res], "(same: PR has no helper extras)")
else:
    print("G0 seed (prereg-literal):", V[res]); print("G0 seed (helper-stricter):", V[res_strict])
