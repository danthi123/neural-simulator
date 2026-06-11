"""Aggregate the multi-seed vocab-ceiling raw JSONs into the findings-doc markdown tables.

Reads research/findings/raw/_vocab_ceiling_V{V}_s{seed}_D{D}.json for the swept cells and prints:
  - the 6-seed per-capability matrix at V=320 (D=128) and (D=256)
  - the V=128 D=128 intermediate-rung table
  - the abstention-moat + shuffled-control summary
  - the per-capability degradation map
Pure stdlib; safe to re-run.
"""
import json
import os

RAW = os.path.join(os.path.dirname(__file__))
CAPS = ["what_qa", "who_qa", "one_attribute", "embedded_clause", "negation_yesno",
        "two_attribute", "generation", "dialogue"]
CAP_SHORT = {
    "what_qa": "what", "who_qa": "who", "one_attribute": "1-attr",
    "embedded_clause": "clause", "negation_yesno": "neg", "two_attribute": "2-attr",
    "generation": "gen", "dialogue": "dialog",
}
SEEDS = [42, 43, 44, 45, 46, 47]


def load(V, seed, D):
    p = os.path.join(RAW, f"_vocab_ceiling_V{V}_s{seed}_D{D}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cell(m, cap):
    r = m[cap]
    return f"{r['correct']}/{r['attempted']}"


def matrix_table(V, D):
    print(f"\n### V={V}, D={D} -- 6-seed capability matrix\n")
    hdr = "| seed | " + " | ".join(CAP_SHORT[c] for c in CAPS) + " | **abstention** | shuffled (false hits) | verdict |"
    sep = "|------|" + "|".join(["-----"] * len(CAPS)) + "|----------------|------------------------|---------|"
    print(hdr)
    print(sep)
    # column-wise pass tallies
    cap_pass = {c: 0 for c in CAPS}
    ab_pass = 0
    sh_pass = 0
    n = 0
    for s in SEEDS:
        d = load(V, s, D)
        if d is None:
            print(f"| {s} | " + " | ".join(["--"] * len(CAPS)) + " | -- | -- | MISSING |")
            continue
        n += 1
        m = d["matrix"]
        cells = [cell(m, c) for c in CAPS]
        ab = m["abstention"]
        sc = m["shuffled_control"]
        ab_str = f"{ab['correct']}/{ab['attempted']}"
        ab_ok = ab["correct"] == ab["attempted"]
        sh_ok = sc["false_hits"] == 0
        if ab_ok:
            ab_pass += 1
        if sh_ok:
            sh_pass += 1
        for c in CAPS:
            if m[c]["correct"] == m[c]["attempted"]:
                cap_pass[c] += 1
        row = (f"| {s} | " + " | ".join(cells) + f" | **{ab_str}**"
               f" | {sc['false_hits']}/{sc['attempted']} | {d['verdict']} |")
        print(row)
    # summary line
    print(f"\n**Per-capability seeds-passing (of {n}):** " +
          ", ".join(f"{CAP_SHORT[c]} {cap_pass[c]}/{n}" for c in CAPS) +
          f"  |  abstention {ab_pass}/{n}  |  shuffled-clean {sh_pass}/{n}")
    return cap_pass, ab_pass, sh_pass, n


def main():
    print("=" * 100)
    print("VOCAB-CEILING MULTI-SEED AGGREGATE")
    print("=" * 100)

    print("\n## V=320 (the ceiling)\n")
    d128 = matrix_table(320, 128)
    d256 = matrix_table(320, 256)

    print("\n## V=128 (intermediate rung)\n")
    v128 = matrix_table(128, 128)

    # degradation map
    print("\n## PER-CAPABILITY DEGRADATION MAP\n")
    cap_pass_320_128, ab320_128, _, n320_128 = d128
    cap_pass_320_256, ab320_256, _, n320_256 = d256
    cap_pass_128_128, ab128, _, n128 = v128
    print("| capability | V=128 D=128 | V=320 D=128 | V=320 D=256 | min-D @ V=320 |")
    print("|------------|-------------|-------------|-------------|---------------|")
    for c in CAPS:
        v128c = f"{cap_pass_128_128.get(c, 0)}/{n128}" if n128 else "--"
        v320_128c = f"{cap_pass_320_128.get(c, 0)}/{n320_128}" if n320_128 else "--"
        v320_256c = f"{cap_pass_320_256.get(c, 0)}/{n320_256}" if n320_256 else "--"
        # min-D verdict
        if n320_128 and cap_pass_320_128.get(c, 0) == n320_128:
            mind = "128"
        elif n320_256 and cap_pass_320_256.get(c, 0) == n320_256:
            mind = ">=256"
        else:
            mind = ">256 / unresolved"
        print(f"| {CAP_SHORT[c]} | {v128c} | {v320_128c} | {v320_256c} | {mind} |")
    print(f"\n| abstention (moat) | {ab128}/{n128} | {ab320_128}/{n320_128} | "
          f"{ab320_256}/{n320_256} | n/a (must be 100%) |")


if __name__ == "__main__":
    main()
