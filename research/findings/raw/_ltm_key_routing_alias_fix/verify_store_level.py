"""Repro script for the LTM key-routing alias fix (research/runners/sharded_phasor_store.py,
`build_alias_index`/`_resolve_alias`). Store-level checks through the REAL `ShardedPhasorStore` read path
(query_patient), against the actually-shipped `wikidata_core_15k` bundle -- no mocking.

Run (light path, numpy backend):
  SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_ltm_key_routing_alias_fix/verify_store_level.py
"""
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _REPO)

from research.runners.sharded_phasor_store import ShardedPhasorStore  # noqa: E402

BUNDLE = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")


def main():
    print("Loading bundle:", BUNDLE)
    ltm = ShardedPhasorStore.load(BUNDLE)
    print("total_facts:", ltm.total_facts())
    print()
    print("=== STORE-LEVEL CHECKS (direct query_patient through the real ShardedPhasorStore path) ===")

    # 1. `_portal`/`_core`-keyed entities retrieve from their BARE surface form
    r1 = ltm.query_patient("canada", "member_of")
    print("query_patient('canada', 'member_of') =", r1, " (expect 'united_nations')")
    r1b = ltm.query_patient("berlin", "country")
    print("query_patient('berlin', 'country') =", r1b, " (expect 'federal_republic_of_germany')")
    r1c = ltm.query_patient("dorset", "country")
    print("query_patient('dorset', 'country') =", r1c, " (expect 'united_kingom')")
    r1d = ltm.query_patient("ska", "instance_of")
    print("query_patient('ska', 'instance_of') [_core suffix] =", r1d, " (expect 'genre_of_music')")

    # 2. an already-working (non-suffixed) entity is UNAFFECTED
    r2 = ltm.query_patient("chelsea_fc", "country")
    print("query_patient('chelsea_fc', 'country') =", r2, " (expect 'united_kingom', unchanged)")

    # 3. MOAT: a genuinely nonexistent entity still returns None (abstain)
    r3 = ltm.query_patient("definitely_not_real_xyz", "country")
    print("query_patient('definitely_not_real_xyz', 'country') =", r3, " (expect None)")

    # 3b. MOAT: a nonexistent RELATION on a REAL suffix-keyed entity still abstains
    r3b = ltm.query_patient("canada", "definitely_not_a_real_relation_xyz")
    print("query_patient('canada', 'definitely_not_a_real_relation_xyz') =", r3b, " (expect None)")

    # 3c. MOAT: the suffixed key itself still works directly (unaffected)
    r3c = ltm.query_patient("canada_portal", "member_of")
    print("query_patient('canada_portal', 'member_of') =", r3c, " (expect 'united_nations', unchanged)")

    idx = ltm.build_alias_index()
    print()
    print("alias index size:", len(idx))
    for k in sorted(idx):
        print("  ", k, "->", idx[k])

    assert r1 == "united_nations", "FAIL: canada bare-form retrieval"
    assert r1b == "federal_republic_of_germany", "FAIL: berlin bare-form retrieval"
    assert r1c == "united_kingom", "FAIL: dorset bare-form retrieval"
    assert r1d == "genre_of_music", "FAIL: ska _core bare-form retrieval"
    assert r2 == "united_kingom", "FAIL: chelsea_fc regression"
    assert r3 is None, "MOAT BREACH: nonexistent entity returned a non-None answer"
    assert r3b is None, "MOAT BREACH: nonexistent relation on a real entity returned a non-None answer"
    assert r3c == "united_nations", "FAIL: direct suffixed-key lookup regressed"
    print()
    print("ALL STORE-LEVEL CHECKS PASSED")


if __name__ == "__main__":
    main()
