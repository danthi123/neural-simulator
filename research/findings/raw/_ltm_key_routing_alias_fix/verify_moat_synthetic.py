"""Repro script for the LTM key-routing alias fix: ADVERSARIAL synthetic moat checks -- cases NOT present in
the real 15k bundle, proving the fallback's two safety invariants hold in general, not just on the shipped
data:
  (A) an AMBIGUOUS suffix-strip (two distinct stored keys strip to the same bare form) must NOT resolve --
      an ambiguous bare form abstains rather than guessing which of two entities the user meant.
  (B) a bare form that is ALREADY a real, distinctly-keyed entity must NEVER be shadowed by a co-existing
      suffixed entity -- the real entity's own facts always win.

Run:
  SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_ltm_key_routing_alias_fix/verify_moat_synthetic.py
"""
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _REPO)

from research.runners.sharded_phasor_store import ShardedPhasorStore  # noqa: E402


def main():
    vocab = ["foo", "foo_portal", "foo_core", "bar", "bar_portal", "x", "y", "z", "rel1", "rel2"]
    s = ShardedPhasorStore(n_shards=4, seed=1, D=32, vocab=vocab)

    # CASE A: ambiguous strip -> must NOT resolve
    s.store("foo_portal", "rel1", "x")
    s.store("foo_core", "rel1", "y")
    idx = s.build_alias_index()
    print("CASE A alias index:", idx)
    assert "foo" not in idx, f"MOAT BREACH: ambiguous bare form 'foo' resolved to {idx.get('foo')}"
    r = s.query_patient("foo", "rel1")
    print("query_patient('foo','rel1') [ambiguous case] =", r, "(expect None)")
    assert r is None, "MOAT BREACH: ambiguous bare form produced an answer"

    # CASE B: bare form ALREADY exists as its own distinct entity -> must never be shadowed
    s.store("bar", "rel2", "z")
    s.store("bar_portal", "rel2", "y")
    idx2 = s.build_alias_index(force=True)
    print("CASE B alias index:", idx2)
    assert "bar" not in idx2, f"MOAT BREACH: real entity 'bar' got shadowed by alias -> {idx2.get('bar')}"
    r2 = s.query_patient("bar", "rel2")
    print("query_patient('bar','rel2') [real entity, must not be shadowed] =", r2, "(expect 'z', bar's own fact)")
    assert r2 == "z", f"MOAT BREACH: 'bar' resolved to the wrong (shadowed) fact: {r2}"
    r2b = s.query_patient("bar_portal", "rel2")
    print("query_patient('bar_portal','rel2') [direct, unaffected] =", r2b, "(expect 'y')")
    assert r2b == "y"

    print()
    print("ALL SYNTHETIC MOAT-ADVERSARIAL CHECKS PASSED")


if __name__ == "__main__":
    main()
