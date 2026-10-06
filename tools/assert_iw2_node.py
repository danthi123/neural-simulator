#!/usr/bin/env python3
"""Node preflight for integration-wave-2 job lines: refuse (exit 3) to run on a node that would build a DEGRADED brain.

Wave-1 review lesson, built in from the start (research/packets/integration-wave1/HANDOFF.md "Adversarial review"):
a pool revision dir without data/corpus/ builds a degraded brain with no error anywhere in the replies (research/
FAILURE_LOG.md 2026-09-23 and 2026-09-25), and a node without the knowledge-bundle data lake builds a brain with no
long-term memory. load_bearing_fraction.py already refuses without the corpus (its CORPUS GUARD, exit 3); nothing in a
load-bearing shard checked the LTM bundle. Every wave-2 job line therefore starts, right after `cd <revision dir>`, with

    .venv/bin/python tools/assert_iw2_node.py &&

which exits 3 unless the four corpus files the CORPUS GUARD names exist under data/corpus/ of the working directory
AND the shipped LTM default resolves a knowledge bundle there (webapp.server._default_ltm_bundle_dir, the SAME
resolver the handler uses, imported, not re-implemented). It prints the resolved bundle name, so the dispatch log
records which LTM each shard ran with. It builds no brain.

  .venv/bin/python tools/assert_iw2_node.py            # exit 0 ok, 3 degraded
"""
import os
import sys

CORPUS_FILES = ("tinystories.txt", "wikitext.txt", "simplewiki.txt", "websters1913.json")


def check(root=None, resolver=None):
    """Returns (ok, message). `resolver` (tests) replaces the server's LTM resolver: () -> (ship_default_on, dir)."""
    root = root or os.getcwd()
    missing = [f for f in CORPUS_FILES if not os.path.exists(os.path.join(root, "data", "corpus", f))]
    if missing:
        return False, "data/corpus/ in %s is missing %s" % (root, missing)
    if resolver is None:
        sys.path.insert(0, root)
        from webapp import server as _server  # noqa: E402  (imports only; no app start, no brain build)

        def resolver():
            on = _server._ltm_ship_default_on()
            return on, (_server._default_ltm_bundle_dir() if on else None)
    on, d = resolver()
    if not on:
        return False, "the shipped LTM default is switched off in this environment (BRAIN_LTM_SHIP_DEFAULT=%r)" % (
            os.environ.get("BRAIN_LTM_SHIP_DEFAULT"))
    if not d:
        return False, "the shipped LTM default resolves no knowledge bundle (BRAIN_DATA_ROOT=%r)" % (
            os.environ.get("BRAIN_DATA_ROOT"))
    return True, "corpus ok; ltm_bundle=%s" % os.path.basename(d)


def main():
    ok, msg = check()
    if not ok:
        print("⛔ assert_iw2_node: %s. Refusing to run: the shard would measure a degraded brain. Sync the corpus / "
              "knowledge bundles (tools/pool_provision.sh, tools/pool_sync_assets.sh) and re-queue." % msg,
              file=sys.stderr, flush=True)
        return 3
    print("assert_iw2_node: %s" % msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
