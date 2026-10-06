"""tools/pool_queue.sh withdraw (cloud Claude session 7, RULING transitive-engagement): removes queued, undispatched
lines by a fixed substring under the dispatcher's lock; the match count must equal --expect, a reason is required, and
every withdrawn line is logged verbatim so it can be re-added."""
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _queue(tmp_path):
    q = tmp_path / "pool.queue"
    now = int(time.time())
    lines = ["%d\tcd x && run --out _shards/s2p-R-tr-abc1234/s%d/lb.json  #checked:prereg y ;" % (now, i)
             for i in range(12)]
    lines += ["%d\tcd x && run --out _shards/s2p-R-wr-abc1234/s%d/lb.json  #checked:prereg z ;" % (now, i)
              for i in range(2)]
    q.write_text("\n".join(lines) + "\n")
    return q


def _run(q, *args):
    env = dict(os.environ, POOL_QUEUE_PATH=str(q))
    return subprocess.run(["bash", os.path.join(REPO, "tools", "pool_queue.sh"), "withdraw", *args],
                          capture_output=True, text=True, env=env, cwd=REPO)


def test_withdraw_exact_count(tmp_path):
    q = _queue(tmp_path)
    r = _run(q, "/s2p-R-tr-", "--expect", "12", "--reason", "RULING x")
    assert r.returncode == 0, r.stderr
    rest = q.read_text().splitlines()
    assert len(rest) == 2 and all("s2p-R-wr-" in l for l in rest)
    log = (tmp_path / "withdrawn.log").read_text().splitlines()
    assert len(log) == 12 and all("\tRULING x\t" in l and "s2p-R-tr-" in l for l in log)


def test_withdraw_refuses_a_count_mismatch_and_changes_nothing(tmp_path):
    q = _queue(tmp_path)
    before = q.read_text()
    r = _run(q, "-abc1234/s", "--expect", "12", "--reason", "too broad")   # matches all 14
    assert r.returncode == 3 and q.read_text() == before and not (tmp_path / "withdrawn.log").exists()


def test_withdraw_requires_a_reason(tmp_path):
    q = _queue(tmp_path)
    assert _run(q, "/s2p-R-tr-", "--expect", "12").returncode == 2


def test_withdraw_refuses_a_short_pattern(tmp_path):
    q = _queue(tmp_path)
    assert _run(q, "s2p-R-", "--expect", "14", "--reason", "r").returncode == 2
