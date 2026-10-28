"""research/runners/c3_topic_stop_score.py: the reader of the C3 topic-stop dev battery (packet
research/coordination/c3-topic-stop-PACKET.md section 7, D1-D8) and of its gate-seed battery
(research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md, G1-G10 and DEC).

The dev tests read the 192 committed dev shards (research/findings/raw/_c3_topic_stop/dev, revision 79c6f5128) and pin the
dev verdict the battery was read at (D1-D8 PASS on 11/11 counted seeds, the attribution 29 vs 0) and the informational
numbers the prereg's caveats quote. The gate tests use the bar-sanity spec's synthetic six-gate-seed cell sets (the dev
shards relabelled): the failing direction first (a broken brain FAILs, a cell that cannot be read is UNDEFINED, never
PASS), then the perfect brain PASSes. No brain is built.
"""
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import c3_topic_stop_score as S  # noqa: E402
from research.runners import stage2_exit_gate as G  # noqa: E402

DEV_ROOT = os.path.join(REPO, S.DEV_DIR)
HAVE_DEV = all(os.path.exists(S.cell_path(DEV_ROOT, s, t)) for s in S.DEV_SEEDS for t in S.TAGS)
SPEC = os.path.join(REPO, "docs/drafts/c3-topic-stop-gate/bar_sanity_spec_c3ts_gate.py")
LINES = os.path.join(REPO, "research/coordination/c3-topic-stop-gate-lines.txt")
needs_dev = pytest.mark.skipif(not HAVE_DEV, reason="the dev shards are not in this checkout")


@pytest.fixture(scope="module")
def dev():
    return S.score(S.load_cells(DEV_ROOT, S.DEV_SEEDS), S.DEV_SEEDS, "dev", S.DEV_PIN, counted=S.DEV_COUNTED,
                   needs={"5": 9})


@pytest.fixture(scope="module")
def spec():
    s = importlib.util.spec_from_file_location("c3ts_gate_spec", SPEC)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _gate(cells, spec):
    return S.score(cells, S.GATE_SEEDS, "gate", spec.PIN)


# ── the dev read ────────────────────────────────────────────────────────────────────────────────────────────────────
@needs_dev
def test_dev_read_reproduces_the_dev_verdict(dev):
    assert dev["n_defined"] == 192
    for b in S.PACKET_BARS:
        t = dev["bars"][b]
        assert t["verdict"] == "PASS" and t["pass"] == 11 and t["of"] == 11, (b, t)
    assert dev["bars"]["5"]["need"] == 9 and dev["bars"]["1"]["need"] == 11
    att = dev["bars"]["4"]["attribution"]
    assert (att["treatment_intact_a"], att["control_null_toad_diffs"], att["treatment_reversed_reported"]) == (29, 0, 24)
    assert att["fraction"] == 1.0 and att["verdict"] == "PASS"
    assert dev["DEC"]["verdict"] == "PASS"
    assert "9" in dev["rows"] and 9 not in dev["counted"]          # the design seed is read, never counted


@needs_dev
def test_dev_informational_numbers_the_prereg_quotes(dev):
    rows = [dev["rows"][str(s)] for s in S.DEV_COUNTED]
    info = [r["info"] for r in rows]
    # (a) Lpinned passes every C3 clause in both orders; (b) pconst too, its vector = C on 9/11 and 11/11
    assert all(i["Lpinned_intact_a"]["all"] and i["Lpinned_reversed"]["all"] for i in info)
    assert all(i["pconst_intact_a"]["all"] and i["pconst_reversed"]["all"] for i in info)
    assert sum(i["pconst_intact_a"]["equals_C_vector"] for i in info) == 9
    assert sum(i["pconst_reversed"]["equals_C_vector"] for i in info) == 11
    # (c) the host count rule k=1 reproduces C's ask decision on 117 of 121 abstain probes
    assert sum(i["null_count"]["1"][0] for i in info) == 117 and sum(i["null_count"]["1"][1] for i in info) == 121
    # (e) C intact_b is an exact replay of intact_a (the responses compare equal as JSON)
    assert all(r["2"]["responses_identical"] for r in rows)
    # (f) every applied C probe: asked == (want_decision_hz >= threshold)
    assert all(i[t]["host_threshold_agrees"][0] == i[t]["host_threshold_agrees"][1] > 0
               for i in info for t in ("C_intact_a", "C_reversed"))
    # (g) the library answer rate: 7 of 19 replies answered, flag on and flag off, every counted seed
    assert all((r["AR"]["pairs"]["C_intact_a"]["on_answered"], r["AR"]["pairs"]["C_intact_a"]["off_answered"],
                r["AR"]["pairs"]["C_intact_a"]["n"]) == (7, 7, 19) for r in rows)
    # (a) seed 12 lesion_P intact_a still asked all three novel gaps
    s12 = dev["rows"]["12"]["info"]["lesionP_intact_a"]["asked"]
    assert s12[0] == s12[2] == s12[4] == "1"


@needs_dev
def test_gate_mode_on_dev_shards_is_undefined_without_the_prereg_argv():
    res = S.score(S.load_cells(DEV_ROOT, (7,)), (7,), "gate", S.DEV_PIN)
    assert res["n_defined"] == 0 and res["DEC"]["verdict"] == "UNDEFINED"
    assert all("--allow-gate-seed --prereg" in "; ".join(v) for v in res["cells"].values())


# ── the gate read on synthetic six-gate-seed sets ───────────────────────────────────────────────────────────────────
@needs_dev
@pytest.mark.parametrize("gen,want", [
    ("null_no_stop", "FAIL"), ("null_stop_order_carried", "FAIL"), ("null_learnable_not_asked", "FAIL"),
    ("null_moats_asked", "FAIL"), ("null_nondeterministic", "FAIL"), ("null_permuted_inert", "FAIL"),
    ("null_lesionL_inert", "FAIL"), ("null_lesionL_one_order", "FAIL"), ("null_lesionP_inert", "FAIL"),
    ("null_lesion_curiosity_asks", "FAIL"), ("null_answer_lost", "FAIL"), ("null_recalled_differs", "FAIL"),
    ("null_lesionP_not_held", "FAIL"), ("null_lesionL_not_held", "FAIL"), ("null_one_seed_no_stop", "FAIL"),
    ("null_one_seed_lesionP_inert", "FAIL"), ("null_one_seed_answer_lost", "FAIL"),
    ("null_missing_cell", "UNDEFINED"), ("null_sidecar_off_pin", "UNDEFINED"), ("null_sidecar_no_prereg", "UNDEFINED"),
    ("null_engagement_missing", "UNDEFINED"), ("null_flag_off_not_off", "UNDEFINED"),
    ("null_sidecar_env_wrong", "UNDEFINED"),
    ("perfect", "PASS"), ("perfect_goat_acquired", "PASS"), ("info_informational_cell_missing", "PASS")])
def test_gate_decision_on_synthetic_sets(spec, gen, want):
    for k in range(3):
        res = _gate(getattr(spec, gen)(np.random.default_rng(k)), spec)
        assert res["DEC"]["verdict"] == want, (gen, k, {b: v["verdict"] for b, v in res["bars"].items()})


@needs_dev
def test_one_seed_lesionP_inert_fails_g5_although_a_9_of_11_rate_would_admit_it(spec):
    res = _gate(spec.null_one_seed_lesionP_inert(np.random.default_rng(0)), spec)
    t = res["bars"]["5"]
    assert (t["pass"], t["need"], t["verdict"]) == (5, 6, "FAIL") and 5 / 6 >= 9 / 11


@needs_dev
def test_the_scorer_does_not_mutate_its_input(spec):
    cells = spec.perfect(np.random.default_rng(0))
    before = {k: id(v["shard"]["responses"]) for k, v in cells.items()}
    snap = repr(sorted((k, sorted(v["shard"]["responses"])) for k, v in cells.items()))
    _gate(cells, spec)
    assert {k: id(v["shard"]["responses"]) for k, v in cells.items()} == before
    assert repr(sorted((k, sorted(v["shard"]["responses"])) for k, v in cells.items())) == snap


# ── the pool lines ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_emitted_lines_shape():
    lines = S.emit_lines()
    assert len(lines) == 6 * 16
    for cmd, why in lines:
        m = re.search(r"--seed (\d+) --group ask --arm (\S+) --allow-gate-seed --prereg (\S+) --out (\S+)$", cmd)
        assert m, cmd
        seed, arm, prereg, out = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        tag = re.search(r"ask__(\S+)\.json$", out).group(1)
        assert seed in S.GATE_SEEDS and prereg == S.PREREG and S.TAGS[tag][0] == arm
        assert out == "%s/s%d/ask__%s.json" % (S.GATE_DIR, seed, tag)
        assert cmd.startswith("cd ~/derisk-pool/revisions/<sha> && bash tools/mem_ok.sh 9 1 && mkdir -p %s/s%d && "
                              "bash tools/memcap.sh 10 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 " % (S.GATE_DIR, seed))
        envs = dict(re.findall(r"(BRAIN_CURIOSITY[A-Z_]*)=(\S+)", cmd))
        assert envs == S.env_of(tag), (tag, envs)
        assert "prereg %s" % S.PREREG in why and why.endswith("mem_gb=9 max_age_h=240")
        assert "\t" not in cmd and "\t" not in why and "\n" not in why


def test_committed_lines_file_is_the_emitted_lines():
    with open(LINES) as f:
        got = [l.rstrip("\n") for l in f if l.strip()]
    assert got == ["%s\t%s" % cw for cw in S.emit_lines()]


def test_every_line_parses_in_the_runner(monkeypatch):
    """A dry argparse parse of every line through stage2_exit_gate.main (run_shard stubbed: no brain)."""
    seen = []

    def stub(seed, group, arm, out, backend="numpy", labels=None, allow_gate_seed=False, prereg=None):
        seen.append((seed, group, arm, out, allow_gate_seed, prereg))
        return 0
    monkeypatch.setattr(G, "run_shard", stub)
    for cmd, _why in S.emit_lines():
        argv = cmd.split("-m research.runners.stage2_exit_gate ", 1)[1].split()
        monkeypatch.setattr(sys, "argv", ["stage2_exit_gate"] + argv)
        with pytest.raises(SystemExit) as e:
            G.main()
        assert e.value.code == 0
    assert len(seen) == 96 and all(x[4] is True and x[5] == S.PREREG and x[1] == "ask" for x in seen)


def test_closure_holds_the_arms_code():
    c = set(S.import_closure())
    for f in ("research/runners/stage2_exit_gate.py", "webapp/server.py", "webapp/curiosity_topic_stop_chat.py",
              "research/runners/curiosity_topic_stop_organ.py", "research/runners/_curiosity_lp_trace_derisk.py",
              "webapp/da_curiosity_drives_chat.py", "webapp/da_mode_drives_chat.py", "webapp/one_brain.py",
              "sim/bridge.py", "sim/neuromodulators.py"):
        assert f in c, f


# ── G3's moat convention (prereg section 9 (d)) and the scorer's read files (section 5) ─────────────────────────────
@needs_dev
def test_dev_g3_reads_both_moat_conventions(dev):
    """G3 (the shard's own acquisition): g_crab excluded, moat 0.0 against gap 1.0. The Stage-2 aggregate's convention
    (intact_a's acquisition): g_newt excluded, g_crab an asked moat, moat 0.5 against gap 1.0, margin 0.0. Every dev seed."""
    for s in S.DEV_SEEDS:
        r3 = dev["rows"][str(s)]["3"]
        assert (r3["ok"], r3["ask_rate_gap"], r3["ask_rate_moat"], r3["moats_undefined"]) == (True, 1.0, 0.0, ["g_crab"])
        ac = r3["aggconv"]
        assert (ac["discriminates"], ac["tracks"], ac["ask_rate_gap"], ac["ask_rate_moat"], ac["margin"],
                ac["moats_undefined"]) == (True, True, 1.0, 0.5, 0.0, ["g_newt"]), s
    g = dev["bars"]["3"]["aggconv_reported"]
    assert (g["discriminates"], g["tracks"], g["of"], g["not_tracking_seeds"]) == (11, 11, 11, [])


@needs_dev
def test_one_permuted_gap_unasked_passes_g3_and_fails_the_aggregate_convention(spec):
    """The caveat (d) case: G3 and DEC still PASS, while the reported aggregate-convention read names the seed."""
    cells = spec.perfect(np.random.default_rng(0))
    spec._ask(cells, 43, "C_permuted", "m_yak", False)
    res = _gate(cells, spec)
    assert res["bars"]["3"]["verdict"] == "PASS" and res["DEC"]["verdict"] == "PASS"
    ac = res["rows"]["43"]["3"]["aggconv"]
    assert (ac["discriminates"], ac["tracks"]) == (False, False) and ac["margin"] < 0
    g = res["bars"]["3"]["aggconv_reported"]
    assert g["not_discriminating_seeds"] == [43] and g["not_tracking_seeds"] == [43] and g["discriminates"] == 5
    assert "aggconv (Stage-2 aggregate moat convention" in S.render(res)


@needs_dev
def test_read_files_are_every_repo_module_the_scorer_loads():
    """READ_FILES (what `--read-check` compares) = the repo files a full dev read and an identity-canary read load,
    besides the scorer itself."""
    import subprocess
    code = ("import os, sys; sys.path.insert(0, %r); os.environ['SIM_NO_PROVENANCE'] = '1'\n"
            "from research.runners import c3_topic_stop_score as S\n"
            "S.render(S.score(S.load_cells(os.path.join(%r, S.DEV_DIR), S.DEV_SEEDS), S.DEV_SEEDS, 'dev', S.DEV_PIN,"
            " counted=S.DEV_COUNTED, needs={'5': 9}))\n"
            "d = os.path.join(%r, S.DEV_DIR)\n"
            "S.render_canary(S.identity_canary(d, d, S.CANARY_SEEDS, S.DEV_PIN))\n"
            "c = S.load_cell(d, 7, 'C_intact_a')['shard']; assert S.compare_shards(c, c)['identical']\n"
            "r = os.path.realpath(%r)\n"
            "fs = {os.path.relpath(os.path.realpath(m.__file__), r) for m in list(sys.modules.values())"
            " if getattr(m, '__file__', None) and os.path.realpath(m.__file__).startswith(r + os.sep)}\n"
            "print('\\n'.join(sorted(fs)))\n") % (REPO, REPO, REPO, REPO)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=dict(os.environ, SIM_BACKEND="numpy")).stdout.split()
    assert set(out) - {"research/runners/c3_topic_stop_score.py"} == set(S.READ_FILES)


# ── the identity canary (prereg section 5, Edit 1) ──────────────────────────────────────────────────────────────────
CANARY_LINES = os.path.join(REPO, "research/coordination/c3-topic-stop-canary-lines.txt")
F_PIN = "e" * 40


def _canary_prov(seed, tag, pin=F_PIN):
    arm = S.TAGS[tag][0]
    env = {"SIM_BACKEND": "numpy", "POOL_JOB_ID": "canary-%d-%s" % (seed, tag), "POOL_JOB_MEM_GB": "9"}
    env.update(S.env_of(tag))
    return {"git_sha": pin, "git_dirty": False, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
            "source_manifest_verified_at_exit": True, "env": env, "host": "node-canary", "run_id": "x",
            "argv": ["research/runners/stage2_exit_gate.py", "--shard", "--seed", str(seed), "--group", "ask", "--arm",
                     arm, "--allow-gate-seed", "--prereg", S.PREREG, "--out", "%s/s%d/ask__%s.json"
                     % (S.CANARY_DIR, seed, tag)]}


@pytest.fixture(scope="module")
def canary_base(tmp_path_factory):
    """A canary directory that reproduces the dev shards of seeds 7 and 8 exactly, except for the host / timing fields
    (different wall times and peak RSS, as another pool host would write), with gate-mode sidecars at F_PIN."""
    root = tmp_path_factory.mktemp("canary")
    for s in S.CANARY_SEEDS:
        os.makedirs(os.path.join(root, "s%d" % s))
        for tag in S.TAGS:
            with open(S.cell_path(DEV_ROOT, s, tag)) as f:
                sh = json.load(f)
            sh["timings_s"] = {k: v + 1.5 for k, v in sh["timings_s"].items()}
            sh["wall_s"] = sh["elapsed_seconds"] = sh["wall_s"] + 77.0
            sh["import_s"] = sh["import_s"] + 0.3
            sh["process"] = dict(sh["process"], peak_rss_mb=sh["process"]["peak_rss_mb"] + 123.4)
            p = S.cell_path(str(root), s, tag)
            with open(p, "w") as f:
                json.dump(sh, f)
            with open(p + ".prov.json", "w") as f:
                json.dump(_canary_prov(s, tag), f)
    return str(root)


def _plant(base, tmp_path, seed, tag, mutate=None, prov=None, drop=False):
    """A canary dir of symlinks to `base`, with one cell's shard mutated (mutate(shard)), its sidecar replaced, or
    dropped."""
    root = tmp_path / "planted"
    for s in S.CANARY_SEEDS:
        (root / ("s%d" % s)).mkdir(parents=True)
        for t in S.TAGS:
            src = S.cell_path(base, s, t)
            dst = S.cell_path(str(root), s, t)
            if (s, t) != (seed, tag):
                os.symlink(src, dst)
                os.symlink(src + ".prov.json", dst + ".prov.json")
                continue
            if drop:
                continue
            with open(src) as f:
                sh = json.load(f)
            if mutate:
                mutate(sh)
            with open(dst, "w") as f:
                json.dump(sh, f)
            with open(dst + ".prov.json", "w") as f:
                json.dump(prov if prov is not None else _canary_prov(s, t), f)
    return str(root)


def _probe(sh, applied=True):
    """The first probe label whose topic-stop trace is applied (or the first probe, applied=False)."""
    for t in sh["turns"]:
        ts = S.ts_of(sh["responses"].get(t["label"]))
        if t["role"] == "probe" and (not applied or (isinstance(ts, dict) and ts.get("applied"))):
            return t["label"]
    raise AssertionError("no probe")


@needs_dev
def test_canary_reproducing_the_dev_shards_reads_identical_on_all_32_arms(canary_base):
    res = S.identity_canary(canary_base, DEV_ROOT, S.CANARY_SEEDS, F_PIN)
    assert res["n_arms"] == 32 and res["n_identical"] == 32 and res["verdict"] == "PASS", \
        {k: a.get("why") or a.get("first") for k, a in res["arms"].items() if a["status"] != "IDENTICAL"}
    assert all(a["n_compared"] > 1000 for a in res["arms"].values())       # not a vacuous compare
    assert "committed check: git status" in res["reference_note"]
    txt = S.render_canary(res)
    assert sum(1 for ln in txt.split("\n") if ln.startswith("  canary ") and " IDENTICAL (" in ln) == 32
    assert "CANARY PASS: 32/32 IDENTICAL" in txt
    for f in S.CANARY_EXCLUDED:
        assert f in txt


@needs_dev
def test_canary_dev_against_itself_reads_identical(tmp_path):
    """Dev vs itself: every committed dev shard (all 12 dev seeds, 16 arms) compared with an independent re-load of
    itself has no difference over >1000 compared leaves; and a canary directory whose shards ARE the committed dev
    files of seeds 7 and 8 (symlinks, byte for byte; only the sidecars are gate-mode ones at F_PIN) reads CANARY PASS,
    32 of 32 IDENTICAL, through the CLI."""
    for s in S.DEV_SEEDS:
        for t in S.TAGS:
            a, b = S.load_cell(DEV_ROOT, s, t)["shard"], S.load_cell(DEV_ROOT, s, t)["shard"]
            assert a is not b
            r = S.compare_shards(a, b)
            assert r["identical"] and r["n_diffs"] == 0 and r["first"] is None and r["n_compared"] > 1000, (s, t)
    root = tmp_path / "dev_self"
    for s in S.CANARY_SEEDS:
        (root / ("s%d" % s)).mkdir(parents=True)
        for t in S.TAGS:
            dst = S.cell_path(str(root), s, t)
            os.symlink(S.cell_path(DEV_ROOT, s, t), dst)
            with open(dst + ".prov.json", "w") as f:
                json.dump(_canary_prov(s, t), f)
    res = S.identity_canary(str(root), DEV_ROOT, S.CANARY_SEEDS, F_PIN)
    assert res["verdict"] == "PASS" and res["n_identical"] == 32 and res["n_arms"] == 32
    assert S.main(["--identity-canary", str(root), "--dev-dir", DEV_ROOT, "--seeds", "7,8", "--pin", F_PIN,
                   "--no-closure"]) == 0


@needs_dev
def test_canary_exclusions_are_what_differs_between_the_two_intact_runs(monkeypatch):
    """Prereg section 5: on all 12 dev seeds C_intact_b (the same arm and seed run again, on another pool host on 9
    seeds) differs from C_intact_a only in CANARY_EXCLUDED and the arm's own name (`arm`, `session`), and every
    excluded field differs on some seed, so the exclusion list is exactly the host / timing noise, no wider."""
    seen, hosts = set(), 0
    for s in S.DEV_SEEDS:
        a, b = S.load_cell(DEV_ROOT, s, "C_intact_a"), S.load_cell(DEV_ROOT, s, "C_intact_b")
        hosts += a["prov"].get("host") != b["prov"].get("host")
        assert {p for p, _x, _y in S.walk_diffs(a["shard"], b["shard"])} == {"arm", "session"}, s
        monkeypatch.setattr(S, "CANARY_EXCLUDED", ())
        for p, _x, _y in S.walk_diffs(a["shard"], b["shard"]):
            top = p.split(".")[0] if not p.startswith("process.") else p
            seen.add(top)
        monkeypatch.undo()
    assert seen == set(S.CANARY_EXCLUDED) | {"arm", "session"}
    assert hosts == 9


@needs_dev
def test_canary_one_planted_field_difference_differs_with_its_path(canary_base, tmp_path):
    with open(S.cell_path(canary_base, 8, "C_intact_a")) as f:
        lab = _probe(json.load(f))

    def bump(sh):
        ts = sh["responses"][lab]["curiosity"]["curiosity_topic_stop"]
        ts["want_decision_hz"] = ts["want_decision_hz"] + 1e-9
    root = _plant(canary_base, tmp_path, 8, "C_intact_a", mutate=bump)
    res = S.identity_canary(root, DEV_ROOT, S.CANARY_SEEDS, F_PIN)
    a = res["arms"]["8/C_intact_a"]
    want = "responses.%s.curiosity.curiosity_topic_stop.want_decision_hz" % lab
    assert a["status"] == "DIFFERS" and a["n_diffs"] == 1 and a["first"]["path"] == want
    assert a["first_decision"]["path"] == want and S.decision_bearing(want)
    assert res["n_identical"] == 31 and res["verdict"] == "STOP"
    txt = S.render_canary(res)
    assert "canary 8/C_intact_a DIFFERS at %s" % want in txt and "CANARY STOP: 31/32" in txt


@needs_dev
@pytest.mark.parametrize("what", ["answer", "abstained", "recalled_svo", "reward_us_spikes", "stp_tau_d_ms_type",
                                  "lesion_readback", "curiosity_da", "affect", "extra_key", "turn_dropped"])
def test_canary_each_kind_of_difference_differs(canary_base, tmp_path, what):
    tag = {"lesion_readback": "lesionL_intact_a", "curiosity_da": "off_intact_a"}.get(what, "C_intact_a")
    with open(S.cell_path(canary_base, 7, tag)) as f:
        sh0 = json.load(f)
    if what == "curiosity_da":
        lab = next(t["label"] for t in sh0["turns"]
                   if isinstance(((sh0["responses"][t["label"]] or {}).get("curiosity") or {}).get("curiosity_da"), dict))
    else:
        lab = _probe(sh0, applied=what not in ("answer", "abstained", "recalled_svo", "affect", "extra_key"))

    def mutate(sh):
        r = sh["responses"][lab]
        ts = (r.get("curiosity") or {}).get("curiosity_topic_stop")
        if what == "answer":
            r["answer"] = r["answer"] + " "
        elif what == "abstained":
            r["abstained"] = not r["abstained"]
        elif what == "recalled_svo":
            r["recalled_svo"] = ["toad", "eat", "fly"]
        elif what == "reward_us_spikes":
            ts["phasic_da"]["reward_us_spikes"] += 1
        elif what == "stp_tau_d_ms_type":                    # 12000.0 -> 12000: the value equal, the type not
            ts["held"]["stp_tau_d_ms"] = int(ts["held"]["stp_tau_d_ms"])
        elif what == "lesion_readback":
            ts["held"]["stp_tau_d_ms"] = 201.0
        elif what == "curiosity_da":
            r["curiosity"]["curiosity_da"]["eff_threshold"] = r["curiosity"]["curiosity_da"]["eff_threshold"] * 1.01
        elif what == "affect":
            r["affect"]["pos_rate"] = 0.5
        elif what == "extra_key":
            r["new_trace"] = {"on": False}
        elif what == "turn_dropped":
            del sh["responses"][lab]
    root = _plant(canary_base, tmp_path, 7, tag, mutate=mutate)
    res = S.identity_canary(root, DEV_ROOT, S.CANARY_SEEDS, F_PIN)
    a = res["arms"]["7/%s" % tag]
    assert a["status"] == "DIFFERS" and res["verdict"] == "STOP", (what, a)
    assert a["first"]["path"].startswith("responses.%s" % lab), (what, a["first"])
    if what in ("answer", "abstained", "recalled_svo", "reward_us_spikes", "stp_tau_d_ms_type", "lesion_readback",
                "curiosity_da"):
        assert a["first_decision"] is not None and a["first_decision"]["path"] == a["first"]["path"]
    if what == "extra_key":
        assert a["first"]["dev"] == "<absent>"
    if what == "turn_dropped":
        assert a["first"]["canary"] == "<absent>"


@needs_dev
def test_canary_first_differing_path_follows_the_dev_shard_order(canary_base, tmp_path):
    with open(S.cell_path(canary_base, 7, "C_reversed")) as f:
        labels = [t["label"] for t in json.load(f)["turns"]]

    def two(sh):
        sh["responses"][labels[-1]]["answer"] += "!"
        sh["responses"][labels[1]]["answer"] += "!"
    root = _plant(canary_base, tmp_path, 7, "C_reversed", mutate=two)
    a = S.identity_canary(root, DEV_ROOT, S.CANARY_SEEDS, F_PIN)["arms"]["7/C_reversed"]
    assert a["n_diffs"] == 2 and a["first"]["path"] == "responses.%s.answer" % labels[1]


@needs_dev
@pytest.mark.parametrize("field", ["timings_s", "wall_s", "import_s", "elapsed_seconds", "process.peak_rss_mb"])
def test_canary_ignores_exactly_the_listed_host_timing_fields(canary_base, tmp_path, field):
    def m(sh):
        if field == "timings_s":
            sh["timings_s"] = {"x": 1.0}
        elif field == "process.peak_rss_mb":
            sh["process"]["peak_rss_mb"] = 1.0
        else:
            sh[field] = -1.0
    root = _plant(canary_base, tmp_path, 8, "pconst_reversed", mutate=m)
    assert S.identity_canary(root, DEV_ROOT, S.CANARY_SEEDS, F_PIN)["arms"]["8/pconst_reversed"]["status"] == "IDENTICAL"
    assert set(S.CANARY_EXCLUDED) == {"timings_s", "import_s", "wall_s", "elapsed_seconds", "process.peak_rss_mb"}


@needs_dev
def test_canary_non_excluded_shard_level_fields_are_compared(canary_base, tmp_path):
    def m(sh):
        sh["process"]["ltm_bundle"] = "other_bundle"
    root = _plant(canary_base, tmp_path, 8, "off_permuted", mutate=m)
    a = S.identity_canary(root, DEV_ROOT, S.CANARY_SEEDS, F_PIN)["arms"]["8/off_permuted"]
    assert a["status"] == "DIFFERS" and a["first"]["path"] == "process.ltm_bundle" and a["first_decision"] is None


@needs_dev
@pytest.mark.parametrize("case", ["missing", "off_pin", "no_prereg", "env_wrong", "dirty"])
def test_canary_cell_that_cannot_be_read_is_undefined_never_identical(canary_base, tmp_path, case):
    prov = _canary_prov(7, "lesionP_reversed")
    if case == "off_pin":
        prov["git_sha"] = "d" * 40
    elif case == "no_prereg":
        prov["argv"] = [x for x in prov["argv"] if x not in ("--allow-gate-seed", "--prereg", S.PREREG)]
    elif case == "env_wrong":
        prov["env"].pop("BRAIN_CURIOSITY_PHASIC_DA_LESION")
    elif case == "dirty":
        prov["git_dirty"] = True
    root = _plant(canary_base, tmp_path, 7, "lesionP_reversed", prov=prov, drop=(case == "missing"))
    res = S.identity_canary(root, DEV_ROOT, S.CANARY_SEEDS, F_PIN)
    a = res["arms"]["7/lesionP_reversed"]
    assert a["status"] == "UNDEFINED" and res["verdict"] == "STOP" and res["n_undefined"] == 1
    assert all(w.startswith("canary: ") for w in a["why"])


@needs_dev
def test_canary_pin_must_be_the_canary_revision(canary_base):
    res = S.identity_canary(canary_base, DEV_ROOT, S.CANARY_SEEDS, "d" * 40)
    assert res["n_undefined"] == 32 and res["verdict"] == "STOP"


@needs_dev
def test_canary_dev_reference_must_be_the_committed_file(canary_base, monkeypatch):
    rel = os.path.relpath(S.cell_path(DEV_ROOT, 7, "C_permuted"), REPO)
    monkeypatch.setattr(S, "_git", lambda *a: " M %s\n" % rel if a[0] == "status" else "")
    res = S.identity_canary(canary_base, DEV_ROOT, S.CANARY_SEEDS, F_PIN)
    a = res["arms"]["7/C_permuted"]
    assert a["status"] == "UNDEFINED" and "not the committed file" in "; ".join(a["why"])
    assert res["n_identical"] == 31 and res["verdict"] == "STOP"


@needs_dev
def test_canary_dev_reference_must_be_measured_at_the_dev_pin(canary_base, tmp_path):
    """A dev root whose sidecars name another revision is not the dev reference."""
    dev2 = tmp_path / "dev2"
    for s in S.CANARY_SEEDS:
        (dev2 / ("s%d" % s)).mkdir(parents=True)
        for t in S.TAGS:
            src = S.cell_path(DEV_ROOT, s, t)
            os.symlink(src, S.cell_path(str(dev2), s, t))
            with open(src + ".prov.json") as f:
                pv = json.load(f)
            if (s, t) == (8, "Lpinned_reversed"):
                pv["git_sha"] = "c" * 40
            with open(S.cell_path(str(dev2), s, t) + ".prov.json", "w") as f:
                json.dump(pv, f)
    res = S.identity_canary(canary_base, str(dev2), S.CANARY_SEEDS, F_PIN)
    assert res["reference_note"].startswith("committed check skipped")
    a = res["arms"]["8/Lpinned_reversed"]
    assert a["status"] == "UNDEFINED" and all(w.startswith("dev reference: ") for w in a["why"])
    assert res["n_identical"] == 31


def test_canary_refuses_gate_seeds():
    for gs in S.GATE_SEEDS:
        with pytest.raises(ValueError):
            S.identity_canary("x", "y", (7, gs), F_PIN)
        with pytest.raises(ValueError):
            S.emit_canary_lines(seeds=(gs,))


@needs_dev
def test_canary_cli_exit_codes(canary_base, tmp_path, capsys):
    assert S.main(["--identity-canary", canary_base, "--dev-dir", DEV_ROOT, "--pin", F_PIN, "--no-closure",
                   "--out", str(tmp_path / "c.json")]) == 0
    out = capsys.readouterr().out
    assert "CANARY PASS: 32/32 IDENTICAL" in out
    with open(tmp_path / "c.json") as f:
        assert json.load(f)["verdict"] == "PASS"

    def m(sh):
        sh["responses"][sh["turns"][0]["label"]]["answer"] = "x"
    root = _plant(canary_base, tmp_path, 7, "C_intact_b", mutate=m)
    assert S.main(["--identity-canary", root, "--dev-dir", DEV_ROOT, "--pin", F_PIN, "--no-closure"]) == 1
    assert "canary 7/C_intact_b DIFFERS at responses." in capsys.readouterr().out
    with pytest.raises(SystemExit):
        S.main(["--identity-canary", root, "--pin", "abc"])


# ── the canary lines ────────────────────────────────────────────────────────────────────────────────────────────────
def test_canary_lines_are_the_gate_lines_except_seed_and_out():
    gate = {}
    for cmd, _why in S.emit_lines():
        m = re.search(r"--seed (\d+) .*ask__(\S+)\.json$", cmd)
        gate[(int(m.group(1)), m.group(2))] = cmd
    lines = S.emit_canary_lines()
    assert len(lines) == 2 * 16
    keys = []
    for cmd, why in lines:
        m = re.search(r"--seed (\d+) --group ask --arm (\S+) --allow-gate-seed --prereg (\S+) --out (\S+)$", cmd)
        assert m, cmd
        s, arm, out = int(m.group(1)), m.group(2), m.group(4)
        tag = re.search(r"ask__(\S+)\.json$", out).group(1)
        keys.append((s, tag))
        assert s in S.CANARY_SEEDS and s not in S.GATE_SEEDS and S.TAGS[tag][0] == arm
        assert out == "%s/s%d/ask__%s.json" % (S.CANARY_DIR, s, tag)
        assert cmd.startswith("cd ~/derisk-pool/revisions/<sha> && ")
        # byte-identical to the gate line of the same arm once the seed and the output directory are swapped back
        g = cmd.replace("%s/s%d" % (S.CANARY_DIR, s), "%s/s42" % S.GATE_DIR).replace("--seed %d " % s, "--seed 42 ")
        assert g == gate[(42, tag)]
        assert why.startswith("prereg %s (F <sha>); " % S.PREREG) and "IDENTITY CANARY" in why
        assert why.endswith("mem_gb=9 max_age_h=240") and "gate seed %d" % s not in why
        assert "\t" not in cmd and "\t" not in why and "\n" not in why
    assert keys == [(s, t) for s in S.CANARY_SEEDS for t in S.TAGS]


def test_committed_canary_lines_file_is_the_emitted_canary_lines():
    with open(CANARY_LINES) as f:
        got = [l.rstrip("\n") for l in f if l.strip()]
    assert got == ["%s\t%s" % cw for cw in S.emit_canary_lines()]


def test_every_canary_line_parses_in_the_runner(monkeypatch):
    seen = []

    def stub(seed, group, arm, out, backend="numpy", labels=None, allow_gate_seed=False, prereg=None):
        seen.append((seed, group, arm, out, allow_gate_seed, prereg))
        return 0
    monkeypatch.setattr(G, "run_shard", stub)
    for cmd, _why in S.emit_canary_lines():
        argv = cmd.split("-m research.runners.stage2_exit_gate ", 1)[1].split()
        monkeypatch.setattr(sys, "argv", ["stage2_exit_gate"] + argv)
        with pytest.raises(SystemExit) as e:
            G.main()
        assert e.value.code == 0
    assert len(seen) == 32 and {x[0] for x in seen} == set(S.CANARY_SEEDS)
    assert all(x[4] is True and x[5] == S.PREREG and x[3].startswith(S.CANARY_DIR + "/") for x in seen)


def test_the_gate_lines_file_carries_the_sha_placeholder_not_a_revision():
    for path in (LINES, CANARY_LINES):
        with open(path) as f:
            body = f.read()
        assert body.count("cd ~/derisk-pool/revisions/<sha> && ") == len(body.strip().split("\n"))
        assert not re.search(r"revisions/[0-9a-f]{7,40}", body)
