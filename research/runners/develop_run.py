"""Self-driven, RESUMABLE longitudinal develop run -- the hands-off artificial-life run.

Wraps the VALIDATED `develop_gpu` day-loop (research/runners/_longitudinal_develop_loop_gpu.py)
for unattended operation, so it can be driven with NO Claude in the loop:

  * a STABLE persistent lineage  -> resumes across restarts (re-run = continue where it stopped)
  * a PAUSE sentinel file        -> stop cleanly at the next day boundary (zero completed work lost)
  * per-day BUNDLES              -> the chat console can --load a day-N brain and talk to it
  * the TinyStories corpus       -> the brain keeps growing vocab+facts for days (won't plateau)

Everything lives under bridges/developed/run3day/ (lineage/, bundles/, PAUSE).

Usage (or just use scripts/develop.ps1 which wraps these):
  start / resume :  SIM_BACKEND=cupy python -m research.runners.develop_run
  pause          :  create the file bridges/developed/run3day/PAUSE   (it stops at the next day)
  resume         :  delete that PAUSE file, then re-run the start command
  status         :  python -m research.runners.develop_run --status     (no GPU needed)

The day-loop persists each completed day to the lineage BEFORE the pause check, so stopping
(PAUSE, Ctrl-C, power loss) never loses a completed day -- the next start resumes from it.
This is a thin orchestration wrapper: NO sim/ edit, reuse-by-import of the validated loop.
"""
import os
import sys
import time
import argparse
from pathlib import Path

# everything for this run is self-contained under one stable root
ROOT = os.path.join("bridges", "developed", "run3day")
PAUSE_FILE = os.path.join(ROOT, "PAUSE")
# Day bundles save DIRECTLY under ROOT as run3day/day_<N> (DEPTH 2 below bridges/developed/),
# because the dashboard's _scan_developed_bundles only scans depth 1 + depth 2 -- a bundles/
# subdir (depth 3) would be invisible to the brain picker.
BUNDLE_ROOT = ROOT
LINEAGE_ROOT = os.path.join(ROOT, "lineage")
LINEAGE_NAME = "develop_3day"


def _status():
    """Print the current developmental state (day / vocab / facts / tier). CPU-only, no GPU."""
    from sim.lineage import BridgeLineage
    lin = BridgeLineage(LINEAGE_NAME, root=Path(LINEAGE_ROOT))
    if not lin.exists():
        print("[develop_run] no run yet -- start it to begin development at day 0.")
        return 0
    from research.runners._longitudinal_develop_loop_gpu import _load_state
    st = _load_state(lin)
    paused = "YES (remove the PAUSE file to resume)" if os.path.exists(PAUSE_FILE) else "no"
    print(f"[develop_run] day={st.day}  vocab={len(st.vocab)}  facts={len(st.facts)}  tier={st.current_tier}")
    print(f"  paused={paused}")
    print(f"  developed-day brains (load a day_<N>/ in the dashboard brain picker): "
          f"{os.path.abspath(BUNDLE_ROOT)}\\day_<N>")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Self-driven resumable longitudinal develop run.")
    ap.add_argument("--n-days", type=int, default=3650,
                    help="max simulated days for THIS invocation (resume continues from the saved day; "
                         "default 3650 ~= 10 sim-years, far more than a 3-day window reaches -- the PAUSE "
                         "sentinel / Ctrl-C is how you stop, not this cap)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-windows-per-day", type=int, default=2500,
                    help="stream-window budget per day (caps per-day wall-clock)")
    ap.add_argument("--n-hub", type=int, default=200, help="stream-cortex hub (context-word) count")
    ap.add_argument("--n-per", type=int, default=12, help="neurons per concept (population code)")
    ap.add_argument("--D", type=int, default=128, help="composer phasor dimension")
    ap.add_argument("--develop-D", type=int, default=None,
                    help="the develop-loop phasor dimension for BOTH the StreamCortex AND the conversational "
                         "composer (they MUST match or the learned codes are dropped by _inject_grounded). "
                         "Default None = use --D. Raise it (e.g. 512) to lift the recall/abstention margin at "
                         "100s of concepts (FHRR capacity ~sqrt(D)); pair with --corpus-curriculum (which kills "
                         "the multi-turn WM-loop's ~quadratic VRAM so a bigger-D 100s-concept run fits 24GB).")
    ap.add_argument("--corpus-path", default=None,
                    help="plain-text corpus shard; default = the wired data/corpus/tinystories.txt")
    ap.add_argument("--status", action="store_true", help="print day/vocab/facts and exit (no GPU)")
    ap.add_argument("--corpus-curriculum", action="store_true",
                    help="scale the run with a CORPUS-GROUNDED curriculum (thousands of TinyStories-frequent "
                         "concepts + real corpus SVO facts) instead of the hardcoded ~24-concept demo schedule")
    ap.add_argument("--brain-npz", default="bridges/firstchat/brain3000pos_w7000.npz_seed42.npz",
                    help="(--corpus-curriculum) the trained brain whose high-frequency-first vocab orders the curriculum")
    ap.add_argument("--facts-json", default="research/findings/raw/_facts3000.json",
                    help="(--corpus-curriculum) the corpus-extracted SVO facts asserted as the brain's vocab grows")
    ap.add_argument("--concepts-per-day", type=int, default=24,
                    help="(--corpus-curriculum) new concepts introduced per simulated day")
    ap.add_argument("--max-concepts", type=int, default=None,
                    help="(--corpus-curriculum) optional cap on total concepts (None = the full corpus vocab)")
    a = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    os.makedirs(BUNDLE_ROOT, exist_ok=True)
    os.makedirs(LINEAGE_ROOT, exist_ok=True)

    if a.status:
        return _status()

    if os.path.exists(PAUSE_FILE):
        print(f"[develop_run] PAUSE file present -> not starting.\n  remove it to run: {os.path.abspath(PAUSE_FILE)}",
              flush=True)
        return 0

    from sim.lineage import BridgeLineage
    from research.runners._longitudinal_develop_loop_gpu import (
        GPUGradedCurriculum, StreamCortex, develop_gpu)
    from research.runners.developed_brain_io import save_developed_brain

    if a.corpus_curriculum:
        from research.runners._corpus_develop_curriculum import (
            CorpusGradedCurriculum, load_concepts_and_facts)
        concepts, facts = load_concepts_and_facts(a.brain_npz, a.facts_json)
        if a.max_concepts:
            concepts = concepts[:int(a.max_concepts)]
        curriculum = CorpusGradedCurriculum(concepts, facts, concepts_per_day=a.concepts_per_day)
        print(f"[develop_run] CORPUS curriculum: {len(concepts)} concepts, {len(facts)} corpus facts, "
              f"{curriculum.n_authored_days()} days @ {a.concepts_per_day}/day", flush=True)
    else:
        curriculum = GPUGradedCurriculum()
    full_vocab = curriculum.full_vocab()
    lineage = BridgeLineage(LINEAGE_NAME, root=Path(LINEAGE_ROOT))
    resume = lineage.exists()

    # The develop-loop phasor dimension (StreamCortex AND composer; --develop-D overrides --D). They MUST stay equal
    # or the stream-learned grounded codes (length develop_D) are silently dropped by _inject_grounded against the
    # composer's D. ONE value drives BOTH below, so they cannot diverge. The KNOWLEDGE-SCALING combo (scoping
    # 2026-06-27): use_multiturn=False on the corpus-curriculum path kills the multi-turn WM loop's ~quadratic VRAM
    # (so a bigger-D 100s-of-concepts run fits 24GB), and develop_D threads the composer dimension recall needs.
    develop_D = int(a.develop_D) if a.develop_D is not None else int(a.D)
    # The corpus-curriculum (scaling) path drops the multi-turn discourse WM loop -- the per-day battery (recall /
    # heldout / retain / chain / moat) needs no cross-turn anaphora, and the WM loop is the ~quadratic-VRAM OOM at
    # 100s of concepts. The hardcoded ~24-concept demo schedule keeps the multi-turn agent (byte-identical).
    use_multiturn = not a.corpus_curriculum

    print("=" * 100, flush=True)
    print(f"[develop_run] {'RESUME' if resume else 'START (day 0)'}  "
          f"backend={os.environ.get('SIM_BACKEND')}  max-days-this-run={a.n_days}  "
          f"D={develop_D}  multi_turn={use_multiturn}", flush=True)
    print(f"  PAUSE  : create  {os.path.abspath(PAUSE_FILE)}   -> stops cleanly at the next day boundary",
          flush=True)
    print(f"  RESUME : remove that file, re-run this command (auto-resumes from the saved day)", flush=True)
    print(f"  BUNDLES: {os.path.abspath(BUNDLE_ROOT)}  (load a day_<N>/ in the chat console to talk to it)",
          flush=True)
    print("=" * 100, flush=True)

    # one shared stream cortex for this invocation; develop_gpu re-hears the cumulative vocab on resume so the
    # freshly-built cortex re-acquires the developed codes (the validated resume path).
    shared = StreamCortex(full_vocab, a.seed, n_hub=a.n_hub, n_per=a.n_per, D=develop_D,
                          verbose=True, corpus_path=a.corpus_path)

    def per_day_hook(day_index, state, grounded, agent):
        """Save a console-loadable developed-brain bundle for this day."""
        bdir = os.path.join(BUNDLE_ROOT, f"day_{day_index}")
        comp = getattr(agent, "agent", agent).composer
        try:
            save_developed_brain(agent, bdir, seed=int(a.seed), D=int(getattr(comp, "D", develop_D)),
                                 composer_kind="rf", develop_state=state, lineage_name="developed_brain",
                                 extra_metadata={"provenance": "develop_run", "day": int(day_index)})
            print(f"    [bundle] day-{day_index} -> {bdir}  (load it in the console)", flush=True)
        except Exception as exc:  # bundle save is non-fatal -- the day is already persisted to the lineage
            print(f"    [bundle] day-{day_index} save failed (non-fatal): {exc!r}", flush=True)

    def should_continue():
        """Polled before each day. The prior day is already durably persisted, so a PAUSE here loses nothing."""
        return not os.path.exists(PAUSE_FILE)

    t0 = time.time()
    try:
        per_day, _assembly = develop_gpu(
            lineage, curriculum, a.n_days, seed=a.seed, consolidation_on=True, plasticity_on=True,
            max_windows_per_day=a.max_windows_per_day, n_hub=a.n_hub, n_per=a.n_per, D=develop_D,
            resume=resume, verbose=True, _shared_cortex=shared, per_day_save_hook=per_day_hook,
            corpus_path=a.corpus_path, should_continue=should_continue, use_multiturn=use_multiturn)
    except KeyboardInterrupt:
        print("\n[develop_run] interrupted (Ctrl-C). The last COMPLETED day is persisted; re-run to resume.",
              flush=True)
        try:
            shared.close()
        except Exception:
            pass
        return 0

    try:
        shared.close()
    except Exception:
        pass

    reason = "PAUSE sentinel" if os.path.exists(PAUSE_FILE) else "reached --n-days for this invocation"
    print(f"\n[develop_run] stopped ({reason}) after {len(per_day)} day(s) this invocation, "
          f"wall {round(time.time() - t0, 1)}s.", flush=True)
    print("  resume: re-run the same command (auto-resumes).  status: --status.  "
          "chat: load a day_<N>/ bundle in the dashboard brain picker.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
