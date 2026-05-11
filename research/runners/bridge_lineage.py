"""Bridge Lineage Manager — CLI for inspect / list / diff / rollback / fork.

Companion runner for the BridgeLineage class (sim/lineage.py). Lets users
inspect their persistent training lineages without writing Python.

Usage examples:
    # List all lineages
    python -m research.runners.bridge_lineage list

    # Show details of a specific lineage
    python -m research.runners.bridge_lineage show main

    # Show history snapshots for a lineage
    python -m research.runners.bridge_lineage history main

    # Roll back to a specific snapshot
    python -m research.runners.bridge_lineage rollback main \
        --to 2026-05-10T22-00-00-123

    # Fork a new experiment branch
    python -m research.runners.bridge_lineage fork main experiment_v3

    # Prune history (keep last N snapshots)
    python -m research.runners.bridge_lineage prune main --keep-last 10

    # Compare two lineages (or one lineage's current vs a history snap)
    python -m research.runners.bridge_lineage diff main \
        --from <snapshot_id> --to current

Design doc: docs/plans/2026-05-10-bridge-lineage-design.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so this can be invoked as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sim.lineage import BridgeLineage, LINEAGE_ROOT


def _fmt_bytes(n: int) -> str:
    """Human-friendly byte size."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB"):
        n = n / 1024
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"


def cmd_list(args) -> int:
    """List all known lineages."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineages = BridgeLineage.list_all(root=root)
    if not lineages:
        print(f"[no lineages found under {root}/]")
        return 0
    print(f"Lineages under {root}/")
    print("-" * 60)
    for L in sorted(lineages, key=lambda l: l.name):
        try:
            meta = L.read_metadata()
            tier = meta.current_tier
            cumul = meta.cumulative_training_events
            n_history = len(L.list_history())
            updated = meta.last_updated_at or "(unknown)"
            parent = (f" <- {meta.parent_lineage}"
                       if meta.parent_lineage else "")
            print(f"  {L.name:<20} {tier:<10} "
                  f"{cumul:>8d} events  "
                  f"{n_history:>3d} snaps  "
                  f"updated {updated}{parent}")
        except Exception as e:
            print(f"  {L.name:<20} <metadata error: {e}>")
    return 0


def cmd_show(args) -> int:
    """Show detailed metadata for one lineage."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineage = BridgeLineage(args.name, root=root)
    if not lineage.exists():
        print(f"ERROR: lineage '{args.name}' does not exist under {root}/",
              file=sys.stderr)
        return 2
    meta = lineage.read_metadata()
    print(f"=== Lineage: {args.name} ===")
    print(f"  Schema version: {meta.schema_version}")
    print(f"  Created:        {meta.created_at}")
    print(f"  Last updated:   {meta.last_updated_at}")
    print(f"  Current tier:   {meta.current_tier}")
    print(f"  Vocab size:     {len(meta.vocab)}"
          + (f"  ({meta.vocab[:8]}...)" if len(meta.vocab) > 8
              else f"  ({meta.vocab})"))
    print(f"  Cumulative events: {meta.cumulative_training_events}")
    if meta.arch:
        print(f"  Arch: {json.dumps(meta.arch, indent=4)[1:-1].strip()}")
    if meta.parent_lineage:
        print(f"  Parent: {meta.parent_lineage}")
        print(f"  Branched at: {meta.branched_at}")
    if meta.tags:
        print(f"  Tags: {meta.tags}")
    # File sizes
    try:
        size = lineage.current_path.stat().st_size
        print(f"  current.simstate.h5: {_fmt_bytes(size)}")
    except OSError:
        pass
    # History summary
    snapshots = lineage.list_history()
    if snapshots:
        print(f"  History: {len(snapshots)} snapshots")
        oldest = snapshots[0].name.replace("-checkpoint.simstate.h5", "")
        newest = snapshots[-1].name.replace("-checkpoint.simstate.h5", "")
        print(f"    oldest: {oldest}")
        print(f"    newest: {newest}")
    # Growth events
    if meta.growth_events:
        print(f"  Growth events ({len(meta.growth_events)}):")
        for e in meta.growth_events[-args.n_events:]:
            kind = e.get("kind", "?")
            desc = e.get("description", "")
            at = e.get("at", "?")
            print(f"    [{at}] {kind:<20} {desc}")
    # Accuracy history
    if meta.accuracy_history:
        print(f"  Accuracy history ({len(meta.accuracy_history)} points):")
        for p in meta.accuracy_history[-args.n_events:]:
            metric = p.get("metric", "?")
            value = p.get("value", 0)
            ctx = p.get("context", "")
            at = p.get("at", "?")
            print(f"    [{at}] {metric:<20} {value:.3f}  {ctx}")
    return 0


def cmd_history(args) -> int:
    """List history snapshots for one lineage."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineage = BridgeLineage(args.name, root=root)
    snapshots = lineage.list_history()
    if not snapshots:
        print(f"[no history snapshots for '{args.name}']")
        return 0
    print(f"History snapshots for '{args.name}' ({len(snapshots)} total):")
    print("-" * 70)
    for snap in snapshots:
        snap_id = snap.name.replace("-checkpoint.simstate.h5", "")
        try:
            size = snap.stat().st_size
            print(f"  {snap_id:<35} {_fmt_bytes(size):>10}")
        except OSError:
            print(f"  {snap_id:<35} <size unknown>")
    return 0


def cmd_rollback(args) -> int:
    """Roll back to a specific history snapshot."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineage = BridgeLineage(args.name, root=root)
    if not lineage.exists():
        print(f"ERROR: lineage '{args.name}' does not exist", file=sys.stderr)
        return 2
    if not args.to:
        print("ERROR: --to <snapshot_id> required", file=sys.stderr)
        return 2
    try:
        lineage.rollback_to(args.to)
        print(f"[rollback] '{args.name}' restored to snapshot {args.to}")
        # Append growth event
        meta = lineage.read_metadata()
        meta.add_growth_event(
            kind="rollback",
            description=f"Rolled back to snapshot {args.to}",
            target_snapshot=args.to,
        )
        lineage.write_metadata(meta)
        return 0
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def cmd_fork(args) -> int:
    """Fork a new lineage from an existing one."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    parent = BridgeLineage(args.parent, root=root)
    if not parent.exists():
        print(f"ERROR: parent lineage '{args.parent}' does not exist",
              file=sys.stderr)
        return 2
    try:
        new_lineage = parent.fork(args.child)
        print(f"[fork] '{args.parent}' -> '{args.child}' "
              f"(new lineage at {new_lineage.root}/)")
        return 0
    except FileExistsError:
        print(f"ERROR: lineage '{args.child}' already exists",
              file=sys.stderr)
        return 2


def cmd_prune(args) -> int:
    """Prune history to keep only the last N snapshots."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineage = BridgeLineage(args.name, root=root)
    if not lineage.exists():
        print(f"ERROR: lineage '{args.name}' does not exist", file=sys.stderr)
        return 2
    before = len(lineage.list_history())
    lineage.prune_history(keep_last=args.keep_last)
    after = len(lineage.list_history())
    removed = before - after
    print(f"[prune] '{args.name}': {before} -> {after} snapshots "
          f"({removed} removed, kept last {args.keep_last})")
    return 0


def cmd_memory_stats(args) -> int:
    """Show BridgeMemory state for a lineage (mirrors /api/bridge-memory)."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineage = BridgeLineage(args.name, root=root)
    if not lineage.exists():
        print(f"ERROR: lineage '{args.name}' does not exist", file=sys.stderr)
        return 2
    meta = lineage.read_metadata()
    bindings = []
    n_forgets = 0
    n_consolidations = 0
    last_consolidation = None
    for e in meta.growth_events:
        kind = e.get("kind", "")
        if kind == "memory_bind":
            md = e.get("metadata", {})
            bindings.append({
                "key": md.get("key", ""),
                "value": md.get("value", ""),
                "target_action": md.get("target_action", ""),
                "confidence": md.get("confidence", 0.0),
                "at": e.get("at", ""),
            })
        elif kind == "memory_forget":
            n_forgets += 1
        elif kind == "memory_consolidate":
            n_consolidations += 1
            last_consolidation = e.get("at", "")

    print(f"=== Memory stats for '{args.name}' ===")
    print(f"  Bindings:       {len(bindings)}")
    print(f"  Forgets:        {n_forgets}")
    print(f"  Consolidations: {n_consolidations}")
    if last_consolidation:
        print(f"  Last consolidation: {last_consolidation}")
    print(f"  Current tier:   {meta.current_tier}")
    print(f"  Vocab size:     {len(meta.vocab)}")
    print(f"  Cumulative training events: {meta.cumulative_training_events}")
    if bindings:
        print(f"\n  Recent bindings (last {min(args.last, len(bindings))}):")
        for b in bindings[-args.last:]:
            conf = b.get("confidence", 0)
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
            print(f"    [{b.get('at', '?')}] {b.get('key', ''):<20} "
                  f"-> {b.get('value', ''):<10} (action={b.get('target_action', '?')}, "
                  f"conf={conf_str})")
    return 0


def cmd_list_shards(args) -> int:
    """List per-pathway shards exported for a lineage."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineage = BridgeLineage(args.name, root=root)
    if not lineage.exists():
        print(f"ERROR: lineage '{args.name}' does not exist", file=sys.stderr)
        return 2
    names = lineage.list_shards()
    if not names:
        print(f"[no shards exported for '{args.name}'; run export-shards "
              f"after loading a bridge from this lineage]")
        return 0
    print(f"Shards for '{args.name}' ({len(names)} pathways):")
    shard_root = lineage.root / "shards"
    for name in names:
        path = shard_root / f"{name}.npz"
        try:
            size = path.stat().st_size
            print(f"  {name:<40} {_fmt_bytes(size):>10}")
        except OSError:
            print(f"  {name:<40} <missing>")
    return 0


def cmd_growth_log(args) -> int:
    """Render the lineage's growth log to stdout (or --write to _growth_log.md)."""
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineage = BridgeLineage(args.name, root=root)
    if not lineage.exists():
        print(f"ERROR: lineage '{args.name}' does not exist", file=sys.stderr)
        return 2
    if args.write:
        path = lineage.write_growth_log()
        print(f"[growth-log] wrote {path}")
    else:
        print(lineage.render_growth_log())
    return 0


def cmd_diff(args) -> int:
    """Compare two lineage states (or one lineage's current vs a snapshot).

    For now, this is a metadata-level diff (vocab, tier, arch, accuracy
    history). A weight-level diff would require loading the bridge for
    both states, which requires GPU. Adding that as a future extension.
    """
    root = Path(args.root) if args.root else LINEAGE_ROOT
    lineage = BridgeLineage(args.name, root=root)
    if not lineage.exists():
        print(f"ERROR: lineage '{args.name}' does not exist", file=sys.stderr)
        return 2

    # Helper to resolve a "side" into a metadata dict
    def _resolve_side(spec: str) -> dict:
        if spec == "current":
            return lineage.read_metadata().to_dict()
        # Otherwise treat as a snapshot ID
        path = lineage.history_dir / f"{spec}-checkpoint.metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot metadata not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        meta_a = _resolve_side(args.from_)
        meta_b = _resolve_side(args.to)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"=== diff: {args.from_} -> {args.to} ===")
    keys = set(meta_a.keys()) | set(meta_b.keys())
    diffs = 0
    for k in sorted(keys):
        a = meta_a.get(k)
        b = meta_b.get(k)
        if a == b:
            continue
        diffs += 1
        if isinstance(a, list) and isinstance(b, list):
            print(f"  {k}: {len(a)} items -> {len(b)} items")
        elif isinstance(a, dict) and isinstance(b, dict):
            sub_diffs = []
            for sk in set(a.keys()) | set(b.keys()):
                if a.get(sk) != b.get(sk):
                    sub_diffs.append(f"{sk}: {a.get(sk)} -> {b.get(sk)}")
            if sub_diffs:
                print(f"  {k}:")
                for s in sub_diffs:
                    print(f"    {s}")
        else:
            print(f"  {k}: {a} -> {b}")
    if diffs == 0:
        print("  (no metadata differences)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=str, default=None,
                    help=f"Lineage root directory (default: {LINEAGE_ROOT})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # list
    p_list = sub.add_parser("list", help="List all lineages")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = sub.add_parser("show", help="Show details for a lineage")
    p_show.add_argument("name", type=str, help="Lineage name")
    p_show.add_argument("--n-events", type=int, default=10,
                         help="How many recent growth events / accuracy "
                              "points to display (default 10)")
    p_show.set_defaults(func=cmd_show)

    # history
    p_hist = sub.add_parser("history", help="List history snapshots")
    p_hist.add_argument("name", type=str, help="Lineage name")
    p_hist.set_defaults(func=cmd_history)

    # rollback
    p_rb = sub.add_parser("rollback", help="Restore a history snapshot")
    p_rb.add_argument("name", type=str, help="Lineage name")
    p_rb.add_argument("--to", type=str, required=True,
                       help="Snapshot ID to restore "
                            "(e.g. '2026-05-10T22-00-00-123')")
    p_rb.set_defaults(func=cmd_rollback)

    # fork
    p_fork = sub.add_parser("fork", help="Fork a new lineage")
    p_fork.add_argument("parent", type=str, help="Parent lineage name")
    p_fork.add_argument("child", type=str, help="New (child) lineage name")
    p_fork.set_defaults(func=cmd_fork)

    # prune
    p_pr = sub.add_parser("prune", help="Prune history snapshots")
    p_pr.add_argument("name", type=str, help="Lineage name")
    p_pr.add_argument("--keep-last", type=int, default=30,
                       help="Number of recent snapshots to keep (default 30)")
    p_pr.set_defaults(func=cmd_prune)

    # memory-stats (Path 3 BridgeMemory inspection)
    p_ms = sub.add_parser("memory-stats",
                            help="Show BridgeMemory state for a lineage")
    p_ms.add_argument("name", type=str, help="Lineage name")
    p_ms.add_argument("--last", type=int, default=10,
                       help="Show last N bindings (default 10)")
    p_ms.set_defaults(func=cmd_memory_stats)

    # list-shards (tiering Phase 3 Strategy C inspection)
    p_ls = sub.add_parser("list-shards",
                            help="List per-pathway shards (if exported)")
    p_ls.add_argument("name", type=str, help="Lineage name")
    p_ls.set_defaults(func=cmd_list_shards)

    # growth-log
    p_gl = sub.add_parser("growth-log",
                            help="Render lineage growth log as markdown")
    p_gl.add_argument("name", type=str, help="Lineage name")
    p_gl.add_argument("--write", action="store_true",
                       help="Write to _growth_log.md (default: print to stdout)")
    p_gl.set_defaults(func=cmd_growth_log)

    # diff
    p_diff = sub.add_parser("diff", help="Diff two lineage states (metadata)")
    p_diff.add_argument("name", type=str, help="Lineage name")
    p_diff.add_argument("--from", dest="from_", type=str, default="current",
                         help="Source state: 'current' or a snapshot ID")
    p_diff.add_argument("--to", type=str, default="current",
                         help="Target state: 'current' or a snapshot ID")
    p_diff.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
