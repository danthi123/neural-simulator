#!/usr/bin/env python3
"""safe_copy_into.py -- copy result files from a PRIVATE staging tree into a directory an agent can rewrite, without
ever following a symlink the agent planted on the destination side (2026-09-30, trust boundary of the server move:
docs/plans/2026-09-30-home-to-server-MIGRATION.md section 10).

WHY. On srv-host-0 `tools/pool_sync.sh` runs as the unprivileged user `sim`, and its destination
(`<workspace>/research/findings/raw`) sits in a tree the Hermes agent owns. rsync follows symlinks in the destination
PATH it is given (reproduced: `raw -> /elsewhere` put every pulled file in /elsewhere), so a planted
`research/findings -> /home/sim/runtime/...` would steer the pull into files `sim` can write. pool_sync therefore
pulls into a staging directory only `sim` can write, and this script moves the files across:

  * the destination is walked one component at a time from a trusted ANCHOR (the workspace's parent, the dataset
    mount, which the agent cannot replace) with O_NOFOLLOW|O_DIRECTORY, so a symlinked directory anywhere on the way
    is refused, and a directory moved away mid-walk keeps its descent inside the agent's own tree;
  * each file is written to a fresh name (O_CREAT|O_EXCL|O_NOFOLLOW) in the opened directory and renamed over the
    target, so an existing symlink or hard link at the target is replaced, never written through; a target that is a
    symlink is REFUSED rather than replaced, and reported, because nothing legitimate puts one in raw/;
  * rsync -u semantics: a file is copied when the target is absent, or older (whole seconds), or the same second with
    a different size; the source mtime is kept. Regular files only (the staging tree came from rsync --safe-links).

Stdlib only, so it runs under the system python3 with -I and needs no venv.

  python3 -I tools/safe_copy_into.py --anchor /home/dant123/Projects \\
          --dest neural-simulator/research/findings/raw  <stage-dir> [<stage-dir> ...]

Prints one line per copied file (except *.prov.json, first 30), then
`safe_copy: copied N, unchanged U, refused R`. Exit 0, or 2 when the anchor or the destination cannot be opened.
"""
from __future__ import annotations

import argparse
import errno
import os
import stat
import sys

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
SHOW_MAX = 30


class Copier:
    def __init__(self, dest_fd: int) -> None:
        self.dest_fd = dest_fd
        self.copied = self.unchanged = self.refused = 0
        self.shown = 0
        self.seq = 0

    def refuse(self, rel: str, why: str) -> None:
        self.refused += 1
        print(f"  REFUSED {rel!r}: {why}", file=sys.stderr)

    def open_subdir(self, parent_fd: int, name: str, rel: str) -> int | None:
        """Open (creating if absent) a real directory `name` under parent_fd; never through a symlink."""
        try:
            return os.open(name, DIR_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            try:
                os.mkdir(name, 0o777, dir_fd=parent_fd)   # umask (0002 in the unit) -> 0775; setgid is inherited
            except FileExistsError:
                pass
            try:
                return os.open(name, DIR_FLAGS, dir_fd=parent_fd)
            except OSError as e:
                self.refuse(rel, f"cannot open the directory just created ({e.strerror})")
                return None
        except OSError as e:
            if e.errno in (errno.ELOOP, errno.ENOTDIR):
                self.refuse(rel, "destination directory is a symlink or not a directory")
            else:
                self.refuse(rel, f"cannot open destination directory ({e.strerror})")
            return None

    def copy_file(self, src_path: str, src_st: os.stat_result, dir_fd: int, name: str, rel: str) -> None:
        try:
            dst_st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            dst_st = None
        if dst_st is not None:
            if stat.S_ISLNK(dst_st.st_mode):
                self.refuse(rel, "destination is a symlink")
                return
            if not stat.S_ISREG(dst_st.st_mode):
                self.refuse(rel, "destination exists and is not a regular file")
                return
            s_sec, d_sec = int(src_st.st_mtime), int(dst_st.st_mtime)
            if s_sec < d_sec or (s_sec == d_sec and src_st.st_size == dst_st.st_size):
                self.unchanged += 1
                return
        self.seq += 1
        tmp = f".{name[:120]}.simsync.{os.getpid()}.{self.seq}"
        try:
            sfd = os.open(src_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError as e:
            self.refuse(rel, f"cannot read the staged copy ({e.strerror})")
            return
        try:
            try:
                tfd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666,
                              dir_fd=dir_fd)
            except OSError as e:
                self.refuse(rel, f"cannot create a temporary file beside it ({e.strerror})")
                return
            try:
                while True:
                    buf = os.read(sfd, 1 << 20)
                    if not buf:
                        break
                    view = memoryview(buf)
                    while view:
                        view = view[os.write(tfd, view):]
                os.utime(tfd, ns=(src_st.st_atime_ns, src_st.st_mtime_ns))
            finally:
                os.close(tfd)
            try:
                os.rename(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            except OSError as e:
                os.unlink(tmp, dir_fd=dir_fd)
                self.refuse(rel, f"cannot rename into place ({e.strerror})")
                return
        finally:
            os.close(sfd)
        self.copied += 1
        if not rel.endswith(".prov.json") and self.shown < SHOW_MAX:
            self.shown += 1
            print(f"      {rel}")

    def walk(self, src_dir: str, dir_fd: int, rel_prefix: str) -> None:
        try:
            entries = sorted(os.scandir(src_dir), key=lambda e: e.name)
        except OSError as e:
            self.refuse(rel_prefix or ".", f"cannot list the staged directory ({e.strerror})")
            return
        for ent in entries:
            rel = f"{rel_prefix}{ent.name}"
            try:
                st = ent.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode):
                sub = self.open_subdir(dir_fd, ent.name, rel + "/")
                if sub is None:
                    continue
                try:
                    self.walk(ent.path, sub, rel + "/")
                finally:
                    os.close(sub)
            elif stat.S_ISREG(st.st_mode):
                self.copy_file(ent.path, st, dir_fd, ent.name, rel)
            # symlinks and special files in the staging tree are never copied


def open_dest(anchor: str, dest_rel: str) -> int:
    """The anchor is trusted (opened normally); every component of dest_rel must be an existing real directory."""
    fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    for comp in [c for c in dest_rel.split("/") if c]:
        if comp in (".", ".."):
            os.close(fd)
            raise ValueError(f"--dest may not contain {comp!r}")
        try:
            nfd = os.open(comp, DIR_FLAGS, dir_fd=fd)
        finally:
            os.close(fd)
        fd = nfd
    return fd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--anchor", required=True, help="trusted directory the destination is resolved from")
    ap.add_argument("--dest", required=True, help="destination, relative to --anchor; every component must exist")
    ap.add_argument("src", nargs="+", help="private staging directories whose CONTENTS are copied into --dest")
    a = ap.parse_args(argv)
    try:
        dest_fd = open_dest(a.anchor, a.dest)
    except (OSError, ValueError) as e:
        print(f"safe_copy: ⛔ cannot open {a.dest!r} under {a.anchor!r} without following a symlink: {e}",
              file=sys.stderr)
        return 2
    c = Copier(dest_fd)
    try:
        for src in a.src:
            if os.path.isdir(src) and not os.path.islink(src):
                c.walk(src, dest_fd, "")
    finally:
        os.close(dest_fd)
    print(f"safe_copy: copied {c.copied}, unchanged {c.unchanged}, refused {c.refused}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
