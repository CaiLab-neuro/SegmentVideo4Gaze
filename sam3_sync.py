#!/usr/bin/env python3
"""
sam3_sync.py — push local annotation edits to a remote project directory.

Copies only the small metadata/annotation files needed for --refine:
  project.json, concept_metadata.json, refinements.json, redetect sentinels,
  mask_anchors/ (user-drawn anchor masks)

Skips large or server-generated data:
  masks/, cond_states/, inference_state.pkl, obj_ptr_priors.npz, video files, frame images

Usage:
    python sam3_sync.py <local_project_dir> <remote_dest>
    python sam3_sync.py --dry-run <local_project_dir> <remote_dest>

Examples:
    python sam3_sync.py ./test_sam3_concept user@server:/data/projects/test_sam3_concept
    python sam3_sync.py --dry-run ./my_project user@192.168.1.10:/home/user/projects/my_project

Windows notes:
  rsync is not built into Windows. This script tries, in order:
    1. rsync         (available with Git for Windows if rsync was selected, or Cygwin)
    2. rsync-win.exe (standalone Windows rsync port, if present on PATH or in known install dirs)
    3. wsl rsync     (available if WSL is installed: wsl --install, then: wsl apt install rsync)
  If none are found, install one of the above or use WinSCP / MobaXterm as a GUI alternative.

Manual rsync (if this script can't run in your environment):
  The essential command is:
    rsync -rtu --exclude masks/ --exclude cond_states/ \
      --exclude inference_state.pkl --exclude obj_ptr_priors.npz \
      --exclude '*.mp4' --exclude '*.avi' --exclude '*.jpg' \
      <local_project_dir>/ user@host:/path/to/project
  -u skips files that are newer on the server (e.g. updated there by --refine), and
  -t keeps modification times so that comparison stays meaningful. The trailing /
  on the source copies the folder's contents rather than nesting it.

  Tip for shared group folders (several users, setgid directories): this script also
  adds --chmod=Fg+rw,Fo+r,F-x,Dg+rwx,Do+rx so synced files stay group-writable, and
  avoids -a. -a implies -p/-g, which copy the source's permissions and group; when
  the source is Windows (no setgid, no matching Unix group), that strips setgid from
  new server directories and later files there end up in the writer's personal group.
  If your rsync is too old for --chmod, the script retries without it; then run
  "chmod -R g+rw" on the server project afterwards.
"""

import argparse
import os
import shutil
import subprocess
import sys


# No "*.png" here: user-drawn mask anchors live in instances/<id>/mask_anchors/*.png
# and --refine needs them on the server. Propagated masks are covered by "masks/".
EXCLUDES = [
    "masks/",
    "cond_states/",
    "inference_state.pkl",
    "obj_ptr_priors.npz",
    "*.mp4",
    "*.avi",
    "*.jpg",
    "*.jpeg",
]


# Common Windows installation paths for standalone rsync ports (cwRsync, DeltaCopy, etc.),
# plus rsync-win.exe (https://github.com/nheinemann/rsync-win / similar standalone builds).
# These are only searched when rsync is not on PATH.
_WIN_RSYNC_SEARCH_PATHS = [
    r"C:\Program Files\cwRsync\bin\rsync.exe",
    r"C:\Program Files (x86)\cwRsync\bin\rsync.exe",
    r"C:\cwRsync\bin\rsync.exe",
    r"C:\Program Files\DeltaCopy\rsync.exe",
    r"C:\Program Files (x86)\DeltaCopy\rsync.exe",
    r"C:\Tools\rsync\rsync.exe",
]

_WIN_RSYNC_WIN_SEARCH_PATHS = [
    r"C:\Tools\rsync-win\rsync-win.exe",
    r"C:\Program Files\rsync-win\rsync-win.exe",
    r"C:\Program Files (x86)\rsync-win\rsync-win.exe",
]


def find_rsync():
    """Return the rsync invocation to use, or None if not found.

    Search order on Windows:
      1. rsync on PATH (Git for Windows, Cygwin, or any port already configured)
      2. Known installation paths for cwRsync / DeltaCopy standalone ports
      3. rsync-win.exe on PATH, or in known install dirs (fallback standalone port)
      4. wsl rsync (WSL must be installed and have rsync)
    On Linux/Mac only step 1 is tried.
    """
    if shutil.which("rsync"):
        return ["rsync"]
    if sys.platform == "win32":
        for candidate in _WIN_RSYNC_SEARCH_PATHS:
            if os.path.isfile(candidate):
                return [candidate]
        rsync_win = shutil.which("rsync-win.exe") or shutil.which("rsync-win")
        if rsync_win:
            return [rsync_win]
        for candidate in _WIN_RSYNC_WIN_SEARCH_PATHS:
            if os.path.isfile(candidate):
                return [candidate]
        if shutil.which("wsl") and _wsl_has_rsync():
            return ["wsl", "rsync"]
    return None


def _wsl_has_rsync():
    try:
        result = subprocess.run(["wsl", "which", "rsync"],
                                capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def _to_wsl_path(path: str) -> str:
    """Convert a Windows path to its WSL mount path (e.g. C:\foo -> /mnt/c/foo)."""
    try:
        result = subprocess.run(
            ["wsl", "wslpath", path.replace("\\", "/")],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Push local SAM3 annotation edits to a remote project directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Show what would be transferred without copying anything.")
    parser.add_argument("local_project_dir",
                        help="Path to the local SAM3 project folder.")
    parser.add_argument("remote_dest",
                        help="rsync destination, e.g. user@host:/path/to/project")
    args = parser.parse_args()

    rsync_cmd = find_rsync()
    if rsync_cmd is None:
        print("ERROR: rsync not found.")
        if sys.platform == "win32":
            print(
                "\nOn Windows, install one of:\n"
                "  - Git for Windows (enable rsync during install)\n"
                "  - rsync-win.exe standalone port (place it on PATH or in one of the\n"
                "    paths listed below)\n"
                "  - WSL:  wsl --install  then  wsl apt install rsync\n"
                "  - cwRsync standalone: https://itefix.net/cwrsync\n"
                "  - DeltaCopy (bundles rsync.exe)\n"
                "  - Cygwin with the rsync package\n"
                "\nAlso searched these paths and did not find rsync.exe:\n"
                + "\n".join(f"  {p}" for p in _WIN_RSYNC_SEARCH_PATHS) +
                "\n\nAnd these paths and did not find rsync-win.exe:\n"
                + "\n".join(f"  {p}" for p in _WIN_RSYNC_WIN_SEARCH_PATHS) +
                "\n\nOr use WinSCP / MobaXterm for a graphical alternative."
            )
        sys.exit(1)

    # Trailing slash on source: rsync copies the *contents*, not the directory itself
    src = args.local_project_dir.rstrip("/\\") + "/"
    dst = args.remote_dest

    # WSL rsync needs the local path in Linux form
    if rsync_cmd == ["wsl", "rsync"]:
        src = _to_wsl_path(src)

    # -rlt rather than -a: skipping -p/-g lets new server dirs inherit setgid and
    # group from their parent (a Windows source has neither to copy). --chmod keeps
    # files group-writable despite the server umask; F-x drops the execute bits
    # Windows rsync ports tend to report on plain data files.
    chmod_opt = "--chmod=Fg+rw,Fo+r,F-x,Dg+rwx,Do+rx"
    base = rsync_cmd + ["-rltuv"]
    tail = []
    if args.dry_run:
        tail.append("--dry-run")
    for pattern in EXCLUDES:
        tail += ["--exclude", pattern]
    tail += [src, dst]

    print(f"Source : {args.local_project_dir}")
    print(f"Dest   : {dst}")
    if args.dry_run:
        print("(dry run — no files will be transferred)")

    cmd = base + [chmod_opt] + tail
    print(f"Command: {' '.join(cmd)}")
    print()
    rc, stderr = _run_rsync(cmd)

    # Old rsync ports (pre-2.6.7, e.g. some DeltaCopy / rsync-win.exe builds) reject
    # --chmod with exit code 1 before transferring anything. Retry once without it.
    if rc == 1 and "chmod" in stderr.lower():
        print("\nWARNING: this rsync build does not support --chmod; retrying without it.")
        print("Synced files may not be group-writable on the server. Afterwards run there:")
        print("  chmod -R g+rw <remote project dir>")
        cmd = base + tail
        print(f"Command: {' '.join(cmd)}")
        print()
        rc, _ = _run_rsync(cmd)

    sys.exit(rc)


def _run_rsync(cmd):
    """Run rsync, streaming stdout live; capture stderr (echoed) for error sniffing."""
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, errors="replace")
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode, result.stderr or ""


if __name__ == "__main__":
    main()
