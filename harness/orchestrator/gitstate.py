"""Per-round git checkpoints and reverts over the agent workspace.

The harness ``_git_init`` already gives the workspace a throwaway repo whose
base commit is the task's starting state, and ``final.patch`` is captured as
``git add -A; git diff --cached`` (staged vs the base commit). This module must
therefore NEVER move HEAD or reset the index in a way that hides earlier work
from that diff. Checkpoints are created with ``write-tree``/``commit-tree``:
pure object writes that change nothing observable to the harness diff flow.

A restore puts the staged tree and worktree back to the checkpoint's content
and deletes files that appeared after it, so a failing round can be reverted
without discarding anything the checkpoint had.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "vulcanbench",
    "GIT_AUTHOR_EMAIL": "bot@vulcanbench",
    "GIT_COMMITTER_NAME": "vulcanbench",
    "GIT_COMMITTER_EMAIL": "bot@vulcanbench",
}


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(_GIT_ENV)
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )


def stage(workspace: Path) -> None:
    _git(workspace, "add", "-A")


def diff_stat(workspace: Path) -> str:
    """Staged-vs-base diffstat (what changed since the task started)."""
    stage(workspace)
    return _git(workspace, "diff", "--cached", "--stat").stdout.strip()


def diff_text(workspace: Path, limit: int = 6_000) -> str:
    stage(workspace)
    diff = _git(workspace, "diff", "--cached").stdout
    if len(diff) > limit:
        omitted = len(diff) - limit
        return diff[:limit] + f"\n...[diff clipped: {omitted} chars omitted]"
    return diff


def file_list(workspace: Path, limit: int = 120) -> str:
    stage(workspace)
    listing = _git(workspace, "ls-files").stdout.splitlines()
    shown = listing[:limit]
    out = "\n".join(shown)
    if len(listing) > limit:
        out += f"\n...[{len(listing) - limit} more files]"
    return out


def _changed_and_untracked(workspace: Path) -> set[str]:
    stage(workspace)
    changed = _git(workspace, "diff", "--cached", "--name-only").stdout.splitlines()
    untracked = _git(workspace, "ls-files", "--others", "--exclude-standard").stdout.splitlines()
    return {p.strip() for p in [*changed, *untracked] if p.strip()}


def checkpoint(workspace: Path) -> dict[str, Any]:
    """Snapshot the current worktree as a commit object (HEAD/index untouched).

    Returns the object id plus the changed-vs-base path set, which a later
    restore uses to delete files that appeared after the snapshot.
    """
    stage(workspace)
    tree = _git(workspace, "write-tree").stdout.strip()
    if not tree:
        raise RuntimeError(f"git write-tree failed: {_git(workspace, 'status').stderr}")
    ref = _git(workspace, "commit-tree", tree, "-m", "orchestrator checkpoint").stdout.strip()
    if not ref:
        raise RuntimeError("git commit-tree failed")
    return {"ref": ref, "paths": sorted(_changed_and_untracked(workspace))}


def restore(workspace: Path, cp: dict[str, Any]) -> None:
    """Restore the workspace to a :func:`checkpoint` snapshot."""
    ref = str(cp["ref"])
    ref_paths = set(_git(workspace, "ls-tree", "-r", "--name-only", ref).stdout.splitlines())
    gone = _changed_and_untracked(workspace) - ref_paths
    for rel in sorted(gone):
        (workspace / rel).unlink(missing_ok=True)
    proc = _git(workspace, "checkout", ref, "--", ".")
    if proc.returncode != 0:
        raise RuntimeError(f"git checkout {ref[:8]} -- . failed: {proc.stderr.strip()}")
    stage(workspace)
