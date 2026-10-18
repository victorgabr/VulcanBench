"""The durable filesystem ledger: the only coordination state in the loop.

Lives under ``<run_dir>/ledger/``, deliberately OUTSIDE the agent workspace:
the captured ``final.patch`` stays pure (no ledger files in the diff), the
hidden verifier never sees them, and the ledger itself is preserved as a run
artifact beside the trace. Manager decisions mutate it; worker prompts receive
bounded renderings of it, never whole files.

Layout mirrors the paper's v2 scaffold: ``task.md`` (normalized statement),
``plan.md`` (3 to 6 sentence strategy), ``tasks.json`` (curated dynamic list),
``notes.md`` (compact durable conclusions), ``verification.md`` (append-only
objective checks), ``best.md`` (current best candidate pointer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TASK_STATUSES = frozenset({"pending", "in_progress", "done", "blocked", "superseded"})

# Prompt-injection caps. The ledger exists to keep worker contexts bounded, so
# every render is clipped with an explicit marker rather than pasted whole.
NOTES_CAP = 6_000
VERIFICATION_CAP = 4_000
BEST_CAP = 3_000
PLAN_CAP = 2_000


def clip(text: str, limit: int) -> str:
    """Clip from the head (older material) and keep the tail, marked.

    Notes and verification logs grow chronologically, so the newest material is
    the relevant material when a cap bites.
    """
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"...[older content clipped: {omitted} chars omitted]\n{text[-limit:]}"


class Ledger:
    """Filesystem-backed ledger with the single ``in_progress`` invariant."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------

    @property
    def task_md(self) -> Path:
        return self.root / "task.md"

    @property
    def plan_md(self) -> Path:
        return self.root / "plan.md"

    @property
    def tasks_json(self) -> Path:
        return self.root / "tasks.json"

    @property
    def notes_md(self) -> Path:
        return self.root / "notes.md"

    @property
    def verification_md(self) -> Path:
        return self.root / "verification.md"

    @property
    def best_md(self) -> Path:
        return self.root / "best.md"

    def file_names(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_file())

    # -- task statement / plan ----------------------------------------------

    def write_task(self, text: str) -> None:
        self.task_md.write_text(text, encoding="utf-8")

    def task_text(self) -> str:
        return self.task_md.read_text(encoding="utf-8")

    def set_plan(self, text: str) -> None:
        self.plan_md.write_text(text.strip() + "\n", encoding="utf-8")

    def plan_text(self) -> str:
        if not self.plan_md.exists():
            return ""
        return self.plan_md.read_text(encoding="utf-8")

    # -- dynamic task list ----------------------------------------------------

    def seed_tasks(self, entries: list[dict[str, Any]]) -> None:
        """Replace the task list with validated seeds (3 to 6 falsifiable tasks)."""
        tasks: list[dict[str, Any]] = []
        for entry in entries:
            tid = str(entry.get("id", "")).strip()
            desc = str(entry.get("desc", "")).strip()
            if not tid or not desc:
                raise ValueError(f"task entry needs id and desc, got {entry!r}")
            if any(t["id"] == tid for t in tasks):
                raise ValueError(f"duplicate task id {tid!r}")
            tasks.append(
                {"id": tid, "desc": desc, "status": "pending", "result": None, "evidence": []}
            )
        self._save_tasks(tasks)

    def tasks(self) -> list[dict[str, Any]]:
        if not self.tasks_json.exists():
            return []
        data = json.loads(self.tasks_json.read_text(encoding="utf-8"))
        return list(data)

    def _save_tasks(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks_json.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    def get(self, task_id: str) -> dict[str, Any] | None:
        for task in self.tasks():
            if task["id"] == task_id:
                return task
        return None

    def add_task(self, task_id: str, desc: str) -> None:
        """Append a genuinely new task created by evidence (never pre-queued work)."""
        tasks = self.tasks()
        if any(t["id"] == task_id for t in tasks):
            return
        tasks.append(
            {"id": task_id, "desc": desc, "status": "pending", "result": None, "evidence": []}
        )
        self._save_tasks(tasks)

    def mark(
        self,
        task_id: str,
        status: str,
        *,
        result: str | None = None,
        evidence: list[str] | None = None,
    ) -> None:
        """Set a task's status, enforcing the one-active-worker invariant.

        A task may be ``in_progress`` only when nothing else is; ``done``
        entries carry a result and at least one evidence line (the engine
        attaches the verify outcome or labels a bare manager assertion).
        """
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown status {status!r}; allowed: {sorted(TASK_STATUSES)}")
        tasks = self.tasks()
        if status == "in_progress":
            active = [t["id"] for t in tasks if t["status"] == "in_progress" and t["id"] != task_id]
            if active:
                raise ValueError(f"serial invariant: task(s) already in_progress: {active}")
        for task in tasks:
            if task["id"] != task_id:
                continue
            task["status"] = status
            if result is not None:
                task["result"] = result
            if evidence:
                task["evidence"] = list(dict.fromkeys([*task["evidence"], *evidence]))
            if status == "done" and not task["evidence"]:
                task["evidence"] = ["manager assertion (unverified)"]
            self._save_tasks(tasks)
            return
        raise ValueError(f"unknown task id {task_id!r}")

    def in_progress_ids(self) -> list[str]:
        return [t["id"] for t in self.tasks() if t["status"] == "in_progress"]

    # -- notes / verification / best -------------------------------------------

    def append_notes(self, lines: list[str]) -> None:
        lines = [ln.strip() for ln in lines if ln.strip()]
        if not lines:
            return
        existing = self.notes_md.read_text(encoding="utf-8") if self.notes_md.exists() else ""
        merged = existing + "\n".join(f"- {ln}" for ln in lines) + "\n"
        self.notes_md.write_text(clip(merged, NOTES_CAP), encoding="utf-8")

    def notes_text(self) -> str:
        if not self.notes_md.exists():
            return ""
        return self.notes_md.read_text(encoding="utf-8")

    def append_verification(self, entry: str) -> None:
        existing = (
            self.verification_md.read_text(encoding="utf-8")
            if self.verification_md.exists()
            else ""
        )
        self.verification_md.write_text(
            clip(existing + entry.rstrip() + "\n", VERIFICATION_CAP), encoding="utf-8"
        )

    def verification_text(self) -> str:
        if not self.verification_md.exists():
            return ""
        return self.verification_md.read_text(encoding="utf-8")

    def set_best(self, text: str) -> None:
        self.best_md.write_text(text.strip() + "\n", encoding="utf-8")

    def best_text(self) -> str:
        if not self.best_md.exists():
            return ""
        return self.best_md.read_text(encoding="utf-8")

    # -- bounded renders for prompts --------------------------------------------

    def render_tasks(self) -> str:
        tasks = self.tasks()
        if not tasks:
            return "(no tasks yet)"
        lines = []
        for task in tasks:
            line = f"{task['id']} [{task['status']}] {task['desc']}"
            if task.get("result"):
                line += f" -> {task['result']}"
            if task.get("evidence"):
                line += f" | evidence: {len(task['evidence'])}"
            lines.append(line)
        return "\n".join(lines)
