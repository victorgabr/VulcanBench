"""Ledger self-orchestration engine: manager to one worker to verifier.

Serial by construction. There is no threading, no asyncio, no parallel
dispatch: every role runs inline in the order the manager chose, and the
manager dispatches exactly one bounded task per round. The paper's anti-
anchoring pass ("fresh perspective") runs serially and once per stall, never
as an ensemble; the manager holds a task list but only ever a *selection*,
never a queue of concurrent workers.

Budget parity with the plain ``vulcan`` loop is a benchmark-integrity
requirement: wall-clock (``_RunDeadline``), total model-loop steps and the
per-run cost cap are the same knobs the baseline uses, so orchestrated and
baseline columns stay comparable at equal budgets. The round cap (soft 6,
hard 10) is an additional engine-internal bound; the paper's bound is 10.

Grading integrity: the engine withholds the last ``grading_reserve_s`` of the
wall clock from every model phase, so the hidden verifier always runs against
the final workspace instead of being zeroed by an exhausted deadline.

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.agent.loop import (
    _complete_provider,
    _remaining_timeout,
    _run_model_loop,
    _RunDeadline,
)
from harness.agent.protocol import RunCommandArgs, get_openai_tool_schemas
from harness.orchestrator import gitstate, prompts
from harness.orchestrator.config import OrchestrationConfig
from harness.orchestrator.ledger import Ledger, clip
from harness.pricing import cost_usd

READONLY_TOOLS = frozenset({"list_files", "read_file", "search_code"})
ALL_TOOL_NAMES = frozenset(
    {
        "list_files",
        "read_file",
        "search_code",
        "edit_file",
        "run_command",
        "run_tests",
        "run_lint",
        "run_build",
        "security_scan",
    }
)

# Prompt-injection caps. The ledger exists to keep worker contexts bounded, so
# renders are clipped with explicit markers instead of pasted whole.
TASK_CLIP = 1_500
NOTES_TAIL = 2_500
VERIFY_TAIL = 1_800
BEST_CLIP = 3_000
PLAN_CLIP = 2_000
WORKER_TAIL = 2_500
DIFFSTAT_CLIP = 1_200


class _ReserveDeadline(_RunDeadline):
    """View of the run deadline with the last ``reserve_s`` withheld.

    The engine's model phases consume this view, so the hidden verifier and
    evaluation stages always start with at least ``reserve_s`` of wall clock
    left. This mirrors how the plain ``vulcan`` loop keeps that room by
    self-limiting its steps: a stop under the reserve is self-limitation, not
    a budget breach, so it never sets the shared ``exceeded`` flag and the run
    is scored honestly (no "run budget exceeded before verification" zeroes).
    """

    def __init__(self, inner: _RunDeadline, reserve_s: float) -> None:
        super().__init__(inner.started_mono, inner.timeout_s)
        self._inner = inner
        self._reserve_s = max(0.0, reserve_s)

    def remaining_s(self) -> float | None:
        remaining = self._inner.remaining_s()
        if remaining is None:
            return None
        return max(0.0, remaining - self._reserve_s)


@dataclass
class OrchestrationOutcome:
    prompt_tokens: int
    completion_tokens: int
    finished: bool
    cost_capped: bool
    steps: int
    report: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RoleStats:
    llm_calls: int = 0
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _tool_schemas(names: frozenset[str]) -> list[dict[str, Any]]:
    return [t for t in get_openai_tool_schemas() if t["function"]["name"] in names]


def _parse_decision(content: str) -> dict[str, Any] | None:
    """First balanced JSON object in the reply, or None.

    Models wrap JSON in prose or fences often enough that a bare
    ``json.loads`` is not the contract; raw_decode from the first brace takes
    the first complete object and ignores the rest.
    """
    text = content.strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _first_lines(text: str, limit: int = 300) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out = "\n".join(lines[:8])
    if len(out) > limit:
        out = out[:limit] + "..."
    return out


def _external_verify_command(workspace: Path) -> str:
    """Fallback verification whose exit code is real.

    ``default_test_command`` suffixes ``|| true`` for the agent's run_tests
    tool; as the engine's external gate that would make every round PASS and
    let the finalizer claim completion on an unchanged workspace. A workspace
    with no discoverable suite gates on ``true``: the in-loop check is
    scaffolding, and the hidden verifier stays the scoring gate.
    """
    ws = workspace.resolve()
    if (ws / "Cargo.toml").exists():
        return "cargo test --quiet 2>&1"
    if (ws / "go.mod").exists():
        return "go test ./... 2>&1"
    if (ws / "package.json").exists():
        return "if [ -d tests ] || [ -d test ]; then node --test 2>&1; else npm test 2>&1; fi"
    if (ws / "pom.xml").exists() or (ws / "build.gradle").exists():
        return "mvn -q test 2>&1 || ./gradlew test 2>&1"
    has_suite = (
        (ws / "tests").is_dir()
        or (ws / "test").is_dir()
        or (ws / "conftest.py").exists()
        or bool(list(ws.glob("test_*.py")))
    )
    return "python -m pytest -q --tb=short 2>&1" if has_suite else "true"


def run_orchestration(  # noqa: PLR0912, PLR0915  (linear serial loop: guards, rounds, exits)
    *,
    task: Any,
    provider: Any,
    pricing_model: str,
    effort: dict[str, Any] | None,
    config: OrchestrationConfig,
    max_steps: int,
    max_run_cost: float | None,
    deadline: Any,
    collector: Any,
    executor: Any,
    workspace: Path,
    run_dir: Path,
    run_id: str,
) -> OrchestrationOutcome:
    """Run the serial loop once for one task; mirrors ``_run_model_loop`` shape.

    Returns token totals, the finished flag (manager sign-off with a passing
    verification, or a finalizer that verified clean), the cost-cap flag and
    the total step count, plus an ``orchestration`` report for the summary.

    Single-threaded serial invariant (documented, not enforced by locks): the
    manager acts once per round, exactly one worker runs at a time, and the
    next decision waits for the previous worker AND its external verification.
    """
    ledger = Ledger(run_dir / "ledger")
    roles: dict[str, _RoleStats] = {}
    spend = {"prompt": 0, "completion": 0, "steps": 0, "calls": 0}
    state = {"cost_capped": False}
    model_deadline = _ReserveDeadline(deadline, config.grading_reserve_s)

    def record(role: str, prompt_tokens: int, completion_tokens: int, steps: int) -> None:
        st = roles.setdefault(role, _RoleStats())
        st.llm_calls += 1
        st.steps += steps
        st.prompt_tokens += prompt_tokens
        st.completion_tokens += completion_tokens
        spend["prompt"] += prompt_tokens
        spend["completion"] += completion_tokens
        spend["steps"] += steps
        spend["calls"] += 1

    def run_cost() -> float | None:
        return cost_usd(pricing_model, int(spend["prompt"]), int(spend["completion"]))

    def steps_left() -> int:
        return max_steps - spend["steps"]

    def budget_stop() -> str | None:
        if deadline.is_expired():
            deadline.record_exceeded(collector)
            return "deadline"
        if max_run_cost is not None:
            cost = run_cost()
            if cost is not None and cost >= max_run_cost:
                collector.record(
                    "cost_cap_exceeded",
                    {"type": "run", "max_run_cost": max_run_cost, "cost": round(cost, 4)},
                )
                state["cost_capped"] = True
                return "cost-cap"
        if steps_left() <= 0:
            collector.record("step_budget_exhausted", {"max_steps": max_steps})
            return "step-budget"
        return None

    # -- role adapters ---------------------------------------------------------

    def one_shot(role: str, *, system: str, user: str) -> str | None:
        if not model_deadline.ensure_time(collector, f"llm_request:{role}"):
            return None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            resp = _complete_provider(provider, messages, [], model_deadline.remaining_s(), effort)
        except Exception:
            if model_deadline.is_expired():
                return None
            raise
        usage = resp.usage
        collector.record("llm_request", {"role": role, "messages": 2, "tools": []})
        collector.record(
            "llm_response",
            {
                "role": role,
                "content": resp.content,
                "has_tool_calls": bool(resp.tool_calls),
                "usage": usage.model_dump(),
            },
        )
        record(role, usage.prompt_tokens, usage.completion_tokens, 1)
        content: str | None = resp.content
        return content

    def bounded_worker(
        role: str, *, system: str, user: str, steps: int, allowlist: frozenset[str]
    ) -> tuple[str | None, int]:
        """One bounded-context model loop; returns (last content, steps used)."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        pt, ct, _fin, _cap, used, content = _run_model_loop(
            provider,
            _tool_schemas(allowlist),
            messages,
            steps,
            collector,
            executor,
            run_id,
            model_deadline,
            effort=effort,
            role=role,
            tool_allowlist=allowlist,
        )
        record(role, pt, ct, used)
        return content, used

    # -- ledger scaffolding ------------------------------------------------------

    task_md = "\n".join(
        [
            f"# Task {task.task_id}",
            "",
            "## Issue",
            str(task.issue).strip(),
            "",
            "## Execution constraints",
            f"- wall-clock budget {deadline.timeout_s}s shared by every role, less a"
            f" {config.grading_reserve_s}s reserve held back for final grading",
            f"- total model-loop steps {max_steps} shared by every role",
            f"- manager rounds: soft cap {config.max_rounds}, hard ceiling {config.hard_max_rounds}",
            "- strictly serial: manager -> one worker -> verification -> manager",
            "- do not search for or read hidden test material; verify with the repo's own suite",
        ]
    )
    ledger.write_task(task_md)
    collector.record(
        "orchestration_start",
        {
            "engine": "ledger-v1",
            "pricing_model": pricing_model,
            "max_rounds": config.max_rounds,
            "hard_max_rounds": config.hard_max_rounds,
            "brainstorm_steps": config.brainstorm_steps,
            "worker_max_steps": config.worker_max_steps,
            "finalizer_steps": config.finalizer_steps,
            "max_steps": max_steps,
            "max_run_cost": max_run_cost,
        },
    )

    finished = False
    termination = "unknown"
    rounds = 0
    stall = 0
    fresh_used = 0
    verify_failures = 0
    reverts = 0
    finalizer_used = False
    last_task_id: str | None = None
    last_round_block = ""
    last_verify_ok = False

    def manager_message(*, seed: bool, round_block: str) -> str:
        return prompts.manager_user_message(
            task_clip=clip(str(task.issue), TASK_CLIP),
            plan=clip(ledger.plan_text(), PLAN_CLIP),
            tasks_render=ledger.render_tasks(),
            notes_tail=clip(ledger.notes_text(), NOTES_TAIL),
            verification_tail=clip(ledger.verification_text(), VERIFY_TAIL),
            best=clip(ledger.best_text(), BEST_CLIP),
            diff_stat=clip(gitstate.diff_stat(workspace), DIFFSTAT_CLIP),
            last_round_block=round_block,
            seed_instruction=(
                "This is the first call: seed 3 to 6 bounded falsifiable tasks, write"
                " the plan, then name exactly one as next (or null if the task is"
                " already complete)."
                if seed
                else "Apply any ledger updates, then name exactly one next task or"
                " set complete. JSON only."
            ),
        )

    def manager_call(*, seed: bool, round_block: str) -> dict[str, Any] | None:
        system = prompts.manager_system_prompt()
        user = manager_message(seed=seed, round_block=round_block)
        content = one_shot("manager", system=system, user=user)
        if content is None:
            return None
        decision = _parse_decision(content)
        if decision is None:
            # The retry resends the full ledger context: manager calls are
            # stateless, so asking for JSON alone would leave it blind.
            content = one_shot(
                "manager",
                system=system,
                user=user + "\n\nYour previous reply did not parse as JSON. Reply with "
                "ONLY the JSON object described in the system prompt.",
            )
            if content is None:
                return None
            decision = _parse_decision(content)
        return decision

    def apply_decision(decision: dict[str, Any]) -> None:
        if decision.get("plan"):
            ledger.set_plan(str(decision["plan"]))
        notes = decision.get("notes")
        if isinstance(notes, list):
            ledger.append_notes([str(n) for n in notes])
        if decision.get("best"):
            ledger.set_best(str(decision["best"]))
        for upd in decision.get("task_updates") or []:
            if not isinstance(upd, dict):
                continue
            try:
                evidence = [str(e) for e in (upd.get("evidence") or [])]
                ledger.mark(
                    str(upd.get("id", "")),
                    str(upd.get("status", "")),
                    result=upd.get("result"),
                    evidence=evidence or None,
                )
            except ValueError as exc:
                ledger.append_notes([f"rejected task_update: {exc}"])
        for new in decision.get("new_tasks") or []:
            if isinstance(new, dict) and new.get("id"):
                ledger.add_task(str(new["id"]), str(new.get("desc", "")))

    # -- initial ideation (single, serial; skipped for cheap tasks) -----------------

    if (
        max_steps >= config.skip_brainstorm_below_steps
        and budget_stop() is None
        and not model_deadline.is_expired()
    ):
        b_content, _used = bounded_worker(
            "brainstorm",
            system=prompts.brainstorm_system_prompt(),
            user=prompts.brainstorm_user_message(
                task_clip=clip(str(task.issue), TASK_CLIP),
                file_list=gitstate.file_list(workspace),
            ),
            steps=min(config.brainstorm_steps, max(1, steps_left() - config.min_worker_steps)),
            allowlist=READONLY_TOOLS,
        )
        if b_content:
            ledger.append_notes([f"brainstorm: {clip(b_content, 1_200)}"])

    # -- seed decision ---------------------------------------------------------------

    decision = manager_call(seed=True, round_block="")
    if decision is not None:
        if decision.get("tasks"):
            try:
                ledger.seed_tasks([t for t in decision["tasks"] if isinstance(t, dict)])
            except ValueError as exc:
                ledger.append_notes([f"rejected task seed: {exc}"])
        apply_decision(decision)
    else:
        termination = "deadline" if deadline.is_expired() else "manager-unreachable"

    # -- serial rounds -----------------------------------------------------------------

    while decision is not None:
        stop = budget_stop()
        if stop:
            termination = stop
            break
        if model_deadline.is_expired():
            termination = "deadline"
            break
        if rounds >= min(config.max_rounds, config.hard_max_rounds):
            termination = "round-budget"
            break
        nxt = decision.get("next")
        if not isinstance(nxt, dict) or not nxt.get("task_id"):
            if decision.get("complete"):
                # Sign-off counts only when the latest external check passed;
                # an unverified completion claim routes to the finalizer.
                finished = last_verify_ok
                termination = (
                    "manager-complete" if last_verify_ok else "manager-complete-unverified"
                )
            else:
                termination = "manager-no-task"
            break
        task_id = str(nxt["task_id"])
        brief = str(nxt.get("brief", "")).strip() or str(nxt.get("desc", ""))
        if ledger.get(task_id) is None:
            ledger.add_task(task_id, brief or f"manager-selected task {task_id}")

        if task_id == last_task_id:
            stall += 1
        else:
            stall = 0

        if stall >= 2:
            if (
                fresh_used < config.max_fresh_perspectives
                and steps_left() >= 2 * config.min_worker_steps
            ):
                fresh_content, _used = bounded_worker(
                    "fresh-perspective",
                    system=prompts.fresh_perspective_system_prompt(),
                    user=prompts.fresh_perspective_user_message(
                        task_clip=clip(str(task.issue), TASK_CLIP),
                        file_list=gitstate.file_list(workspace),
                        symptom=clip(last_round_block, 1_000)
                        or "the loop keeps repeating one task",
                    ),
                    steps=min(10, max(1, steps_left() - config.min_worker_steps)),
                    allowlist=READONLY_TOOLS,
                )
                fresh_used += 1
                stall = 0
                if fresh_content:
                    ledger.append_notes([f"fresh perspective: {clip(fresh_content, 1_200)}"])
                last_round_block = (
                    "An independent reviewer (no access to plan/notes) diagnosed the "
                    f"stall:\n{clip(fresh_content or '(no output)', WORKER_TAIL)}\n"
                    "Choose a task that is NOT the stalled one, or complete."
                )
                decision = manager_call(seed=False, round_block=last_round_block)
                if decision is not None:
                    apply_decision(decision)
                continue
            termination = "no-progress"
            break

        if steps_left() < config.min_worker_steps:
            termination = "step-budget"
            break

        try:
            checkpoint = gitstate.checkpoint(workspace)
        except RuntimeError as exc:
            checkpoint = None
            ledger.append_notes([f"checkpoint failed: {exc}"])

        ledger.mark(task_id, "in_progress")
        task_row = ledger.get(task_id) or {}
        rounds += 1
        w_content, _used = bounded_worker(
            f"worker:{task_id}",
            system=prompts.worker_system_prompt(),
            user=prompts.worker_user_message(
                task_clip=clip(str(task.issue), TASK_CLIP),
                plan=clip(ledger.plan_text(), PLAN_CLIP),
                notes_tail=clip(ledger.notes_text(), NOTES_TAIL),
                task_id=task_id,
                task_desc=str(task_row.get("desc", "")),
                brief=brief,
                diff_stat=clip(gitstate.diff_stat(workspace), DIFFSTAT_CLIP),
                file_list=gitstate.file_list(workspace),
            ),
            steps=min(config.worker_max_steps, steps_left()),
            allowlist=ALL_TOOL_NAMES,
        )

        if w_content is None:
            # Truncated attempt: persist conclusions before anything else so a
            # cut-off worker never leaves the ledger blind.
            stat = gitstate.diff_stat(workspace)
            summary = one_shot(
                "summarizer",
                system=prompts.summarizer_system_prompt(),
                user=prompts.summarizer_user_message(partial_tail="", diff_stat=stat),
            )
            if summary:
                ledger.append_notes(
                    [f"worker {task_id} cut off mid-attempt: {clip(summary, 1_000)}"]
                )

        verify_cmd = str(decision.get("verify_cmd") or "").strip()
        if not verify_cmd:
            verify_cmd = _external_verify_command(workspace)
        try:
            vres = executor.run_command(
                RunCommandArgs(
                    cmd=verify_cmd,
                    timeout=_remaining_timeout(600, model_deadline.remaining_s),
                )
            )
        except Exception as exc:  # a verify tooling failure counts as a failed check
            vres = {"stdout": "", "stderr": f"verify runner error: {exc}", "exit_code": 127}
        exit_code = int(vres.get("exit_code", 1)) if isinstance(vres, dict) else 1
        last_verify_ok = exit_code == 0
        verdict = "PASS" if last_verify_ok else "FAIL"
        evidence_line = f"verify round {rounds}: {verify_cmd} exit={exit_code} {verdict}"
        reverted = False
        if not last_verify_ok:
            verify_failures += 1
            if decision.get("revert_on_failure", False) and checkpoint is not None:
                try:
                    gitstate.restore(workspace, checkpoint)
                    reverts += 1
                    reverted = True
                    collector.record("orchestration_revert", {"round": rounds, "task_id": task_id})
                except RuntimeError as exc:
                    ledger.append_notes([f"revert failed after {task_id}: {exc}"])
        ledger.append_verification(
            f"[round {rounds}] {task_id}: {verify_cmd}\nexit={exit_code} {verdict}"
            + ("\nworkspace reverted to pre-round checkpoint" if reverted else "")
            + f"\n{_first_lines(str(vres.get('stdout', '')) + chr(10) + str(vres.get('stderr', '')))}"
        )
        collector.record(
            "orchestration_verify",
            {
                "round": rounds,
                "task_id": task_id,
                "cmd": verify_cmd,
                "exit_code": exit_code,
                "ok": last_verify_ok,
                "reverted": reverted,
            },
        )

        if not last_verify_ok:
            ledger.mark(task_id, "pending", evidence=[evidence_line])
            last_task_id = task_id
            last_round_block = (
                f"Worker {task_id} output tail:\n{clip(w_content or '(truncated)', WORKER_TAIL)}\n\n"
                f"Verification FAILED: {evidence_line}"
                + ("\nThe workspace was reverted to the pre-round checkpoint." if reverted else "")
                + "\nThe next task must fix, revert or switch approach."
            )
            decision = manager_call(seed=False, round_block=last_round_block)
            if decision is not None:
                apply_decision(decision)
            continue

        ledger.mark(
            task_id,
            "done",
            result=str(decision.get("reason") or brief),
            evidence=[evidence_line],
        )
        last_task_id = task_id
        last_round_block = (
            f"Worker {task_id} output tail:\n{clip(w_content or '(no text output)', WORKER_TAIL)}"
            f"\n\nVerification: {evidence_line}"
        )
        decision = manager_call(seed=False, round_block=last_round_block)
        if decision is not None:
            apply_decision(decision)
    if termination == "unknown":
        expired_window = deadline.is_expired() or model_deadline.is_expired()
        termination = "deadline" if expired_window else "manager-unreachable"

    # -- finalizer (unresolved cases only) ------------------------------------------------

    final_verify_ok: bool | None = None
    if (
        not finished
        and termination != "cost-cap"
        and steps_left() >= config.min_worker_steps
        and not model_deadline.is_expired()
    ):
        finalizer_used = True
        f_content, _used = bounded_worker(
            "finalizer",
            system=prompts.finalizer_system_prompt(),
            user=prompts.finalizer_user_message(
                task_clip=clip(str(task.issue), TASK_CLIP),
                plan=clip(ledger.plan_text(), PLAN_CLIP),
                notes_tail=clip(ledger.notes_text(), NOTES_TAIL),
                verification_tail=clip(ledger.verification_text(), VERIFY_TAIL),
                best=clip(ledger.best_text(), BEST_CLIP),
                diff_stat=clip(gitstate.diff_stat(workspace), DIFFSTAT_CLIP),
            ),
            steps=min(config.finalizer_steps, steps_left()),
            allowlist=ALL_TOOL_NAMES,
        )
        if f_content:
            ledger.append_notes([f"finalizer: {clip(f_content, 1_200)}"])
        final_cmd = _external_verify_command(workspace)
        try:
            fres = executor.run_command(
                RunCommandArgs(
                    cmd=final_cmd, timeout=_remaining_timeout(600, model_deadline.remaining_s)
                )
            )
            final_verify_ok = (
                int(fres.get("exit_code", 1)) == 0 if isinstance(fres, dict) else False
            )
        except Exception:
            final_verify_ok = False
        ledger.append_verification(f"[finalizer] {final_cmd}\nexit_ok={final_verify_ok}")
        if final_verify_ok:
            finished = True

    report = {
        "engine": "ledger-v1",
        "termination": termination,
        "rounds": rounds,
        "llm_calls": spend["calls"],
        "verify_failures": verify_failures,
        "reverts": reverts,
        "fresh_perspectives": fresh_used,
        "finalizer_used": finalizer_used,
        "final_verify_ok": final_verify_ok,
        "roles": {
            role: {
                "llm_calls": st.llm_calls,
                "steps": st.steps,
                "prompt_tokens": st.prompt_tokens,
                "completion_tokens": st.completion_tokens,
            }
            for role, st in sorted(roles.items())
        },
        "ledger_files": ledger.file_names(),
        "soft_max_rounds": config.max_rounds,
        "hard_max_rounds": config.hard_max_rounds,
    }
    collector.record("orchestration_end", report)

    return OrchestrationOutcome(
        prompt_tokens=spend["prompt"],
        completion_tokens=spend["completion"],
        finished=finished,
        cost_capped=state["cost_capped"],
        steps=spend["steps"],
        report=report,
    )
