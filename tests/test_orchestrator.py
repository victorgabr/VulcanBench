"""Ledger-orchestration engine: selection, ledger invariants, serial loop.

The end-to-end tests drive the real agent loop with a scripted provider that
dispatches on the ``VULCANBENCH_ORCH_ROLE`` sentinel each role prompt carries,
so the serial order, the external verification gate, and the revert-on-failure
path are exercised against the real verifier and real git.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import harness.agent.loop as loop_mod
from harness.agent.loop import _RunDeadline, run_agent
from harness.agent.providers import LLMProvider, LLMResponse, TokenUsage, ToolInvocation
from harness.cli import _execution_spec
from harness.orchestrator.config import (
    ORCHESTRATOR_ENGINE,
    OrchestrationConfig,
    strip_engine_prefix,
)
from harness.orchestrator.engine import (
    _external_verify_command,
    _parse_decision,
    _ReserveDeadline,
    run_orchestration,
)
from harness.orchestrator.gitstate import checkpoint, restore
from harness.orchestrator.ledger import Ledger

HELLO_LINE = 'print("hello from vulcanbench")\n'


class ScriptedProvider(LLMProvider):
    """Role-aware script for the serial loop; mirrors MockProvider's hello-world solve.

    ``verify_cmds`` maps manager call index (1-based) to the verify_cmd the
    manager supplies in its decision, letting tests force a failing round.
    """

    def __init__(
        self,
        *,
        verify_cmds: dict[int, str] | None = None,
        revert: bool | None = True,
        skip_second_edit: bool = False,
    ) -> None:
        super().__init__("scripted")
        self.verify_cmds = verify_cmds or {}
        self.revert = revert
        self.skip_second_edit = skip_second_edit
        self.calls: dict[str, int] = {}
        self.role_order: list[str] = []

    def _manager_json(self, payload: dict[str, Any]) -> str:
        """Apply the revert-knob policy: None omits the key (engine default)."""
        if self.revert is None:
            payload.pop("revert_on_failure", None)
        else:
            payload["revert_on_failure"] = self.revert
        return json.dumps(payload)

    @property
    def name(self) -> str:
        return "mock"

    def complete(  # noqa: PLR0911
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        del tools, timeout_s, effort
        usage = TokenUsage(prompt_tokens=50, completion_tokens=20)
        if any("VULCANBENCH_JUDGE" in str(m.get("content", "")) for m in messages):
            return LLMResponse(content='{"score": 80, "rationale": "mock judge"}', usage=usage)
        system = str(messages[0].get("content", ""))
        role = system.partition("VULCANBENCH_ORCH_ROLE: ")[2].split("\n", 1)[0]
        self.calls[role] = self.calls.get(role, 0) + 1
        n = self.calls[role]
        self.role_order.append(role)

        if role == "brainstorm":
            if n == 1:
                # Try to cross the read-only boundary: the engine must deny it.
                return LLMResponse(
                    tool_calls=[
                        ToolInvocation(
                            id="b1",
                            name="edit_file",
                            arguments={
                                "path": "premature.py",
                                "old_string": "",
                                "new_string": "x\n",
                            },
                        )
                    ],
                    usage=usage,
                )
            return LLMResponse(
                content="Two routes: write the file, or import a module.", usage=usage
            )

        if role == "manager":
            if n == 1:
                return LLMResponse(
                    content=self._manager_json(
                        {
                            "tasks": [
                                {
                                    "id": "T1",
                                    "desc": "create hello.py with the required print line",
                                },
                                {
                                    "id": "T2",
                                    "desc": "re-create hello.py if the first attempt was reverted",
                                },
                                {"id": "T3", "desc": "clean up scratch files"},
                            ],
                            "plan": "Hard part is the exact stdout text. Write the file, gate on test -s.",
                            "next": {
                                "task_id": "T1",
                                "brief": "create hello.py with the exact print line",
                            },
                            "verify_cmd": self.verify_cmds.get(1, "test -s hello.py"),
                            "complete": False,
                            "reason": "seed",
                        }
                    ),
                    usage=usage,
                )
            # Ledger-aware: complete once a task shows [done] (verification
            # passed); otherwise retry the write with a corrected gate.
            ledger_view = str(messages[-1].get("content", ""))
            nxt_task = None if "[done]" in ledger_view else {2: "T2"}.get(n)
            if nxt_task is not None:
                return LLMResponse(
                    content=self._manager_json(
                        {
                            "task_updates": [
                                {
                                    "id": "T1",
                                    "status": "blocked",
                                    "result": "reverted",
                                    "evidence": ["verify failed"],
                                }
                            ],
                            "next": {
                                "task_id": nxt_task,
                                "brief": "write hello.py again; keep it minimal",
                            },
                            "verify_cmd": self.verify_cmds.get(n, "test -s hello.py"),
                            "complete": False,
                            "reason": "retry with corrected gate",
                        }
                    ),
                    usage=usage,
                )
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_updates": [
                            {
                                "id": "T1",
                                "status": "done",
                                "result": "hello.py present",
                                "evidence": ["verify pass"],
                            }
                        ],
                        "next": None,
                        "complete": True,
                        "reason": "acceptance criterion met and verification passed",
                    }
                ),
                usage=usage,
            )

        if role == "worker":
            workers_seen = self.calls.get("worker", 0)
            # Odd worker calls issue the edit; even ones close with text.
            prior_tool_turns = sum(1 for m in messages if m.get("role") == "tool")
            repeat_write = self.skip_second_edit and self.calls.get("worker", 0) >= 3
            if prior_tool_turns == 0 and not repeat_write:
                return LLMResponse(
                    tool_calls=[
                        ToolInvocation(
                            id=f"w{workers_seen}",
                            name="edit_file",
                            arguments={
                                "path": "hello.py",
                                "old_string": "",
                                "new_string": HELLO_LINE,
                            },
                        )
                    ],
                    usage=usage,
                )
            return LLMResponse(content="wrote hello.py and checked the print line", usage=usage)

        # fresh-perspective / summarizer / finalizer: text-only.
        return LLMResponse(content="bounded review complete", usage=usage)


def _orchestrated_run(
    tmp_path: Path, provider: ScriptedProvider, **overrides: Any
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "task_id": "hello-world",
        "model": f"{ORCHESTRATOR_ENGINE}:mock:scripted",
        "output_dir": tmp_path,
        "tasks_root": Path("tasks/v1"),
        "provider": provider,
        "max_steps": 30,
        "timeout_s": 300.0,
    }
    kwargs.update(overrides)
    return run_agent(**kwargs)


# -- selection and pricing strip ------------------------------------------------


def test_strip_engine_prefix() -> None:
    assert strip_engine_prefix("orchestrator:openai:gpt-5.4-mini") == "openai:gpt-5.4-mini"
    assert strip_engine_prefix("openai:gpt-5.4-mini") == "openai:gpt-5.4-mini"
    assert strip_engine_prefix("codex:gpt-5.4") == "codex:gpt-5.4"


def test_execution_spec_orchestrator() -> None:
    assert (
        _execution_spec("openai:gpt-5.4-mini", "orchestrator", "auto")
        == "orchestrator:openai:gpt-5.4-mini"
    )
    # Idempotent when the spec already carries the engine prefix.
    assert (
        _execution_spec("orchestrator:openai:gpt-5.4-mini", "orchestrator", "auto")
        == "orchestrator:openai:gpt-5.4-mini"
    )
    with pytest.raises(ValueError, match="subscription"):
        _execution_spec("openai:gpt-5.4-mini", "orchestrator", "subscription")
    with pytest.raises(ValueError, match="cannot wrap vendor CLI"):
        _execution_spec("codex:gpt-5.4", "orchestrator", "auto")
    with pytest.raises(ValueError, match="provider:model"):
        _execution_spec("gpt-5.4-mini", "orchestrator", "auto")
    with pytest.raises(ValueError, match="orchestrator"):
        _execution_spec("openai:gpt-5.4-mini", "nonsense", "auto")


def test_resolve_run_engine_prefix() -> None:
    adapter, provider, _meta, spec = loop_mod._resolve_run_engine(
        "orchestrator:mock:synthetic", None, None, "local", False
    )
    assert adapter is None
    assert spec == "mock:synthetic"
    assert provider is not None and provider.name == "mock"

    # An injected provider stays injected; the prefix still selects the engine.
    scripted = ScriptedProvider()
    _, prov2, _, spec2 = loop_mod._resolve_run_engine(
        "orchestrator:mock:scripted", scripted, None, "local", False
    )
    assert prov2 is scripted and spec2 == "mock:scripted"

    with pytest.raises(ValueError, match="cannot wrap a vendor CLI"):
        loop_mod._resolve_run_engine("orchestrator:codex:gpt-5.4", None, None, "local", False)


# -- ledger and git primitives ---------------------------------------------------


def test_ledger_serial_invariant_and_persistence(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "ledger")
    led.write_task("# Task demo\n")
    led.seed_tasks(
        [
            {"id": "T1", "desc": "first"},
            {"id": "T2", "desc": "second"},
            {"id": "T3", "desc": "third"},
        ]
    )
    led.mark("T1", "in_progress")
    with pytest.raises(ValueError, match="serial invariant"):
        led.mark("T2", "in_progress")
    with pytest.raises(ValueError, match="unknown status"):
        led.mark("T1", "shipped")
    with pytest.raises(ValueError, match="unknown task id"):
        led.mark("T9", "done")
    led.mark("T1", "done", result="ok")  # no evidence given
    assert "manager assertion (unverified)" in led.get("T1")["evidence"]

    # A fresh Ledger over the same directory replays the persisted state.
    reopened = Ledger(tmp_path / "ledger")
    assert [t["id"] for t in reopened.tasks()] == ["T1", "T2", "T3"]
    assert reopened.get("T1")["status"] == "done"
    led.append_verification("verify round 1: pytest exit=0 PASS")
    assert "round 1" in reopened.verification_text()

    with pytest.raises(ValueError, match="duplicate"):
        led.seed_tasks([{"id": "T1", "desc": "a"}, {"id": "T1", "desc": "b"}])


def test_git_checkpoint_restore(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "t",
                "GIT_COMMITTER_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_EMAIL": "t@t",
                "HOME": str(tmp_path),
            },
        )

    git("init", "-q")
    (tmp_path / "base.py").write_text("base\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")

    (tmp_path / "good.py").write_text("good\n", encoding="utf-8")
    cp = checkpoint(tmp_path)

    # A bad round edits tracked work and adds untracked junk.
    (tmp_path / "base.py").write_text("corrupted\n", encoding="utf-8")
    (tmp_path / "junk.log").write_text("junk\n", encoding="utf-8")
    restore(tmp_path, cp)

    assert (tmp_path / "base.py").read_text(encoding="utf-8") == "base\n"
    assert (tmp_path / "good.py").read_text(encoding="utf-8") == "good\n"
    assert not (tmp_path / "junk.log").exists()


def test_external_verify_command_preserves_exit_codes(tmp_path: Path) -> None:
    assert _external_verify_command(tmp_path) == "true"  # no discoverable suite
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    cmd = _external_verify_command(tmp_path)
    assert "pytest" in cmd and "|| true" not in cmd
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    assert "|| true" not in _external_verify_command(tmp_path)


def test_parse_decision_tolerates_fences() -> None:
    assert _parse_decision('```json\n{"complete": true}\n```') == {"complete": True}
    assert _parse_decision("prose only, no json") is None


# -- end-to-end serial loop -------------------------------------------------------


def test_orchestrated_run_solves_hello_world(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    res = _orchestrated_run(tmp_path, provider)
    summary = res["summary"]
    run_dir = tmp_path / res["run_id"]

    # The real verifier gates functional success on the orchestrated workspace.
    assert summary["scores"]["functional"] == 1.0
    assert summary["finished"] is True
    assert summary["cost_usd"] == 0.0  # mock pricing, identical to the plain loop

    orch = summary["orchestration"]
    assert orch["termination"] == "manager-complete"
    assert orch["rounds"] == 1
    assert orch["verify_failures"] == 0 and orch["reverts"] == 0
    assert orch["finalizer_used"] is False
    assert set(orch["roles"]) >= {"manager", "brainstorm", "worker:T1"}
    assert orch["ledger_files"]

    # Strictly serial order: brainstorm, seed manager, one worker turn pair, sign-off.
    assert provider.role_order[:6] == [
        "brainstorm",
        "brainstorm",
        "manager",
        "worker",
        "worker",
        "manager",
    ]

    # The read-only gate denied brainstorm's premature edit attempt.
    trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    denied = [
        e
        for e in trace
        if e["type"] == "tool_observation"
        and "not available to role" in str(e.get("data", {}).get("error", ""))
    ]
    assert denied and denied[0]["data"]["result"] is None
    assert not (run_dir / "workspace" / "premature.py").exists()

    # Every engine model call is attributed to its role in the trace.
    roles = {e["data"].get("role") for e in trace if e["type"] == "llm_response"}
    assert {"manager", "brainstorm", "worker:T1"} <= roles

    # Ledger files were persisted under the run directory.
    assert (run_dir / "ledger" / "tasks.json").exists()
    tasks = json.loads((run_dir / "ledger" / "tasks.json").read_text())
    assert [t["id"] for t in tasks] == ["T1", "T2", "T3"]
    assert tasks[0]["status"] == "done"

    # Pricing bills the stripped spec: an orchestrated run is comparable at
    # equal budgets with the plain vulcan column.
    assert summary["manifest"]["model"] == "orchestrator:mock:scripted"
    assert (run_dir / "final.patch").read_text().count("hello.py") >= 1


def test_orchestrated_failed_verification_reverts_then_repairs(tmp_path: Path) -> None:
    provider = ScriptedProvider(verify_cmds={1: "test -e never-created-marker.txt"})
    res = _orchestrated_run(tmp_path, provider)
    summary = res["summary"]

    orch = summary["orchestration"]
    assert orch["termination"] == "manager-complete"
    assert orch["rounds"] == 2
    assert orch["verify_failures"] == 1 and orch["reverts"] == 1
    assert summary["finished"] is True

    # Round 1's write was reverted (checkpoint restore), round 2's survived.
    workspace = tmp_path / res["run_id"] / "workspace"
    hello = workspace / "hello.py"
    assert hello.exists() and hello.read_text() == HELLO_LINE

    trace = (tmp_path / res["run_id"] / "trace.jsonl").read_text()
    assert '"orchestration_revert"' in trace


def test_orchestrated_step_budget_stops_before_worker(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    res = _orchestrated_run(tmp_path, provider, max_steps=4)
    summary = res["summary"]
    orch = summary["orchestration"]
    assert orch["termination"] == "step-budget"
    assert orch["rounds"] == 0
    assert summary["finished"] is False
    # The engine refused to start a worker it could not budget.
    assert "worker" not in provider.calls
    # Cheap task: brainstorm was skipped below the activation gate.
    assert "brainstorm" not in provider.calls
    assert summary["scores"]["functional"] == 0.0


def test_config_default_budgets_are_serial_only() -> None:
    cfg = OrchestrationConfig.default()
    assert cfg.max_rounds <= cfg.hard_max_rounds
    assert cfg.max_fresh_perspectives == 1  # never an ensemble
    assert run_orchestration.__name__ == "run_orchestration"


# -- grading reserve and revert policy ----------------------------------------


def test_reserve_deadline_withholds_grading_tail() -> None:
    inner = _RunDeadline(time.monotonic(), 2.0)
    shim = _ReserveDeadline(inner, 1.5)
    rem = shim.remaining_s()
    assert rem is not None and rem <= 0.5
    time.sleep(1.0)
    # The engine's view expires and self-limits while the real run deadline
    # still holds the reserved slice, and a shim stop never flags the budget.
    assert shim.remaining_s() == 0.0 and shim.is_expired()
    assert not inner.is_expired() and not inner.exceeded


def test_orchestrated_failed_round_keeps_work_by_default(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        verify_cmds={1: "test -e never-created-marker.txt"},
        revert=None,
        skip_second_edit=True,
    )
    res = _orchestrated_run(tmp_path, provider)
    summary = res["summary"]
    orch = summary["orchestration"]

    assert orch["termination"] == "manager-complete"
    assert orch["rounds"] == 2
    assert orch["verify_failures"] == 1
    assert orch["reverts"] == 0
    # A failed round keeps its edits: the surviving hello.py passes the
    # repair round's gate and is graded by the real verifier.
    workspace = tmp_path / res["run_id"] / "workspace"
    hello = workspace / "hello.py"
    assert hello.exists() and hello.read_text() == HELLO_LINE
    assert summary["scores"]["functional"] == 1.0
    assert '"orchestration_revert"' not in (tmp_path / res["run_id"] / "trace.jsonl").read_text()


class StallingProvider(LLMProvider):
    """Never signs off: dispatches bounded rounds until the wall clock runs out."""

    def __init__(self, call_s: float = 0.4) -> None:
        super().__init__("scripted")
        self.call_s = call_s
        self.n = 0

    @property
    def name(self) -> str:
        return "mock"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        del tools, timeout_s, effort
        time.sleep(self.call_s)
        usage = TokenUsage(prompt_tokens=50, completion_tokens=20)
        if any("VULCANBENCH_JUDGE" in str(m.get("content", "")) for m in messages):
            return LLMResponse(content='{"score": 80, "rationale": "mock judge"}', usage=usage)
        role = (
            str(messages[0].get("content", ""))
            .partition("VULCANBENCH_ORCH_ROLE: ")[2]
            .split("\n", 1)[0]
        )
        if role == "manager":
            self.n += 1
            task_id = f"T{self.n}"
            return LLMResponse(
                content=json.dumps(
                    {
                        "next": {"task_id": task_id, "brief": "inspect the repo; do not edit"},
                        "verify_cmd": "true",
                        "complete": False,
                    }
                ),
                usage=usage,
            )
        return LLMResponse(content="probed; no edits", usage=usage)


def test_orchestrated_run_reserves_wall_clock_for_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        OrchestrationConfig,
        "default",
        classmethod(lambda cls: replace(cls(), grading_reserve_s=2.0)),
    )
    res = _orchestrated_run(tmp_path, StallingProvider(), timeout_s=4.0)
    summary = res["summary"]
    orch = summary["orchestration"]

    # The engine self-limits on its reserve view and reports the deadline stop,
    # but the hidden verifier runs against the final workspace instead of
    # being zeroed by an exhausted run budget.
    assert orch["termination"] == "deadline"
    assert orch["rounds"] >= 1
    assert summary["finished"] is False
    assert summary["scores"].get("budget_exceeded") is not True
