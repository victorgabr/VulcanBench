"""omp coding agent adapter: pinned executable, operator endpoint, audited.

omp (a pi fork) runs as an external CLI under the same outer sandbox the Pi
adapter uses. The executable and its SHA256 are pinned through the
environment; the model spec ("provider/model") is handed to ``--model``
verbatim, so this adapter is model-agnostic. omp keeps mutable state under
$HOME/.omp, so each run gets a scratch HOME holding a copy of the operator
config: VULCANBENCH_OMP_CONFIG_DIR when set, the user's own ~/.omp otherwise
(local-config auth).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from harness.agent.cli_agents import (
    CliAgentOutcome,
    HarnessCapabilities,
    HarnessPreflight,
    _subscription_env,
)
from harness.agent.muse_code import boundary_profile
from harness.agent.pi_agent import _bwrap_argv, add_usage, sandbox_context
from harness.agent.providers import ProviderError
from harness.pricing import cost_usd
from harness.redaction import sanitize

# omp switches analogous to Pi's DISABLED_RESOURCES: everything ambient that
# could differ between machines or sessions is off. --no-title also stops
# omp's automatic title generation, which would otherwise add a background
# model call outside the requested model's accounting.
DISABLED_RESOURCES = (
    "--no-session",
    "--no-extensions",
    "--no-skills",
    "--no-rules",
    "--no-title",
)


def pinned_install() -> tuple[Path, str, Path | None]:
    """omp executable, its pinned SHA256, and the optional profile directory."""
    binary = Path(os.environ.get("VULCANBENCH_OMP_BIN", ""))
    expected = os.environ.get("VULCANBENCH_OMP_SHA256", "")
    if not str(binary) or not binary.is_absolute() or not binary.is_file():
        raise ProviderError("Set VULCANBENCH_OMP_BIN to an absolute file path")
    if not os.access(binary, os.X_OK):
        raise ProviderError("VULCANBENCH_OMP_BIN is not executable")
    actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    if len(expected) != 64 or actual != expected:
        raise ProviderError("omp executable SHA256 mismatch")
    config = os.environ.get("VULCANBENCH_OMP_CONFIG_DIR", "")
    if not config:
        return binary.resolve(), actual, None
    config_path = Path(config)
    if not config_path.is_absolute() or not config_path.is_dir():
        raise ProviderError("Set VULCANBENCH_OMP_CONFIG_DIR to an absolute directory")
    return binary.resolve(), actual, config_path.resolve()


def _version(binary: Path) -> str | None:
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=_subscription_env(),
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    return out if proc.returncode == 0 and out else None


class OmpAdapter:
    harness_id = "omp"

    def capabilities(self) -> HarnessCapabilities:
        return HarnessCapabilities(
            self.harness_id,
            "omp",
            "omp",
            True,
            True,
            True,
            True,
            False,
            "macOS seatbelt or Linux bubblewrap outer sandbox; isolated writes; read deny",
        )

    def preflight(self) -> HarnessPreflight:
        try:
            binary, _, config = pinned_install()
        except ProviderError as exc:
            return HarnessPreflight(self.harness_id, False, None, False, None, detail=str(exc))
        version = _version(binary)
        has_config = config is not None or (Path.home() / ".omp").is_dir()
        return HarnessPreflight(
            self.harness_id,
            bool(version),
            version,
            has_config,
            "local-config" if has_config else None,
            detail=(
                "omp keeps its own provider config; live authentication is checked by the run"
                if has_config
                else "no omp profile found: set VULCANBENCH_OMP_CONFIG_DIR or configure ~/.omp"
            ),
        )

    def run_task(self, **kwargs: Any) -> CliAgentOutcome:
        return self._run(**kwargs)

    def _run(  # noqa: PLR0912, PLR0915 - one process lifecycle and receipt archive
        self,
        *,
        workspace: str | Path,
        prompt: str,
        model: str,
        priced_spec: Any,
        max_turns: int,
        collector: Any,
        stream_log_path: str | Path | None = None,
        timeout_s: float | None = None,
        network: bool = False,
        max_run_cost: float | None = None,
        effort: str | None = None,
        agent_container: Any = None,
        **kwargs: Any,
    ) -> CliAgentOutcome:
        del kwargs, max_turns, network
        if agent_container:
            raise ProviderError("omp runs locally under the outer sandbox, not in a container")
        provider, sep, provider_model = model.partition("/")
        if not sep or not provider_model:
            raise ProviderError("omp model must be a provider/model spec")
        if effort is None:
            raise ProviderError("omp requires an explicit thinking level")
        if timeout_s is None or timeout_s <= 0:
            raise ProviderError("omp requires a positive wall-clock timeout")
        binary, binary_sha256, config = pinned_install()
        checked = self.preflight()
        # HarnessPreflight.ready means subscription-ready; omp bills an
        # operator-declared endpoint, so ready is not the gate here.
        if not (checked.available and checked.authenticated):
            raise ProviderError(f"omp preflight failed: {checked.detail}")
        workspace = Path(workspace).resolve()
        scratch = Path(tempfile.mkdtemp(prefix="vb-omp-session-")).resolve()
        (scratch / "tmp").mkdir()
        env = _subscription_env()
        env.update(
            {
                "TMPDIR": str(scratch / "tmp"),
                "TMP": str(scratch / "tmp"),
                "TEMP": str(scratch / "tmp"),
            }
        )
        # omp writes session daemons, caches and locks under $HOME/.omp, so
        # the process gets a scratch HOME holding a copy of the operator
        # config; the real home stays read-only under the outer sandbox.
        agent_home = scratch / "home"
        agent_home.mkdir()
        config_source = config if config is not None else Path.home() / ".omp"
        if not config_source.is_dir():
            raise ProviderError(f"omp config source {config_source} is not a directory")
        # The source's run/ tree is live runtime state (daemon sockets, locks);
        # the sandboxed omp recreates what it needs under the staged home.
        shutil.copytree(
            config_source,
            agent_home / ".omp",
            ignore=lambda src, names: {"run"} if Path(src) == config_source else set(),
        )
        env["HOME"] = str(agent_home)
        if sys.platform == "darwin":
            profile = boundary_profile(workspace, scratch)
            profile_path = scratch / "boundary.sb"
            profile_path.write_text(profile)
            confinement = ["/usr/bin/sandbox-exec", "-f", str(profile_path)]
            boundary_note = (
                "macOS outer sandbox; workspace/session writes; shared-temp/checkout read deny; "
                "explicit TMPDIR context; no web tools; shell network not isolated; credentials "
                "in the agent environment"
            )
            context = sandbox_context(scratch)
        elif shutil.which("bwrap"):
            confinement = _bwrap_argv(workspace, scratch)
            boundary_note = (
                "Linux bubblewrap outer sandbox; host read-only; workspace/session writes only; "
                "benchmark checkout and vendor agent dirs masked with tmpfs; network shared for "
                "local inference; explicit TMPDIR context; no web tools"
            )
            context = sandbox_context(scratch)
        else:
            confinement = []
            boundary_note = (
                "no OS confinement: neither seatbelt nor bubblewrap is active, the agent ran "
                "directly on the host"
            )
            context = (
                f"Execution environment: write temporary files inside the workspace or "
                f"$TMPDIR ({scratch / 'tmp'}).\n\n"
            )
        cmd = [
            *confinement,
            str(binary),
            "--mode",
            "json",
            *DISABLED_RESOURCES,
            "--model",
            model,
            "--thinking",
            effort,
        ]
        cmd.append(context + prompt)
        session_id = str(uuid.uuid4())
        collector.record(
            "cli_agent_start",
            {
                "harness": self.harness_id,
                "argv": [*cmd[:-1], "<task prompt>"],
                "harness_version": checked.version,
                "omp_sha256": binary_sha256,
                "omp_bin": str(binary),
                "omp_config_source": str(config_source),
                "requested_model": model,
                "requested_effort": effort,
                "billing": "api",
                "max_run_cost_usd": max_run_cost,
                "execution_boundary": boundary_note,
                "max_turns_note": "omp has no step cap; the wall-clock timeout bounds the run",
            },
        )
        outcome = CliAgentOutcome(
            harness=self.harness_id,
            requested_model=model,
            session_id=session_id,
            harness_version=checked.version,
            auth_method="pinned-config-dir" if config else "local-config",
            billing="api",
            cost_basis="operator-endpoint-declared",
            execution_boundary=boundary_note,
        )
        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        errors: list[str] = []
        assert proc.stdout is not None and proc.stderr is not None
        stdout, stderr = proc.stdout, proc.stderr

        def drain() -> None:
            for line in stderr:
                errors.append(line)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()

        def kill() -> None:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)

        def on_timeout() -> None:
            outcome.timed_out = True
            kill()

        timer = threading.Timer(timeout_s, on_timeout)
        timer.daemon = True
        timer.start()
        totals: dict[str, Any] = dict(
            input=0, output=0, cacheRead=0, cacheWrite=0, reasoning=0, calls=0, pi_cost_usd=0.0
        )
        last_stop: str | None = None
        last_error = ""
        settled = False
        try:
            with Path(stream_log_path or scratch / "stdout.jsonl").open("w") as log:
                for line in stdout:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    log.write(json.dumps(sanitize(event)) + "\n")
                    log.flush()
                    kind = event.get("type")
                    if kind == "agent_end" and event.get("isTerminal"):
                        settled = True
                    if kind != "message_end":
                        continue
                    message = event.get("message") or {}
                    if message.get("role") != "assistant":
                        continue
                    add_usage(message, provider, provider_model, totals)
                    last_stop = message.get("stopReason")
                    last_error = str(message.get("errorMessage") or "")
                    text = "".join(
                        part.get("text", "")
                        for part in message.get("content") or []
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                    if text:
                        collector.record("assistant_text", {"text": text})
                    if max_run_cost is not None:
                        spent = cost_usd(
                            priced_spec,
                            totals["input"] + totals["cacheRead"] + totals["cacheWrite"],
                            totals["output"],
                            cached_input_tokens=totals["cacheRead"],
                        )
                        if spent is not None and spent > max_run_cost:
                            outcome.cost_capped = True
                            kill()
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                kill()
                proc.wait()
            reader.join(timeout=5)
        if stream_log_path:
            archive = Path(stream_log_path).parent / "omp-session"
            archive.mkdir(exist_ok=True)
            (archive / "stderr.txt").write_text(str(sanitize("".join(errors))))
            if sys.platform == "darwin":
                (archive / "boundary.sb").write_text(profile)
        outcome.prompt_tokens = totals["input"] + totals["cacheRead"] + totals["cacheWrite"]
        outcome.completion_tokens = totals["output"]
        outcome.cached_input_tokens = totals["cacheRead"]
        outcome.reasoning_output_tokens = totals["reasoning"]
        outcome.num_turns = totals["calls"]
        outcome.reported_model = model if totals["calls"] else None
        outcome.model_identity_confidence = (
            "provider-reported" if totals["calls"] else "requested-only"
        )
        # pi's add_usage accumulates the CLI-reported cost under pi_cost_usd.
        outcome.cli_reported_cost_usd = round(totals["pi_cost_usd"], 6) if totals["calls"] else None
        outcome.finished = bool(settled and last_stop == "stop" and proc.returncode == 0)
        outcome.subtype = last_stop or "no-assistant-message"
        omp_totals = dict(totals)
        omp_totals["omp_cost_usd"] = omp_totals.pop("pi_cost_usd")
        collector.record("omp_usage", omp_totals)
        if not (outcome.timed_out or outcome.cost_capped or outcome.finished):
            detail = last_error or "".join(errors)[-2000:]
            raise ProviderError(
                f"omp failed, exit={proc.returncode}, stop={last_stop}: {detail[:500]}"
            )
        collector.record("cli_agent_result", outcome.summary())
        return outcome
