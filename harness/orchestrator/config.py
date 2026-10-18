"""Engine selection constants and the serial-loop budget configuration.

No heavy imports here: the CLI uses :func:`strip_engine_prefix` for pricing
checks without loading the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Model-spec prefix selecting this engine (e.g. ``orchestrator:openai:gpt-5.4-mini``).
ORCHESTRATOR_ENGINE = "orchestrator"


def strip_engine_prefix(spec: str) -> str:
    """Return the underlying ``provider:model`` spec of an engine-prefixed spec.

    >>> strip_engine_prefix("orchestrator:openai:gpt-5.4-mini")
    'openai:gpt-5.4-mini'
    >>> strip_engine_prefix("openai:gpt-5.4-mini")
    'openai:gpt-5.4-mini'
    """
    if spec.startswith(ORCHESTRATOR_ENGINE + ":"):
        return spec[len(ORCHESTRATOR_ENGINE) + 1 :]
    return spec


@dataclass(frozen=True)
class OrchestrationConfig:
    """Budgets for the serial manager-worker loop.

    The paper's scaffold bounds the loop at 10 manager-to-worker cycles; the
    default here is softer (6) with 10 as the hard ceiling, and every budget
    (wall-clock, total model-loop steps, per-run cost cap) is shared with the
    plain ``vulcan`` loop so an orchestrated run stays comparable to the
    baseline columns at equal budgets.
    """

    #: Soft cap on manager-to-worker rounds before the loop stops for budget.
    max_rounds: int = 6
    #: Hard ceiling no decision may exceed (paper's bound).
    hard_max_rounds: int = 10
    #: Model-loop steps granted to the single brainstorm pass.
    brainstorm_steps: int = 10
    #: Per-round cap on a worker's model-loop steps.
    worker_max_steps: int = 40
    #: Finalizer worker's model-loop steps (only when sign-off was not clean).
    finalizer_steps: int = 20
    #: At most this many anti-anchoring passes per run (serial, one per stall).
    max_fresh_perspectives: int = 1
    #: Do not start a worker with fewer remaining steps than this.
    min_worker_steps: int = 6
    #: Tasks whose total step budget is below this skip the brainstorm pass
    #: (the activation gate: cheap tasks do not pay for orchestration).
    skip_brainstorm_below_steps: int = 24
    #: Wall clock held back from every model phase so the hidden verifier and
    #: evaluation always run against the final workspace. The engine stops
    #: starting new work once less than this remains.
    grading_reserve_s: int = 60

    @classmethod
    def default(cls) -> OrchestrationConfig:
        return cls()
