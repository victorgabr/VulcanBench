"""Ledger-based self-orchestration engine (manager to one worker to verifier).

The engine replaces the plain single-loop ReAct solve phase with a strictly
serial manager-worker control flow over a persistent filesystem ledger, per
"Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM
Coding Performance" (arXiv:2608.26480v1), adapted to repository work.

Serial by construction: the whole engine is single-threaded and single-tasked.
There is no threading, asyncio or fan-out anywhere in this package, and no
model role other than the manager dispatches other roles.

Engine selection and pricing helpers live in :mod:`harness.orchestrator.config`
(no heavy imports so the CLI can use them without loading the loop); the
engine itself is imported lazily from ``harness.agent.loop`` to avoid a
circular import.
"""

from harness.orchestrator.config import (
    ORCHESTRATOR_ENGINE,
    OrchestrationConfig,
    strip_engine_prefix,
)

__all__ = ["ORCHESTRATOR_ENGINE", "OrchestrationConfig", "strip_engine_prefix"]
