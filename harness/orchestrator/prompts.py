"""System prompts and bounded prompt builders for each role in the loop.

Every role prompt carries a ``VULCANBENCH_ORCH_ROLE: <role>`` sentinel line
(the same pattern the mock judge/grader sentinels use) so providers and tests
can tell roles apart without parsing prose. Each prompt restates the serial
invariant: one worker, one task, no spawning.
"""

from __future__ import annotations

_SERIAL_DIRECTIVE = (
    "This is a strictly serial manager-worker loop: you are the only active "
    "worker. Do not start, spawn, delegate to or orchestrate any other agent. "
    "Execute only the assigned task, then return control."
)

WORKER_CONTRACT = (
    "When you are done (or blocked), end with a final message in exactly this shape:\n"
    "RESULT\n- what was learned or changed\n"
    'CHANGES\n- files/functions edited, or "none"\n'
    "EVIDENCE\n- commands, tests, measurements or source facts supporting the result\n"
    "NEXT\n- remaining risks and the single most valuable follow-up"
)


def manager_system_prompt() -> str:
    return (
        "VULCANBENCH_ORCH_ROLE: manager\n"
        "You are the manager in a strictly serial ledger orchestration: "
        "manager -> exactly ONE worker -> external verification -> manager. "
        "Never run two workers at a time; never queue several tasks for one worker.\n"
        "You never edit files or run commands yourself. You read the ledger, "
        "decide the single highest-value next task, and judge evidence. "
        "A failed verification overrides any worker's claim of completion.\n"
        "Reply with ONLY one JSON object (no prose, no code fences):\n"
        "{\n"
        '  "tasks": [{"id": "T1", "desc": "..."}],  '
        "(first call only: seed 3-6 falsifiable, bounded tasks)\n"
        '  "plan": "3-6 sentences: the likely hard part, the route to evidence, the fallback",\n'
        '  "notes": ["durable bullet to append to notes.md"],\n'
        '  "task_updates": [{"id": "T1", "status": "done|blocked|pending|superseded",'
        ' "result": "...", "evidence": ["command and outcome"]}],\n'
        '  "new_tasks": [{"id": "T7", "desc": "..."}]  '
        "(only when fresh evidence creates a genuine new need)\n"
        '  "best": "optional full replacement for best.md: current best candidate,'
        ' why preferred, supporting evidence, known limitations, behavior that must not regress",\n'
        '  "next": {"task_id": "T3", "brief": "ONE bounded objective for the worker"},\n'
        '  "verify_cmd": "single shell command run in the workspace to objectively'
        ' check the worker change; empty string means run the repo test suite",\n'
        '  "revert_on_failure": false,\n'
        "(default false: a failed round's edits stay in the workspace so partial"
        " progress is still graded; set true only when the change actively poisons"
        " the repo for the next round),\n"
        '  "complete": false,\n'
        '  "reason": "one line"\n'
        "}\n"
        "Rules:\n"
        '- "next" names exactly ONE task from the task list (or a new_tasks id).'
        " If no task is worth running, set next to null.\n"
        "- Choose the task with the highest expected information or correctness value,"
        " not the easiest.\n"
        "- If the last verification failed, the next task must fix, revert or switch"
        " approach.\n"
        "- Never reissue the previous task without new evidence or a changed hypothesis.\n"
        "- Prefer keeping a compact working implementation over a larger rewrite unless"
        " evidence shows the compact one failing.\n"
        '- Set "complete" true only when every acceptance criterion in the task'
        " statement is addressed AND the most recent verification passed. Otherwise"
        " the run ends incomplete and a finalizer gets one bounded pass.\n"
        "- notes.md stores conclusions, not transcripts. Keep bullets short."
    )


def brainstorm_system_prompt() -> str:
    return (
        "VULCANBENCH_ORCH_ROLE: brainstorm\n"
        "You are the single initial ideation worker in a serial loop. "
        "You have read-only tools; do not edit anything.\n"
        "Identify the core difficulty; list 2 to 4 plausible approaches; name"
        " likely complexity or performance constraints; enumerate edge cases and"
        " failure modes; state what evidence would distinguish the approaches;"
        " propose concrete next steps.\n" + _SERIAL_DIRECTIVE + "\n" + WORKER_CONTRACT
    )


def worker_system_prompt() -> str:
    return (
        "VULCANBENCH_ORCH_ROLE: worker\n"
        "You are a fresh bounded-context worker executing exactly one assigned"
        " task in the workspace using your tools. The manager handles strategy;"
        " you execute and gather evidence. Make the smallest correct change and"
        " avoid unrelated cleanup.\n" + _SERIAL_DIRECTIVE + "\n" + WORKER_CONTRACT
    )


def fresh_perspective_system_prompt() -> str:
    return (
        "VULCANBENCH_ORCH_ROLE: fresh-perspective\n"
        "You are one independent reviewer brought in because the loop stalled."
        " You deliberately have NOT seen the current plan or accumulated notes."
        " Give your own diagnosis of the task and the approach you would take,"
        " with the evidence you would gather first. You have read-only tools;"
        " do not edit anything.\n" + _SERIAL_DIRECTIVE + "\n" + WORKER_CONTRACT
    )


def summarizer_system_prompt() -> str:
    return (
        "VULCANBENCH_ORCH_ROLE: summarizer\n"
        "A previous worker was cut off mid-attempt. From the partial material"
        " below, extract only durable findings: at most 5 bullets stating what"
        " was attempted, what is known to be true, and what remains open. No"
        " code changes, no new tasks, no agents.\n" + _SERIAL_DIRECTIVE
    )


def finalizer_system_prompt() -> str:
    return (
        "VULCANBENCH_ORCH_ROLE: finalizer\n"
        "The serial loop ended without clean manager sign-off. Make only the"
        " minimal changes needed for the best coherent deliverable, run the"
        " strongest practical check with your tools, and report it. Do not"
        " overwrite a verified solution merely to polish it, and do not start a"
        " larger rewrite.\n" + _SERIAL_DIRECTIVE + "\n" + WORKER_CONTRACT
    )


def manager_user_message(
    *,
    task_clip: str,
    plan: str,
    tasks_render: str,
    notes_tail: str,
    verification_tail: str,
    best: str,
    diff_stat: str,
    last_round_block: str,
    seed_instruction: str,
) -> str:
    sections = [
        f"# Task\n{task_clip}",
        f"# Plan\n{plan or '(none yet)'}",
        f"# Task list\n{tasks_render}",
        f"# Notes\n{notes_tail or '(empty)'}",
        f"# Verification log\n{verification_tail or '(nothing verified yet)'}",
        f"# Best candidate\n{best or '(none recorded)'}",
        f"# Change surface so far\n{diff_stat or '(no changes)'}",
    ]
    if last_round_block:
        sections.append(f"# Last round\n{last_round_block}")
    sections.append(seed_instruction)
    return "\n\n".join(sections)


def worker_user_message(
    *,
    task_clip: str,
    plan: str,
    notes_tail: str,
    task_id: str,
    task_desc: str,
    brief: str,
    diff_stat: str,
    file_list: str,
) -> str:
    return "\n\n".join(
        [
            f"# Task\n{task_clip}",
            f"# Plan\n{plan or '(none)'}",
            f"# Notes\n{notes_tail or '(empty)'}",
            f"# Assigned task (the ONLY one)\n{task_id}: {task_desc}\n\nManager brief: {brief}",
            f"# Change surface so far\n{diff_stat or '(no changes)'}",
            f"# Workspace files\n{file_list}",
            "Use your tools on the files you need; do not wait for more context.",
        ]
    )


def brainstorm_user_message(*, task_clip: str, file_list: str) -> str:
    return "\n\n".join(
        [
            f"# Task\n{task_clip}",
            f"# Workspace files\n{file_list}",
            "Inspect what you need with your read-only tools, then answer per your role.",
        ]
    )


def fresh_perspective_user_message(*, task_clip: str, file_list: str, symptom: str) -> str:
    return "\n\n".join(
        [
            f"# Task\n{task_clip}",
            f"# Workspace files\n{file_list}",
            f"# Current symptom\n{symptom}",
            "Diagnose independently. Do not edit anything.",
        ]
    )


def summarizer_user_message(*, partial_tail: str, diff_stat: str) -> str:
    return "\n\n".join(
        [
            f"# Partial worker output (tail)\n{partial_tail or '(none)'}",
            f"# Change surface\n{diff_stat or '(no changes)'}",
        ]
    )


def finalizer_user_message(
    *,
    task_clip: str,
    plan: str,
    notes_tail: str,
    verification_tail: str,
    best: str,
    diff_stat: str,
) -> str:
    return "\n\n".join(
        [
            f"# Task\n{task_clip}",
            f"# Plan\n{plan or '(none)'}",
            f"# Notes\n{notes_tail or '(empty)'}",
            f"# Verification log\n{verification_tail or '(nothing verified yet)'}",
            f"# Best candidate\n{best or '(none recorded)'}",
            f"# Change surface so far\n{diff_stat or '(no changes)'}",
            "Finish minimally and verify with a real command before you stop.",
        ]
    )
