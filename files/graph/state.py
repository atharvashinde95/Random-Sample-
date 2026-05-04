"""
LangGraph state definition for the scheduling workflow.
"""
from __future__ import annotations
from typing import Any, TypedDict


class SchedulingGraphState(TypedDict, total=False):
    run_id: str
    csv_path: str
    output_dir: str
    user_input: str

    llm_problem_understanding: dict[str, Any]

    csv_react_trace: list[dict[str, Any]]
    csv_validation_report: dict[str, Any]
    csv_validation_status: str
    csv_ready_for_cpsat: bool

    final_summary: dict[str, Any]

    errors: list[str]
    warnings: list[str]
