"""
LangChain tool wrappers around the deterministic CSV functions in csv_tools.py.

We intentionally expose only FOUR clean tools to the ReAct agent:
    1. inspect_csv_tool
    2. validate_required_fields_tool
    3. validate_data_quality_tool
    4. validate_constraint_readiness_tool

Each tool returns a JSON string so the ReAct agent can read the observation.
None of these wrappers call any LLM.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from app.tools import csv_tools


def _dumps(result: Any) -> str:
    return json.dumps(result, indent=2, default=str)


def _parse_input(tool_input: Any) -> dict[str, Any]:
    """
    Accept dict, JSON string, single-quoted JSON, or raw csv_path string.
    Always returns a dict.
    """
    if isinstance(tool_input, dict):
        return tool_input

    text = str(tool_input).strip()

    # Strip wrapping quotes some models add
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            text = str(parsed)
        except Exception:
            text = text[1:-1]

    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Plain path string
    return {"csv_path": text}


# ---------------------------------------------------------------------------
# Four clean tools
# ---------------------------------------------------------------------------

@tool
def inspect_csv_tool(tool_input: str) -> str:
    """Inspect the CSV: returns columns, dtypes, row count, preview rows, profile.
    Input: JSON {"csv_path": "..."} or a raw path."""
    args = _parse_input(tool_input)
    return _dumps(csv_tools.inspect_csv(args.get("csv_path", "")))


@tool
def validate_required_fields_tool(tool_input: str) -> str:
    """Check that all LLM-required fields exist in the CSV.
    Input: JSON {"csv_path": "...", "required_fields": ["job_id", ...]}"""
    args = _parse_input(tool_input)
    try:
        result = csv_tools.validate_required_fields(
            args["csv_path"], args.get("required_fields", []) or [],
        )
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    return _dumps(result)


@tool
def validate_data_quality_tool(tool_input: str) -> str:
    """Check data quality across required fields: numeric, missing, negative.
    Input: JSON {"csv_path": "...", "required_fields": [...], "numeric_fields": [...]}"""
    args = _parse_input(tool_input)
    try:
        result = csv_tools.validate_data_quality(
            args["csv_path"],
            args.get("required_fields", []) or [],
            args.get("numeric_fields", []) or [],
        )
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    return _dumps(result)


@tool
def validate_constraint_readiness_tool(tool_input: str) -> str:
    """Check whether the CSV is ready for the LLM-extracted constraints
    (duration calculation, demand/inventory availability, due-date readiness,
    machine-assignment readiness).
    Input: JSON {"csv_path": "...", "duration_required": bool,
                 "due_required": bool, "machine_id_required": bool,
                 "demand_inventory_required": bool}"""
    args = _parse_input(tool_input)
    try:
        result = csv_tools.validate_constraint_readiness(
            args["csv_path"],
            duration_required=bool(args.get("duration_required", False)),
            due_required=bool(args.get("due_required", False)),
            machine_id_required=bool(args.get("machine_id_required", False)),
            demand_inventory_required=bool(args.get("demand_inventory_required", False)),
        )
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    return _dumps(result)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
LANGCHAIN_CSV_TOOLS = [
    inspect_csv_tool,
    validate_required_fields_tool,
    validate_data_quality_tool,
    validate_constraint_readiness_tool,
]
