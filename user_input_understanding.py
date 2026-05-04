"""
Step 1 of the MVP: LLM-only user input understanding.

This is intentionally NOT an agent. It is a single LLM call (with one
JSON-repair retry) that returns a structured representation of the user's
scheduling requirement.

Naming note from the manager:
    Do not call this BusinessRuleExtractorAgent.
    The first step is LLM-only.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.llm import ollama_client
from app.llm.prompts import build_json_repair_prompt, build_understanding_prompt
from app.utils import logger


class LLMUnderstandingError(RuntimeError):
    """Raised when the LLM cannot produce parseable JSON, even after repair."""


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    """Drop ```json ... ``` fences if a chatty model adds them anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    """Find the first {...} block that parses as JSON. Best-effort."""
    text = _strip_code_fences(text)
    # Quick path
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Brace-walking fallback
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _parse_json(text: str) -> dict[str, Any] | None:
    candidate = _extract_first_json_object(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Light post-processing - the small model sometimes drops fields
# ---------------------------------------------------------------------------
_DEFAULT_SHAPE: dict[str, Any] = {
    "problem_type": "production_scheduling",
    "is_cpsat_compatible": False,
    "cpsat_compatibility_reason": "",
    "objective": {
        "name": "",
        "description": "",
        "is_supported_for_cpsat": False,
    },
    "variables": [],
    "constraints": [],
    "business_rules": [],
    "global_parameters": {},
    "required_csv_fields": [],
    "optional_csv_fields": [],
    "constraint_validation": {
        "are_constraints_valid": True,
        "has_contradictions": False,
        "contradictions": [],
        "warnings": [],
    },
    "missing_inputs": [],
    "assumptions": [],
}


def _ensure_shape(parsed: dict[str, Any]) -> dict[str, Any]:
    """Guarantee every top-level key exists so downstream code is safe."""
    out = json.loads(json.dumps(_DEFAULT_SHAPE))  # deep copy
    out.update(parsed or {})
    # Ensure nested defaults too
    if not isinstance(out.get("objective"), dict):
        out["objective"] = dict(_DEFAULT_SHAPE["objective"])
    cv = out.get("constraint_validation")
    if not isinstance(cv, dict):
        out["constraint_validation"] = dict(_DEFAULT_SHAPE["constraint_validation"])
    else:
        for k, v in _DEFAULT_SHAPE["constraint_validation"].items():
            cv.setdefault(k, v)
    # List-typed fields
    for k in (
        "variables",
        "constraints",
        "business_rules",
        "required_csv_fields",
        "optional_csv_fields",
        "missing_inputs",
        "assumptions",
    ):
        if not isinstance(out.get(k), list):
            out[k] = []
    if not isinstance(out.get("global_parameters"), dict):
        out["global_parameters"] = {}
    return out


def _enforce_implied_fields(result: dict[str, Any], user_input: str) -> dict[str, Any]:
    """
    Deterministic safety net: scan user_input for keywords and force
    the implied CSV fields into required_csv_fields.

    The small LLM often drops fields; this guarantees correctness.
    """
    text = user_input.lower()
    required: list[str] = list(result.get("required_csv_fields") or [])
    added: list[str] = []

    # job_id is always required for scheduling
    if "job_id" not in required:
        required.append("job_id")
        added.append("job_id")

    # Duration formula keywords
    if "production_qty" in text or "quantity" in text:
        if "production_qty" not in required:
            required.append("production_qty")
            added.append("production_qty")

    if "processing_time" in text:
        if "processing_time_min_per_unit" not in required:
            required.append("processing_time_min_per_unit")
            added.append("processing_time_min_per_unit")

    # Demand / inventory keywords
    if "demand" in text:
        if "demand_qty" not in required:
            required.append("demand_qty")
            added.append("demand_qty")

    if "inventory" in text:
        if "inventory_qty" not in required:
            required.append("inventory_qty")
            added.append("inventory_qty")

    if added:
        result["required_csv_fields"] = required
        logger.info("LLM", f"Post-fix added implied fields: {added}")

    return result


# ---------------------------------------------------------------------------
# Deterministic constraint post-fix
# ---------------------------------------------------------------------------
# Each rule: (constraint_name, keyword_groups, type, description, required_csv_fields)
# A keyword_group is a list of synonyms — at least one must appear.
# Multiple keyword_groups means ALL groups must match (logical AND).
_IMPLIED_CONSTRAINT_RULES: list[dict[str, Any]] = [
    {
        "name": "schedule_all_jobs",
        "keyword_groups": [
            ["plan all", "schedule all", "all pending", "all jobs",
             "all orders", "pending jobs", "pending orders"],
        ],
        "type": "hard",
        "description": "All pending jobs from the CSV should be considered for scheduling.",
        "required_csv_fields": ["job_id"],
    },
    {
        "name": "machine_capacity",
        "keyword_groups": [
            ["machine", "production line", "production lines", "lines",
             "oven", "ovens", "resource", "resources"],
        ],
        "type": "hard",
        "description": "The factory has a fixed number of identical production machines/lines.",
        "required_csv_fields": [],
    },
    {
        "name": "working_time_limit",
        "keyword_groups": [
            ["shift", "hours per day", "hour shift", "working day",
             "working minutes", "minutes per day", "8 hours", "8-hour"],
        ],
        "type": "hard",
        "description": "The factory works a fixed-length shift per day (working minutes per day).",
        "required_csv_fields": [],
    },
    {
        "name": "duration_calculation",
        "keyword_groups": [
            ["production quantity", "production_qty", "quantity"],
            ["processing time", "processing_time", "process time"],
        ],
        "type": "hard",
        "description": "Each job duration requires production quantity and processing time information.",
        "required_csv_fields": ["production_qty", "processing_time_min_per_unit"],
    },
    {
        "name": "demand_inventory_validation",
        "keyword_groups": [
            ["demand", "customer demand"],
            ["inventory", "stock", "stock availability", "availability"],
        ],
        "type": "hard",
        "description": "Customer demand and available stock must be checked before scheduling.",
        "required_csv_fields": ["demand_qty", "inventory_qty"],
    },
]


_OBJECTIVE_MAKESPAN_PHRASES = (
    "finish as early as possible",
    "as early as possible",
    "as low as possible",
    "as soon as possible",
    "minimize completion time",
    "minimize total completion",
    "minimize overall completion",
    "reduce completion time",
    "reduce total completion",
    "reduce overall completion",
    "complete the plan quickly",
    "finish quickly",
    "minimize makespan",
)


def _enforce_implied_constraints(
    result: dict[str, Any], user_input: str
) -> dict[str, Any]:
    """
    Deterministic safety net: scan user_input for keywords and add any
    obviously-implied constraints that the small LLM dropped.

    This does NOT replace the LLM output. It only adds missing constraints
    whose names are not already present.
    """
    text = user_input.lower()

    constraints: list[dict[str, Any]] = list(result.get("constraints") or [])
    existing_names = {
        (c.get("name") or "").strip().lower() for c in constraints if isinstance(c, dict)
    }

    added_names: list[str] = []
    for rule in _IMPLIED_CONSTRAINT_RULES:
        if rule["name"].lower() in existing_names:
            continue
        # All keyword groups must match (AND); within a group, ANY synonym (OR).
        all_groups_match = all(
            any(kw in text for kw in group) for group in rule["keyword_groups"]
        )
        if not all_groups_match:
            continue
        constraints.append({
            "name": rule["name"],
            "type": rule["type"],
            "description": rule["description"],
            "required_csv_fields": list(rule["required_csv_fields"]),
        })
        added_names.append(rule["name"])
        existing_names.add(rule["name"].lower())

    if added_names:
        result["constraints"] = constraints
        logger.info(
            "LLM",
            f"Post-fix added implied constraints: {added_names}",
        )

        # Union: pull every per-constraint required_csv_field into the
        # top-level required_csv_fields so a natural phrase like "stock"
        # still produces the right CSV check downstream.
        required = list(result.get("required_csv_fields") or [])
        added_fields: list[str] = []
        for c in constraints:
            for f in c.get("required_csv_fields", []) or []:
                if f and f not in required:
                    required.append(f)
                    added_fields.append(f)
        if added_fields:
            result["required_csv_fields"] = required
            logger.info(
                "LLM",
                f"Post-fix added implied fields from constraints: {added_fields}",
            )

    # Objective post-fix: map natural-language phrases to minimize_makespan.
    objective = result.get("objective") or {}
    if not isinstance(objective, dict):
        objective = {}
    obj_name = (objective.get("name") or "").strip()
    if not obj_name or obj_name.lower() in {"unknown", "none", ""}:
        if any(phrase in text for phrase in _OBJECTIVE_MAKESPAN_PHRASES):
            objective["name"] = "minimize_makespan"
            objective.setdefault(
                "description",
                "Minimize the overall completion time (makespan) across all jobs.",
            )
            objective["is_supported_for_cpsat"] = True
            result["objective"] = objective
            logger.info("LLM", "Post-fix mapped objective to minimize_makespan")

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def understand(user_input: str) -> dict[str, Any]:
    """
    Run the LLM-only understanding step.

    Order of operations:
      1. Make sure Ollama is reachable and the model is installed.
      2. Call the model with the extraction prompt.
      3. Parse JSON. If broken, run ONE repair attempt.
      4. Normalise shape and report progress to the terminal.
      5. Enforce implied fields from user keywords (deterministic).
    """
    logger.info("INPUT", "User scheduling requirement received")
    logger.info("OLLAMA", "Calling model qwen2.5:3b")

    # Will raise OllamaError / OllamaModelMissingError if anything is wrong.
    ollama_client.ensure_ollama_ready()

    prompt = build_understanding_prompt(user_input)
    raw = ollama_client.generate(prompt, format_json=True)

    parsed = _parse_json(raw)

    if parsed is None:
        logger.warn("LLM", "First response was not valid JSON, trying repair")
        repaired_raw = ollama_client.generate(
            build_json_repair_prompt(raw), format_json=True
        )
        parsed = _parse_json(repaired_raw)

    if parsed is None:
        raise LLMUnderstandingError(
            "LLM did not return parseable JSON, even after a repair attempt."
        )

    result = _ensure_shape(parsed)

    # Deterministic post-fix: guarantee fields implied by user keywords
    result = _enforce_implied_fields(result, user_input)

    # Deterministic post-fix: guarantee constraints implied by user keywords
    result = _enforce_implied_constraints(result, user_input)

    # Terminal breadcrumbs - mirrors the spec exactly
    logger.info("LLM", f"Problem type extracted: {result.get('problem_type')}")
    logger.info(
        "LLM",
        f"Objective extracted: {result.get('objective', {}).get('name', 'unknown')}",
    )
    variables = result.get("variables", []) or []
    logger.info(
        "LLM",
        f"Variables extracted: {len(variables)} variable(s)",
    )
    for v in variables:
        if not isinstance(v, dict):
            continue
        name = v.get("name", "?")
        vtype = v.get("type", "?")
        desc = v.get("description", "")
        line = f"  - {name} (type: {vtype})"
        if desc:
            line += f" - {desc}"
        logger.info("LLM", line)

    constraints = result.get("constraints", []) or []
    logger.info(
        "LLM",
        f"Constraints extracted: {len(constraints)} constraint(s)",
    )
    for c in constraints:
        if not isinstance(c, dict):
            continue
        name = c.get("name", "?")
        ctype = c.get("type", "hard")
        desc = c.get("description", "")
        fields = c.get("required_csv_fields") or []
        line = f"  - {name} ({ctype})"
        if desc:
            line += f" - {desc}"
        if fields:
            line += f" [needs: {', '.join(fields)}]"
        logger.info("LLM", line)
    logger.info(
        "LLM",
        f"Required CSV fields: {result.get('required_csv_fields', [])}",
    )
    logger.info(
        "LLM",
        f"CP-SAT compatibility checked: "
        f"{'YES' if result.get('is_cpsat_compatible') else 'NO'}",
    )
    cv = result.get("constraint_validation", {})
    logger.info(
        "LLM",
        f"Constraint contradiction check completed: "
        f"contradictions={'YES' if cv.get('has_contradictions') else 'NO'}, "
        f"warnings={len(cv.get('warnings', []))}",
    )

    return result

