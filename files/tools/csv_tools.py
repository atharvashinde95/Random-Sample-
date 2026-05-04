"""
Deterministic CSV inspection tools used by the CSVValidationReActAgent.

These are plain Python functions. They are NOT agents and they do NOT call
any LLM. The agent decides which tool to call; the tool returns a fact.

Every tool returns a dict so the agent can store structured observations
in the ReAct trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


# ---------------------------------------------------------------------------
# Internal cache - load the CSV once per agent run
# ---------------------------------------------------------------------------
_loaded: dict[str, pd.DataFrame] = {}


def _normalize_csv_path(csv_path: Any) -> str:
    """
    Accept raw path, JSON string, quoted JSON string, or dict.
    Return a real filesystem path string.
    """
    if isinstance(csv_path, dict):
        return str(csv_path.get("csv_path", ""))

    text = str(csv_path).strip()

    # Remove outer quotes if present
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        try:
            text = json.loads(text)
            if isinstance(text, dict):
                return str(text.get("csv_path", ""))
            text = str(text)
        except Exception:
            text = text[1:-1]

    # Parse JSON object string
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "csv_path" in data:
                return str(data["csv_path"])
        except Exception:
            pass

    return text


def _load(csv_path: Any) -> pd.DataFrame:
    csv_path = _normalize_csv_path(csv_path)
    if csv_path not in _loaded:
        _loaded[csv_path] = pd.read_csv(csv_path)
    return _loaded[csv_path]


def reset_cache() -> None:
    _loaded.clear()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def inspect_csv_schema(csv_path: Any) -> dict[str, Any]:
    """List columns and dtypes."""
    csv_path = _normalize_csv_path(csv_path)
    p = Path(csv_path)
    if not p.exists():
        return {"ok": False, "error": f"CSV not found at {csv_path}"}
    df = _load(csv_path)
    return {
        "ok": True,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "row_count": int(len(df)),
    }


def preview_csv_rows(csv_path: str, n: int = 3) -> dict[str, Any]:
    """First few rows, mostly for sanity-checking in the trace."""
    df = _load(csv_path)
    return {
        "ok": True,
        "rows": df.head(n).to_dict(orient="records"),
    }


def check_required_columns(
    csv_path: str, required_fields: list[str]
) -> dict[str, Any]:
    """Are all LLM-required CSV fields actually present in the file?"""
    df = _load(csv_path)
    cols = set(df.columns)
    found = [f for f in required_fields if f in cols]
    missing = [f for f in required_fields if f not in cols]
    return {
        "ok": len(missing) == 0,
        "found": found,
        "missing": missing,
    }


def check_numeric_columns(
    csv_path: str, fields: list[str]
) -> dict[str, Any]:
    """For each requested field, is it numeric? (or coercible to numeric)"""
    df = _load(csv_path)
    result: dict[str, dict[str, Any]] = {}
    all_ok = True
    for f in fields:
        if f not in df.columns:
            result[f] = {"present": False, "numeric": False}
            all_ok = False
            continue
        coerced = pd.to_numeric(df[f], errors="coerce")
        # numeric if at most a small fraction failed coercion AND original
        # dtype was numeric, OR everything coerced cleanly.
        original_numeric = pd.api.types.is_numeric_dtype(df[f])
        coercion_ok = coerced.notna().sum() == df[f].notna().sum()
        is_numeric = bool(original_numeric or coercion_ok)
        result[f] = {
            "present": True,
            "numeric": is_numeric,
            "dtype": str(df[f].dtype),
        }
        if not is_numeric:
            all_ok = False
    return {"ok": all_ok, "details": result}


def check_missing_values(
    csv_path: str, fields: list[str]
) -> dict[str, Any]:
    """How many NaN values per requested field."""
    df = _load(csv_path)
    result: dict[str, int] = {}
    all_ok = True
    for f in fields:
        if f not in df.columns:
            result[f] = -1  # column missing
            all_ok = False
            continue
        n_missing = int(df[f].isna().sum())
        result[f] = n_missing
        if n_missing > 0:
            all_ok = False
    return {"ok": all_ok, "missing_per_field": result}


def check_negative_values(
    csv_path: str, fields: list[str]
) -> dict[str, Any]:
    """Negative values are invalid for quantities, durations, inventory, etc."""
    df = _load(csv_path)
    result: dict[str, Any] = {}
    all_ok = True
    for f in fields:
        if f not in df.columns:
            result[f] = {"present": False}
            all_ok = False
            continue
        coerced = pd.to_numeric(df[f], errors="coerce")
        n_neg = int((coerced < 0).sum())
        result[f] = {"present": True, "negative_count": n_neg}
        if n_neg > 0:
            all_ok = False
    return {"ok": all_ok, "details": result}


def check_duration_calculation_possible(csv_path: str) -> dict[str, Any]:
    """Is `duration = production_qty * processing_time_min_per_unit` doable?"""
    df = _load(csv_path)
    needed = ["production_qty", "processing_time_min_per_unit"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return {"ok": False, "reason": f"missing fields: {missing}"}
    qty = pd.to_numeric(df["production_qty"], errors="coerce")
    pt = pd.to_numeric(df["processing_time_min_per_unit"], errors="coerce")
    if qty.isna().any() or pt.isna().any():
        return {
            "ok": False,
            "reason": "non-numeric or missing values in inputs",
        }
    durations = (qty * pt).astype("int64", errors="ignore")
    return {
        "ok": True,
        "sample_durations_minutes": durations.head(5).tolist(),
        "min_duration": int(durations.min()),
        "max_duration": int(durations.max()),
        "total_workload_minutes": int(durations.sum()),
    }


def check_due_fields_if_required(
    csv_path: str, required: bool
) -> dict[str, Any]:
    """Only meaningful if the LLM extracted a 'respect due dates' constraint."""
    if not required:
        return {"ok": True, "applicable": False}
    df = _load(csv_path)
    has_due_day = "due_day" in df.columns
    has_due_time = "due_time" in df.columns
    if not (has_due_day or has_due_time):
        return {
            "ok": False,
            "applicable": True,
            "reason": "Neither due_day nor due_time present in CSV",
        }
    return {
        "ok": True,
        "applicable": True,
        "has_due_day": has_due_day,
        "has_due_time": has_due_time,
    }


def check_machine_fields_if_required(
    csv_path: str, required: bool
) -> dict[str, Any]:
    """Only meaningful when the LLM says specific machine assignments matter."""
    if not required:
        return {
            "ok": True,
            "applicable": False,
            "note": "machine_id not required - machine count is a global parameter",
        }
    df = _load(csv_path)
    if "machine_id" not in df.columns:
        return {
            "ok": False,
            "applicable": True,
            "reason": "machine_id column missing",
        }
    n_empty = int(df["machine_id"].isna().sum())
    return {
        "ok": n_empty == 0,
        "applicable": True,
        "empty_count": n_empty,
    }


def summarize_csv_profile(csv_path: str) -> dict[str, Any]:
    """High-level shape summary, useful as the agent's last observation."""
    df = _load(csv_path)
    return {
        "ok": True,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_list": list(df.columns),
    }


# ---------------------------------------------------------------------------
# Combined, agent-facing helpers (4 simple operations)
# ---------------------------------------------------------------------------
def inspect_csv(csv_path: Any) -> dict[str, Any]:
    """Combined: schema + preview rows + profile in one observation."""
    schema = inspect_csv_schema(csv_path)
    if not schema.get("ok"):
        return schema
    preview = preview_csv_rows(csv_path, n=3)
    profile = summarize_csv_profile(csv_path)
    return {
        "ok": True,
        "columns": schema.get("columns", []),
        "dtypes": schema.get("dtypes", {}),
        "row_count": schema.get("row_count", 0),
        "preview_rows": preview.get("rows", []),
        "profile": {
            "rows": profile.get("rows", 0),
            "columns": profile.get("columns", 0),
            "column_list": profile.get("column_list", []),
        },
    }


def validate_required_fields(
    csv_path: Any, required_fields: list[str]
) -> dict[str, Any]:
    """Pass-through that names the agent operation clearly."""
    return check_required_columns(csv_path, required_fields)


def validate_data_quality(
    csv_path: Any,
    required_fields: list[str],
    numeric_fields: list[str],
) -> dict[str, Any]:
    """Combined: numeric + missing + negative checks across all fields."""
    numeric = check_numeric_columns(csv_path, numeric_fields) if numeric_fields else {
        "ok": True, "details": {},
    }
    missing = check_missing_values(csv_path, required_fields) if required_fields else {
        "ok": True, "missing_per_field": {},
    }
    negative = check_negative_values(csv_path, numeric_fields) if numeric_fields else {
        "ok": True, "details": {},
    }

    errors: list[str] = []
    warnings: list[str] = []

    if not numeric.get("ok", True):
        bad = [f for f, d in numeric.get("details", {}).items() if not d.get("numeric")]
        if bad:
            errors.append(f"Non-numeric required fields: {bad}")

    for f, n in missing.get("missing_per_field", {}).items():
        if isinstance(n, int) and n > 0:
            warnings.append(f"Field '{f}' has {n} missing value(s)")

    for f, d in negative.get("details", {}).items():
        if d.get("negative_count", 0) > 0:
            errors.append(
                f"Field '{f}' has {d['negative_count']} negative value(s)"
            )

    ok = not errors and numeric.get("ok", True) and missing.get("ok", True) and negative.get("ok", True)

    return {
        "ok": bool(ok),
        "numeric_field_checks": numeric.get("details", {}),
        "missing_value_checks": missing.get("missing_per_field", {}),
        "negative_value_checks": negative.get("details", {}),
        "errors": errors,
        "warnings": warnings,
    }


def validate_constraint_readiness(
    csv_path: Any,
    duration_required: bool = False,
    due_required: bool = False,
    machine_id_required: bool = False,
    demand_inventory_required: bool = False,
) -> dict[str, Any]:
    """Combined readiness check for the LLM-extracted constraints."""
    schema = inspect_csv_schema(csv_path)
    cols = schema.get("columns", []) if schema.get("ok") else []

    duration_possible = (
        bool(check_duration_calculation_possible(csv_path).get("ok"))
        if duration_required else None
    )
    due_check = (
        bool(check_due_fields_if_required(csv_path, True).get("ok"))
        if due_required else None
    )
    machine_check = check_machine_fields_if_required(csv_path, machine_id_required)

    if demand_inventory_required:
        demand_inventory_possible: Any = (
            "demand_qty" in cols and "inventory_qty" in cols
        )
    else:
        demand_inventory_possible = None

    return {
        "ok": True,
        "duration_calculation_possible": duration_possible,
        "demand_inventory_check_possible": demand_inventory_possible,
        "due_date_check_possible": due_check,
        "machine_assignment_data_required": bool(machine_id_required),
        "machine_check": machine_check,
    }


# ---------------------------------------------------------------------------
# Registry consumed by the CSV agent
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "inspect_csv": inspect_csv,
    "validate_required_fields": validate_required_fields,
    "validate_data_quality": validate_data_quality,
    "validate_constraint_readiness": validate_constraint_readiness,
}
