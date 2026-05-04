"""
CSVValidationReActAgent
=======================

The ONLY agent in this MVP.

This version uses a real LangChain ReAct agent:
  - LangChain tools wrapping deterministic CSV functions
  - ReAct prompting (Thought / Action / Observation / Final Answer)
  - AgentExecutor with intermediate_steps capture

After the LangChain ReAct run, deterministic post-validation builds the
structured JSON report so output is always reliable regardless of LLM
parsing quirks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_OPTIONS
from app.tools import csv_tools
from app.tools.langchain_csv_tools import LANGCHAIN_CSV_TOOLS
from app.utils import logger


# ---------------------------------------------------------------------------
# ReAct prompt (no Hub required — defined locally)
# ---------------------------------------------------------------------------
_REACT_TEMPLATE = """You are a CSV validation ReAct agent for a production scheduling system.

You must validate the CSV using the constraints extracted by the LLM.
Use ONLY the available tools. Do not guess — always call a tool.

CSV path: {csv_path}

LLM extracted required CSV fields: {required_fields}
LLM extracted constraints: {constraints_summary}
LLM extracted objective: {objective}
Global parameters: {global_parameters}

You have access to exactly four tools:
{tools}

Use this format EXACTLY:

Thought: I need to think about what to do next
Action: tool_name
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to give a final answer
Final Answer: CSV validation status: VALID or INVALID. Ready for CP-SAT: YES or NO. [details]

Every tool call MUST have all three lines: Thought, Action, Action Input.
Do NOT repeat the same tool call with the same input.

Execute these FOUR steps in order, exactly once each:

Step 1 - Inspect the CSV (schema + preview + profile):
Action: inspect_csv_tool
Action Input: {{"csv_path": "{csv_path}"}}

Step 2 - Verify ALL LLM-required fields are present:
Action: validate_required_fields_tool
Action Input: {{"csv_path": "{csv_path}", "required_fields": {required_fields}}}

Step 3 - Validate data quality (numeric + missing + negative) across all required fields:
Action: validate_data_quality_tool
Action Input: {{"csv_path": "{csv_path}", "required_fields": {required_fields}, "numeric_fields": {numeric_fields}}}

Step 4 - Check whether the CSV is ready for the extracted constraints:
Action: validate_constraint_readiness_tool
Action Input: {{"csv_path": "{csv_path}", "duration_required": {duration_required}, "due_required": {due_required}, "machine_id_required": {machine_id_required}, "demand_inventory_required": {demand_inventory_required}}}

After all four steps, output Final Answer.

Tool names: {tool_names}

User task:
{input}

Begin!

{agent_scratchpad}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CSVAgentResult:
    csv_path: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    validation_status: str = "UNKNOWN"
    ready_for_cpsat: bool = False
    report: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CSVValidationReActAgent:
    """LangChain ReAct CSV validator driven by the LLM's extracted constraints."""

    def __init__(self, csv_path: str, llm_understanding: dict[str, Any]):
        self.csv_path = csv_path
        self.llm_understanding = llm_understanding

        self._required_fields: list[str] = list(
            llm_understanding.get("required_csv_fields") or []
        )
        constraints = llm_understanding.get("constraints", [])
        constraint_names = " ".join(
            (c.get("name", "") + " " + c.get("description", "")).lower()
            for c in constraints
        )
        self._duration_required = (
            "duration" in constraint_names
            or "production_qty" in self._required_fields
        )
        self._due_required = any(
            kw in constraint_names
            for kw in ("due", "deadline", "tardiness")
        )
        self._machine_id_required = (
            "machine_id" in self._required_fields
            or "specific machine" in constraint_names
            or "machine id" in constraint_names
        )

        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            repeat_penalty=OLLAMA_OPTIONS.get("repeat_penalty", 1.2),
            num_predict=OLLAMA_OPTIONS.get("num_predict", 2048),
        )

    def _build_prompt(self) -> PromptTemplate:
        constraints = self.llm_understanding.get("constraints", [])
        constraints_summary = "; ".join(
            c.get("name", "") + ": " + c.get("description", "")
            for c in constraints
        ) or "none specified"
        objective = (
            self.llm_understanding.get("objective", {}).get("name", "unknown")
        )
        global_params = json.dumps(
            self.llm_understanding.get("global_parameters", {}), default=str
        )

        # Numeric fields = all required minus known string identifiers
        _STRING_FIELDS = {"job_id", "product_name"}
        numeric_fields = [
            f for f in self._required_fields if f not in _STRING_FIELDS
        ]

        # Use forward slashes in path so JSON action inputs are always valid
        from pathlib import Path
        escaped_csv_path = Path(self.csv_path).as_posix()

        demand_inventory_required = any(
            f in self._required_fields for f in ("demand_qty", "inventory_qty")
        )

        return PromptTemplate.from_template(_REACT_TEMPLATE).partial(
            csv_path=escaped_csv_path,
            required_fields=json.dumps(self._required_fields),
            numeric_fields=json.dumps(numeric_fields),
            constraints_summary=constraints_summary,
            objective=objective,
            global_parameters=global_params,
            duration_required=str(bool(self._duration_required)).lower(),
            due_required=str(bool(self._due_required)).lower(),
            machine_id_required=str(bool(self._machine_id_required)).lower(),
            demand_inventory_required=str(bool(demand_inventory_required)).lower(),
        )

    def _build_trace(
        self, intermediate_steps: list[tuple[Any, Any]]
    ) -> list[dict[str, Any]]:
        """Convert LangChain intermediate_steps into our standard trace format."""
        trace = []
        parser_warning_count = 0
        for i, (action, observation) in enumerate(intermediate_steps, start=1):
            raw_log: str = getattr(action, "log", "") or ""
            tool_name = getattr(action, "tool", "unknown")

            # Filter parser error steps — count silently, don't spam terminal
            if tool_name == "_Exception":
                parser_warning_count += 1
                trace.append({
                    "step": i,
                    "type": "parser_warning",
                    "message": str(observation),
                    "raw_log": raw_log,
                })
                continue

            # Extract thought from the log (everything before Action:)
            thought = raw_log.strip()
            if "Action:" in thought:
                thought = thought[: thought.index("Action:")].strip()
            if thought.lower().startswith("thought:"):
                thought = thought[len("thought:"):].strip()

            step = {
                "step": i,
                "thought": thought or "(see raw_log)",
                "action": tool_name,
                "action_input": getattr(action, "tool_input", ""),
                "observation": observation,
                "raw_log": raw_log,
            }
            trace.append(step)

            # Print to terminal
            logger.react_thought(step["thought"])
            logger.react_action(
                f"{step['action']}({json.dumps(step['action_input'], default=str)})"
            )
            logger.react_observation(
                json.dumps(observation, indent=2, default=str)
                if not isinstance(observation, str)
                else observation
            )

        if parser_warning_count > 0:
            logger.warn(
                "CSV AGENT",
                f"{parser_warning_count} parser warning(s) suppressed (see trace JSON for details)"
            )
        return trace

    def _deterministic_report(
        self, columns_found: list[str], errors: list[str], warnings: list[str]
    ) -> dict[str, Any]:
        """Run deterministic checks to build a reliable JSON report."""
        # numeric checks
        # Identify numeric fields from all required fields
        # (all fields except known string identifiers)
        _STRING_FIELDS = {"job_id", "product_name"}
        numeric_candidates = [
            f for f in self._required_fields
            if f in columns_found and f not in _STRING_FIELDS
        ]
        numeric_checks = (
            csv_tools.check_numeric_columns(self.csv_path, numeric_candidates)
            if numeric_candidates else {}
        )
        missing_checks = (
            csv_tools.check_missing_values(self.csv_path, self._required_fields)
            if self._required_fields else {}
        )
        negative_checks = (
            csv_tools.check_negative_values(self.csv_path, numeric_candidates)
            if numeric_candidates else {}
        )
        duration_possible = (
            csv_tools.check_duration_calculation_possible(self.csv_path).get("ok")
            if self._duration_required else None
        )
        due_check = (
            csv_tools.check_due_fields_if_required(self.csv_path, True).get("ok")
            if self._due_required else None
        )
        machine_check = csv_tools.check_machine_fields_if_required(
            self.csv_path, self._machine_id_required
        )

        # Demand / inventory check
        demand_inventory_required = any(
            f in self._required_fields for f in ["demand_qty", "inventory_qty"]
        )
        if demand_inventory_required:
            demand_inventory_check_possible = (
                "demand_qty" in columns_found and "inventory_qty" in columns_found
            )
            if not demand_inventory_check_possible:
                missing_di = [
                    f for f in ["demand_qty", "inventory_qty"]
                    if f not in columns_found
                ]
                errors.append(
                    f"Demand/inventory check required but missing columns: {missing_di}"
                )
        else:
            demand_inventory_check_possible = None

        # Collect errors from deterministic checks
        req_check = csv_tools.check_required_columns(
            self.csv_path, self._required_fields
        )
        missing_required = req_check.get("missing", [])
        if missing_required and f"Missing required CSV fields: {missing_required}" not in errors:
            errors.append(f"Missing required CSV fields: {missing_required}")

        if not numeric_checks.get("ok", True):
            bad = [
                f for f, d in numeric_checks.get("details", {}).items()
                if not d.get("numeric")
            ]
            if bad:
                errors.append(f"Non-numeric required fields: {bad}")

        if not missing_checks.get("ok", True):
            for f, n in missing_checks.get("missing_per_field", {}).items():
                if n > 0:
                    warnings.append(f"Field '{f}' has {n} missing value(s)")

        if not negative_checks.get("ok", True):
            for f, d in negative_checks.get("details", {}).items():
                if d.get("negative_count", 0) > 0:
                    errors.append(
                        f"Field '{f}' has {d['negative_count']} negative value(s)"
                    )

        if self._duration_required and not duration_possible:
            errors.append("Duration calculation (production_qty * processing_time_min_per_unit) not possible")

        if self._due_required and not due_check:
            warnings.append("Due-date constraint declared but no due_day/due_time in CSV")

        if self._machine_id_required and not machine_check.get("ok"):
            errors.append(f"Machine assignment data invalid: {machine_check.get('reason', machine_check)}")

        return {
            "csv_path": self.csv_path,
            "required_fields_from_llm": self._required_fields,
            "columns_found": columns_found,
            "missing_required_fields": missing_required,
            "numeric_field_checks": numeric_checks.get("details", {}),
            "missing_value_checks": missing_checks.get("missing_per_field", {}),
            "negative_value_checks": negative_checks.get("details", {}),
            "constraint_specific_checks": {
                "duration_calculation_possible": duration_possible,
                "demand_inventory_check_possible": demand_inventory_check_possible,
                "due_date_check_possible": due_check,
                "machine_assignment_data_required": self._machine_id_required,
                "machine_check": machine_check,
            },
            "warnings": warnings,
            "errors": errors,
        }

    def _deterministic_verification_trace(
        self,
        existing_trace: list[dict[str, Any]],
        columns_found: list[str],
    ) -> list[dict[str, Any]]:
        """
        Append demo-safe verification steps to the trace.

        We always run these — even if the LangChain ReAct agent already
        called the same tools — because qwen2.5:3b sometimes shrinks the
        field list and the manager demo must visibly show every required
        field being validated.

        These trace entries are still part of the SAME CSVValidationReActAgent.
        They are NOT a second agent.
        """
        from pathlib import Path
        _STRING_FIELDS = {"job_id", "product_name"}
        numeric_fields = [
            f for f in self._required_fields
            if f in columns_found and f not in _STRING_FIELDS
        ]
        display_csv_path = Path(self.csv_path).as_posix()

        # Skip duplicate-emission detection: we always emit our deterministic
        # block so the demo is consistent. The trace JSON keeps both passes.
        next_step = (existing_trace[-1]["step"] + 1) if existing_trace else 1

        verification_steps: list[dict[str, Any]] = []

        def _emit(thought: str, action: str, tool_input: dict[str, Any], obs: Any) -> None:
            nonlocal next_step
            entry = {
                "step": next_step,
                "type": "deterministic_verification",
                "thought": thought,
                "action": action,
                "action_input": tool_input,
                "observation": obs,
            }
            verification_steps.append(entry)
            next_step += 1
            logger.react_thought(thought)
            logger.react_action(f"{action}({json.dumps(tool_input, default=str)})")
            logger.react_observation(json.dumps(obs, indent=2, default=str))

        demand_inventory_required = any(
            f in self._required_fields for f in ("demand_qty", "inventory_qty")
        )

        # Header thought so the manager sees this is intentional.
        logger.react_thought(
            "I will now deterministically verify all LLM-required fields "
            "for demo-safe validation."
        )

        # 1. Inspect the CSV (schema + preview + profile)
        _emit(
            "I need to inspect the CSV.",
            "inspect_csv_tool",
            {"csv_path": display_csv_path},
            csv_tools.inspect_csv(self.csv_path),
        )

        # 2. Verify ALL LLM-required fields exist
        _emit(
            "I need to verify all required fields from the LLM.",
            "validate_required_fields_tool",
            {"csv_path": display_csv_path, "required_fields": self._required_fields},
            csv_tools.validate_required_fields(self.csv_path, self._required_fields),
        )

        # 3. Validate data quality (numeric + missing + negative)
        _emit(
            "I need to validate numeric, missing, and negative values.",
            "validate_data_quality_tool",
            {
                "csv_path": display_csv_path,
                "required_fields": self._required_fields,
                "numeric_fields": numeric_fields,
            },
            csv_tools.validate_data_quality(
                self.csv_path, self._required_fields, numeric_fields
            ),
        )

        # 4. Check whether the CSV is ready for the extracted constraints
        _emit(
            "I need to check whether the CSV is ready for the extracted constraints.",
            "validate_constraint_readiness_tool",
            {
                "csv_path": display_csv_path,
                "duration_required": bool(self._duration_required),
                "due_required": bool(self._due_required),
                "machine_id_required": bool(self._machine_id_required),
                "demand_inventory_required": bool(demand_inventory_required),
            },
            csv_tools.validate_constraint_readiness(
                self.csv_path,
                duration_required=bool(self._duration_required),
                due_required=bool(self._due_required),
                machine_id_required=bool(self._machine_id_required),
                demand_inventory_required=bool(demand_inventory_required),
            ),
        )

        return verification_steps

    def _build_deterministic_final_answer(self, report: dict[str, Any]) -> str:
        """Build the final answer from deterministic report, not LLM output."""
        status = report.get("validation_status", "UNKNOWN")
        ready = "YES" if report.get("ready_for_cpsat") else "NO"

        errors = report.get("errors", [])
        warnings = report.get("warnings", [])
        missing = report.get("missing_required_fields", [])

        if status == "VALID":
            return (
                "CSV validation status: VALID.\n"
                "The CSV contains all data required by the user constraints extracted by the LLM.\n"
                "Ready for CP-SAT: YES."
            )

        return (
            "CSV validation status: INVALID.\n"
            f"Ready for CP-SAT: {ready}.\n"
            f"Missing required fields: {missing}\n"
            f"Errors: {errors}\n"
            f"Warnings: {warnings}"
        )

    def run(self) -> CSVAgentResult:
        csv_tools.reset_cache()
        errors: list[str] = []
        warnings: list[str] = []

        prompt = self._build_prompt()
        agent = create_react_agent(self._llm, LANGCHAIN_CSV_TOOLS, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=LANGCHAIN_CSV_TOOLS,
            verbose=False,          # we do our own pretty printing
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

        # --- Run the LangChain ReAct agent ---
        agent_execution_errors = []
        try:
            result = executor.invoke({"input": f"Validate CSV at: {self.csv_path}"})
            intermediate_steps = result.get("intermediate_steps", [])
            final_answer_text: str = result.get("output", "")
        except Exception as exc:
            agent_execution_errors.append(str(exc))
            warnings.append(f"ReAct agent execution warning: {exc}")
            intermediate_steps = []
            final_answer_text = f"Agent warning: {exc}"

        # --- Build trace for terminal + JSON ---
        trace = self._build_trace(intermediate_steps)
        # Store raw LLM final answer — do NOT print it as official result
        agent_raw_final_answer = final_answer_text

        # --- Get columns found for report ---
        schema = csv_tools.inspect_csv_schema(self.csv_path)
        columns_found: list[str] = schema.get("columns", []) if schema.get("ok") else []

        if not schema.get("ok"):
            errors.append(schema.get("error", "CSV file could not be inspected"))
            report = {
                "csv_path": self.csv_path,
                "validation_status": "INVALID",
                "ready_for_cpsat": False,
                "required_fields_from_llm": self._required_fields,
                "columns_found": [],
                "missing_required_fields": self._required_fields,
                "numeric_field_checks": {},
                "missing_value_checks": {},
                "negative_value_checks": {},
                "constraint_specific_checks": {},
                "warnings": warnings,
                "errors": errors,
                "agent_execution_errors": agent_execution_errors,
                "agent_raw_final_answer": agent_raw_final_answer,
                "agent_final_answer": "CSV validation status: INVALID. CSV file could not be inspected.",
            }
            deterministic_final = self._build_deterministic_final_answer(report)
            report["agent_final_answer"] = deterministic_final
            logger.react_final(deterministic_final)
            return CSVAgentResult(
                csv_path=self.csv_path,
                trace=trace,
                final_answer=deterministic_final,
                validation_status="INVALID",
                ready_for_cpsat=False,
                report=report,
            )

        # --- Deterministic verification trace (demo-safe) ---
        # qwen2.5:3b sometimes reduces the field list during ReAct. We
        # always run a deterministic verification pass so the terminal
        # demo visibly validates every required field.
        verification_steps = self._deterministic_verification_trace(
            trace, columns_found
        )
        trace.extend(verification_steps)

        # --- Deterministic report ---
        report = self._deterministic_report(columns_found, errors, warnings)
        report["validation_status"] = "INVALID" if report["errors"] else "VALID"
        report["ready_for_cpsat"] = report["validation_status"] == "VALID"
        report["agent_raw_final_answer"] = agent_raw_final_answer
        report["agent_execution_errors"] = agent_execution_errors

        deterministic_final_answer = self._build_deterministic_final_answer(report)
        report["agent_final_answer"] = deterministic_final_answer

        logger.react_final(deterministic_final_answer)

        return CSVAgentResult(
            csv_path=self.csv_path,
            trace=trace,
            final_answer=deterministic_final_answer,
            validation_status=report["validation_status"],
            ready_for_cpsat=report["ready_for_cpsat"],
            report=report,
        )



