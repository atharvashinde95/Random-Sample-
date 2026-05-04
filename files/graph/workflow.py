"""
LangGraph workflow definition.

Flow: START → llm_problem_understanding_node → csv_validation_react_agent_node → final_summary_node → END
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.csv_validation_react_agent import CSVValidationReActAgent
from app.graph.state import SchedulingGraphState
from app.llm import user_input_understanding
from app.utils import logger
from app.utils.json_writer import write_json


# ---------------------------------------------------------------------------
# Node 1: LLM problem understanding (NOT an agent)
# ---------------------------------------------------------------------------

def llm_problem_understanding_node(state: SchedulingGraphState) -> SchedulingGraphState:
    logger.graph("Node: llm_problem_understanding_node")
    logger.section("STEP 1: LLM USER INPUT UNDERSTANDING")

    understanding = user_input_understanding.understand(state["user_input"])

    output_dir = Path(state["output_dir"])
    out_path = output_dir / "llm_problem_understanding.json"
    write_json(out_path, understanding)
    logger.saved(str(out_path))
    logger.graph("Node completed: llm_problem_understanding_node")

    return {**state, "llm_problem_understanding": understanding}


# ---------------------------------------------------------------------------
# Node 2: CSV validation ReAct agent
# ---------------------------------------------------------------------------

def csv_validation_react_agent_node(state: SchedulingGraphState) -> SchedulingGraphState:
    logger.graph("Node: csv_validation_react_agent_node")
    logger.section("STEP 2: CSV VALIDATION AGENT WITH LANGCHAIN ReAct")

    agent = CSVValidationReActAgent(
        csv_path=state["csv_path"],
        llm_understanding=state["llm_problem_understanding"],
    )
    result = agent.run()

    output_dir = Path(state["output_dir"])

    # Serialise trace (dataclasses → dicts)
    trace_jsonable = result.trace  # already list[dict]

    trace_path = output_dir / "csv_react_trace.json"
    report_path = output_dir / "csv_validation_report.json"
    write_json(trace_path, trace_jsonable)
    write_json(report_path, result.report)
    logger.saved(str(trace_path))
    logger.saved(str(report_path))

    logger.graph("Node completed: csv_validation_react_agent_node")

    return {
        **state,
        "csv_react_trace": trace_jsonable,
        "csv_validation_report": result.report,
        "csv_validation_status": result.validation_status,
        "csv_ready_for_cpsat": result.ready_for_cpsat,
    }


# ---------------------------------------------------------------------------
# Node 3: Final summary
# ---------------------------------------------------------------------------

def final_summary_node(state: SchedulingGraphState) -> SchedulingGraphState:
    logger.graph("Node: final_summary_node")

    llm = state.get("llm_problem_understanding", {})
    cv = llm.get("constraint_validation", {})

    ready_for_cpsat = (
        bool(llm.get("is_cpsat_compatible"))
        and bool(cv.get("are_constraints_valid"))
        and not bool(cv.get("has_contradictions"))
        and bool(state.get("csv_ready_for_cpsat"))
    )

    final_summary: dict[str, Any] = {
        "problem_type": llm.get("problem_type"),
        "is_cpsat_compatible": llm.get("is_cpsat_compatible"),
        "objective": llm.get("objective", {}).get("name"),
        "constraints_valid": cv.get("are_constraints_valid"),
        "contradictions_found": cv.get("has_contradictions"),
        "csv_validation_status": state.get("csv_validation_status"),
        "ready_for_cpsat": ready_for_cpsat,
        "next_step": "Build CP-SAT model later. Not implemented in this MVP.",
    }

    logger.final_summary(final_summary)

    output_dir = Path(state["output_dir"])

    summary_path = output_dir / "final_summary.json"
    write_json(summary_path, final_summary)
    logger.saved(str(summary_path))

    # Save full LangGraph state trace (excluding large raw traces for readability)
    state_trace: dict[str, Any] = {
        "run_id": state.get("run_id"),
        "csv_path": state.get("csv_path"),
        "user_input": state.get("user_input"),
        "llm_problem_understanding": state.get("llm_problem_understanding"),
        "csv_validation_status": state.get("csv_validation_status"),
        "csv_ready_for_cpsat": state.get("csv_ready_for_cpsat"),
        "final_summary": final_summary,
        "errors": state.get("errors", []),
        "warnings": state.get("warnings", []),
    }
    state_trace_path = output_dir / "langgraph_state_trace.json"
    write_json(state_trace_path, state_trace)
    logger.saved(str(state_trace_path))

    logger.graph("END")

    return {**state, "final_summary": final_summary}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_workflow():
    workflow = StateGraph(SchedulingGraphState)

    workflow.add_node("llm_problem_understanding_node", llm_problem_understanding_node)
    workflow.add_node("csv_validation_react_agent_node", csv_validation_react_agent_node)
    workflow.add_node("final_summary_node", final_summary_node)

    workflow.add_edge(START, "llm_problem_understanding_node")
    workflow.add_edge("llm_problem_understanding_node", "csv_validation_react_agent_node")
    workflow.add_edge("csv_validation_react_agent_node", "final_summary_node")
    workflow.add_edge("final_summary_node", END)

    return workflow.compile()
