"""
Orchestrator entry point.
Builds the LangGraph workflow and runs it.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from app.config import DEFAULT_CSV, DEFAULT_USER_INPUT, OUTPUTS_DIR
from app.graph.workflow import build_workflow
from app.utils import logger
from app.utils.json_writer import create_run_dir


def main() -> None:
    logger.banner()

    # --- Run ID + paths ---
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = create_run_dir(run_id)
    csv_path = DEFAULT_CSV

    logger.info("RUN", f"run_id = {run_id}")
    logger.info("RUN", f"output dir = {output_dir}")
    logger.info("RUN", f"csv = {csv_path}")
    logger.info("CSV", f"CSV exists: {csv_path.exists()}")

    if not csv_path.exists():
        logger.error("CSV", f"CSV file not found: {csv_path}")
        logger.error("CSV", "Place the dataset at this path or update DEFAULT_CSV in config.py")
        return

    # --- User input ---
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        try:
            user_input = input(
                "\nEnter scheduling requirement (ENTER for default):\n> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            user_input = ""
    if not user_input:
        user_input = DEFAULT_USER_INPUT
        logger.info("INPUT", "Using default requirement")

    # --- Build and run graph ---
    logger.graph("START")

    initial_state = {
        "run_id": run_id,
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "user_input": user_input,
        "errors": [],
        "warnings": [],
    }

    graph = build_workflow()
    graph.invoke(initial_state)
