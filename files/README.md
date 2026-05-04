# Production Scheduling Validation MVP

A terminal-first MVP that proves an LLM can understand a natural-language
production-scheduling requirement and that a CSV validation agent can
validate the dataset against the extracted constraints.

> No UI. No web server. No CP-SAT solver. Just terminal + JSON.

---

## Architecture

```
User Input
  ↓
LLM Problem Understanding
  ↓
CSV Validation Agent
  ↓
Final Validation Summary
```

Internally:

- **LangGraph** orchestrates the workflow:
  `START → llm_problem_understanding_node → csv_validation_react_agent_node → final_summary_node → END`
- **LangChain** powers the CSV ReAct agent and its `@tool` wrappers.

**Important:**
Step 1 is LLM-only — it is not an agent.
The only actual agent is `CSVValidationReActAgent`.

---

## CSV agent: 4 clean tools

The agent uses just four tools:

1. `inspect_csv_tool` — schema + preview + profile
2. `validate_required_fields_tool` — checks every LLM-required column exists
3. `validate_data_quality_tool` — numeric, missing, negative checks across required fields
4. `validate_constraint_readiness_tool` — duration / demand-inventory / due / machine readiness

---

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Start Ollama

```bash
ollama serve
ollama pull qwen2.5:3b
```

---

## 3. Run

```bash
python run.py
```

Press ENTER to use the default requirement, or type your own.

---

## 4. Output files

Every run saves to `outputs/terminal_runs/<run_id>/`:

```
llm_problem_understanding.json   # LLM extraction result
csv_react_trace.json             # Full ReAct Thought/Action/Observation trace
csv_validation_report.json       # Structured per-field validation report
langgraph_state_trace.json       # LangGraph state snapshot
final_summary.json               # One-screen summary
```

---

## 5. What is implemented

- LangGraph workflow orchestration (3 nodes)
- LLM-only problem understanding (Step 1, not an agent)
- LangChain ReAct CSV validation agent (the only agent) with 4 clean tools
- Full Thought / Action / Observation trace — terminal + JSON
- Deterministic post-validation for reliable JSON report
- Rich terminal output
- Per-run JSON output folder

## 6. What is NOT implemented (intentionally)

- No CP-SAT solver
- No web UI / Flask / FastAPI
- No additional agents

---

## File layout

```
app/
├── main.py
├── config.py
├── graph/
│   ├── state.py
│   └── workflow.py
├── llm/
│   ├── ollama_client.py
│   ├── prompts.py
│   └── user_input_understanding.py
├── agents/
│   └── csv_validation_react_agent.py
├── tools/
│   ├── csv_tools.py
│   └── langchain_csv_tools.py
└── utils/
    ├── logger.py
    └── json_writer.py
data/
└── production_scheduling_basic_dataset.csv
outputs/terminal_runs/<run_id>/
requirements.txt
run.py
README.md
```
