"""
Prompt templates for the LLM-only problem understanding step.

We keep the schema description tight and explicit because qwen2.5:3b
is small - the more deterministic the contract, the better.
"""

from __future__ import annotations

# The exact JSON schema we want the LLM to fill in. Embedded as text so the
# model copies the keys verbatim.
PROBLEM_UNDERSTANDING_SCHEMA = """
{
  "problem_type": "string (e.g. production_scheduling, job_shop, flow_shop)",
  "is_cpsat_compatible": true,
  "cpsat_compatibility_reason": "short explanation",
  "objective": {
    "name": "string (e.g. minimize_makespan, minimize_tardiness)",
    "description": "string",
    "is_supported_for_cpsat": true
  },
  "variables": [
    {"name": "string", "type": "string", "description": "string"}
  ],
  "constraints": [
    {
      "name": "string",
      "type": "hard | soft",
      "description": "string",
      "required_csv_fields": ["string"]
    }
  ],
  "business_rules": ["string"],
  "global_parameters": {
    "machine_count": 0,
    "working_minutes_per_day": 0,
    "machines_are_identical": true
  },
  "required_csv_fields": ["string"],
  "optional_csv_fields": ["string"],
  "constraint_validation": {
    "are_constraints_valid": true,
    "has_contradictions": false,
    "contradictions": ["string"],
    "warnings": ["string"]
  },
  "missing_inputs": ["string"],
  "assumptions": ["string"]
}
""".strip()


def build_understanding_prompt(user_input: str) -> str:
    """Main extraction prompt. JSON-only output."""
    return f"""You are a constraint-programming problem analyst.
Your job is to read a natural-language production-scheduling requirement and
extract a strict, machine-readable description of it.

You must answer THREE things explicitly:
1. Is the user input compatible with Google OR-Tools CP-SAT?
2. Are the user's constraints internally valid (well-formed)?
3. Are any constraints contradictory with each other?

CP-SAT compatibility heuristics:
- Discrete decisions, integer durations, finite machines/resources -> compatible.
- Continuous variables, stochastic objectives, non-linear cost terms -> NOT compatible.

Contradiction examples to look for:
- "schedule all jobs" + "skip some required jobs"
- "use only machine M1" + "must run only on M2"
- "machines are identical" + machine-specific rule that conflicts
- "check inventory" + "ignore inventory"
- impossible deadlines that can be proven from user input alone (without CSV)

If a contradiction can only be proven once we see the CSV data, put it in
`constraint_validation.warnings`, NOT in `contradictions`.

For each constraint, list the CSV fields it requires in `required_csv_fields`.
If the user only specifies a machine COUNT (e.g. "2 identical machines"),
do NOT add `machine_id` as a required CSV field - it is a global parameter.

CRITICAL FIELD EXTRACTION RULES:
- `job_id` is ALWAYS required for any scheduling problem.
- If the user mentions "production_qty" or "quantity", add `production_qty` to required_csv_fields.
- If the user mentions "processing_time" or "processing_time_min_per_unit", add `processing_time_min_per_unit` to required_csv_fields.
- If the user says "duration = production_qty * processing_time_min_per_unit" or similar, BOTH `production_qty` AND `processing_time_min_per_unit` are required.
- If the user says "check demand" or mentions "demand", add `demand_qty` to required_csv_fields.
- If the user says "check inventory" or mentions "inventory" or "stock", add `inventory_qty` to required_csv_fields.
- The top-level `required_csv_fields` must be the UNION of all per-constraint required fields plus `job_id`.

CRITICAL CONSTRAINT EXTRACTION RULES (the `constraints` array MUST NOT be empty
for production scheduling):
- If the user says plan/schedule all jobs/orders/pending jobs, ADD a constraint
  named `schedule_all_jobs` (type: hard, required_csv_fields: ["job_id"]).
- If the user mentions machines, production lines, ovens, resources, or any
  machine count, ADD a constraint named `machine_capacity` (type: hard,
  required_csv_fields: []).
- If the user mentions shift length, working day, "8 hours", "X hours per day",
  or working minutes, ADD a constraint named `working_time_limit` (type: hard,
  required_csv_fields: []).
- If the user mentions BOTH a quantity-like word AND a processing-time-like
  word (e.g. "production quantity" + "processing time"), ADD a constraint
  named `duration_calculation` (type: hard,
  required_csv_fields: ["production_qty", "processing_time_min_per_unit"]).
- If the user mentions demand/customer demand AND inventory/stock/availability,
  ADD a constraint named `demand_inventory_validation` (type: hard,
  required_csv_fields: ["demand_qty", "inventory_qty"]).

OBJECTIVE MAPPING RULES:
- "finish as early as possible", "reduce/minimize total/overall completion
  time", "complete the plan quickly", "as low as possible" -> objective.name
  MUST be `minimize_makespan`.
- "minimize lateness/tardiness/delays" -> objective.name = `minimize_tardiness`.
- "balance load across machines" -> objective.name = `balance_load`.

The `constraints` array MUST contain at least one entry whenever the user
describes a scheduling task. Returning an empty `constraints` array is invalid.

Output STRICT JSON only. No prose, no markdown fences.
Match this schema exactly (fill values, keep keys):

{PROBLEM_UNDERSTANDING_SCHEMA}

USER REQUIREMENT:
\"\"\"
{user_input}
\"\"\"

Return ONLY the JSON object.
""".strip()


def build_json_repair_prompt(broken_text: str) -> str:
    """Used after the first call returns non-parseable JSON."""
    return f"""The following text was supposed to be a single JSON object
matching a fixed schema, but it could not be parsed.

Fix it. Output STRICT JSON only. No prose, no markdown fences, no comments.
Keep the same keys and intent.

BROKEN OUTPUT:
\"\"\"
{broken_text}
\"\"\"
""".strip()
