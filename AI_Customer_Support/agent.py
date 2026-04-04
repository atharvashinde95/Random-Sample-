"""
LangGraph Agent — the brain of the helpdesk assistant.

Architecture (ReAct loop):
    User message
        │
        ▼
    ┌──────────────┐
    │  LLM Node    │  ← Decides: answer directly OR call a tool
    └──────┬───────┘
           │  tool_calls present?
    ┌──────▼───────┐
    │  Tool Node   │  ← Executes the tool, returns result
    └──────┬───────┘
           │  loops back to LLM
    ┌──────▼───────┐
    │  LLM Node    │  ← Reads tool result, writes final response
    └──────────────┘

Concepts demonstrated:
  - ToolNode / routing: LangGraph handles LLM→Tool→LLM automatically
  - Short-term memory: MemorySaver stores this session's messages per thread_id
  - Context injection: employee_id, department injected via system prompt — never typed by user
  - Long-term memory: preferences loaded at agent-creation time from memory_store
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS
import memory_store
from mock_data import EMPLOYEES

load_dotenv()


# ── LLM setup ───────────────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://openai.generative.engine.capgemini.com/v1"
        ),
        api_key=os.getenv("GENERATIVE_ENGINE_API_KEY", "YOUR_API_KEY_HERE"),
        model=os.getenv("MODEL_NAME", "openai.gpt-4o"),
        temperature=0.3,        # lower = more deterministic tool decisions
        max_tokens=1024,
    )


# ── System prompt ────────────────────────────────────────────────────────────

def _build_system_prompt(employee_id: str) -> str:
    """
    Inject hidden context (employee_id, dept, prefs) that the user never types.
    This is the 'Context' concept — personalisation without exposing raw data.
    """
    emp = EMPLOYEES.get(employee_id, {})
    prefs = memory_store.get_all_preferences(employee_id)

    lang  = prefs.get("preferred_language", "English")
    style = prefs.get("response_style", "detailed")

    pref_lines = "\n".join(f"    {k}: {v}" for k, v in prefs.items()) if prefs else "    (none saved yet)"

    return f"""You are a helpful Employee Helpdesk AI Assistant.

═══ SESSION CONTEXT (injected by the app — never shown to user) ════════════
  Employee ID  : {employee_id}
  Name         : {emp.get('name', 'Unknown')}
  Department   : {emp.get('department', 'Unknown')}
  Location     : {emp.get('location', 'Unknown')}
═══ LONG-TERM PREFERENCES (loaded from store) ══════════════════════════════
{pref_lines}
════════════════════════════════════════════════════════════════════════════

INSTRUCTIONS:
1. Always pass employee_id="{employee_id}" to any tool that requires it — the user should never need to say their ID.
2. Respond in {lang}.  Keep responses {style}.
3. NEVER guess leave balances, ticket details, or account data — always call the right tool.
4. When a user says "remember X" or "set my preference to Y", call set_user_preference immediately.
5. Be concise, warm, and professional.
6. If you just updated a preference, mention that it will be remembered for future sessions too.
"""


# ── Agent factory ────────────────────────────────────────────────────────────

def create_agent(employee_id: str):
    """
    Build a fresh LangGraph ReAct agent for a given employee.
    Each employee gets an isolated MemorySaver (short-term memory).
    Long-term preferences are baked into the system prompt.
    """
    llm = _build_llm()
    memory = MemorySaver()
    system_msg = SystemMessage(content=_build_system_prompt(employee_id))

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        checkpointer=memory,
        state_modifier=system_msg,   # prepends system message to every call
    )
    return agent


# ── Run helper ───────────────────────────────────────────────────────────────

def run_agent(agent, user_message: str, thread_id: str) -> list:
    """
    Send one user message to the agent and return ALL messages produced
    (including intermediate tool calls and results).
    LangGraph + MemorySaver automatically maintains history via thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    return result["messages"]


def parse_run_result(messages: list) -> dict:
    """
    Extract tool calls, tool results, and the final text response
    from the list of LangGraph messages.
    Returns: { "tool_calls": [...], "tool_results": [...], "response": str }
    """
    from langchain_core.messages import AIMessage, ToolMessage

    tool_calls   = []
    tool_results = []
    final_text   = ""

    for msg in messages:
        if isinstance(msg, AIMessage):
            # May contain tool_calls OR the final text (or both in streaming)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc["name"],
                        "args": tc["args"],
                    })
            if msg.content and not msg.tool_calls:
                final_text = msg.content   # last AIMessage with text = final answer

        elif isinstance(msg, ToolMessage):
            tool_results.append({
                "tool":    msg.name,
                "content": str(msg.content),
            })

    return {
        "tool_calls":   tool_calls,
        "tool_results": tool_results,
        "response":     final_text,
    }
