"""
Agent — the brain of the helpdesk assistant.

FIXED BUGS (caused the blank Streamlit screen):
  1. Name collision: the line `create_agent = create_agent_instance` overwrote
     the imported `create_agent` from langchain, so calling it became infinite
     recursion → crash before Streamlit could render anything.
  2. Fixed by aliasing the library import as `_lc_create_agent`.

Compatible with:
  - LangChain 1.x  → from langchain.agents import create_agent
  - LangChain 0.3.x fallback → from langgraph.prebuilt import create_react_agent
  This file auto-detects which is installed.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS
import memory_store
from mock_data import EMPLOYEES

load_dotenv()

# ── Import the right agent builder depending on installed version ─────────────
# LangChain 1.x  →  langchain.agents.create_agent       (new standard)
# LangChain 0.3x →  langgraph.prebuilt.create_react_agent (old standard)
try:
    from langchain.agents import create_agent as _build_agent
    _AGENT_MODE = "langchain1x"
except ImportError:
    from langgraph.prebuilt import create_react_agent as _build_agent   # type: ignore
    _AGENT_MODE = "langgraph_prebuilt"


# ── LLM setup ────────────────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://openai.generative.engine.capgemini.com/v1"
        ),
        api_key=os.getenv("GENERATIVE_ENGINE_API_KEY", "YOUR_API_KEY_HERE"),
        model=os.getenv("MODEL_NAME", "openai.gpt-4o"),
        temperature=0.3,
        max_tokens=1024,
    )


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(employee_id: str) -> str:
    emp   = EMPLOYEES.get(employee_id, {})
    prefs = memory_store.get_all_preferences(employee_id)

    lang  = prefs.get("preferred_language", "English")
    style = prefs.get("response_style", "detailed")

    pref_lines = (
        "\n".join(f"    {k}: {v}" for k, v in prefs.items())
        if prefs else "    (none saved yet)"
    )

    return f"""You are a helpful Employee Helpdesk AI Assistant.

=== SESSION CONTEXT (injected by the app — never shown to user) ===
  Employee ID  : {employee_id}
  Name         : {emp.get('name', 'Unknown')}
  Department   : {emp.get('department', 'Unknown')}
  Location     : {emp.get('location', 'Unknown')}
=== LONG-TERM PREFERENCES (loaded from store) ===
{pref_lines}
===================================================================

INSTRUCTIONS:
1. Always pass employee_id="{employee_id}" to any tool that requires it.
   The user should never need to say their own ID.
2. Respond in {lang}. Keep responses {style}.
3. NEVER guess leave balances, ticket details, or account data — always call the right tool.
4. When a user says "remember X" or "set my preference to Y", call set_user_preference immediately.
5. Be concise, warm, and professional.
6. If you just updated a preference, confirm that it will be remembered in future sessions too.
"""


# ── Agent factory ─────────────────────────────────────────────────────────────

def create_agent(employee_id: str):
    """
    Build an agent for the given employee.
    Works with LangChain 1.x (create_agent) or 0.3.x (create_react_agent).
    Short-term memory via MemorySaver; long-term via system prompt injection.
    """
    llm    = _build_llm()
    memory = MemorySaver()
    prompt = _build_system_prompt(employee_id)

    if _AGENT_MODE == "langchain1x":
        # LangChain 1.x API
        agent = _build_agent(
            model=llm,
            tools=ALL_TOOLS,
            system_prompt=prompt,
            checkpointer=memory,
        )
    else:
        # LangGraph 0.3.x / LangChain 0.3.x API
        agent = _build_agent(
            model=llm,
            tools=ALL_TOOLS,
            state_modifier=prompt,   # called state_modifier in older API
            checkpointer=memory,
        )

    return agent


# ── Run helper ────────────────────────────────────────────────────────────────

def run_agent(agent, user_message: str, thread_id: str) -> list:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    return result["messages"]


def parse_run_result(messages: list) -> dict:
    tool_calls   = []
    tool_results = []
    final_text   = ""

    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"name": tc["name"], "args": tc["args"]})
            if msg.content and not msg.tool_calls:
                final_text = msg.content

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
