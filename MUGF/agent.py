"""
Agent — helpdesk assistant brain.
Uses: langgraph.prebuilt.create_react_agent (langgraph==0.2.60)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS
import memory_store
from mock_data import EMPLOYEES

load_dotenv()


def _build_llm():
    return ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://openai.generative.engine.capgemini.com/v1"),
        api_key=os.getenv("GENERATIVE_ENGINE_API_KEY", "YOUR_API_KEY_HERE"),
        model=os.getenv("MODEL_NAME", "openai.gpt-4o"),
        temperature=0.3,
        max_tokens=1024,
    )


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

=== SESSION CONTEXT ===
  Employee ID  : {employee_id}
  Name         : {emp.get('name', 'Unknown')}
  Department   : {emp.get('department', 'Unknown')}
  Location     : {emp.get('location', 'Unknown')}
=== SAVED PREFERENCES ===
{pref_lines}
========================

INSTRUCTIONS:
1. Always pass employee_id="{employee_id}" to every tool that needs it.
2. Respond in {lang}. Keep responses {style}.
3. NEVER guess data — always call the right tool.
4. When user says "remember X", call set_user_preference immediately.
5. Be concise, warm, and professional.
"""


def create_agent(employee_id: str):
    """Create a fresh agent for the given employee."""
    agent = create_react_agent(
        model=_build_llm(),
        tools=ALL_TOOLS,
        prompt=_build_system_prompt(employee_id),   # 'prompt=' is correct for langgraph 0.2.x
        checkpointer=MemorySaver(),
    )
    return agent


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
            tool_results.append({"tool": msg.name, "content": str(msg.content)})

    return {
        "tool_calls":   tool_calls,
        "tool_results": tool_results,
        "response":     final_text,
    }
