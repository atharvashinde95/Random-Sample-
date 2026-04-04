# 🤖 Employee Helpdesk AI — Demo

A fully working AI-powered employee support assistant that demonstrates 6 key LLM concepts in one conversation.

## Concepts Demonstrated

| Concept | How it's shown |
|---|---|
| 🔧 **Tool Calling** | LLM calls `get_leave_balance()`, `get_ticket_status()` etc. instead of guessing |
| 🧠 **Short-term Memory** | LangGraph `MemorySaver` keeps the full conversation in-session per `thread_id` |
| 💾 **Long-term Memory** | Preferences saved to `preferences_store.json`, loaded on next session |
| 👤 **Context Injection** | `employee_id`, department, location passed in system prompt — user never types it |
| ⚙️ **State Updates** | `set_user_preference()` mutates the store and changes behaviour immediately |
| 🔀 **Graph Routing** | LangGraph ReAct loop: `LLM → ToolNode → LLM` only when a tool is needed |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Edit .env and set GENERATIVE_ENGINE_API_KEY

# 3. Run the app
streamlit run app.py
```

## Demo Script (present this to your manager)

**Scene 1 — Context injection**
> The app passes `employee_id` silently. User never types their ID.

**Scene 2 — Tool calling**
> "What is my leave balance?"
> → LLM calls `get_leave_balance(employee_id="EMP001")`

**Scene 3 — Specific tool routing**
> "What is the status of TKT-2451?"
> → LLM calls `get_ticket_status(ticket_id="TKT-2451")`

**Scene 4 — State update (command)**
> "Remember that I prefer Hindi responses"
> → LLM calls `set_user_preference(preference_key="preferred_language", preference_value="Hindi")`
> → Saved to `preferences_store.json`

**Scene 5 — Long-term memory in action**
> Close browser. Reopen. Select same employee.
> → System prompt now includes `preferred_language: Hindi`
> → Assistant responds in Hindi automatically!

## Project Structure

```
helpdesk_demo/
├── app.py              # Streamlit UI (run this)
├── agent.py            # LangGraph ReAct agent + system prompt
├── tools.py            # All @tool definitions (what LLM can call)
├── mock_data.py        # Fake HR/IT database
├── memory_store.py     # Long-term preference persistence (JSON)
├── requirements.txt
└── .env.example
```

## What to say about the LLM config screenshot

> "This configures an OpenAI-compatible endpoint using `openai.gpt-4o`.
> I control `temperature` and `top_p` for response quality, and I pass `tools`
> so the model can call backend functions when needed.
> That means the model can answer normally — or trigger real actions
> like a database lookup, a memory update, or a state change.
> LangGraph then routes the flow: User → LLM → Tool (if needed) → LLM → Response."
