"""
Streamlit UI — Employee Helpdesk AI Demo
Run with:  streamlit run app.py
"""

import uuid
import streamlit as st
from agent import create_agent, run_agent, parse_run_result
from mock_data import EMPLOYEES
import memory_store

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Helpdesk AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tool call pill */
.tool-badge {
    display: inline-block;
    background: #7c3aed22;
    border: 1px solid #7c3aed;
    color: #a78bfa;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    font-family: monospace;
    margin: 2px 4px 2px 0;
}
/* Concept label */
.concept-tag {
    display: inline-block;
    background: #0ea5e922;
    border: 1px solid #0ea5e9;
    color: #38bdf8;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 11px;
    margin-right: 4px;
}
.pref-chip {
    background: #10b98122;
    border: 1px solid #10b981;
    color: #34d399;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    display: inline-block;
    margin: 2px 4px 2px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":        [],
        "thread_id":       str(uuid.uuid4()),
        "employee_id":     "EMP001",
        "agent":           None,
        "pending_input":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _reset_session(new_emp_id: str = None):
    emp_id = new_emp_id or st.session_state.employee_id
    st.session_state.messages      = []
    st.session_state.thread_id     = str(uuid.uuid4())
    st.session_state.employee_id   = emp_id
    st.session_state.agent         = create_agent(emp_id)
    st.session_state.pending_input = None


# Ensure agent is created on first run
if st.session_state.agent is None:
    st.session_state.agent = create_agent(st.session_state.employee_id)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Helpdesk Demo")
    st.caption("LangGraph · Tool Calling · Memory")
    st.divider()

    # ---- Employee selection (simulates SSO/login context injection) ----
    st.markdown("### 👤 Employee Login")
    st.caption("Simulates app passing `employee_id` as hidden context")

    emp_labels = {f"{v['name']} ({k})": k for k, v in EMPLOYEES.items()}
    chosen_label = st.selectbox(
        "Select Employee",
        list(emp_labels.keys()),
        index=list(emp_labels.values()).index(st.session_state.employee_id),
        label_visibility="collapsed",
    )
    chosen_id = emp_labels[chosen_label]

    if chosen_id != st.session_state.employee_id:
        _reset_session(chosen_id)
        st.rerun()

    emp = EMPLOYEES[st.session_state.employee_id]
    st.info(
        f"**{emp['name']}**  \n"
        f"{emp['department']} · {emp['location']}  \n"
        f"ID: `{st.session_state.employee_id}`"
    )

    st.divider()

    # ---- Long-term preferences (from memory store) ----
    st.markdown("### 💾 Saved Preferences")
    st.caption("Persisted across sessions (long-term memory)")
    prefs = memory_store.get_all_preferences(st.session_state.employee_id)
    if prefs:
        for k, v in prefs.items():
            st.markdown(f'<span class="pref-chip">🔖 **{k}** = {v}</span>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Preferences", use_container_width=True):
            memory_store.clear_preferences(st.session_state.employee_id)
            _reset_session()
            st.rerun()
    else:
        st.caption("No preferences saved yet.  \nTry: _'Remember I prefer Hindi'_")

    st.divider()

    # ---- Quick demo questions ----
    st.markdown("### ⚡ Demo Script")
    st.caption("Click to run the demo story")

    demo_steps = [
        ("1️⃣  Leave balance",        "What is my leave balance?"),
        ("2️⃣  Recent tickets",       "Show my last 3 support tickets"),
        ("3️⃣  Specific ticket",      "What is the status of TKT-2451?"),
        ("4️⃣  Set preference",       "Remember that I prefer Hindi responses"),
        ("5️⃣  Account summary",      "Summarize my account details"),
        ("6️⃣  Check preferences",    "What preferences have you saved for me?"),
    ]
    for label, question in demo_steps:
        if st.button(label, use_container_width=True, key=f"btn_{label}"):
            st.session_state.pending_input = question

    st.divider()

    if st.button("🔄 New Session", use_container_width=True):
        _reset_session()
        st.rerun()

    # ---- Architecture diagram ----
    st.markdown("### 🔀 Agent Flow")
    st.code(
        "User Message\n"
        "    ↓\n"
        "LLM Node  ←─────────┐\n"
        "  │ tool_call?       │\n"
        "  ↓ yes              │\n"
        "Tool Node            │\n"
        "  │ result           │\n"
        "  └─────────────────→┘\n"
        "    ↓ no tool_call\n"
        "Final Response",
        language="text"
    )


# ── Main chat area ───────────────────────────────────────────────────────────
st.markdown("## 🤖 Employee Helpdesk Assistant")
st.caption(
    "Powered by **LangGraph** · **Tool Calling** · **Short + Long-term Memory** · **Context Injection**"
)

# Concepts legend
cols = st.columns(6)
concepts = [
    ("🔧", "Tool Calling"),
    ("🧠", "Short-term Memory"),
    ("💾", "Long-term Memory"),
    ("🔀", "Graph Routing"),
    ("👤", "Context Injection"),
    ("⚙️", "State Updates"),
]
for col, (icon, label) in zip(cols, concepts):
    col.markdown(
        f'<div style="text-align:center; background:#ffffff08; border-radius:8px; padding:6px 2px; font-size:12px;">'
        f'{icon}<br><b>{label}</b></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        if role == "user":
            st.write(msg["content"])
        else:
            # Show tool calls made
            if msg.get("tool_calls"):
                tc_html = "".join(
                    f'<span class="tool-badge">🔧 {tc["name"]}({", ".join(f"{k}={v!r}" for k,v in tc["args"].items())})</span>'
                    for tc in msg["tool_calls"]
                )
                st.markdown(
                    f'<div style="margin-bottom:6px">'
                    f'<span class="concept-tag">TOOL CALLING</span> {tc_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                # Tool results in an expander
                if msg.get("tool_results"):
                    with st.expander("📦 Tool Results (raw data from backend)", expanded=False):
                        for tr in msg["tool_results"]:
                            st.markdown(f"**`{tr['tool']}`**")
                            st.code(tr["content"], language="text")

            st.write(msg["content"])


# ── Handle input (chat box OR quick button) ──────────────────────────────────
user_input = st.session_state.pop("pending_input", None) or st.chat_input(
    "Ask anything… e.g. 'What is my leave balance?' or 'Remember I prefer brief responses'"
)

if user_input:
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                raw_messages = run_agent(
                    st.session_state.agent,
                    user_input,
                    st.session_state.thread_id,
                )
                parsed = parse_run_result(raw_messages)

                tool_calls   = parsed["tool_calls"]
                tool_results = parsed["tool_results"]
                response     = parsed["response"]

                # ── Display tool calls ──
                if tool_calls:
                    tc_html = "".join(
                        f'<span class="tool-badge">🔧 {tc["name"]}({", ".join(f"{k}={v!r}" for k,v in tc["args"].items())})</span>'
                        for tc in tool_calls
                    )
                    st.markdown(
                        f'<div style="margin-bottom:6px">'
                        f'<span class="concept-tag">TOOL CALLING</span> {tc_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Display tool results ──
                if tool_results:
                    with st.expander("📦 Tool Results (raw data from backend)", expanded=True):
                        for tr in tool_results:
                            st.markdown(f"**`{tr['tool']}`**")
                            st.code(tr["content"], language="text")

                # ── Display final response ──
                st.write(response)

                # ── Save to history ──
                st.session_state.messages.append({
                    "role":         "assistant",
                    "content":      response,
                    "tool_calls":   tool_calls,
                    "tool_results": tool_results,
                })

            except Exception as exc:
                err = f"⚠️ Error: {exc}"
                st.error(err)
                import traceback
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                st.session_state.messages.append({
                    "role": "assistant", "content": err
                })

    st.rerun()
