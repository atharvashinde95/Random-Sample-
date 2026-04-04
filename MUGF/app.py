"""
Streamlit UI — Employee Helpdesk AI
Run: streamlit run app.py
"""

import uuid
import streamlit as st
from agent import create_agent, run_agent, parse_run_result
from mock_data import EMPLOYEES
import memory_store

st.set_page_config(
    page_title="Employee Helpdesk AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
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


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    for k, v in {
        "messages":      [],
        "thread_id":     str(uuid.uuid4()),
        "employee_id":   "EMP001",
        "agent":         None,
        "pending_input": None,
    }.items():
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


if st.session_state.agent is None:
    st.session_state.agent = create_agent(st.session_state.employee_id)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Helpdesk Demo")
    st.caption("LangGraph · Tool Calling · Memory")
    st.divider()

    st.markdown("### 👤 Employee Login")
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

    st.markdown("### 💾 Saved Preferences")
    prefs = memory_store.get_all_preferences(st.session_state.employee_id)
    if prefs:
        for k, v in prefs.items():
            st.markdown(f'<span class="pref-chip">🔖 **{k}** = {v}</span>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Preferences", use_container_width=True):
            memory_store.clear_preferences(st.session_state.employee_id)
            _reset_session()
            st.rerun()
    else:
        st.caption("No preferences saved yet.\nTry: _'Remember I prefer Hindi'_")

    st.divider()

    st.markdown("### ⚡ Demo Script")
    demo_steps = [
        ("1️⃣  Leave balance",     "What is my leave balance?"),
        ("2️⃣  Recent tickets",    "Show my last 3 support tickets"),
        ("3️⃣  Specific ticket",   "What is the status of TKT-2451?"),
        ("4️⃣  Set preference",    "Remember that I prefer Hindi responses"),
        ("5️⃣  Account summary",   "Summarize my account details"),
        ("6️⃣  Check preferences", "What preferences have you saved for me?"),
    ]
    for label, question in demo_steps:
        if st.button(label, use_container_width=True, key=f"btn_{label}"):
            st.session_state.pending_input = question

    st.divider()
    if st.button("🔄 New Session", use_container_width=True):
        _reset_session()
        st.rerun()

    st.markdown("### 🔀 Agent Flow")
    st.code(
        "User Message\n    ↓\nLLM Node  ←────┐\n  │ tool_call?   │\n"
        "  ↓ yes         │\nTool Node        │\n  │ result       │\n"
        "  └─────────────┘\n    ↓ no tool\nFinal Response",
        language="text"
    )


# ── Main chat ─────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Employee Helpdesk Assistant")
st.caption("Powered by **LangGraph** · Tool Calling · Short + Long-term Memory · Context Injection")

cols = st.columns(6)
for col, (icon, label) in zip(cols, [
    ("🔧", "Tool Calling"), ("🧠", "Short-term Memory"), ("💾", "Long-term Memory"),
    ("🔀", "Graph Routing"), ("👤", "Context Injection"), ("⚙️", "State Updates"),
]):
    col.markdown(
        f'<div style="text-align:center;background:#ffffff08;border-radius:8px;padding:6px 2px;font-size:12px;">'
        f'{icon}<br><b>{label}</b></div>', unsafe_allow_html=True
    )

st.divider()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            if msg.get("tool_calls"):
                tc_html = "".join(
                    f'<span class="tool-badge">🔧 {tc["name"]}({", ".join(f"{k}={v!r}" for k,v in tc["args"].items())})</span>'
                    for tc in msg["tool_calls"]
                )
                st.markdown(
                    f'<div style="margin-bottom:6px"><span class="concept-tag">TOOL CALLING</span> {tc_html}</div>',
                    unsafe_allow_html=True
                )
                if msg.get("tool_results"):
                    with st.expander("📦 Tool Results", expanded=False):
                        for tr in msg["tool_results"]:
                            st.markdown(f"**`{tr['tool']}`**")
                            st.code(tr["content"], language="text")
            st.write(msg["content"])

# Handle input
user_input = st.session_state.pop("pending_input", None) or st.chat_input(
    "Ask anything… e.g. 'What is my leave balance?'"
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                raw_messages = run_agent(
                    st.session_state.agent,
                    user_input,
                    st.session_state.thread_id,
                )
                parsed       = parse_run_result(raw_messages)
                tool_calls   = parsed["tool_calls"]
                tool_results = parsed["tool_results"]
                response     = parsed["response"]

                if tool_calls:
                    tc_html = "".join(
                        f'<span class="tool-badge">🔧 {tc["name"]}({", ".join(f"{k}={v!r}" for k,v in tc["args"].items())})</span>'
                        for tc in tool_calls
                    )
                    st.markdown(
                        f'<div style="margin-bottom:6px"><span class="concept-tag">TOOL CALLING</span> {tc_html}</div>',
                        unsafe_allow_html=True
                    )

                if tool_results:
                    with st.expander("📦 Tool Results", expanded=True):
                        for tr in tool_results:
                            st.markdown(f"**`{tr['tool']}`**")
                            st.code(tr["content"], language="text")

                st.write(response)
                st.session_state.messages.append({
                    "role": "assistant", "content": response,
                    "tool_calls": tool_calls, "tool_results": tool_results,
                })

            except Exception as exc:
                import traceback
                err = f"⚠️ Error: {exc}"
                st.error(err)
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                st.session_state.messages.append({"role": "assistant", "content": err})

    st.rerun()
