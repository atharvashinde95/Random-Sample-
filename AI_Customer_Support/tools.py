"""
Tools — each decorated with @tool so LangChain/LangGraph can bind them to the LLM.

Concept demonstrated:
  - Tool Calling: LLM decides WHEN to call these, not the user
  - ToolNode routing: LangGraph sends control here, then returns to LLM
  - Command / state update: set_user_preference mutates the long-term store
"""

import json
from langchain_core.tools import tool
from mock_data import EMPLOYEES, LEAVE_BALANCES, TICKETS, ALL_TICKETS
import memory_store


@tool
def get_employee_info(employee_id: str) -> str:
    """
    Fetch profile and account details for an employee.
    Use when the user asks about their account, profile, or basic information.
    """
    emp = EMPLOYEES.get(employee_id.upper())
    if not emp:
        return f"Employee '{employee_id}' not found in the system."
    return (
        f"Name       : {emp['name']}\n"
        f"Department : {emp['department']}\n"
        f"Email      : {emp['email']}\n"
        f"Manager    : {emp['manager']}\n"
        f"Joined     : {emp['joining_date']}\n"
        f"Location   : {emp['location']}\n"
        f"Grade      : {emp['grade']}"
    )


@tool
def get_leave_balance(employee_id: str) -> str:
    """
    Return the current leave balance (casual, sick, earned) for an employee.
    Use when the user asks about leave days, vacation balance, or time off.
    """
    bal = LEAVE_BALANCES.get(employee_id.upper())
    if not bal:
        return f"Leave data not found for '{employee_id}'."
    return (
        f"Leave Balance:\n"
        f"  Casual Leave  : {bal['casual']} days remaining\n"
        f"  Sick Leave    : {bal['sick']} days remaining\n"
        f"  Earned Leave  : {bal['earned']} days remaining\n"
        f"  Used This Year: {bal['total_used_this_year']} days"
    )


@tool
def get_recent_tickets(employee_id: str, count: int = 3) -> str:
    """
    List the most recent IT/HR support tickets raised by an employee.
    'count' controls how many to return (default 3, max 10).
    Use when the user asks to see their tickets, requests, or support history.
    """
    tickets = TICKETS.get(employee_id.upper(), [])
    if not tickets:
        return f"No support tickets found for '{employee_id}'."
    count = max(1, min(count, 10))
    recent = tickets[::-1][:count]   # newest first
    lines = [f"Last {len(recent)} ticket(s) for {employee_id}:\n"]
    for t in recent:
        lines.append(
            f"  [{t['id']}] {t['title']}\n"
            f"    Status: {t['status']}  |  Priority: {t['priority']}  |  Date: {t['created']}\n"
        )
    return "\n".join(lines).strip()


@tool
def get_ticket_status(ticket_id: str) -> str:
    """
    Get the full details and current status for a specific support ticket by ID.
    Use when the user mentions a specific ticket number like TKT-2451.
    """
    ticket = ALL_TICKETS.get(ticket_id.upper())
    if not ticket:
        return f"Ticket '{ticket_id}' not found. Please check the ticket ID."
    return (
        f"Ticket      : {ticket['id']}\n"
        f"Title       : {ticket['title']}\n"
        f"Status      : {ticket['status']}\n"
        f"Priority    : {ticket['priority']}\n"
        f"Category    : {ticket['category']}\n"
        f"Created     : {ticket['created']}\n"
        f"Last Updated: {ticket['updated']}\n"
        f"Details     : {ticket['description']}"
    )


@tool
def set_user_preference(employee_id: str, preference_key: str, preference_value: str) -> str:
    """
    Save a personal preference for the employee. This persists across sessions (long-term memory).
    Common keys: 'preferred_language', 'response_style', 'notification_type'.
    Examples: preferred_language=Hindi, response_style=brief, notification_type=email.
    Use when the user says 'remember that I prefer...', 'set my language to...', etc.
    """
    memory_store.set_preference(employee_id.upper(), preference_key, preference_value)
    return f"✅ Preference saved — {preference_key}: '{preference_value}'. I'll remember this for future sessions."


@tool
def get_user_preferences(employee_id: str) -> str:
    """
    Retrieve all saved long-term preferences for the employee.
    Use when the user asks what preferences are saved, or to verify a setting.
    """
    prefs = memory_store.get_all_preferences(employee_id.upper())
    if not prefs:
        return f"No preferences saved yet for '{employee_id}'."
    lines = [f"Saved preferences for {employee_id}:"]
    for k, v in prefs.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ── exported list ────────────────────────────────────────────────────────────
ALL_TOOLS = [
    get_employee_info,
    get_leave_balance,
    get_recent_tickets,
    get_ticket_status,
    set_user_preference,
    get_user_preferences,
]
