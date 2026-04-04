"""
Tools — each decorated with @tool so the LLM can call them.
"""

from langchain_core.tools import tool
from mock_data import EMPLOYEES, LEAVE_BALANCES, TICKETS, ALL_TICKETS
import memory_store


@tool
def get_employee_info(employee_id: str) -> str:
    """Fetch profile and account details for an employee."""
    emp = EMPLOYEES.get(employee_id.upper())
    if not emp:
        return f"Employee '{employee_id}' not found."
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
    """Return current leave balance for an employee."""
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
    """List the most recent support tickets for an employee."""
    tickets = TICKETS.get(employee_id.upper(), [])
    if not tickets:
        return f"No tickets found for '{employee_id}'."
    recent = tickets[::-1][:max(1, min(count, 10))]
    lines = [f"Last {len(recent)} ticket(s):\n"]
    for t in recent:
        lines.append(
            f"  [{t['id']}] {t['title']}\n"
            f"    Status: {t['status']}  |  Priority: {t['priority']}  |  Date: {t['created']}\n"
        )
    return "\n".join(lines).strip()


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Get full details for a specific support ticket by ID (e.g. TKT-2451)."""
    ticket = ALL_TICKETS.get(ticket_id.upper())
    if not ticket:
        return f"Ticket '{ticket_id}' not found."
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
    """Save a personal preference for the employee that persists across sessions."""
    memory_store.set_preference(employee_id.upper(), preference_key, preference_value)
    return f"✅ Saved — {preference_key}: '{preference_value}'. This will be remembered next session too."


@tool
def get_user_preferences(employee_id: str) -> str:
    """Get all saved preferences for the employee."""
    prefs = memory_store.get_all_preferences(employee_id.upper())
    if not prefs:
        return f"No preferences saved for '{employee_id}'."
    return "\n".join(f"  {k}: {v}" for k, v in prefs.items())


ALL_TOOLS = [
    get_employee_info,
    get_leave_balance,
    get_recent_tickets,
    get_ticket_status,
    set_user_preference,
    get_user_preferences,
]
