"""
Long-term memory store — persists employee preferences across sessions.

Concept demonstrated:
  - "Store" in LangGraph terminology
  - Unlike short-term (MemorySaver), this survives session restarts
  - In production this would be a DB like Redis or Postgres
"""

import json
import os

STORE_FILE = "preferences_store.json"


def _load() -> dict:
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save(store: dict) -> None:
    with open(STORE_FILE, "w") as f:
        json.dump(store, f, indent=2)


def get_all_preferences(employee_id: str) -> dict:
    """Return all saved preferences for an employee (empty dict if none)."""
    return _load().get(employee_id, {})


def set_preference(employee_id: str, key: str, value: str) -> None:
    """Upsert a single preference for an employee and persist to disk."""
    store = _load()
    if employee_id not in store:
        store[employee_id] = {}
    store[employee_id][key] = value
    _save(store)


def clear_preferences(employee_id: str) -> None:
    """Remove all saved preferences for an employee."""
    store = _load()
    store.pop(employee_id, None)
    _save(store)
