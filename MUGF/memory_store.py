"""Long-term memory — persists employee preferences across sessions."""

import json, os

STORE_FILE = "preferences_store.json"


def _load() -> dict:
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE) as f:
            return json.load(f)
    return {}


def _save(store: dict) -> None:
    with open(STORE_FILE, "w") as f:
        json.dump(store, f, indent=2)


def get_all_preferences(employee_id: str) -> dict:
    return _load().get(employee_id, {})


def set_preference(employee_id: str, key: str, value: str) -> None:
    store = _load()
    store.setdefault(employee_id, {})[key] = value
    _save(store)


def clear_preferences(employee_id: str) -> None:
    store = _load()
    store.pop(employee_id, None)
    _save(store)
