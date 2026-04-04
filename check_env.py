"""
Run this BEFORE `streamlit run app.py` to diagnose issues.
Usage:  python check_env.py
"""

import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}\n")

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  ✅  {label}")
    except Exception as e:
        print(f"  ❌  {label}")
        print(f"       → {e}")
        errors.append(label)

print("── Checking packages ──────────────────────────────────")
check("langchain",        lambda: __import__("langchain"))
check("langchain_openai", lambda: __import__("langchain_openai"))
check("langchain_core",   lambda: __import__("langchain_core"))
check("langgraph",        lambda: __import__("langgraph"))
check("openai",           lambda: __import__("openai"))
check("streamlit",        lambda: __import__("streamlit"))
check("dotenv",           lambda: __import__("dotenv"))

print("\n── Checking langchain version & API ───────────────────")
try:
    import langchain, langchain_openai, langgraph
    print(f"  langchain        : {langchain.__version__}")
    print(f"  langchain_openai : {langchain_openai.__version__}")
    print(f"  langgraph        : {langgraph.__version__}")
except Exception as e:
    print(f"  Could not get versions: {e}")

check("langchain.agents.create_agent exists",
      lambda: getattr(__import__("langchain.agents", fromlist=["create_agent"]), "create_agent"))

print("\n── Checking project files ─────────────────────────────")
check("mock_data imports",   lambda: __import__("mock_data"))
check("memory_store imports", lambda: __import__("memory_store"))
check("tools imports",       lambda: __import__("tools"))
check("agent imports",       lambda: __import__("agent"))

print("\n── Checking .env file ─────────────────────────────────")
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GENERATIVE_ENGINE_API_KEY", "")
if api_key and api_key != "YOUR_API_KEY_HERE":
    print("  ✅  GENERATIVE_ENGINE_API_KEY is set")
else:
    print("  ❌  GENERATIVE_ENGINE_API_KEY is missing or still placeholder")
    print("       → Edit your .env file and set your real API key")
    errors.append("API key not set")

print("\n" + "─" * 54)
if errors:
    print(f"❌  {len(errors)} problem(s) found:")
    for e in errors:
        print(f"   • {e}")
    print("\nFix the above, then run:  streamlit run app.py")
else:
    print("✅  Everything looks good! Run:  streamlit run app.py")
