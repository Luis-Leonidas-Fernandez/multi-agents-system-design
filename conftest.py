import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Provide a dummy API key so graph.py can instantiate agents at module scope
# without a real .env file. Tests that need LLM calls mock get_llm() directly.
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")
