"""Wrapper Arena-compatible qui utilise DeepSeek via llm_decide."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from llm_decide import decide_with_llm

_cache = {}
try:
    with open(os.path.join(os.path.dirname(__file__), "../equity_cache.json")) as f:
        _cache = json.load(f)
    print(f"[decide_llm] cache loaded: {len(_cache)} spots")
except FileNotFoundError:
    print("[decide_llm] no cache, fallback equity 0.5")

def decide(table, deadline_s=10.0, research_context=None):
    return decide_with_llm(table, _cache, deadline_s=deadline_s)