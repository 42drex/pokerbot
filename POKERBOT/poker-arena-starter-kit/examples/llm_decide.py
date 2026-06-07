import json
from montecarlo import monte_carlo_equity
from ranges import RANGES


def build_prompt(hole, board, equity, pot_odds, allowed):
    return f"""
spot: BTN_vs_BB

hole_cards: {hole}
board: {board}

equity: {equity}
pot_odds: {pot_odds}

allowed_actions: {allowed}

RULES:
- You MUST use hole cards + board
- No invented sizes
- Output JSON ONLY:
{{
  "action": "fold|call|raise",
  "reason": "..."
}}
"""


def llm_decide(table):
    hero = table.get("hero_range", RANGES["BTN"])
    villain = table.get("villain_range", RANGES["BB"])

    hole = table.get("hole", [])
    board = table.get("board", [])

    equity = monte_carlo_equity(hero, villain, iterations=1500)["hero_equity"]

    pot_odds = table.get("pot_odds", 0.0)

    prompt = build_prompt(hole, board, equity, pot_odds, ["fold", "call", "raise"])

    print("\n[PROMPT]\n", prompt)

    raw = call_llm(prompt)  # <-- TON LLM EXISTANT

    try:
        return json.loads(raw)
    except Exception:
        return {"action": "fold", "reason": "parse error"}