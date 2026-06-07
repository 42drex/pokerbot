"""
Monte Carlo equity estimator.

hero_range / villain_range can be:
  - a range string like "22+, AKs+"  (parsed via combos_from_range)
  - a list of (card_int, card_int) tuples  (used directly, e.g. real hole cards)
"""

import random
from treys import Card, Evaluator
from ranges import combos_from_range

evaluator = Evaluator()


def _all_deck_cards():
    ranks = "23456789TJQKA"
    suits = "shdc"
    return [Card.new(r + s) for r in ranks for s in suits]


def _resolve_combos(range_input):
    """
    Accept either a range string or an already-expanded list of combos.
    """
    if isinstance(range_input, str):
        return combos_from_range(range_input)
    # assume list of (int, int) tuples
    return list(range_input)


def _sample_hand(combos, dead):
    candidates = [(a, b) for (a, b) in combos if a not in dead and b not in dead]
    if not candidates:
        return None
    return random.choice(candidates)


def monte_carlo_equity(hero_range, villain_range, board=None, iterations=3000):
    hero_combos = _resolve_combos(hero_range)
    villain_combos = _resolve_combos(villain_range)
    board = board or []

    wins = ties = valid = 0

    full_deck = _all_deck_cards()
    board_set = set(board)

    for _ in range(iterations):
        dead = set(board_set)

        hero = _sample_hand(hero_combos, dead)
        if hero is None:
            continue
        dead.update(hero)

        villain = _sample_hand(villain_combos, dead)
        if villain is None:
            continue
        dead.update(villain)

        available = [c for c in full_deck if c not in dead]
        need = 5 - len(board)

        if len(available) < need:
            continue

        runout = random.sample(available, need)
        full_board = board + runout

        hero_score = evaluator.evaluate(full_board, list(hero))
        villain_score = evaluator.evaluate(full_board, list(villain))

        if hero_score < villain_score:
            wins += 1
        elif hero_score == villain_score:
            ties += 1

        valid += 1

    if valid == 0:
        return {"hero_equity": 0.5, "villain_equity": 0.5, "tie_rate": 0.0, "iterations_run": 0}

    return {
        "hero_equity": round((wins + ties * 0.5) / valid, 4),
        "villain_equity": round((valid - wins - ties * 0.5) / valid, 4),
        "tie_rate": round(ties / valid, 4),
        "iterations_run": valid,
    }


if __name__ == "__main__":
    from ranges import RANGES

    print("BTN vs BB (range vs range)")
    print(monte_carlo_equity(RANGES["BTN"], RANGES["BB"], iterations=2000))

    print("\nAsKd vs BB (exact hand)")
    hole = [(Card.new("As"), Card.new("Kd"))]
    print(monte_carlo_equity(hole, RANGES["BB"], iterations=2000))
