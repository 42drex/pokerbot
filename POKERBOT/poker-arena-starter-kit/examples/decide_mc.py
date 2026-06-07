"""
Arena-compatible decide() for dev.fun Poker Eval.

Drop this file into the starter kit as:
  examples/agent.py  (replace the decide() function)
OR load it via:
  ./pokerkit run --agent examples/decide_mc.py

The file is self-contained: it imports our range+MC engine from the
pokerbot/ folder (adjust sys.path below if needed).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

# ── Path setup ───────────────────────────────────────────────────────────────
# If you placed pokerbot/ next to examples/, this just works.
# Adjust if your directory layout differs.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_HERE, _ROOT, os.path.join(_ROOT, "pokerbot")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ranges import RANGES, combos_from_range
from montecarlo import monte_carlo_equity

# ── Constants ─────────────────────────────────────────────────────────────────

# How many MC iterations per decision (tune speed vs accuracy)
MC_ITERS_POSTFLOP = 400
MC_ITERS_PREFLOP  = 200   # preflop table is faster anyway

# Raise edge required above pot odds
RAISE_EDGE = 0.08
CALL_EDGE  = 0.01

# Preflop equity table (fast fallback, no MC needed preflop)
_PREFLOP_EQ: dict[str, float] = {
    "AA": 0.85, "KK": 0.82, "QQ": 0.80, "JJ": 0.77, "TT": 0.75,
    "99": 0.72, "88": 0.69, "77": 0.66, "66": 0.63, "55": 0.60,
    "44": 0.57, "33": 0.54, "22": 0.50,
    "AKs": 0.67, "AKo": 0.65, "AQs": 0.66, "AQo": 0.64,
    "AJs": 0.65, "AJo": 0.63, "ATs": 0.64,
    "KQs": 0.63, "KQo": 0.61, "KJs": 0.62,
    "QJs": 0.60, "JTs": 0.58, "T9s": 0.54, "98s": 0.52,
}

_RANKS = "23456789TJQKA"
_FALLBACK_REASONING = '{vr: "std", ke: "legal", pp: "pot control"}'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hand_class(hole: list[str]) -> str:
    """Convert ['As','Kd'] → 'AKo'  |  ['Qh','Qd'] → 'QQ'"""
    if len(hole) != 2:
        return ""
    r1, s1 = hole[0][0].upper(), hole[0][-1].lower()
    r2, s2 = hole[1][0].upper(), hole[1][-1].lower()
    if r1 not in _RANKS or r2 not in _RANKS:
        return ""
    if _RANKS.index(r1) < _RANKS.index(r2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def _infer_position(table: dict) -> str:
    """
    Rough position from seat number and number of players.
    Arena doesn't expose BTN seat directly, so we approximate:
    seat 1 = BTN, seat 2 = SB, seat 3 = BB, rest = CO/UTG.
    """
    seats = table.get("seats") or []
    n = len(seats)
    self_num = table.get("selfSeatNumber") or 1

    # Rank our seat relative to total players
    rank = self_num % max(n, 1)
    if rank == 0:
        return "BTN"
    if rank == 1:
        return "SB"
    if rank == 2:
        return "BB"
    if rank == n - 1:
        return "CO"
    return "UTG"


def _get_equity(hole: list[str], board: list[str],
                street: str, deadline_s: float) -> float:
    """
    Preflop: use lookup table (fast, accurate enough).
    Postflop: run Monte Carlo with our range-based engine.
    """
    if deadline_s < 2.0:
        cls = _hand_class(hole)
        return _PREFLOP_EQ.get(cls, 0.45)

    if not board:
        # Preflop
        cls = _hand_class(hole)
        eq = _PREFLOP_EQ.get(cls)
        if eq is not None:
            return eq
        # Fallback MC for unlisted hands
        result = monte_carlo_equity(
            RANGES["BTN"], RANGES["BB"],
            iterations=MC_ITERS_PREFLOP,
        )
        return result["hero_equity"]

    # Postflop: convert Arena card strings to treys format, run MC
    from treys import Card
    def to_treys(c: str) -> int:
        r = "T" if c.startswith("10") else c[0].upper()
        s = c[-1].lower()
        return Card.new(r + s)

    try:
        board_ints = [to_treys(c) for c in board]
        result = monte_carlo_equity(
            RANGES["BTN"], RANGES["BB"],   # villain range approximation
            board=board_ints,
            iterations=MC_ITERS_POSTFLOP,
        )
        return result["hero_equity"]
    except Exception:
        cls = _hand_class(hole)
        return _PREFLOP_EQ.get(cls, 0.45)


def _build_reasoning(action: str, equity: float, pot_odds: float,
                     table: dict) -> str:
    """YAML flow style, max 150 chars."""
    board  = table.get("boardCards") or []
    street = table.get("street") or "Preflop"
    pos    = _infer_position(table)

    plan_map = {"Preflop": "see flop", "Flop": "barrel T",
                "Turn": "check R", "River": "showdown"}
    pp = f"{pos} {plan_map.get(street, 'pot ctrl')}"[:28]

    feats: list[str] = []
    if board:
        suits = [c[-1].lower() for c in board]
        for s in set(suits):
            if suits.count(s) >= 2:
                feats.append(f"FD-{s}")
        ranks = [c[0].upper() for c in board]
        if len(set(ranks)) < len(ranks):
            feats.append("paired")
    bf = "[" + ",".join(feats[:2]) + "]" if feats else "[dry]"

    ke = f"{int(round(equity * 100))}% eq"
    parts = [
        f'vr: "range-mc"',
        f'ke: "{ke}"',
        f'bf: {bf}',
        f'pp: "{pp}"',
    ]
    if action in ("bet", "raise", "all-in"):
        sr = f"po {int(round(pot_odds * 100))}% FE"
        parts.append(f'sr: "{sr}"')

    yaml = "{" + ", ".join(parts) + "}"
    return yaml if len(yaml) <= 150 else _FALLBACK_REASONING


# ── Main decide() — Arena entry point ────────────────────────────────────────

def decide(
    table: dict,
    deadline_s: float = 10.0,
    research_context: Optional[dict] = None,
) -> dict:
    """
    Range-based Monte Carlo poker agent.
    Compatible with Arena Starter Kit's decide() contract.
    """
    allowed   = table.get("allowedActions") or {}
    available = allowed.get("availableActions") or []

    # Hard deadline fallback
    if deadline_s < 2.0:
        if allowed.get("canCheck"):
            return {"action": "check",
                    "message": "deadline tight",
                    "reasoning": _FALLBACK_REASONING}
        return {"action": "fold",
                "message": "deadline tight",
                "reasoning": _FALLBACK_REASONING}

    # Extract game state
    self_seat_num = table.get("selfSeatNumber")
    seats         = table.get("seats") or []
    self_seat     = next((s for s in seats if s.get("seatNumber") == self_seat_num), {})
    hole          = list(self_seat.get("holeCards") or [])
    board         = list(table.get("boardCards") or [])
    street        = table.get("street") or "Preflop"

    pot        = int(table.get("potChips") or 0)
    call_chips = int(allowed.get("callChips") or 0)
    pot_odds   = call_chips / max(pot + call_chips, 1) if call_chips else 0.0

    # Equity estimation
    equity = _get_equity(hole, board, street, deadline_s)

    # Raise/call/fold thresholds
    raise_thresh = pot_odds + RAISE_EDGE
    call_thresh  = pot_odds + CALL_EDGE
    spr = (int(self_seat.get("stackChips") or 100)) / max(pot, 1)

    # ── Decision logic ────────────────────────────────────────────────────────
    action_name: str
    amount: Optional[int] = None

    if call_chips == 0:
        # No bet to face — check or bet for value
        if equity > 0.70 and allowed.get("canBet"):
            br      = allowed.get("betRange") or {}
            min_bet = int(br.get("min") or max(int(pot * 0.5), 1))
            max_bet = int(br.get("max") or min_bet)
            target  = max(min_bet, min(int(pot * 0.66), max_bet))
            action_name, amount = "bet", target
        elif "check" in available:
            action_name = "check"
        elif "call" in available:
            action_name = "call"
        else:
            action_name = "fold"
    else:
        # Facing a bet
        if equity >= raise_thresh and spr > 1.0 and allowed.get("canRaise"):
            rr        = allowed.get("raiseRange") or {}
            min_raise = int(rr.get("min") or call_chips * 2)
            max_raise = int(rr.get("max") or min_raise)
            target    = max(min_raise, min(int(pot * 0.66 + call_chips * 2), max_raise))
            action_name, amount = "raise", target
        elif equity >= call_thresh and "call" in available:
            action_name = "call"
        elif "check" in available:
            action_name = "check"
        else:
            action_name = "fold"

    # Strip amount for actions that don't take one
    if action_name in ("fold", "check", "call"):
        amount = None

    reasoning = _build_reasoning(action_name, equity, pot_odds, table)
    eq_pct    = int(round(equity * 100))
    po_pct    = int(round(pot_odds * 100))

    msg_map = {
        "fold":  f"equity {eq_pct}% < price {po_pct}%, folding",
        "check": f"free option, equity {eq_pct}%",
        "call":  f"equity {eq_pct}% > price {po_pct}%, calling",
        "bet":   f"value bet, equity {eq_pct}%",
        "raise": f"raising for value, equity {eq_pct}% vs range",
        "all-in": f"jamming, equity {eq_pct}%",
    }
    message = msg_map.get(action_name, action_name)

    payload: dict = {
        "action":    action_name,
        "message":   message,
        "reasoning": reasoning,
    }
    if amount is not None:
        payload["amount"] = int(amount)

    return payload


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate the table dict from decide-function.md worked example
    test_table = {
        "tableId": "table_test_1",
        "potChips": 3,
        "street": "Preflop",
        "boardCards": [],
        "selfSeatNumber": 4,
        "seats": [
            {"seatNumber": 1, "agentHandle": "villain", "stackChips": 198, "holeCards": []},
            {"seatNumber": 4, "agentHandle": "hero",    "stackChips": 200, "holeCards": ["As", "Ks"]},
        ],
        "actionDeadlineAt": int(time.time() * 1000) + 10_000,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callChips": 2,
            "callToAmount": 2,
            "canCheck": False,
            "canBet": False,
            "canRaise": True,
            "raiseRange": {"min": 4, "max": 200},
        },
    }

    result = decide(test_table, deadline_s=10.0)
    print("AKs preflop:", result)

    # Flop test
    test_table2 = dict(test_table)
    test_table2["street"]     = "Flop"
    test_table2["boardCards"] = ["Ah", "7d", "2c"]
    test_table2["potChips"]   = 10
    test_table2["allowedActions"] = {
        "availableActions": ["check", "bet"],
        "callChips": 0,
        "canCheck": True,
        "canBet": True,
        "betRange": {"min": 5, "max": 100},
        "canRaise": False,
    }
    result2 = decide(test_table2, deadline_s=10.0)
    print("AKs on A72 flop:", result2)
