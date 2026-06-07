"""Arena PokerKit — L1 heuristic agent. EDIT THIS FILE.

Three functions here are the surface builders normally touch:

  - retrieve_solver_context(table)  → Auto Research hook (preflop chart,
                                      postflop solver, opponent HUD).
                                      See examples/research_static_chart.py
                                      for a runnable example you can swap in.

  - estimate_equity(hole, board)    → Monte Carlo equity vs 1 villain.
                                      Backed by `treys`; tune `sims` to
                                      trade speed for accuracy.

  - decide(table, deadline_s, ctx)  → Return one action:
                                      {action, amount?, message, reasoning}.

Everything else is glue. HTTP client, retries, introspection, credential
cache and dry-run scaffolding live in `arena_client.py` and `mock.py`.

CLI:
    uv run examples/agent.py                       # live (uses .env)
    uv run examples/agent.py --competition-id <id> # override env
    uv run examples/agent.py --dry-run             # mock loop, no network
    uv run examples/agent.py --dry-run-scenario queued|stale
    uv run examples/agent.py --max-hands 10        # cap hands (server-settled)
    uv run examples/agent.py --agent path/to/decide.py  # plug-in decide()

You can also use the branded CLI: `pokerkit run --max-hands 10`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Optional

from dotenv import load_dotenv

from arena_client import (
    ArenaClient,
    ArenaError,
    DEFAULT_BASE,
    append_iteration,
    assert_endpoints,
    fetch_introspection,
    load_or_register,
    load_state,
    resolve_terminal_phases,
    save_state,
)

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator, Deck as TreysDeck
    _HAS_TREYS = True
except Exception:
    _HAS_TREYS = False

# Our Monte Carlo engine and range system
try:
    from montecarlo import monte_carlo_equity
    from ranges import RANGES
    _HAS_MONTECARLO = True
except Exception:
    _HAS_MONTECARLO = False


POLL_INTERVAL = 1.0
POLL_JITTER = 0.5
STATUS_REFRESH_S = 8.0


# ─── Auto Research hook ───────────────────────────────────────────
def retrieve_solver_context(table: dict) -> dict:
    return {}


# ─── Card parsing helpers ─────────────────────────────────────────

def _to_treys(card_str: str) -> str:
    """Arena returns 'Ah' / 'AS'. treys wants 'Ah' (rank upper, suit lower).
    Also handle '10x' -> 'Tx'."""
    if not card_str:
        return "2c"
    if card_str.startswith("10"):
        r = "T"
        s = card_str[2].lower() if len(card_str) > 2 else "x"
    else:
        r = card_str[0].upper()
        s = card_str[-1].lower()
    return r + s


def _parse_card_int(c: str) -> int | None:
    """Convert an Arena card string to a treys card int. Returns None on failure."""
    if not c or len(c) < 2:
        return None
    try:
        return TreysCard.new(_to_treys(c))
    except Exception:
        return None


def _parse_cards_to_ints(cards: list[str]) -> list[int]:
    """Parse a list of Arena card strings into treys ints, skipping failures."""
    result = []
    for c in cards:
        parsed = _parse_card_int(c)
        if parsed is not None:
            result.append(parsed)
    return result


# ─── Preflop equity table (fallback) ─────────────────────────────

_PREFLOP_EQUITY = {
    "AA": 0.85, "KK": 0.82, "QQ": 0.80, "JJ": 0.77, "TT": 0.75,
    "99": 0.72, "88": 0.69, "77": 0.66, "66": 0.63, "55": 0.60,
    "44": 0.57, "33": 0.54, "22": 0.50,
    "AKs": 0.67, "AKo": 0.65, "AQs": 0.66, "AQo": 0.64, "AJs": 0.65,
    "AJo": 0.63, "ATs": 0.64, "KQs": 0.63, "KQo": 0.61, "KJs": 0.62,
    "QJs": 0.60, "JTs": 0.58, "T9s": 0.54, "98s": 0.52,
}


def _hand_class(hole: list[str]) -> str:
    ranks = "23456789TJQKA"
    if len(hole) != 2:
        return ""
    r1, s1 = hole[0][0].upper(), hole[0][-1].lower()
    r2, s2 = hole[1][0].upper(), hole[1][-1].lower()
    if r1 not in ranks or r2 not in ranks:
        return ""
    if ranks.index(r1) < ranks.index(r2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def _villain_range_str(table: dict, self_seat_num: int) -> str:
    """Guess villain's positional range from seat info. Falls back to BB."""
    if not _HAS_MONTECARLO:
        return ""
    seats = table.get("seats") or []
    for seat in seats:
        if seat.get("seatNumber") == self_seat_num:
            continue
        pos = (seat.get("position") or "").upper()
        if pos in RANGES:
            return RANGES[pos]
    return RANGES["BB"]


# ─── Hand strength estimation ─────────────────────────────────────

def estimate_equity(hole: list[str], board: list[str], sims: int = 200,
                    deadline_s: float = 10.0) -> float:
    """Monte Carlo equity using the real hole cards vs villain range.
    Falls back to preflop chart when treys/montecarlo unavailable or
    time is tight, or when there is no board yet."""
    cls = _hand_class(hole)

    # Fast path: no time, no treys, or preflop with a known hand class
    if not _HAS_TREYS or deadline_s < 2.0:
        return _PREFLOP_EQUITY.get(cls, 0.45)
    if not board and cls in _PREFLOP_EQUITY:
        return _PREFLOP_EQUITY[cls]

    # Monte Carlo path using real hole cards
    if _HAS_MONTECARLO and len(hole) == 2:
        hole_ints = _parse_cards_to_ints(hole)
        board_ints = _parse_cards_to_ints(board)
        if len(hole_ints) == 2:
            # Pass exact hand as a single-element combo list — bot now sees its cards
            hero_combos = [(hole_ints[0], hole_ints[1])]
            # Infer villain range from table context (set in decide() via closure workaround)
            vrange = _current_villain_range if _current_villain_range else RANGES.get("BB", "")
            try:
                res = monte_carlo_equity(
                    hero_combos,
                    vrange,
                    board=board_ints,
                    iterations=min(sims * 6, 1200),
                )
                return res["hero_equity"]
            except Exception as e:
                print(f"[arena-pokerkit] montecarlo failed: {e}, falling back", file=sys.stderr)

    # Fallback: treys-only random-opponent simulation (original approach)
    try:
        ev = TreysEvaluator()
        hero = [TreysCard.new(_to_treys(c)) for c in hole]
        board_t = [TreysCard.new(_to_treys(c)) for c in board]
        used = set(hero) | set(board_t)
        rng = random.Random(2026)
        wins = ties = 0
        for _ in range(sims):
            deck = TreysDeck()
            deck.cards = [c for c in deck.cards if c not in used]
            rng.shuffle(deck.cards)
            opp = [deck.cards.pop(), deck.cards.pop()]
            runout = []
            needed = 5 - len(board_t)
            for _ in range(needed):
                runout.append(deck.cards.pop())
            full_board = board_t + runout
            hero_rank = ev.evaluate(full_board, hero)
            opp_rank = ev.evaluate(full_board, opp)
            if hero_rank < opp_rank:
                wins += 1
            elif hero_rank == opp_rank:
                ties += 1
        return (wins + 0.5 * ties) / max(sims, 1)
    except Exception:
        return _PREFLOP_EQUITY.get(cls, 0.45)


# Module-level variable so estimate_equity can access villain range
# without changing its signature (which the original agent.py defines).
_current_villain_range: str = ""


# ─── decide() — the part builders edit ────────────────────────────────

def decide(table: dict, deadline_s: float = 10.0,
           research_context: Optional[dict] = None) -> dict:
    """Return one action: {action, amount?, message, reasoning}.

    Reasoning is YAML flow style, max 150 chars, required on benchmark tables:
      {vr: "<range>", ke: "<num+unit>", bf: [<features>], pp: "<plan>", sr: "<size reason>"}
    """
    global _current_villain_range

    allowed = table.get("allowedActions") or {}
    available = allowed.get("availableActions") or []

    # Deadline fallback
    if deadline_s < 2.0:
        if allowed.get("canCheck"):
            return _build("check", None, table, allowed, eq=0.5, po=0.0,
                          msg="deadline tight, taking free option")
        return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                      msg="deadline tight and price not justified")

    self_seat_num = table.get("selfSeatNumber")
    seats = table.get("seats") or []
    self_seat = next((s for s in seats if s.get("seatNumber") == self_seat_num), {})
    hole = list(self_seat.get("holeCards") or [])
    board = list(table.get("boardCards") or [])

    pot = int(table.get("potChips") or 0)
    call_chips = int(allowed.get("callChips") or 0)
    pot_odds = call_chips / max(pot + call_chips, 1) if call_chips else 0.0

    # Set villain range for estimate_equity to consume
    _current_villain_range = _villain_range_str(table, self_seat_num)

    equity = estimate_equity(hole, board, sims=200, deadline_s=deadline_s)

    # Decision tree
    action_name: str
    amount: Optional[int] = None

    if call_chips == 0:
        # Free option: check or bet for value
        if equity > 0.65 and allowed.get("canBet"):
            br = allowed.get("betRange") or {}
            min_bet = int(br.get("min") or max(int(pot * 0.5), 1))
            max_bet = int(br.get("max") or min_bet)
            target = max(min_bet, min(int(pot * 0.66), max_bet))
            action_name, amount = "bet", target
        elif "check" in available:
            action_name = "check"
        elif "call" in available:
            action_name = "call"
        else:
            action_name = "fold"
    else:
        # Facing a bet
        if equity > 0.72 and allowed.get("canRaise"):
            rr = allowed.get("raiseRange") or {}
            min_raise = int(rr.get("min") or call_chips * 2)
            max_raise = int(rr.get("max") or min_raise)
            target = max(min_raise, min(int(pot * 0.75 + call_chips), max_raise))
            action_name, amount = "raise", target
        elif equity >= pot_odds + 0.05 and "call" in available:
            action_name = "call"
            cta = allowed.get("callToAmount")
            if cta is not None:
                amount = int(cta)
        elif equity >= pot_odds - 0.05 and "call" in available:
            # Marginal / breakeven zone → call rather than fold
            action_name = "call"
        elif "fold" in available:
            action_name = "fold"
        elif "check" in available:
            action_name = "check"
        else:
            action_name = "fold"

    # Strip amount from actions that don't take one
    if action_name in ("fold", "check", "call"):
        amount = None

    msg = _human_message(action_name, equity, pot_odds, hole)
    return _build(action_name, amount, table, allowed,
                  eq=equity, po=pot_odds, msg=msg)


def _build(action: str, amount: Optional[int], table: dict, allowed: dict,
           eq: float, po: float, msg: str) -> dict:
    reasoning = _build_reasoning(action, eq, po, table, allowed)
    payload: dict[str, Any] = {
        "action": action,
        "message": msg[:500],
        "reasoning": reasoning,
    }
    if amount is not None:
        payload["amount"] = int(amount)
    return payload


_FALLBACK_REASONING = '{vr: "std", ke: "legal", pp: "pot control"}'


def _build_reasoning(action: str, equity: float, pot_odds: float,
                     table: dict, allowed: dict) -> str:
    board = table.get("boardCards") or []
    street = (table.get("street") or "Preflop")
    self_seat = table.get("selfSeatNumber") or 0
    pos_label = "IP" if self_seat and self_seat % 2 == 0 else "OOP"
    plan_map = {"Preflop": "see flop", "Flop": "barrel T", "Turn": "ck R",
                "River": "showdown"}
    pp = f"{pos_label} {plan_map.get(street, 'pot ctrl')}"[:30]
    if not board:
        bf = "[]"
    else:
        suits = [c[-1].lower() for c in board]
        feats: list[str] = []
        for s in set(suits):
            if suits.count(s) >= 2:
                feats.append(f"FD-{s}")
        ranks = [c[0].upper() for c in board]
        if len(set(ranks)) < len(ranks):
            feats.append("paired")
        bf = "[" + ",".join(feats[:3]) + "]" if feats else "[dry]"
    ke = f"{int(round(equity * 100))}% eq"[:30]
    if action in ("bet", "raise", "all-in"):
        sr = f"po {int(round(pot_odds * 100))}% sized for FE"[:30]
    elif action == "call":
        sr = f"po {int(round(pot_odds * 100))}% covered"[:30]
    else:
        sr = ""
    parts = [
        f'vr: "ln:unknown"',
        f'ke: "{ke}"',
        f'bf: {bf}',
        f'pp: "{pp}"',
    ]
    if sr:
        parts.append(f'sr: "{sr}"')
    yaml = "{" + ", ".join(parts) + "}"
    if len(yaml) <= 150:
        return yaml
    for drop_i in (4, 2):
        if drop_i < len(parts):
            trimmed = parts[:drop_i] + parts[drop_i + 1:]
            candidate = "{" + ", ".join(trimmed) + "}"
            if len(candidate) <= 150:
                return candidate
    return _FALLBACK_REASONING


def _human_message(action: str, equity: float, pot_odds: float, hole: list[str]) -> str:
    eq_pct = int(round(equity * 100))
    po_pct = int(round(pot_odds * 100))
    if action == "fold":
        return f"equity {eq_pct}% short of price {po_pct}%, folding"
    if action == "check":
        return f"taking the free card, equity {eq_pct}%"
    if action == "call":
        return f"equity {eq_pct}% covers price {po_pct}%, calling"
    if action == "bet":
        return f"value bet, hand wants worse to call"
    if action == "raise":
        return f"raising for value, equity {eq_pct}% ahead of range"
    if action == "all-in":
        return f"jamming, equity {eq_pct}% plus fold equity"
    return action


# ─── Live loop (glue — do not edit below this line) ────────────────────

def _safe_research_context(table: dict, retrieve_fn: Any) -> dict:
    if retrieve_fn is None:
        return {}
    try:
        ctx = retrieve_fn(table)
        return ctx if isinstance(ctx, dict) else {}
    except Exception as e:
        print(f"[arena-pokerkit] Auto Research hook failed: {e}, "
              "continuing without context", file=sys.stderr)
        return {}


def _validate_pending_tables(pending: Any) -> list[dict]:
    if not isinstance(pending, dict):
        print(f"[arena-pokerkit] pending-actions returned non-dict "
              f"({type(pending).__name__}); falling back to status poll",
              file=sys.stderr)
        return []
    raw = pending.get("tables")
    if raw is None:
        return []
    if not isinstance(raw, list):
        print(f"[arena-pokerkit] pending-actions `tables` not a list "
              f"({type(raw).__name__}); falling back to status poll",
              file=sys.stderr)
        return []
    valid: list[dict] = []
    for row in raw:
        if not isinstance(row, dict):
            print(f"[arena-pokerkit] skipping malformed pending row "
                  f"(not a dict): {str(row)[:80]}", file=sys.stderr)
            continue
        tid = row.get("tableId")
        if not isinstance(tid, str) or not tid:
            print(f"[arena-pokerkit] skipping pending row without tableId",
                  file=sys.stderr)
            continue
        valid.append(row)
    return valid


def _emit_heartbeat(phase: Any, completed: Any, target: Any, score: Any,
                    pending_count: int, label: str = "",
                    eta_str: str = "") -> None:
    prefix = f"[arena-pokerkit{label}]"
    print(f"{prefix} phase={phase} | "
          f"completedHands={completed}/{target} | "
          f"adjustedBbPer100={score} | "
          f"pending={pending_count}{eta_str}")


def _compute_eta(start_time: float, hands_done: Any, target: Any) -> str:
    try:
        hd = int(hands_done or 0)
        tgt = int(target or 0)
    except (TypeError, ValueError):
        return ""
    if hd <= 0 or tgt <= 0 or tgt <= hd:
        return ""
    elapsed = time.monotonic() - start_time
    if elapsed <= 0:
        return ""
    rate_s_per_hand = elapsed / hd
    remaining = tgt - hd
    eta_s = int(remaining * rate_s_per_hand)
    return f" | ETA {eta_s // 60}m{eta_s % 60:02d}s"


_ACTION_ALIASES = {"all_in": "all-in", "allin": "all-in"}


def _normalize_action_name(action: dict) -> dict:
    if not isinstance(action, dict):
        return action
    name = action.get("action")
    if isinstance(name, str) and name in _ACTION_ALIASES:
        out = dict(action)
        out["action"] = _ACTION_ALIASES[name]
        return out
    return action


def _attempt_credential_repair(client: ArenaClient, args: argparse.Namespace) -> bool:
    try:
        from arena_client import _move_creds_aside, _restore_creds_backup  # noqa: F401
        _move_creds_aside()
        client.api_key = None
        try:
            creds = load_or_register(client, args.handle, args.name, args.quote)
        except Exception:
            _restore_creds_backup()
            raise
        return bool(creds.get("apiKey") or client.api_key)
    except Exception as e:
        print(f"[arena-pokerkit] credential repair failed: {e}", file=sys.stderr)
        return False


def _run_benchmark_loop(
    client: ArenaClient,
    args: argparse.Namespace,
    competition_id: str,
    decide_fn: Any,
    retrieve_fn: Any,
    terminal_phases: set,
    terminal_statuses: set,
    label: str = "",
) -> int:
    state = load_state()
    rng = random.Random()
    hands_acted = 0
    last_completed_hands = 0
    saw_status_refresh = False
    last_status_at = 0.0
    last_heartbeat_at = 0.0
    first_heartbeat_done = False
    credential_repair_used = False
    loop_start_monotonic = time.monotonic()

    _emit_heartbeat(phase="(starting)", completed=0,
                    target="?", score=None,
                    pending_count=0, label=label, eta_str="")
    last_heartbeat_at = time.time()
    first_heartbeat_done = True

    while True:
        tables: list[dict] = []
        try:
            pending = client.get(
                f"/texas/pending-actions?competitionId={competition_id}")
            tables = _validate_pending_tables(pending)
            tables = sorted(tables,
                            key=lambda t: (t.get("actionDeadlineAt") or 0))
        except ArenaError as e:
            print(f"[arena-pokerkit] pending-actions error: {e}", file=sys.stderr)
            if e.status in (401, 403):
                if not credential_repair_used and _attempt_credential_repair(client, args):
                    credential_repair_used = True
                    continue
                print(f"[arena-pokerkit] Credentials rejected mid-match "
                      f"(HTTP {e.status}). Likely .arena-credentials is stale. "
                      f"Run with a fresh handle: --handle <new-handle>",
                      file=sys.stderr)
                return 4
            if e.status == 404:
                raise

        if tables:
            table = tables[0]
            deadline_ms = table.get("actionDeadlineAt") or 0
            deadline_s = (max(0.0, (deadline_ms / 1000.0) - time.time())
                          if deadline_ms else 10.0)
            research_context = _safe_research_context(table, retrieve_fn)
            try:
                action = decide_fn(table, deadline_s=deadline_s,
                                   research_context=research_context)
            except TypeError:
                action = decide_fn(table, deadline_s=deadline_s)
            action = _normalize_action_name(action)
            payload = {"tableId": table["tableId"], **action}
            try:
                client.post("/texas/action", payload)
                hands_acted += 1
                state["hands_played"] = state.get("hands_played", 0) + 1
                state["last_action"] = {
                    "action": action["action"],
                    "amount": action.get("amount"),
                    "at": int(time.time()),
                }
                save_state(state)
            except ArenaError as e:
                if e.status == 409:
                    state["stale_count"] = state.get("stale_count", 0) + 1
                    save_state(state)
                    continue
                if e.status in (401, 403):
                    if not credential_repair_used and _attempt_credential_repair(client, args):
                        credential_repair_used = True
                        continue
                    print(f"[arena-pokerkit] Credentials rejected mid-match "
                          f"(HTTP {e.status}). Likely .arena-credentials is stale. "
                          f"Run with a fresh handle: --handle <new-handle>",
                          file=sys.stderr)
                    return 4
                if e.status == 400:
                    state["rejection_count"] = state.get("rejection_count", 0) + 1
                    save_state(state)
                    try:
                        client.post("/texas/action", {
                            "tableId": table["tableId"],
                            "action": "fold",
                            "message": "fallback after illegal action",
                            "reasoning": _FALLBACK_REASONING,
                        })
                    except ArenaError:
                        pass
                    continue
                raise
            if (args.max_hands and saw_status_refresh
                    and last_completed_hands >= args.max_hands):
                print(f"[arena-pokerkit] hit --max-hands={args.max_hands} "
                      f"(completedHands={last_completed_hands}), stopping")
                return 0

        now = time.time()
        if (not tables) or (now - last_status_at >= STATUS_REFRESH_S):
            status = None
            try:
                status = client.get(
                    f"/texas/benchmark/status?competitionId={competition_id}")
            except ArenaError as e:
                print(f"[arena-pokerkit] status refresh error: {e}",
                      file=sys.stderr)
                if e.status in (401, 403):
                    if not credential_repair_used and _attempt_credential_repair(client, args):
                        credential_repair_used = True
                        continue
                    print(f"[arena-pokerkit] Credentials rejected mid-match "
                          f"(HTTP {e.status}). Likely .arena-credentials is stale. "
                          f"Run with a fresh handle: --handle <new-handle>",
                          file=sys.stderr)
                    return 4
            last_status_at = now
            if isinstance(status, dict):
                match = status.get("match") or {}
                saw_status_refresh = True
                try:
                    last_completed_hands = int(match.get("completedHands") or 0)
                except (TypeError, ValueError):
                    last_completed_hands = 0
                if (args.max_hands
                        and last_completed_hands >= args.max_hands):
                    print(f"[arena-pokerkit{label}] hit --max-hands="
                          f"{args.max_hands} "
                          f"(completedHands={last_completed_hands}), stopping")
                    return 0
                if now - last_heartbeat_at >= 5.0:
                    eta_str = _compute_eta(
                        loop_start_monotonic,
                        match.get("completedHands"),
                        match.get("targetHands"),
                    )
                    _emit_heartbeat(
                        phase=match.get("phase"),
                        completed=match.get("completedHands"),
                        target=match.get("targetHands"),
                        score=match.get("adjustedBbPer100"),
                        pending_count=len(tables),
                        label=label,
                        eta_str=eta_str,
                    )
                    last_heartbeat_at = now
                phase = match.get("phase")
                msstatus = match.get("status")
                if phase in terminal_phases or msstatus in terminal_statuses:
                    print(f"[arena-pokerkit{label}] match terminal "
                          f"({phase}/{msstatus}) | "
                          f"hands={match.get('completedHands')} | "
                          f"adjustedBbPer100={match.get('adjustedBbPer100')}")
                    score = match.get("adjustedBbPer100")
                    if score is not None:
                        try:
                            s = float(score)
                            print(f"[arena-pokerkit{label}] baseline reference: "
                                  "default L1 heuristic typically scores -15 to -5 "
                                  "bb/100 vs the DeepCFR panel.")
                            if s > 5:
                                tag = "🏆 above heuristic baseline — strong"
                            elif s > -5:
                                tag = "✓ within heuristic baseline range"
                            elif s > -15:
                                tag = "↺ below baseline — iterate decide()"
                            else:
                                tag = "⚠ well below baseline — check decide() bugs"
                            print(f"[arena-pokerkit{label}] verdict: {tag} "
                                  f"(your score: {s:+.1f} bb/100)")
                        except (TypeError, ValueError):
                            pass
                    print(f"[arena-pokerkit{label}] match summary: "
                          f"{json.dumps(match, sort_keys=True)}")
                    state["bankroll"] = int(match.get("rawChipDelta") or 0)
                    save_state(state)
                    try:
                        bb = match.get("adjustedBbPer100")
                        bb_val = float(bb) if bb is not None else None
                    except (TypeError, ValueError):
                        bb_val = None
                    try:
                        hands_val = int(match.get("completedHands") or 0)
                    except (TypeError, ValueError):
                        hands_val = None
                    decide_version = os.environ.get(
                        "ARENA_DECIDE_VERSION", "decide() iter")
                    try:
                        append_iteration({
                            "bb_per_100": bb_val,
                            "hands": hands_val,
                            "decide_version": decide_version,
                            "phase": match.get("phase"),
                            "status": match.get("status"),
                        })
                    except Exception as _e:
                        print(f"[arena-pokerkit{label}] could not record "
                              f"iteration: {_e}", file=sys.stderr)
                    return 0

        if not tables:
            time.sleep(POLL_INTERVAL + rng.uniform(-POLL_JITTER, POLL_JITTER))


def run_live_benchmark(args: argparse.Namespace,
                       decide_fn: Optional[Any] = None) -> int:
    load_dotenv()
    api_key = os.environ.get("ARENA_API_KEY") or None
    base = os.environ.get("ARENA_API_BASE", DEFAULT_BASE)
    competition_id = args.competition_id or os.environ.get("ARENA_COMPETITION_ID")
    if not competition_id:
        print(
            "ERROR: no competition specified.\n\n"
            "Either:\n"
            "  cp .env.example .env       # has Poker Eval S1 ID pre-filled\n"
            "or:\n"
            "  uv run examples/agent.py --competition-id seed_poker_eval_s1",
            file=sys.stderr,
        )
        return 2

    decide_fn = decide_fn or decide

    client = ArenaClient(base, api_key=api_key)

    try:
        creds = load_or_register(client, args.handle, args.name, args.quote)
        agent_id = creds.get("agentId") or creds.get("id") or "?"
        print(f"[arena-pokerkit] registered agent={agent_id} base={base}")

        schema = fetch_introspection(client)
        assert_endpoints(schema)
        terminal_phases, terminal_statuses = resolve_terminal_phases(schema)
        print(f"[arena-pokerkit] introspection OK | "
              f"terminal phases={sorted(terminal_phases)} | "
              f"statuses={sorted(terminal_statuses)}")

        try:
            start_resp = client.post("/texas/benchmark/start",
                                     {"competitionId": competition_id})
        except ArenaError as e:
            if e.status == 402:
                print("[arena-pokerkit] competition has entry fee — pay manually or "
                      "pick a free competition", file=sys.stderr)
                return 3
            raise
        if not isinstance(start_resp, dict):
            raise ArenaError(0, str(start_resp)[:200], "benchmark/start malformed")
        match = start_resp.get("match") or {}
        if match.get("phase") in terminal_phases or match.get("status") in terminal_statuses:
            print(f"[arena-pokerkit] benchmark already terminal: phase={match.get('phase')} "
                  f"summary={json.dumps(match, sort_keys=True)}")
            return 0
        print(f"[arena-pokerkit] benchmark started: phase={match.get('phase')} "
              f"target={match.get('targetHands')}")

        return _run_benchmark_loop(
            client=client,
            args=args,
            competition_id=competition_id,
            decide_fn=decide_fn,
            retrieve_fn=retrieve_solver_context,
            terminal_phases=terminal_phases,
            terminal_statuses=terminal_statuses,
            label="",
        )
    finally:
        client.close()


def load_external_decide(path: str) -> Any:
    import importlib.util
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        raise SystemExit(f"[arena-pokerkit] --agent path not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_external_agent_{p.stem}", str(p))
    if spec is None or spec.loader is None:
        raise SystemExit(f"[arena-pokerkit] could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        raise SystemExit(f"[arena-pokerkit] error loading {path}: {e}")
    fn = getattr(mod, "decide", None)
    if not callable(fn):
        raise SystemExit(
            f"[arena-pokerkit] {path} has no top-level `decide(...)` function")
    return fn


# ─── Main / CLI ────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arena PokerKit L1 agent")
    parser.add_argument("--competition-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-scenario",
                        choices=("instant", "queued", "stale"),
                        default="instant")
    parser.add_argument("--max-hands", type=int, default=0)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--handle", default="pokerkit-starter")
    parser.add_argument("--name", default="PokerKit Starter")
    parser.add_argument("--quote", default="probability over swagger")
    args = parser.parse_args(argv)

    if not args.dry_run and not os.path.exists(".arena-credentials") and (
        args.name == "PokerKit Starter"
        or args.quote == "probability over swagger"
    ):
        raise SystemExit(
            "[arena-pokerkit] refusing to register with the placeholder identity.\n"
            "Run Arena's onboarding skill first to get a real handle/name/quote:\n"
            "  https://arena.dev.fun/skills/arena.md\n"
            "It walks Phase 1 (propose Name + Bio to the owner, await confirm)\n"
            "and Phase 2 (register, write .arena-credentials). Then re-run\n"
            "`./pokerkit run` here and it picks up the cached credentials.\n"
            "To bypass for power-user runs, pass --handle X --name Y --quote Z."
        )

    decide_fn = decide
    if args.agent:
        decide_fn = load_external_decide(args.agent)
        print(f"[arena-pokerkit] using external decide() from {args.agent}")

    if args.dry_run:
        from mock import run_mock_benchmark
        return run_mock_benchmark(args, decide_fn=decide_fn,
                                  retrieve_solver_context=retrieve_solver_context)
    return run_live_benchmark(args, decide_fn=decide_fn)


if __name__ == "__main__":
    sys.exit(main())
