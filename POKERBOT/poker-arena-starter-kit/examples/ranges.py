"""
Range system: parses poker ranges into real treys card combos.
"""

from treys import Card

RANKS = "23456789TJQKA"
SUIT_CHARS = "shdc"


def rank_to_char(r: int) -> str:
    return RANKS[r]


def expand_pair(rank: str):
    start = RANKS.index(rank)
    return [(r, r, None) for r in range(start, 13)]


def expand_offsuit(r1: str, r2: str, plus=False):
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    lo, hi = min(i1, i2), max(i1, i2)
    if plus:
        return [(hi, r, False) for r in range(lo, hi)]
    return [(hi, lo, False)]


def expand_suited(r1: str, r2: str, plus=False):
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    lo, hi = min(i1, i2), max(i1, i2)
    if plus:
        return [(hi, r, True) for r in range(lo, hi)]
    return [(hi, lo, True)]


def expand_range(notation: str):
    """
    Returns abstract tuples:
    (hi_rank_idx, lo_rank_idx, suited: True/False/None)
    """
    combos = []

    for token in notation.split(","):
        token = token.strip()
        if not token:
            continue

        plus = token.endswith("+")
        if plus:
            token = token[:-1]

        # pocket pairs
        if len(token) == 2 and token[0] == token[1]:
            combos += expand_pair(token[0])
            continue

        r1, r2 = token[0], token[1]
        suited = None

        if len(token) == 3:
            if token[2] == "s":
                suited = True
            elif token[2] == "o":
                suited = False

        if suited is True:
            combos += expand_suited(r1, r2, plus)
        elif suited is False:
            combos += expand_offsuit(r1, r2, plus)
        else:
            # No suit suffix → expand BOTH suited and offsuit
            combos += expand_suited(r1, r2, plus)
            combos += expand_offsuit(r1, r2, plus)

    return combos


def combos_from_range(range_notation: str):
    """
    Convert range string → actual treys card combos (deduplicated).
    FIX: non-suited/non-offsuit tokens previously generated duplicates
    (e.g. Kh Qh appeared as both suited and a cross-suit combo).
    Now handled by splitting into explicit suited + offsuit paths.
    """
    abstract = expand_range(range_notation)
    seen = set()
    result = []

    for hi, lo, suited in abstract:
        hr = RANKS[hi]
        lr = RANKS[lo]

        if hi == lo:
            # pocket pair: 6 combos (C(4,2))
            for i in range(4):
                for j in range(i + 1, 4):
                    combo = (Card.new(hr + SUIT_CHARS[i]), Card.new(lr + SUIT_CHARS[j]))
                    key = tuple(sorted(combo))
                    if key not in seen:
                        seen.add(key)
                        result.append(combo)

        elif suited is True:
            # 4 suited combos
            for s in SUIT_CHARS:
                combo = (Card.new(hr + s), Card.new(lr + s))
                key = tuple(sorted(combo))
                if key not in seen:
                    seen.add(key)
                    result.append(combo)

        elif suited is False:
            # 12 offsuit combos
            for s1 in SUIT_CHARS:
                for s2 in SUIT_CHARS:
                    if s1 != s2:
                        combo = (Card.new(hr + s1), Card.new(lr + s2))
                        key = tuple(sorted(combo))
                        if key not in seen:
                            seen.add(key)
                            result.append(combo)

    return result


# POSITION RANGES
RANGES = {
    "BTN": "22+, A2s+, A2o+, K9s+, K9o+, Q9s+, J9s+, T9s, 98s",
    "BB":  "22+, A2s+, A2o+, K2s+, Q2s+, J2s+, T2s+, 92s+",
    "CO":  "33+, A4s+, A9o+, KTs+, QTs+, JTs",
    "HJ":  "44+, A9s+, AJo+, KQs",
    "UTG": "66+, ATs+, AJo+",
    "SB":  "22+, A2s+, A5o+, K7s+, Q8s+, J8s+",
}
