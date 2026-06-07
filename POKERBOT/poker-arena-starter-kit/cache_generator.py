import json
import time

from montecarlo import monte_carlo_equity
from ranges import RANGES


def generate_cache(path="equity_cache.json", iterations=3000):
    cache = {}

    positions = list(RANGES.keys())
    pairs = [(h, v) for h in positions for v in positions if h != v]

    print(f"[CACHE GEN] {len(pairs)} matchups | {iterations} iterations each")

    start = time.time()

    for i, (hero, villain) in enumerate(pairs):

        print(f"\n[{i+1}/{len(pairs)}] {hero} vs {villain}")

        try:
            result = monte_carlo_equity(
                RANGES[hero],
                RANGES[villain],
                iterations=iterations
            )
        except Exception as e:
            print(f"[ERROR] {hero} vs {villain} → {e}")
            continue

        key = f"{hero}_vs_{villain}"

        cache[key] = {
            "hero_pos": hero,
            "villain_pos": villain,
            "hero_equity": result["hero_equity"],
            "villain_equity": result["villain_equity"],
            "tie_rate": result["tie_rate"],
        }

        print(f"→ equity {result['hero_equity']:.4f}")

    with open(path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n[DONE] saved {path} | entries={len(cache)}")


if __name__ == "__main__":
    generate_cache()