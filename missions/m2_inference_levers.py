"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    # Extension 3: Prompt caching break-even economics
    # Write cost is ~1.25x base input price (standard provider cache write rate)
    small_pin, _ = MODEL_PRICES["small"]
    large_pin, _ = MODEL_PRICES["large"]
    small_be = pricing.cache_break_even_reads(small_pin * 1.25, small_pin, read_discount=0.10)
    large_be = pricing.cache_break_even_reads(large_pin * 1.25, large_pin, read_discount=0.10)
    # In token_usage.csv, 2400 reqs across 4 shared project prefixes (~600 reqs/prefix/day, ~25 reqs/hr)
    estimated_avg_reads = 25.0

    cache_active = {
        "small": pricing.cache_is_worth_it(estimated_avg_reads, small_pin * 1.25, read_discount=0.10, base_price_per_m=small_pin),
        "large": pricing.cache_is_worth_it(estimated_avg_reads, large_pin * 1.25, read_discount=0.10, base_price_per_m=large_pin),
    }

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching (gated by cache_is_worth_it), batch API
        tier_name = r["route_tier"]
        pin, pout = MODEL_PRICES[tier_name]
        effective_cached = cached if cache_active.get(tier_name, True) else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=effective_cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("\n[Extension 3] Prompt Caching Economics:")
        print(f"  Small model break-even reads: {small_be:.2f} reads (cost ${small_pin*1.25:.2f}/M write, saves ${small_pin*0.90:.2f}/M read)")
        print(f"  Large model break-even reads: {large_be:.2f} reads (cost ${large_pin*1.25:.2f}/M write, saves ${large_pin*0.90:.2f}/M read)")
        print(f"  Estimated dataset prefix reuse: ~{estimated_avg_reads:.0f} reads/prefix -> Cache Worth It: {all(cache_active.values())}")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_break_even": {"small": round(small_be, 2), "large": round(large_be, 2)},
        "cache_is_worth_it": all(cache_active.values()),
    }


if __name__ == "__main__":
    run()
