"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


GPU_INTERRUPTION_RATES = {
    "H100": 0.03,    # ~3% per hour (stable, high-capacity clusters)
    "H200": 0.04,
    "B200": 0.02,
    "A100": 0.08,    # ~8% per hour
    "MI300X": 0.06,
    "L4": 0.12,      # ~12% per hour
    "A10G": 0.20,    # ~20% per hour (high contention / preemption volatility)
}


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
) -> str:
    """Pick a purchasing tier from duty cycle, interruptibility, GPU type, and duration.

    Extension 1 logic:
      1. Interruption rate risk: For interruptible workloads, check GPU preemption rate.
         High preemption (e.g. >=15% on A10G) coupled with short/ad-hoc runs (<=8h/day)
         risks excessive restart overhead or breaks interactive work -> 'on_demand'.
         Otherwise -> 'spot'.
      2. Duration & Commitment lock-in: For uninterruptible workloads with duty cycle >= break-even:
         If job_days is specified and < 180 days (short ad-hoc project), committing to 1-3 years
         locks in wasted unutilized capacity -> 'on_demand'.
         Otherwise (long-running / core production service) -> 'reserved'.
      3. Otherwise -> 'on_demand'.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)

    if interruptible and hours_per_day < 24:
        irate = GPU_INTERRUPTION_RATES.get(gpu_type, 0.05) if gpu_type else 0.05
        if irate >= 0.15 and hours_per_day <= 8:
            return "on_demand"
        return "spot"

    if duty >= be:
        if job_days is not None and job_days < 180:
            return "on_demand"
        return "reserved"

    return "on_demand"


def recommend_commitment_length(job_days: float | None = None) -> str:
    """Compare 3-year vs 1-year reserved commitment based on workload time horizon.

    Returns:
      '3yr' for long-term production services (>= 2 years) where 45% discount is maximized.
      '1yr' for medium-term initiatives (1-2 years) to minimize multi-year lock-in risk.
      'none' for temporary projects (< 1 year) where commitment lock-in exceeds on-demand cost.
    """
    if job_days is None or job_days >= 730:
        return "3yr"
    elif job_days >= 365:
        return "1yr"
    return "none"


def tier_recommendation_matrix() -> list[dict]:
    """Decision matrix for purchasing tiers across GPU type, duty cycle, and interruptibility."""
    matrix = []
    sample_gpus = ["H100", "A100", "A10G", "L4"]
    for g in sample_gpus:
        irate = GPU_INTERRUPTION_RATES.get(g, 0.05)
        for duty_hrs, is_int in [(4, False), (8, True), (18, True), (24, False)]:
            tier = recommend_tier(duty_hrs, is_int, gpu_type=g)
            matrix.append({
                "gpu_type": g,
                "hours_per_day": duty_hrs,
                "interruptible": is_int,
                "interruption_rate": irate,
                "recommended_tier": tier,
            })
    return matrix


def cache_break_even_reads(
    write_cost_per_m: float,
    base_price_per_m: float,
    read_discount: float = 0.10,
) -> float:
    """Break-even reads needed for prompt caching to save money over uncached requests.

    Each read saves: base_price_per_m * (1 - read_discount).
    Break-even reads = write_cost_per_m / (base_price_per_m * (1 - read_discount)).
    """
    savings_per_read = base_price_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return float("inf")
    return write_cost_per_m / savings_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    base_price_per_m: float = 3.00,
) -> bool:
    """Prompt cache pays off when total read savings exceed prefix creation/storage cost.

    Break-even: avg_cache_reads >= write_cost_per_m / (base_price_per_m * (1 - read_discount)).
    """
    be_reads = cache_break_even_reads(write_cost_per_m, base_price_per_m, read_discount)
    return avg_cache_reads >= be_reads


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
