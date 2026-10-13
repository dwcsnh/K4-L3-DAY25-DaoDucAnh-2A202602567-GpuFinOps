import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing


def test_extension_1_tier_recommendation_and_interruption():
    # Stable GPU (H100, ~3% preemption) allows spot for interruptible workloads
    assert pricing.recommend_tier(8, True, gpu_type="H100") == "spot"
    # Volatile GPU (A10G, ~20% preemption) on short/interactive runs routes to on-demand
    assert pricing.recommend_tier(8, True, gpu_type="A10G") == "on_demand"
    # High-duty long-running uninterruptible workloads recommend reserved
    assert pricing.recommend_tier(24, False, gpu_type="A100") == "reserved"
    # Short-lived ad-hoc project (< 180 days) avoids multi-year lock-in
    assert pricing.recommend_tier(24, False, gpu_type="A100", job_days=60) == "on_demand"

    # Commitment length recommendations
    assert pricing.recommend_commitment_length(800) == "3yr"
    assert pricing.recommend_commitment_length(400) == "1yr"
    assert pricing.recommend_commitment_length(90) == "none"

    # Decision matrix completeness
    matrix = pricing.tier_recommendation_matrix()
    assert len(matrix) >= 16
    assert any(m["recommended_tier"] == "on_demand" for m in matrix)
    assert any(m["recommended_tier"] == "spot" for m in matrix)
    assert any(m["recommended_tier"] == "reserved" for m in matrix)


def test_extension_2_mbu_rightsizing():
    r1 = m1_efficiency_audit.run(verbose=False)
    assert "rightsize_recs" in r1
    assert "rightsize_monthly_savings" in r1
    assert r1["rightsize_monthly_savings"] > 0

    recs = {r["gpu_id"]: r for r in r1["rightsize_recs"]}
    # gpu-h100-4 has achieved BW ~0.69 TB/s -> can safely run on A100 (2.0 TB/s)
    if "gpu-h100-4" in recs:
        assert recs["gpu-h100-4"]["proposed_type"] == "A100"
        assert recs["gpu-h100-4"]["monthly_savings"] > 0

    # gpu-a10g-0 has achieved BW ~0.14 TB/s -> can safely run on L4 (0.3 TB/s)
    if "gpu-a10g-0" in recs:
        assert recs["gpu-a10g-0"]["proposed_type"] == "L4"


def test_extension_3_cache_is_worth_it_and_break_even():
    # Anthropic-style: write = 1.25x base, read_discount = 0.10 (savings = 0.90x base)
    # Break-even = 1.25 / 0.90 ~= 1.3889 reads
    be = pricing.cache_break_even_reads(write_cost_per_m=3.75, base_price_per_m=3.00, read_discount=0.10)
    assert abs(be - 1.38888888) < 1e-4

    # When reads < break-even, caching loses money -> False
    assert pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost_per_m=3.75, read_discount=0.10, base_price_per_m=3.00) is False
    # When reads >= break-even, caching is profitable -> True
    assert pricing.cache_is_worth_it(avg_cache_reads=2.0, write_cost_per_m=3.75, read_discount=0.10, base_price_per_m=3.00) is True

    # Integration in M2
    r2 = m2_inference_levers.run(verbose=False)
    assert r2["cache_is_worth_it"] is True
    assert r2["cache_break_even"]["small"] < 2.0
    assert r2["cache_break_even"]["large"] < 2.0
