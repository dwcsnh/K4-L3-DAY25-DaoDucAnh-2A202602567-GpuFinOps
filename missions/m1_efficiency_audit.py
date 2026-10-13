"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0, "mem_used": []})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if "mem_used_gb" in r:
            a["mem_used"].append(num(r["mem_used_gb"]))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "mem_used_gb": round(sum(a["mem_used"]) / len(a["mem_used"]), 1) if a["mem_used"] else 0.0,
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing analysis based on MBU and $/GB-VRAM ---
    vram_costs = {}
    for gtype, c in cat.items():
        od = num(c["on_demand_hr"])
        hbm = num(c["hbm_gb"])
        bw = num(c["peak_bw_tbs"])
        vram_costs[gtype] = {
            "on_demand_hr": od,
            "hbm_gb": hbm,
            "peak_bw_tbs": bw,
            "cost_per_gb_vram": od / hbm if hbm else 0.0,
            "cost_per_tb_bw": od / bw if bw else 0.0,
        }

    rightsize_recs = []
    for s in summary:
        mbu = s["mbu"]
        cur_type = s["gpu_type"]
        cur_info = vram_costs[cur_type]
        cur_od = cur_info["on_demand_hr"]
        achieved_bw = mbu * cur_info["peak_bw_tbs"]
        needed_vram = s.get("mem_used_gb", 0.0)

        # Flag memory-bound GPUs with low MBU (< 0.32) for right-sizing
        if mbu < 0.32:
            candidates = []
            for cand_type, cand_info in vram_costs.items():
                if cand_info["on_demand_hr"] < cur_od:
                    # Target GPU must satisfy bandwidth headroom (>= achieved * 1.15) and VRAM capacity
                    if cand_info["peak_bw_tbs"] >= achieved_bw * 1.15 and cand_info["hbm_gb"] >= needed_vram:
                        candidates.append((cand_info["on_demand_hr"], cand_type, cand_info))
            if candidates:
                candidates.sort()
                best_od, best_type, best_info = candidates[0]
                delta_hr = cur_od - best_od
                delta_month = delta_hr * 24 * 30
                pct = (delta_hr / cur_od) * 100
                rightsize_recs.append({
                    "gpu_id": s["gpu_id"],
                    "current_type": cur_type,
                    "mbu": mbu,
                    "achieved_bw_tbs": round(achieved_bw, 3),
                    "proposed_type": best_type,
                    "current_hr": cur_od,
                    "proposed_hr": best_od,
                    "monthly_savings": round(delta_month, 2),
                    "savings_pct": round(pct, 1),
                    "reason": (f"Target {best_type} provides {best_info['peak_bw_tbs']} TB/s BW (needed {achieved_bw:.2f} TB/s) "
                               f"at ${best_info['cost_per_gb_vram']:.4f}/GB-VRAM vs ${cur_info['cost_per_gb_vram']:.4f}/GB-VRAM"),
                })

    rightsize_monthly_savings = sum(r["monthly_savings"] for r in rightsize_recs)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n[Extension 2] GPU Catalog: $/GB-VRAM and Bandwidth Cost:")
        print(f"{'GPU':8}{'VRAM(GB)':>10}{'BW(TB/s)':>11}{'$/GPU-hr':>11}{'$/GB-VRAM-hr':>15}{'$/(TB/s)-hr':>13}")
        for g, info in sorted(vram_costs.items(), key=lambda x: x[1]["cost_per_gb_vram"]):
            print(f"{g:8}{info['hbm_gb']:>10.0f}{info['peak_bw_tbs']:>11.2f}${info['on_demand_hr']:>10.2f}"
                  f"${info['cost_per_gb_vram']:>14.4f}${info['cost_per_tb_bw']:>12.2f}")

        print("\n[Extension 2] MBU Right-Sizing Recommendations (Memory-Bound GPUs):")
        print(f"{'GPU ID':12}{'Current':9}{'MBU':>6}{'Achieved BW':>14}  {'Proposed':10}{'Cur $/h':>9}{'New $/h':>9}{'Saved/mo':>10}")
        for r in rightsize_recs:
            print(f"{r['gpu_id']:12}{r['current_type']:9}{r['mbu']:>6.3f}{r['achieved_bw_tbs']:>10.2f} TB/s  "
                  f"{r['proposed_type']:10}${r['current_hr']:>8.2f}${r['proposed_hr']:>8.2f}${r['monthly_savings']:>9.0f}")
        print(f"Total potential monthly right-sizing savings: ${rightsize_monthly_savings:,.0f}")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "rightsize_recs": rightsize_recs,
        "rightsize_monthly_savings": round(rightsize_monthly_savings, 2),
        "vram_costs": vram_costs,
    }


if __name__ == "__main__":
    run()
