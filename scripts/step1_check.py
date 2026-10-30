"""
第一步：单队列 + Poisson + M/D/1 对照。

用法（在 snn-proposal/ 目录下）：
    D:\miniconda3\envs\snntorch\python.exe scripts\step1_check.py

做两件事：
    Part A  无限缓冲，三组 rho，对照 M/D/1 解析解（验证模拟器）
    Part B  有限缓冲（容量 16），观察丢弃率与峰值（为后续定参数）

输出：
    results/step1_results.json
    results/step1_warmup.png      滚动平均等待曲线 + warm-up 端点
    results/step1_partB.png       丢弃率与峰值
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from config import CONFIG
from generators_poisson import generate_poisson
from simulator import simulate
from analysis import find_warmup, batch_means_se, percentile, rolling_mean


# ---------------------------------------------------------------- 理论值

def md1_mean_wait(rho, service_time):
    """M/D/1 的平均等待时间（Pollaczek-Khinchine）。
    仅当 rho < 1 且缓冲无限时成立。"""
    return rho * service_time / (2.0 * (1.0 - rho))


# ---------------------------------------------------------------- Part A

def part_a(arrivals):
    """无限缓冲：对照 M/D/1。返回 (标量结果列表, 各组的等待序列)。"""
    print("\n" + "=" * 88)
    print("Part A  infinite buffer, compare with M/D/1 analytic solution")
    print("=" * 88)

    rows = []
    waits_list = []

    for s in CONFIG["service_times"]:
        rho = CONFIG["rate"] * s
        theory = md1_mean_wait(rho, s)

        res = simulate(arrivals, service_time=s, capacity=CONFIG["capacity_validate"])
        waits = np.asarray(res["waits"], dtype=float)

        K = find_warmup(
            waits,
            window=CONFIG["warmup_window"],
            tol=CONFIG["warmup_tol"],
            min_windows=CONFIG["warmup_min_windows"],
            max_frac=CONFIG["warmup_max_frac"],
        )
        waits_ss = waits[K:]
        mean_w = float(waits_ss.mean()) if len(waits_ss) else 0.0
        se = batch_means_se(waits_ss, n_batches=CONFIG["n_batches"])

        error = mean_w - theory
        n_se = abs(error) / se if se > 0 else float("inf")

        rows.append({
            "service_time": float(s),
            "rho": float(rho),
            "theory": float(theory),
            "simulated": mean_w,
            "se": float(se),
            "error": float(error),
            "n_se": float(n_se),
            "warmup_K": int(K),
            "n_effective": int(len(waits_ss)),
            "dropped": int(res["dropped"]),
            "peak": int(res["peak"]),
        })
        waits_list.append(waits)

    header = (f"{'s(ms)':>7} {'rho':>6} {'theory':>9} {'simulated':>10} "
              f"{'SE':>9} {'error':>9} {'err/SE':>8} {'warmup':>7} "
              f"{'n_eff':>8}  verdict")
    print(header)
    print("-" * len(header))
    for r in rows:
        verdict = "OK" if r["n_se"] <= CONFIG["max_n_se"] else "FAIL"
        print(f"{r['service_time']:>7.2f} {r['rho']:>6.2f} {r['theory']:>9.4f} "
              f"{r['simulated']:>10.4f} {r['se']:>9.5f} {r['error']:>+9.4f} "
              f"{r['n_se']:>8.2f} {r['warmup_K']:>7d} {r['n_effective']:>8d}  {verdict}")

    # 单调性检查（用归一化等待 Wq/s）
    norm = [r["simulated"] / r["service_time"] for r in rows]
    mono = all(norm[i] < norm[i + 1] for i in range(len(norm) - 1))
    print(f"\nmonotonicity of (mean_wait / service_time): "
          f"{'PASS' if mono else 'FAIL'}  {[round(v, 3) for v in norm]}")

    return rows, waits_list


# ---------------------------------------------------------------- Part B

def part_b(arrivals):
    """有限缓冲：观察丢弃率与峰值。"""
    print("\n" + "=" * 88)
    print(f"Part B  finite buffer (capacity={CONFIG['capacity_observe']}), observe behaviour")
    print("=" * 88)

    rows = []
    for s in CONFIG["service_times"]:
        rho = CONFIG["rate"] * s
        res = simulate(arrivals, service_time=s, capacity=CONFIG["capacity_observe"])
        waits = np.asarray(res["waits"], dtype=float)

        rows.append({
            "service_time": float(s),
            "rho": float(rho),
            "drop_rate": float(res["dropped"] / len(arrivals)),
            "peak": int(res["peak"]),
            "p50_occ": percentile(res["occupancy"], 50),
            "p95_occ": percentile(res["occupancy"], 95),
            "p99_occ": percentile(res["occupancy"], 99),
            "p95": percentile(waits, 95),
            "p99": percentile(waits, 99),
            "mean_wait": float(waits.mean()) if len(waits) else 0.0,
            "n_accepted": int(res["n_accepted"]),
        })

    header = (f"{'s(ms)':>7} {'rho':>6} {'drop_rate':>10} {'peak':>6} "
              f"{'P50occ':>7} {'P95occ':>7} {'P99occ':>7} "
              f"{'P95wait':>8} {'P99wait':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['service_time']:>7.2f} {r['rho']:>6.2f} "
              f"{r['drop_rate']*100:>9.2f}% {r['peak']:>6d} "
              f"{r['p50_occ']:>7.1f} {r['p95_occ']:>7.1f} {r['p99_occ']:>7.1f} "
              f"{r['p95']:>8.3f} {r['p99']:>8.3f}")

    return rows


# ---------------------------------------------------------------- 作图

def plot_part_a(rows, waits_list, outpath):
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.1 * n))
    if n == 1:
        axes = [axes]

    w = CONFIG["warmup_window"]
    for ax, r, waits in zip(axes, rows, waits_list):
        rm = rolling_mean(waits, w)
        x = np.arange(len(rm)) + w // 2

        ax.plot(x, rm, lw=0.9, color="tab:blue", label="rolling mean wait")
        ax.axhline(r["theory"], color="tab:red", ls="--", lw=1.3,
                   label=f"theory  {r['theory']:.3f} ms")
        ax.axhline(r["simulated"], color="tab:green", ls=":", lw=1.3,
                   label=f"steady   {r['simulated']:.3f} ms")
        ax.axvline(r["warmup_K"], color="gray", ls="-.", lw=1.0,
                   label=f"warm-up end  K={r['warmup_K']}")

        ax.set_ylabel("mean wait (ms)")
        ax.set_title(f"service_time = {r['service_time']:.2f} ms,  "
                     f"rho = {r['rho']:.2f},  err/SE = {r['n_se']:.2f}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("event index")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved: {outpath}")


def plot_part_b(rows, outpath):
    labels = [f"s={r['service_time']:.2f}\nrho={r['rho']:.2f}" for r in rows]
    drop = [r["drop_rate"] * 100.0 for r in rows]
    peak = [r["peak"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax1.bar(labels, drop, color="tab:orange")
    ax1.set_ylabel("drop rate (%)")
    ax1.set_title(f"drop rate (capacity = {CONFIG['capacity_observe']})")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(labels, peak, color="tab:purple")
    ax2.axhline(CONFIG["capacity_observe"], color="red", ls="--", lw=1.2,
                label="capacity")
    ax2.set_ylabel("peak occupancy (packets)")
    ax2.set_title("peak occupancy")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved: {outpath}")


# ---------------------------------------------------------------- 主流程

def main():
    print("=" * 88)
    print("Step 1  single-queue sanity check")
    print("=" * 88)
    print(f"rate = {CONFIG['rate']} events/ms, "
          f"n_events = {CONFIG['n_events']:,}, seed = {CONFIG['seed']}")

    arrivals = generate_poisson(CONFIG["rate"], CONFIG["n_events"], CONFIG["seed"])
    print(f"generated {len(arrivals):,} events, span = {arrivals[-1]:.1f} ms")

    rows_a, waits_a = part_a(arrivals)
    rows_b = part_b(arrivals)

    outdir = os.path.join(PROJ, CONFIG["results_dir"])
    os.makedirs(outdir, exist_ok=True)

    payload = {
        "config": {k: v for k, v in CONFIG.items() if k != "make_plots"},
        "part_a": rows_a,
        "part_b": rows_b,
    }
    jpath = os.path.join(outdir, "step1_results.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {jpath}")

    if CONFIG["make_plots"]:
        plot_part_a(rows_a, waits_a, os.path.join(outdir, "step1_warmup.png"))
        plot_part_b(rows_b, os.path.join(outdir, "step1_partB.png"))

    all_pass = all(r["n_se"] <= CONFIG["max_n_se"] for r in rows_a)
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    print("=" * 88)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
