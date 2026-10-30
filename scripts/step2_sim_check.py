"""
多跳模拟器验收：五节检查。

用法（在 snn-proposal/ 目录下）：
    D:\\miniconda3\\envs\\snntorch\\python.exe scripts\\step2_sim_check.py

§1 手工案例     零竞争下的延迟公式  delay = 3d + (L-1)
§2 退化测试     k=1 退化为单播；单链路退化为已验证的单队列
§3 账目平衡     消息数 / 接受+丢弃 / 链路穿越 / 延迟分解
§4 单调性       rho = 0.2 / 0.6 / 0.8 下延迟与丢弃率
§5 合理性       零负载截距、热点系数
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from config import CONFIG
from topology import Mesh
from multicast import build_messages
from noc_sim import simulate_network, msg_len_flits
from simulator import simulate


PASS, FAIL = [], []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    (PASS if ok else FAIL).append(name)
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------- §1

def section1():
    print("\n=== §1  手工案例（零竞争延迟 = 3d + (L-1)）===")
    hop = CONFIG["hops_cycles"]
    L = msg_len_flits(1, CONFIG["n_end"], CONFIG["n_payload"])

    class OnePort:
        def out_port(self, u, v):
            return 0

    for d in (1, 2, 5):
        msgs = [{"t": 0.0, "src": 0, "path": list(range(d + 1)), "header": [0]}]
        r = simulate_network(msgs, d + 1, OnePort(), n_ports=1,
                             port_capacity=None, hops_cycles=hop,
                             n_end=CONFIG["n_end"], n_payload=CONFIG["n_payload"])
        got = r["per_message"][0]["delay"]
        want = hop * d + (L - 1)
        check(f"{d} 跳延迟 = {want}", abs(got - want) < 1e-9, f"got {got}")

    # 零竞争时等待为 0
    msgs = [{"t": float(t), "src": 0, "path": [0, 1], "header": [1]}
            for t in (0, 100, 200)]
    r = simulate_network(msgs, 2, OnePort(), n_ports=1, port_capacity=None,
                         hops_cycles=hop)
    check("稀疏到达无等待", all(abs(m["wait"]) < 1e-9 for m in r["per_message"]))


# ---------------------------------------------------------------- §2

def section2():
    print("\n=== §2  退化测试 ===")
    topo = Mesh(CONFIG["mesh_n"])
    hop = CONFIG["hops_cycles"]
    L = msg_len_flits(1, CONFIG["n_end"], CONFIG["n_payload"])

    # (a) 单链路 ≡ 单队列（所有消息都走 0 -> 1 的东向端口）
    rng = np.random.default_rng(11)
    arrivals = np.cumsum(rng.exponential(10.0, 3000))
    msgs = [{"t": float(t), "src": 0, "path": [0, 1], "header": [1]} for t in arrivals]
    r = simulate_network(msgs, 2, topo, n_ports=CONFIG["n_output_ports"],
                         port_capacity=None, hops_cycles=hop)
    w_noc = np.array([m["wait"] for m in r["per_message"]])
    sq = simulate(arrivals, service_time=float(L), capacity=None)
    w_sq = np.asarray(sq["waits"], dtype=float)
    check("单链路 wait ≡ 单队列 wait", np.allclose(w_noc, w_sq, atol=1e-9),
          f"max|diff| = {np.max(np.abs(w_noc - w_sq)):.2e}")

    # (b) k=1 时 dual-path 只有 1 条消息，且等于单播
    src = topo.idx(3, 3)
    dst = topo.idx(6, 5)
    m_dual = build_messages(0.0, src, [dst], topo, mode="dual_path")
    m_uni = build_messages(0.0, src, [dst], topo, mode="unicast")
    check("k=1 时 dual-path 消息数 = 1", len(m_dual) == 1, f"len={len(m_dual)}")
    check("k=1 时 dual-path ≡ 单播",
          len(m_dual) == len(m_uni) and m_dual[0]["path"] == m_uni[0]["path"])


# ---------------------------------------------------------------- §3

def section3():
    print("\n=== §3  账目平衡 ===")
    topo = Mesh(CONFIG["mesh_n"])
    rng = np.random.default_rng(12)
    n_src = 16
    per_src = 40
    msgs = []
    for s in range(n_src):
        src = rng.integers(0, topo.n_nodes)
        for t in np.sort(rng.uniform(0, 1000, per_src)):
            dst = rng.integers(0, topo.n_nodes)
            msgs.extend(build_messages(float(t), int(src), [int(dst)],
                                       topo, mode="dual_path"))
    r = simulate_network(msgs, topo.n_nodes, topo, n_ports=CONFIG["n_output_ports"],
                         port_capacity=CONFIG["vc_count"] * CONFIG["vc_depth"],
                         hops_cycles=CONFIG["hops_cycles"])
    n_acc = sum(1 for m in r["per_message"] if not m["dropped"])
    check("接受 + 丢弃 = 总数", n_acc + r["n_dropped"] == len(msgs),
          f"{n_acc} + {r['n_dropped']} = {len(msgs)}")

    want_hops = sum(len(m["path"]) - 1 for m, res in zip(msgs, r["per_message"])
                    if not res["dropped"])
    check("链路穿越 = Σ 路径长度", r["total_hops"] == want_hops,
          f"{r['total_hops']} vs {want_hops}")

    # 延迟分解：delay = wait + 3*hops + (L-1)
    ok = True
    for m, res in zip(msgs, r["per_message"]):
        if res["dropped"]:
            continue
        L = msg_len_flits(len(m["header"]), CONFIG["n_end"], CONFIG["n_payload"])
        expect = res["wait"] + CONFIG["hops_cycles"] * res["hops"] + (L - 1)
        if abs(expect - res["delay"]) > 1e-9:
            ok = False
            break
    check("delay = wait + 3·hops + (L-1)", ok)


# ---------------------------------------------------------------- §4

def port_demand(topo, msgs, n_ports):
    """离线统计每个 (node, port) 的 flit-需求（不含时间维度）。"""
    load = np.zeros((topo.n_nodes, n_ports))
    for m in msgs:
        L = msg_len_flits(len(m["header"]), CONFIG["n_end"], CONFIG["n_payload"])
        p = m["path"]
        for a, b in zip(p[:-1], p[1:]):
            load[a][topo.out_port(a, b)] += L
    return load


def make_scenario(topo, lambda_s, events_per_source, seed=0):
    """
    构造 64 源的事件 + dual-path 消息。

    lambda_s : 每个源的事件率（事件 / cycle）。
               这是直接的控制变量——比"目标 rho"更可靠，因为多跳网络的
               端口负载是**分布**（实测 max/mean = 3.33），用一个平均 rho
               无法确定工作点。
    """
    rng = np.random.default_rng(seed)
    n = topo.n_nodes
    side = CONFIG["mesh_n"]
    sigma = CONFIG["sigma_ref"] * side / CONFIG["mesh_n_ref"]

    def sample_dests(src, k):
        cx, cy = topo.coords(src)
        allv = np.arange(n)
        xs, ys = allv % side, allv // side
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        w = np.exp(-d2 / (2 * sigma ** 2))
        w[src] = 0.0
        w = w / w.sum()
        k = min(int(k), n - 1)
        return rng.choice(allv, size=k, replace=False, p=w).tolist()

    fanout = 2.56
    msgs = []
    for src in range(n):
        ts = np.cumsum(rng.exponential(1.0 / lambda_s, events_per_source))
        for t in ts:
            k = 2 if rng.random() < (1 - (fanout - 2)) else 3
            dsts = sample_dests(src, k)
            msgs.extend(build_messages(float(t), src, dsts, topo,
                                       mode=CONFIG["multicast_mode"]))
    T = float(max(m["t"] for m in msgs)) if msgs else 1.0
    return msgs, T


def section4():
    print("\n=== §4  负载敏感性（每源事件率 lambda_s）===")
    topo = Mesh(CONFIG["mesh_n"])
    cap = CONFIG["vc_count"] * CONFIG["vc_depth"]
    lambdas = CONFIG.get("lambda_s_targets", [0.10, 0.25, 0.40])
    rows = []
    for lam in lambdas:
        msgs, T = make_scenario(topo, lam, events_per_source=200,
                                seed=int(lam * 1000))
        load = port_demand(topo, msgs, CONFIG["n_output_ports"])
        util = load / T                       # 每端口利用率
        r = simulate_network(msgs, topo.n_nodes, topo,
                             n_ports=CONFIG["n_output_ports"],
                             port_capacity=cap,
                             hops_cycles=CONFIG["hops_cycles"],
                             record_occupancy=True)
        delays = np.array([m["delay"] for m in r["per_message"] if not m["dropped"]])
        waits = np.array([m["wait"] for m in r["per_message"] if not m["dropped"]])
        rows.append({
            "lambda_s": lam,
            "T": T,
            "n_msg": len(msgs),
            "util_mean": float(util.mean()),
            "util_max": float(util.max()),
            "mean_delay": float(delays.mean()) if len(delays) else float("nan"),
            "p95_delay": float(np.percentile(delays, 95)) if len(delays) else float("nan"),
            "mean_wait": float(waits.mean()) if len(waits) else float("nan"),
            "drop_rate": r["n_dropped"] / len(msgs) if msgs else 0.0,
            "peak_occ": r["peak_occ"],
        })
        print(f"    lambda_s={lam:.2f}  T={T:8.0f}  util mean/max={util.mean():.3f}/{util.max():.3f}  "
              f"delay={rows[-1]['mean_delay']:7.2f}  wait={rows[-1]['mean_wait']:7.2f}  "
              f"drop={rows[-1]['drop_rate']*100:5.2f}%  peak={rows[-1]['peak_occ']}")

    inc = all(rows[i]["mean_delay"] <= rows[i + 1]["mean_delay"]
              for i in range(len(rows) - 1))
    check("平均延迟随负载单调不减", inc,
          "  ".join(f"{r['mean_delay']:.2f}" for r in rows))
    check("最低负载下几乎无排队", rows[0]["mean_wait"] < 5.0,
          f"wait={rows[0]['mean_wait']:.2f}")
    check("最低负载下无丢弃", rows[0]["drop_rate"] < 0.001,
          f"drop={rows[0]['drop_rate']*100:.3f}%")
    return rows


# ---------------------------------------------------------------- §5

def section5():
    print("\n=== §5  合理性 ===")
    topo = Mesh(CONFIG["mesh_n"])

    # (a) 零负载截距
    msgs, T = make_scenario(topo, 0.01, events_per_source=100, seed=99)
    r = simulate_network(msgs, topo.n_nodes, topo,
                         n_ports=CONFIG["n_output_ports"],
                         port_capacity=CONFIG["vc_count"] * CONFIG["vc_depth"],
                         hops_cycles=CONFIG["hops_cycles"])
    delays = np.array([m["delay"] for m in r["per_message"] if not m["dropped"]])
    mean_hops = np.mean([m["hops"] for m in r["per_message"] if not m["dropped"]])
    L = msg_len_flits(2, CONFIG["n_end"], CONFIG["n_payload"])   # 平均 header≈2
    predicted = CONFIG["hops_cycles"] * mean_hops + (L - 1)
    rel = abs(delays.mean() - predicted) / predicted
    check("零负载延迟 ≈ 3·hops + (L-1)", rel < 0.05,
          f"measured {delays.mean():.3f} vs predicted {predicted:.3f} (rel {rel*100:.1f}%)")

    # (b) 热点系数
    load = np.zeros(topo.n_nodes)
    rng = np.random.default_rng(5)
    for _ in range(4000):
        s = int(rng.integers(0, topo.n_nodes))
        d = int(rng.integers(0, topo.n_nodes))
        if s == d:
            continue
        p = topo.path_rd(s, d) or topo.path_manhattan(s, d)
        for v in p:
            load[v] += 1
    ratio = load.max() / load.mean()
    check("热点系数在 1.2–1.6 之间", 1.2 <= ratio <= 1.6, f"max/mean = {ratio:.3f}")
    return {"mean_hops": float(mean_hops), "zero_load_delay": float(delays.mean()),
            "hotspot_ratio": float(ratio)}


# ---------------------------------------------------------------- main

def main():
    print("=" * 78)
    print("Step 2  多跳模拟器验收")
    print("=" * 78)
    print(f"mesh={CONFIG['mesh_n']}x{CONFIG['mesh_n']}  "
          f"vc={CONFIG['vc_count']}x{CONFIG['vc_depth']}  "
          f"hops={CONFIG['hops_cycles']} cycles  "
          f"multicast={CONFIG['multicast_mode']}")

    section1()
    section2()
    section3()
    rows = section4()
    extra = section5()

    outdir = os.path.join(PROJ, CONFIG["results_dir"])
    os.makedirs(outdir, exist_ok=True)
    payload = {"config": CONFIG, "section4": rows, "section5": extra,
               "passed": PASS, "failed": FAIL}
    path = os.path.join(outdir, "step2_sim_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {path}")

    print("\n" + "=" * 78)
    print(f"PASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("FAILED:")
        for f_ in FAIL:
            print("  -", f_)
    print("=" * 78)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
