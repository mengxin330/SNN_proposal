# mini G1

> Bilingual README — **English first**, 中文见下半部分。

---

# English

## 1. What this is

A reduced-scale experiment for the NSF Future CoRe proposal (SNN multiscale features → hardware cost). The goal: before committing a year to the full Gate G1, determine whether multiscale descriptors add **incremental** predictive power for event-driven queueing cost, after controlling for rate, burstiness, fanout, and static graph structure.

The proposal itself lists this as a pre-submission to-do:

> "Run the pilot decision-value experiment before submission and replace provisional targets with defensible preliminary evidence, **if available**."

## 2. Directory

| Path | Contents |
|---|---|
| `proposal/` | Proposal drafts (two English versions + Chinese translation) |
| `design/` | mini G1 design notes (CN/EN md, EN docx) + proposal gap review |
| `papers/` | 7 primary sources + `README.md` |
| `figures/` | Figures |
| `src/` | 8 modules |
| `scripts/` | 3 validation scripts |
| `tests/` | 67 unit tests |
| `results/` | JSON outputs and plots |

## 3. Status

| Item | State |
|---|---|
| Design | ✅ Done (8 gate criteria mapped one-to-one) |
| Single-queue simulator | ✅ Error vs M/D/1: 1.04 / 0.58 / 0.06 standard errors |
| Multi-hop simulator | ✅ 15/15 acceptance (incl. "degenerate case equals single queue exactly") |
| Four-family generators | ✅ 20/20 acceptance (incl. cross-family separability) |
| Unit tests | ✅ **67 passing** |
| C0–C3 features | ❌ Not written (`mfdfa.py`, `features.py`) |
| Dataset & training | ❌ Not written (`dataset.py`, `train_eval.py`) |
| Full comparison results | ❌ Not run |

**Toolchain verified; the final four-family × four-feature-class comparison is not.**

## 4. Modules

| File | Role |
|---|---|
| `src/config.py` | All parameters |
| `src/simulator.py` | Single-queue FIFO (verified against M/D/1) |
| `src/generators_poisson.py` | Poisson generator |
| `src/generators_multi.py` | Renewal / Hawkes / LRD + unified entry `generate()` |
| `src/topology.py` | 8×8 mesh, Hamiltonian labeling, routing function Rd, shortest paths |
| `src/multicast.py` | Unicast replication / dual-path message construction |
| `src/noc_sim.py` | Multi-hop queueing engine (discrete-event + heapq) |
| `src/analysis.py` | Percentiles, batch means, warm-up detection |

One test file per generator:
`generators_poisson.py` ↔ `tests/test_generators_poisson.py` (8 tests)
`generators_multi.py` ↔ `tests/test_generators_multi.py` (22 tests)

## 5. Validation scripts

| Script | Contents | Result |
|---|---|---|
| `scripts/step1_check.py` | Single queue vs M/D/1 (ρ = 0.5 / 0.7 / 0.85) | **15/15** |
| `scripts/step2_sim_check.py` | Multi-hop simulator, five sections | **15/15** |
| `scripts/step3_family_check.py` | Four-family generators, five sections | **20/20** |

```powershell
cd snn-proposal
D:\miniconda3\envs\snntorch\python.exe -m unittest discover -s tests
D:\miniconda3\envs\snntorch\python.exe scripts\step1_check.py
D:\miniconda3\envs\snntorch\python.exe scripts\step2_sim_check.py
D:\miniconda3\envs\snntorch\python.exe scripts\step3_family_check.py
```

## 6. Pipeline

```
generate traces → sample destinations (locality) → build messages (dual-path)
   → multi-hop queueing → compute C0–C3 → train predictors → leave-one-family-out
```

1. Four families generate event times with a shared mean-rate interface (`rate`).
2. Sample 2–3 destinations per event from $e^{-d^2/2\sigma^2}$, σ = 2.67.
3. dual-path: split by Hamiltonian label, emit at most two messages per event.
4. Per-hop queueing (16 VC × 8 flits, 3 cycles/hop) → delay / wait / occupancy.
5. Compute C0–C3 from the event stream (**not yet implemented**).
6. Train one predictor per feature class with matched budgets.
7. Leave-one-family-out; compare C3 vs C2 error and extraction cost.

### Target quantities

Only two targets are predicted:

| # | Target | Definition | Why |
|---|---|---|---|
| 1 | **tail queueing delay** | P95/P99 of waiting time (cycles) | Listed in the proposal; strips the fixed transmission part, giving the best spread (6.8x measured) |
| 2 | **peak buffer occupancy** | max queue length per (node, port), in flits | Listed in the proposal; measured spread 6 -> 27 flits (4.5x) |

The other four proposal targets are not predicted here:

| Target | Why not |
|---|---|
| link traversals | determined by path length, hence nearly deterministic; no spread across feature classes |
| latency (total) | about 95% is the fixed term 3*d; queueing is only about 5%, giving a spread of only 1.6x |
| memory accesses | requires hardware calibration |
| energy | requires hardware calibration |

The comparison itself is between feature classes: **C3 vs C2**, under leave-one-family-out.

## 7. Key parameters

| Item | Value | Source |
|---|---|---|
| Topology | 8×8 mesh | BookSim `k=8, n=2` |
| VCs × depth | **16 × 8 = 128 flits/port** | BookSim `num_vcs` / `vc_buf_size` |
| Per-hop latency | **3 cycles** | BookSim pipeline (0+1+1+1) |
| Message length L | **3 flits** | 1 end + 1 payload + 1 header |
| End-to-end delay | $3d + (L-1)$ | — |
| Connectivity | $\alpha e^{-d^2/2\sigma^2}$ | Ref. [1] Table 1 (σ = 10 on 30×30) |
| σ scaled to 8×8 | **2.67** | my scaling choice |
| Events per source | **10,000** | Matches Kantelhardt's N = 8192 |
| Load control | **λ_s = 0.02 / 0.035 / 0.05** | Measured (hotspot ρ ≤ 0.8) |
| Multicast | **dual-path** | Lin/McKinley/Ni; Fleury & Fraigniaud §3.2 |

## 8. Three traps found empirically

| # | Finding | Consequence |
|---|---|---|
| 1 | **Port load is highly uneven** (max/mean ≈ 2.8); mean ρ is not a usable control variable | Setting "mean ρ = 0.6" saturates instantly (delay 496, drop 6.5%) |
| 2 | **Hawkes intensity must be pulled back to μ**: $d\lambda=-\beta(\lambda-\mu)dt+\alpha\beta\,dN$ | Dropping the μ term sends λ→0, dt→∞, and the simulation stalls |
| 3 | **M/G/∞ gives $H=(3-\alpha)/2$ only asymptotically** | H = 0.8 yields a measured 0.668 at 10k events → use **measured H** as the label |

## 9. Known deviations (must be stated in any report)

| # | Deviation | Direction |
|---|---|---|
| 1 | Drop on full VC (real hardware uses credit-based backpressure) | Underestimates congestion spreading |
| 2 | Engine is **output-queued** (one FIFO per (node, port)), not BookSim's input-queued + iSLIP | Removes head-of-line blocking → **latency biased low** |
| 3 | Routing is dual-path (Hamiltonian labeling), not BookSim's `dor` | Cannot match BookSim curves number by number |
| 4 | 128 flits/port is large vs real hardware (8–64), though measured peak only reaches 27 | The 128 cap is never exercised |
| 5 | Synthetic traces only; no real SNN recordings | External validity untested |

## 10. Papers

See `papers/README.md`. Seven primary sources:

| File | Used for |
|---|---|
| `s41467-025-62202-1.pdf` | Proposal ref. [1]; σ, α, network parameters |
| `fleury1997-path-based-multicasting.pdf` | dual-path steps, all-port model, delay model, header encoding |
| `kantelhardt2002-mfdfa.pdf` | MFDFA definition; N = 8192 error level |
| `hawkes1971-self-exciting.pdf` | Intensity function, branching ratio, stability condition |
| `ogata1981-thinning.pdf` | Thinning (scanned, no text layer) |
| `cwi-davies-harte-method.pdf` | Davies–Harte four steps + fGn autocovariance |
| `gretsi-fbm-exact-synthesis.pdf` | Exact fBm synthesis (alternate reference) |

Not obtained: Lin/McKinley/Ni (1994, IEEE TPDS) and Davies & Harte (1987, Biometrika) — both closed access.

## 11. Next steps

| Step | Contents | Depends on | Estimate |
|---|---|---|---|
| 1 | `mfdfa.py` + validate on fBm with known H | — | 2–4 h |
| 2 | `features.py` (C0–C3, 19 features) | mfdfa | 2–4 h |
| 3 | `dataset.py` (assemble sample table) | all | 1–2 h |
| 4 | `train_eval.py` (32 models + leave-one-family-out) | dataset | 2–3 h |
| 5 | Full run (4 families × 50 runs × 10k events) | all | **~12 h compute** |

**Total ≈ 2–3 days** (development 7–13 h + compute 12 h + debugging 2–8 h).

Reduced run (4 × 20 × 5k): ≈ 2.5 h, but `h(q)` error rises from 0.03 to 0.04.

## Note

The `.txt` / `.pages.json` / `.transcript.json` files under `papers/` are transcription caches, not the primary sources.

---
---

# 中文

## 1. 这是什么

针对 NSF Future CoRe 提案（SNN 多尺度特征 → 硬件成本）的**缩小规模实验**。目的：在投入完整 Gate G1（约一年）之前，判断多尺度描述子在控制住 rate、突发性、fanout 与静态图结构之后，**是否还有**对事件驱动排队成本的增量预测能力。

proposal 自己把它列为提交前待办：

> "Run the pilot decision-value experiment before submission and replace provisional targets with defensible preliminary evidence, **if available**."

## 2. 目录

| 路径 | 内容 |
|---|---|
| `proposal/` | 提案各版本（英文原稿两版 + 中文译本） |
| `design/` | mini G1 设计文档（中/英 md、英文 docx）+ 提案审查报告 |
| `papers/` | 7 篇一手文献 + `README.md` |
| `figures/` | 图 |
| `src/` | 8 个模块 |
| `scripts/` | 3 个验证脚本 |
| `tests/` | 67 个单元测试 |
| `results/` | JSON 输出与图 |

## 3. 当前状态

| 项 | 状态 |
|---|---|
| 方案设计 | ✅ 完成（8 项判据逐项对应） |
| 单队列模拟器 | ✅ 对照 M/D/1 误差 1.04 / 0.58 / 0.06 个标准误 |
| 多跳模拟器 | ✅ 15/15 验收（含"退化情形完全等于单队列"） |
| 四族生成器 | ✅ 20/20 验收（含跨族区分度） |
| 单元测试 | ✅ **67 个全过** |
| C0–C3 特征 | ❌ 未写（`mfdfa.py`、`features.py`） |
| 数据表与训练 | ❌ 未写（`dataset.py`、`train_eval.py`） |
| 完整对比结果 | ❌ 未跑 |

**工具链已验证；"四族 × 四组特征"的最终对比还没有。**

## 4. 模块

| 文件 | 职责 |
|---|---|
| `src/config.py` | 全部参数 |
| `src/simulator.py` | 单队列 FIFO（已对照 M/D/1 验证） |
| `src/generators_poisson.py` | Poisson 生成器 |
| `src/generators_multi.py` | Renewal / Hawkes / LRD + 统一入口 `generate()` |
| `src/topology.py` | 8×8 mesh、哈密顿标号、路由函数 Rd、最短路 |
| `src/multicast.py` | 单播复制 / dual-path 消息构造 |
| `src/noc_sim.py` | 多跳排队引擎（离散事件 + heapq） |
| `src/analysis.py` | 分位数、批次均值、warm-up 检测 |

每个生成器正好一个测试文件：
`generators_poisson.py` ↔ `tests/test_generators_poisson.py`（8 个）
`generators_multi.py` ↔ `tests/test_generators_multi.py`（22 个）

## 5. 验证脚本

| 脚本 | 内容 | 结果 |
|---|---|---|
| `scripts/step1_check.py` | 单队列对照 M/D/1（ρ = 0.5 / 0.7 / 0.85） | **15/15** |
| `scripts/step2_sim_check.py` | 多跳模拟器五节验收 | **15/15** |
| `scripts/step3_family_check.py` | 四族生成器五节验收 | **20/20** |

```powershell
cd snn-proposal
D:\miniconda3\envs\snntorch\python.exe -m unittest discover -s tests
D:\miniconda3\envs\snntorch\python.exe scripts\step1_check.py
D:\miniconda3\envs\snntorch\python.exe scripts\step2_sim_check.py
D:\miniconda3\envs\snntorch\python.exe scripts\step3_family_check.py
```

## 6. 实验链条

```
生成事件流 → 采样目的（局部性）→ 构造消息（dual-path）
   → 多跳排队 → 算 C0–C3 → 训练预测器 → 留一族比误差
```

1. 四族生成事件时刻，共用"平均事件率"接口（`rate`）。
2. 按 $e^{-d^2/2\sigma^2}$ 采样 2–3 个目的（σ = 2.67）。
3. dual-path：按哈密顿标号分组，每个事件最多发两条消息。
4. 逐跳排队（16 VC × 8 flits，3 cycles/跳）→ 延迟 / 排队 / 占用。
5. 从事件流算 C0–C3（**尚未实现**）。
6. 每组特征同等预算训练一个预测器。
7. 留一族交叉验证，比较 C3 与 C2 的误差与提取耗时。

### 被预测的目标量

只预测两个：

| # | 目标量 | 定义 | 为什么选它 |
|---|---|---|---|
| 1 | **tail queueing delay** | 排队时间的 P95/P99（cycles） | proposal 明确列出；剥离固定传输成分后区分度最好（实测 6.8 倍） |
| 2 | **peak buffer occupancy** | 每个 (node, port) 队列长度的最大值（flits） | proposal 明确列出；实测区分度 6 -> 27 flits（4.5 倍） |

另外四个 proposal 目标量不在此预测：

| 目标量 | 为什么不做 |
|---|---|
| link traversals | 由路径长度决定，几乎确定；各特征组都能完美预测，无区分度 |
| latency（总） | 约 95% 是固定项 3*d，排队只占约 5%，区分度仅 1.6 倍 |
| memory accesses | 需要硬件标定 |
| energy | 需要硬件标定 |

对比本身发生在特征组之间：**C3 vs C2**，在留一族交叉验证下。

## 7. 关键参数

| 项 | 值 | 依据 |
|---|---|---|
| 拓扑 | 8×8 mesh | BookSim `k=8, n=2` |
| VC 数 × 深度 | **16 × 8 = 128 flits/端口** | BookSim `num_vcs` / `vc_buf_size` |
| 每跳延迟 | **3 cycles** | BookSim 流水线（0+1+1+1） |
| 消息长度 L | **3 flits** | 1 end + 1 payload + 1 header |
| 端到端延迟 | $3d + (L-1)$ | — |
| 连接模型 | $\alpha e^{-d^2/2\sigma^2}$ | 文献[1] Table 1（σ = 10 @ 30×30） |
| σ 缩放到 8×8 | **2.67** | 我的缩放选择 |
| 每源事件数 | **10,000** | 对齐 Kantelhardt 的 N = 8192 |
| 负载控制量 | **λ_s = 0.02 / 0.035 / 0.05** | 实测（热点 ρ ≤ 0.8） |
| 多播 | **dual-path** | Lin/McKinley/Ni；Fleury & Fraigniaud §3.2 |

## 8. 三个实测发现的坑

| # | 发现 | 后果 |
|---|---|---|
| 1 | **端口负载极不均**（max/mean ≈ 2.8），平均 ρ 不能当控制变量 | 设"平均 ρ = 0.6"会立刻饱和（延迟 496、丢弃 6.5%） |
| 2 | **Hawkes 的强度必须向 μ 回拉**：$d\lambda=-\beta(\lambda-\mu)dt+\alpha\beta\,dN$ | 漏掉 μ 项会让 λ→0、dt→∞、模拟卡死 |
| 3 | **M/G/∞ 的 $H=(3-\alpha)/2$ 只是渐近关系** | 设 H = 0.8 在 10000 事件下只测到 0.668 → 用**实测 H** 当标签 |

## 9. 已知偏离（任何报告都必须写明）

| # | 偏离 | 方向 |
|---|---|---|
| 1 | VC 满丢弃（真实硬件用信用流控背压） | 低估拥塞扩散 |
| 2 | 引擎是 **output-queued**（每 (node, port) 一条 FIFO），不是 BookSim 的 input-queued + iSLIP | 消除队头阻塞 → **延迟偏低** |
| 3 | 路由用 dual-path（哈密顿标号），不是 BookSim 的 `dor` | 无法与 BookSim 曲线逐个对数 |
| 4 | 128 flits/端口相对真实硬件（8–64）偏大，但实测峰值只到 27 | 128 这个上限从未被用到 |
| 5 | 只用合成事件流，不含真实 SNN 迹 | 外部效度未验证 |

## 10. 文献

见 `papers/README.md`。7 篇一手文献：

| 文件 | 用于 |
|---|---|
| `s41467-025-62202-1.pdf` | 提案[1]；σ、α、网络参数 |
| `fleury1997-path-based-multicasting.pdf` | dual-path 五步、all-port、延迟模型、header 编码 |
| `kantelhardt2002-mfdfa.pdf` | MFDFA 定义；N = 8192 的误差水平 |
| `hawkes1971-self-exciting.pdf` | 强度函数、分支比、稳定条件 |
| `ogata1981-thinning.pdf` | thinning（扫描件，无文本层） |
| `cwi-davies-harte-method.pdf` | Davies–Harte 四步 + fGn 自协方差 |
| `gretsi-fbm-exact-synthesis.pdf` | fBm 精确合成（备选参考） |

未获取全文：Lin/McKinley/Ni（1994, IEEE TPDS）、Davies & Harte（1987, Biometrika）——均闭源。

## 11. 下一步

| 步 | 内容 | 依赖 | 估计 |
|---|---|---|---|
| 1 | `mfdfa.py` + 用已知 H 的 fBm 验证 | — | 2–4 h |
| 2 | `features.py`（C0–C3，19 个特征） | mfdfa | 2–4 h |
| 3 | `dataset.py`（组装样本表） | 全部 | 1–2 h |
| 4 | `train_eval.py`（32 模型 + 留一族） | dataset | 2–3 h |
| 5 | 完整跑（4 族 × 50 次 × 每源 10000 事件） | 全部 | **约 12 h 纯计算** |

**总计约 2–3 天**（开发 7–13 h + 跑 12 h + 调试 2–8 h）。

缩减版（4 族 × 20 次 × 每源 5000）：约 2.5 h，但 `h(q)` 误差从 0.03 升到 0.04。

## 注意

`papers/` 下的 `.txt` / `.pages.json` / `.transcript.json` 是转录缓存，不是原始文献。
