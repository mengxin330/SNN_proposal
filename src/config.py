"""
第一步实验的全部参数。

改参数只改这个文件。
"""

CONFIG = {
    # ==================== 数据生成 ====================
    "rate": 1.0,              # 到达率（事件 / ms）
    "n_events": 400_000,      # 事件数
    "seed": 0,                # 随机种子

    # ==================== 服务时间 ====================
    # 利用率 rho = rate * service_time
    # 0.50 -> rho=0.50, 0.70 -> rho=0.70, 0.85 -> rho=0.85
    "service_times": [0.50, 0.70, 0.85],

    # ==================== 缓冲 ====================
    # 验证组用无限缓冲（这样 M/D/1 公式才成立）
    # 观察组用有限缓冲（看丢弃与峰值的真实行为）
    "capacity_validate": None,
    "capacity_observe": 32,

    # ==================== warm-up 检测 ====================
    "warmup_window": 1000,      # 滚动窗口长度（事件数）
    "warmup_tol": 0.05,         # 平稳判据：尾部窗口均值的极差 < 5%
    "warmup_min_windows": 10,   # 稳态段至少保留的窗口数
    "warmup_max_frac": 0.50,    # warm-up 不得超过总样本的比例

    # ==================== 批次均值法 ====================
    "n_batches": 10,            # 把稳态段分成几批

    # ==================== 判据 ====================
    "max_n_se": 2.0,            # 误差 < 2 个标准误视为通过

    # ==================== 输出 ====================
    "results_dir": "results",
    "make_plots": True,         # 是否画图

    # ==================== 多跳 NoC（对齐 BookSim） ====================
    # 拓扑与路由器
    "mesh_n": 8,                 # 8x8 mesh
    "vc_count": 16,              # BookSim num_vcs
    "vc_depth": 8,               # BookSim vc_buf_size
    "hops_cycles": 3,            # BookSim: routing 0 + vc_alloc 1 + sw_alloc 1 + st 1
    "n_output_ports": 5,         # 北 南 东 西 本地

    # 消息长度（flits）
    "n_end": 1,                  # end-of-message 信号
    "n_payload": 1,              # 数据本身（一个事件）
    "n_header_max": 3,           # 单条 dual-path 消息的 header 地址数上限

    # 负载（三档，做敏感性检查）
    # 注意：多跳网络的端口负载是"分布"（实测 max/mean ≈ 2.8），
    # 一个平均 rho 无法确定工作点，所以直接用"每源事件率"作控制变量。
    # 实测 λ=0.10 时平均 util 已 0.54、热点 1.50，所以三档取更低的值，
    # 使热点利用率约为 0.30 / 0.53 / 0.75。
    "lambda_s_targets": [0.02, 0.035, 0.05],   # 事件 / cycle / 源
    "rho_targets": [0.2, 0.6, 0.8],            # 保留备查

    # 每源事件数
    "events_per_source": 10_000,

    # 连接模型（论文 [1] Table 1）
    "sigma_ref": 10.0,           # 30x30 上的 sigma（归一化单位）
    "mesh_n_ref": 30,
    "alpha_ee": 0.11,            # alpha_{E->E} 中档

    # 多播策略："unicast" | "dual_path"
    "multicast_mode": "dual_path",

    # 是否把到达时刻量化到整数 cycle
    "quantize_arrivals": False,
}
