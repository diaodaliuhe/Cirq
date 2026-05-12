from __future__ import annotations
from typing import List, Tuple, Sequence, Dict, Callable, Optional, Any
import secrets
from dataclasses import dataclass
from scipy.optimize import curve_fit
import numpy as np
import cirq

from cirq.noise.utils.metrics import fidelity
from cirq.noise.utils.noise_builder import make_noise_model
from cirq.noise.utils.compilation_scheme import all_in_one_compile, rb_1q_weak_compile_blockwise

def _survival_prob_from_dm(dm: np.ndarray) -> float:
    """
        默认初态为0态，dm为最终密度矩阵。
    """
    return float(np.real(dm[0, 0]))

@dataclass
class FitResult:
    """
        RB 衰减拟合结果。
    """
    model: str
    params: Dict[str, float]
    param_stderr: Dict[str, Optional[float]]
    F_avg: Optional[float] = None
    EPC: Optional[float] = None
    F_avg_stderr: Optional[float] = None
    EPC_stderr: Optional[float] = None
    yhat: Optional[np.ndarray] = None
    pcov: Optional[np.ndarray] = None

def fit_rb_decay(
    m_list: List[int],
    y_list: List[float],
    d: int,
    fit_curve: str = "exponential",
    y_std: Optional[List[float]] = None,
    shots: Optional[int] = None,
    n_seq_per_m: Optional[int] = None,
    lock_B: bool = True,
    B_window: Optional[float] = None,
    add_m0: bool = True,
) -> FitResult:
    """
    拟合 RB 衰减函数: y(m) = A * p^m + B
    返回 p,A,B 及其标准差，并计算 F_avg 与 EPC 及误差。

    Args:
        m_list: RB 深度列表
        y_list: 对应的生存概率均值列表
        d:      系统维度 = 2**n_qubits
        y_std:  每个 y 的标准差（可选）
        shots:  每个序列的测量次数（可选，若传了则自动计算权重）
        n_seq_per_m: 每个 m 的序列数（可选，若传了则自动计算权重）
        lock_B: 是否锁定 B = 1/d（默认 True）
        B_window: 若放开 B，则给 B 一个有限 window（默认 None，即 ± 0.02）
        add_m0: 是否把 m = 0, y=  1 也纳入拟合（默认 True）
    """
    # 组装数据
    m = np.asarray(m_list, float)
    y = np.asarray(y_list, float)

    if add_m0 and (len(m) == 0 or m[0] != 0):
        m = np.concatenate([[0.0], m])
        y = np.concatenate([[1.0], y])

    # 权重
    if y_std is not None:
        sigma = np.asarray(y_std, float)
        if add_m0:
            sigma = np.concatenate([[1e-6], sigma])  # m=0 近似无噪，给极小权重
    elif shots is not None:
        n_eff = int(n_seq_per_m) if n_seq_per_m is not None else 1
        sigma = np.sqrt(np.maximum(y * (1 - y), 1e-9) / max(shots * n_eff, 1))
        if add_m0:
            sigma = np.concatenate([[1e-6], sigma])
    else:
        sigma = None

    # 模型
    B_star = 1.0 / d
    bounds = None

    if fit_curve == "exponential":
        def model(m, A, p, B): return A * p**m + B
        A0, p0 = float(y.max() - y.min()), 0.99
        if lock_B:
            def model_fixB(m, A, p): return A * p**m + B_star
            popt, pcov = curve_fit(model_fixB, m, y, p0=[A0, p0], bounds=([0, 0], [1, 1]),
            sigma=sigma, absolute_sigma=(sigma is not None),
            maxfev=20000)
            A, p = popt; B = B_star
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [None, None]
            params = {"A": A, "p": p, "B": B}
            param_stderr = {"A": perr[0], "p": perr[1], "B": None}
        else:
            if B_window is None:
                B_lo, B_hi = max(0.0, B_star - 0.02), min(1.0, B_star + 0.02)
            else:
                B_lo, B_hi = max(0.0, B_star - float(B_window)), min(1.0, B_star + float(B_window))
            bounds = ([0.0, 0.0, B_lo], [1.0, 1.0, B_hi])
            popt, pcov = curve_fit(model, m, y, p0=[A0, 0.99, B_star], bounds=bounds,
            sigma=sigma, absolute_sigma=(sigma is not None),
            maxfev=20000)
            A, p, B = popt
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [None]*3
            params = {"A": A, "p": p, "B": B}
            param_stderr = dict(zip(["A", "p", "B"], perr))


    elif fit_curve == "double_exponential":
        def model(m, A1, p1, A2, p2, B): return A1 * p1**m + A2 * p2**m + B
        popt, pcov = curve_fit(model, m, y, p0=[0.4, 0.99, 0.4, 0.9, B_star],
        bounds=([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
        sigma=sigma, absolute_sigma=(sigma is not None),
        maxfev=20000)
        A1, p1, A2, p2, B = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [None]*5
        params = {"A1": A1, "p1": p1, "A2": A2, "p2": p2, "B": B}
        param_stderr = dict(zip(params.keys(), perr))


    elif fit_curve == "stretched_exponential":
        def model(m, A, p, beta, B): return A * np.exp(-(1 - p) * m**beta) + B
        popt, pcov = curve_fit(model, m, y, p0=[0.8, 0.99, 0.8, B_star],
        bounds=([0, 0, 0.1, 0], [1, 1, 2.0, 1]),
        sigma=sigma, absolute_sigma=(sigma is not None),
        maxfev=20000)
        A, p, beta, B = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [None]*4
        params = {"A": A, "p": p, "beta": beta, "B": B}
        param_stderr = dict(zip(params.keys(), perr))


    # 计算 F_avg 和 EPC 若适用（只对单指数 p 有意义）
    if "p" in params:
        p = params["p"]
        F_avg = (d - 1) / d * p + 1.0 / d
        EPC = 1.0 - F_avg
        p_stderr = param_stderr.get("p")
        F_se = ((d - 1) / d) * p_stderr if p_stderr is not None else None
    else:
        F_avg = EPC = F_se = None


    return FitResult(
        model=fit_curve,
        params=params,
        param_stderr=param_stderr,
        F_avg=F_avg,
        EPC=EPC,
        F_avg_stderr=F_se,
        EPC_stderr=F_se,
    )

def get_1q_clifford_library() -> Tuple[np.ndarray, List[cirq.Gate]]:
    """返回 (matrices[24,2,2], gates[24])，供快速采样与查找反演使用。"""
    all_c = list(cirq.SingleQubitCliffordGate.all_single_qubit_cliffords)
    mats = np.array([cirq.unitary(g) for g in all_c])
    gates = [cirq.PhasedXZGate.from_matrix(m) for m in mats]
    return mats, gates

GateSpec = Dict[str, Any]

def clifford_1q_gate_set(base_prob: float = 1.0, include_idle: bool = False) -> List[GateSpec]:
    """
    返回形如 [{"name", "arity", "prob", "sampler"}, ...] 的单比特 Clifford gate set。
    """
    # 取 24 个单比特 Clifford，并用 PhasedXZ 实例化为可 on(q) 的 Gate
    all_c = list(cirq.SingleQubitCliffordGate.all_single_qubit_cliffords)
    gates = [cirq.PhasedXZGate.from_matrix(cirq.unitary(g)) for g in all_c]

    specs: List[GateSpec] = []
    for i, g in enumerate(gates):
        specs.append({
            "name": f"C1_{i:02d}",     # 你也可以换成更有语义的命名
            "arity": 1,
            "prob": base_prob,
            "sampler": (lambda q, rng, _g=g: _g.on(q)),  # 用缺省参数绑定，避免 late binding
        })

    if include_idle:
        specs.append({
            "name": "IDLE",
            "arity": 1,
            "prob": 8,
            "sampler": (lambda q, rng: None),
        })
    return specs

def _y90(sign: int): return cirq.YPowGate(exponent = 0.5 * sign)

def _inv_op(op: cirq.Operation) -> Optional[cirq.Operation]:
    g, qs = op.gate, op.qubits
    if isinstance(g, cirq.YPowGate):  # Y^(±1/2) → 取反号
        return cirq.YPowGate(exponent = - g.exponent).on(*qs)
    if isinstance(g, cirq.CZPowGate) or g == cirq.CZ:  # CZ 自逆
        return g.on(*qs)
    # idle/空操作不需要反演；测量不应在主体中出现
    return None

def _runs_from_mask(mask: List[int]) -> List[Tuple[int, int]]:
    """
        从掩码提取“团块”（连续 1 的半开区间 [l, r)）
        例如 mask = [0,1,1,0,1,1,1,0] → [(1,3), (4,7)]
    """
    runs, n = [], len(mask)
    i = 0
    while i < n:
        if mask[i] == 0:
            i += 1
            continue
        j = i
        while j < n and mask[j] == 1:
            j += 1
        runs.append((i, j))  # [i, j)
        i = j
    return runs

# 团块内部：均匀随机“无重叠配对”匹配（路径图上所有 matchings 等概率）
def _uniform_random_matching_L(L: int, rng: np.random.RandomState) -> List[Tuple[int, int]]:
    if L <= 1:
        return []
    # 计数 DP：M[k] = 长度 k 的路径图匹配数（F_{k+1}）
    M = [0] * (L + 2)
    M[0] = 1; M[1] = 1
    for k in range(2, L + 1):
        M[k] = M[k - 1] + M[k - 2]
    pairs = []
    i = 0
    while i < L:
        w_skip = M[L - (i + 1)]
        w_pair = M[L - (i + 2)] if i + 1 < L else 0
        total = w_skip + w_pair
        if total == 0 or w_pair == 0:
            i += 1
        else:
            if rng.rand() < (w_pair / total):
                pairs.append((i, i + 1))
                i += 2
            else:
                i += 1
    return pairs

# 团块内部：带 2q 密度的简单偏置匹配（从左到右以 p2 选择 (i,i+1)）
def _biased_matching_L(L: int, p2: float, rng: np.random.RandomState) -> List[Tuple[int, int]]:
    pairs = []
    i = 0
    while i < L:
        if i + 1 < L and rng.rand() < p2:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    return pairs

# # ---------- 2) 标准 RB：随机 Clifford + 反演 ----------
# def build_rb_circuit(
#     depth: int,
#     qubits: list[cirq.Qid],
#     rng: np.random.RandomState | None = None,
#     measure: bool = False,
# ) -> cirq.Circuit:
#     """该RB具有以下约束：

#     """
#     if rng is None:
#         rng = np.random.RandomState()
#     mats, gates = get_1q_clifford_library()
#     num_cliffords = len(gates)
#     num_qubits = len(qubits)

#     ops: List[List[cirq.Operation]] = []
#     idxs: List[List[int]] = []
#     # 随机抽索引
#     for (qubit, i) in zip(qubits, num_qubits):
#         idxs[i] = rng.choice(num_cliffords, size=depth)
#         ops[i] = [gates[j].on(qubit) for j in idxs[i]]

#     # 整体矩阵（注意乘法方向与应用顺序一致）
#     global_ops : List[np.ndarray]
#     global_ops = [np.eye(2, dtype=complex) for _ in range(num_qubits)]

#     for (qubit, i) in zip(qubits, num_qubits):
#         for j in idxs[i]:
#             global_ops[i] = mats[j] @ global_ops[i]

#     global_invs : List[np.ndarray] = []
#     for (qubit, i) in zip(qubits, num_qubits):
#         global_invs[i] = global_ops[i].conj().T
#         inv_op = cirq.MatrixGate(global_invs[i]).on(qubit)
#         ops[i].append(inv_op)

#     cir = cirq.Circuit()
#     for (qubit, i) in zip(qubits, num_qubits):
#         cir.append(ops[i])

#     if measure:
#         cir.append(cirq.measure(qubit, key="m"))
#     return cir

def default_gate_set() -> List[Dict]:
    """
    每个条目：
      - name: 标识
      - arity: 1 或 2（IDLE 也按 1 处理，表示在该比特上不放门）
      - prob: 采样概率（会自动归一化）
      - sampler(qubits, rng) -> cirq.Operation | None:
          给出某个比特（或一对比特）返回具体 Operation；IDLE 返回 None
    """
    return [
        {"name": "Y90+",  "arity": 1, "prob": 1.0,
         "sampler": lambda q, rng: _y90(+1).on(q)},
        {"name": "Y90-",  "arity": 1, "prob": 1.0,
         "sampler": lambda q, rng: _y90(-1).on(q)},
        {"name": "Z",     "arity": 1, "prob": 1.0,
         "sampler": lambda q, rng: cirq.Z.on(q)},
        {"name": "CZ",    "arity": 2, "prob": 1.0,
         "sampler": lambda qpair, rng: cirq.CZ(qpair[0], qpair[1])},
        {"name": "IDLE",  "arity": 1, "prob": 1.0,
         "sampler": lambda q, rng: None},  # 在该比特 idle（不放门）
    ]

DEFAULT_RB_GATE_SET = default_gate_set()

def fill_rb_gate_samplers(gate_set: List[Dict]) -> List[Dict]:
    """为 YAML 中加载的 gate set 自动补全 sampler 字段（根据 name 匹配）"""
    def _y90(sign: int): return cirq.YPowGate(exponent = 0.5 * sign)

    for g in gate_set:
        name = g.get("name", "").upper()
        if "sampler" in g and callable(g["sampler"]):
            continue  # 已定义 sampler，不覆盖
        if name == "Y90+":
            g["sampler"] = lambda q, rng: _y90(+1).on(q)
        elif name == "Y90-":
            g["sampler"] = lambda q, rng: _y90(-1).on(q)
        elif name == "H":
            g["sampler"] = lambda q, rng: cirq.H.on(q)
        elif name == "Z":
            g["sampler"] = lambda q, rng: cirq.Z.on(q)
        elif name == "CZ":
            g["sampler"] = lambda qpair, rng: cirq.CZ(qpair[0], qpair[1])
        elif name == "IDLE":
            g["sampler"] = lambda q, rng: None
        else:
            raise ValueError(f"Unrecognized gate name: {name}")
    return gate_set


# def _normalize_probs(gs: List[Dict]) -> np.ndarray:
#     w = np.array([g["prob"] for g in gs], dtype=float)
#     s = w.sum()
#     if s <= 0:  # 全 0 退化为均匀
#         w = np.ones_like(w) / len(w)
#     else:
#         w = w / s
#     return w

# ---- 从 gate_set 计算权重与分布 ----
def _split_and_weights(gate_set: List[Dict]):
    oneq_all   = [g for g in gate_set if g["arity"] == 1]
    twoq       = [g for g in gate_set if g["arity"] == 2]

    # idle: sampler 返回 None 视为 idle
    def _is_idle(g: Dict) -> bool:
        try:
            return g["sampler"] is None or g["sampler"]("SENTINEL", np.random.RandomState()) is None
        except Exception:
            # sampler 签名不同也没关系；按名字兜底
            return g.get("name", "").upper() == "IDLE"

    oneq_idle  = [g for g in oneq_all if _is_idle(g)]
    oneq_nonidle = [g for g in oneq_all if not _is_idle(g)]

    mass_1q_total   = float(sum(g["prob"] for g in oneq_all))      # 含 idle
    mass_1q_nonidle = float(sum(g["prob"] for g in oneq_nonidle))
    mass_1q_idle    = float(sum(g["prob"] for g in oneq_idle))
    mass_2q         = float(sum(g["prob"] for g in twoq))

    # 站点活跃概率：来自 1q 集合的“非 idle 占比”
    if mass_1q_total > 0:
        p_active = 1.0 - (mass_1q_idle / mass_1q_total)
    else:
        # 没有 1q 门：若有 2q，建议把站点都设活跃（让 2q 有机会），否则 0
        p_active = 1.0 if mass_2q > 0 else 0.0

    # 团块内 2q 倾向
    denom = mass_2q + mass_1q_nonidle
    p2 = (mass_2q / denom) if denom > 0 else 0.0

    # 归一化分布（给具体门时用）
    def _norm_dist(lst: List[Dict]):
        w = np.array([g["prob"] for g in lst], dtype=float)
        if w.sum() <= 0:
            return lst, None  # 无可选项
        return lst, (w / w.sum())

    oneq_nonidle, dist_1q = _norm_dist(oneq_nonidle)
    twoq, dist_2q         = _norm_dist(twoq)

    return {
        "oneq_nonidle": oneq_nonidle, "dist_1q": dist_1q,
        "twoq": twoq,                 "dist_2q": dist_2q,
        "p_active": p_active,
        "p2": p2,
    }

# ---- 生成单个 moment（不允许全 idle）----
def _sample_moment(
    qubits: List[cirq.Qid],
    gate_set: List[Dict] = DEFAULT_RB_GATE_SET,
    rng: Optional[np.random.RandomState] = None,
    allow_empty: bool = False, # 若为 False，则全 idle 时重抽
    twoq_mode: str = "uniform",   # "uniform" 或 "biased"
) -> List[cirq.Operation]:
    """
        给定一组 qubits，按 gate_set 概率随机放置 1q/2q 门（禁止全 idle），返回操作列表。
    """
    if rng is None:
        rng = np.random.RandomState()

    info = _split_and_weights(gate_set)
    oneq_nonidle, dist_1q = info["oneq_nonidle"], info["dist_1q"]
    twoq, dist_2q         = info["twoq"], info["dist_2q"]
    p_active, p2          = info["p_active"], info["p2"]

    n = len(qubits)

    while True:
        mask = (rng.rand(n) < p_active).astype(int).tolist()
        if allow_empty or any(mask):
            break

    ops: List[cirq.Operation] = []
    runs = _runs_from_mask(mask)

    for (l, r) in runs:                # 团块局部坐标 [0..L-1]
        L = r - l
        if L <= 0:
            continue
        if twoq_mode == "uniform":
            pairs_local = _uniform_random_matching_L(L, rng)
        else:
            pairs_local = _biased_matching_L(L, p2, rng)

        paired = set()
        for (i_loc, j_loc) in pairs_local:
            i, j = l + i_loc, l + j_loc
            if twoq and dist_2q is not None:
                k = rng.choice(len(twoq), p=dist_2q)    # 选具体 2Q 门
                op = twoq[k]["sampler"]((qubits[i], qubits[j]), rng)
            else:
                op = cirq.CZ(qubits[i], qubits[j])      # 兜底：有 2Q 候选才会走到这，一般不会用到
            if op is not None:
                ops.append(op)
                paired.add(i); paired.add(j)

        # 1Q：剩余活跃未配对，按 oneq_nonidle 权重抽具体门
        for i in range(l, r):
            if i in paired:
                continue
            if oneq_nonidle and dist_1q is not None:
                k = rng.choice(len(oneq_nonidle), p=dist_1q)
                op = oneq_nonidle[k]["sampler"](qubits[i], rng)
            else:
                # 兜底：如果没有 1q 非 idle，就放一个 Y90±
                s = +1 if rng.rand() < 0.5 else -1
                op = _y90(s).on(qubits[i])
            if op is not None:
                ops.append(op)

    return ops
    # for _ in range(max_trials):
    #     used = set()
    #     ops: List[cirq.Operation] = []

    #     # 随机化位置扫描顺序（先决定一个遍历顺序）
    #     order = list(range(n))
    #     rng.shuffle(order)

    #     i = 0
    #     while i < n:
    #         if i in used:
    #             i += 1
    #             continue

    #         # 采样一个门条目
    #         k = rng.choice(len(gs), p=probs)
    #         g = gs[k]
    #         arity = g["arity"]

    #         if arity == 1:
    #             q = qubits[order[i]]
    #             op = g["sampler"](q, rng)  # 可能是 None（IDLE）
    #             if op is not None:
    #                 ops.append(op)
    #                 used.add(order[i])
    #             else:
    #                 # idle：不占用，但也不放门
    #                 pass
    #             i += 1

    #         elif arity == 2:
    #             # line 拓扑：尝试与邻居配对（左右优先随机）
    #             q_idx = order[i]
    #             neighbors = []
    #             if q_idx - 1 >= 0: neighbors.append(q_idx - 1)
    #             if q_idx + 1 < n:  neighbors.append(q_idx + 1)
    #             rng.shuffle(neighbors)

    #             paired = False
    #             for nb in neighbors:
    #                 if nb in used or nb == q_idx:
    #                     continue
    #                 q0, q1 = qubits[min(q_idx, nb)], qubits[max(q_idx, nb)]
    #                 op = g["sampler"]((q0, q1), rng)
    #                 if op is not None:
    #                     ops.append(op)
    #                     used.add(q_idx); used.add(nb)
    #                     paired = True
    #                     break
    #             # 若没配上，就当这次轮空（不放门，不占用）
    #             i += 1
    #         else:
    #             raise ValueError(f"Unsupported arity: {arity}")

    #     if ops:  # 至少一个门则成功
    #         return ops

    # 超过重试次数仍为空，返回空（上层决定如何处理）
    # return []

def build_rb1_circuit(
    qubits: List[cirq.Qid],
    depth: int,
    gate_set: Optional[List[Dict]] = DEFAULT_RB_GATE_SET,
    seed: Optional[int] = None,
    measure: bool = True,
) -> cirq.Circuit:
    """
    构造 depth 个 moment，每个 moment 按 gate_set 概率随机放置 1q/2q 门（禁止全 idle）。
    最后反序 + 逐门取逆，得到 RB 风格可逆电路；可选末尾测量。

    该 RB 具有以下性质：
    - Subset RB: 只对特定的门集进行RB(默认是Ry(π/2)、Ry(-π/2)、CZ、idle)
    - Interleaved RB: 可以对特定的序列片段分析其贡献[TODO]
    """
    rng = np.random.RandomState(seed)
    qs = qubits
    gs = gate_set

    circuit = cirq.Circuit()
    applied_moments: List[List[cirq.Operation]] = []  

    # 逐层生成（不允许全 idle）
    for _ in range(depth):
        ops = _sample_moment(qs, gate_set = gs, rng = rng) 
        while not ops:                                  
            ops = _sample_moment(qs, gate_set = gs, rng = rng)
        circuit.append(cirq.Moment(ops))
        applied_moments.append(ops)

    # 精确反演
    for ops in reversed(applied_moments):
        inv_ops: List[cirq.Operation] = []
        for op in reversed(ops):
            inv = _inv_op(op)
            if inv is not None:
                inv_ops.append(inv)
        if inv_ops:
            circuit.append(cirq.Moment(inv_ops))

    if measure:
        circuit.append(cirq.measure(*qs, key="m"))

    return circuit

def _run_rb_sweep_custom(
    qubits: List[cirq.Qid],
    gate_set: List[Dict[str, Any]],
    timing_cfg: Dict[str, Any],
    noise_cfg: Dict[str, Any],
    interleaved_gate: Optional[Dict[str, Any]],
    depth_list: List[int],
    n_seq: int,
    d: int,
    seed: int = 0,
    use_fidelity: bool = True,
) -> Dict:
    rng = np.random.RandomState(seed)
    mean_vals = []
    seq_vals_all = []


    for m in depth_list:
        print(f"[DEBUG] Running depth = {m}, for interleaved gate = {interleaved_gate['name'] if interleaved_gate else 'None'}")
        vals = []
        for seq_idx in range(n_seq):
            # === 1. 构建参考电路 ===
            circuit0 = build_rb1_circuit(
                qubits=qubits,
                depth=m,
                gate_set=gate_set,
                seed=rng.randint(2**31),
                measure=False,
            )
            circuit = all_in_one_compile(circuit0)

            if seq_idx == 0:  # 只打印前几条
                print(f"         Generated circuit depth (before interleaving): {len(circuit)}")
                if m == 20:
                    print(circuit)

            # === 2. 插入 interleaved 门（如果有） ===
            if interleaved_gate is not None:
                circuit_with_interleaved = cirq.Circuit()
                for moment in circuit:
                    circuit_with_interleaved.append(moment)

                    ops = []
                    if interleaved_gate["arity"] == 1:
                        for q in qubits:
                            op = interleaved_gate["sampler"](q, rng)
                            if op is not None:
                                ops.append(op)

                    elif interleaved_gate["arity"] == 2:
                        for i in range(0, len(qubits) - 1, 2):
                            op = interleaved_gate["sampler"]((qubits[i], qubits[i + 1]), rng)
                            if op is not None:
                                ops.append(op)

                    if ops:
                        circuit_with_interleaved.append(cirq.Moment(ops))

                circuit_with_interleaved.append(circuit[-1])  # 反演
                circuit = circuit_with_interleaved

                if seq_idx == 0:  # 只打印前几条
                    print(f"[DEBUG] m={m} seq={seq_idx} interleaved_gate={interleaved_gate['name']}")
                    print(f"         circuit depth: {len(circuit)}")
                    print(f"         interleaved moment count: {sum(1 for mom in circuit if any(isinstance(op.gate, cirq.Gate) for op in mom))}")
                    if m == 20:
                        print(circuit_with_interleaved)

            # === 3. 模拟 ===
            noise_model, _ = make_noise_model(
                circuit, timing_cfg, noise_cfg,
                seed_base = seed,
                tag_a = int(m), tag_b = int(seq_idx)
            )
            sim = cirq.DensityMatrixSimulator(noise=noise_model)
            rho_noisy = sim.simulate(circuit).final_density_matrix


            if use_fidelity:
                rho_ideal = cirq.DensityMatrixSimulator().simulate(circuit.with_noise(None)).final_density_matrix
                try:
                    fid = float(fidelity(rho_ideal, rho_noisy))
                except:
                    fid = float("nan")
                vals.append(fid)
            else:
                p_surv = _survival_prob_from_dm(rho_noisy)
                vals.append(p_surv)
        
        
        mean_vals.append(float(np.nanmean(vals)))
        seq_vals_all.append(vals)

    

    y_std = [np.nanstd(v, ddof=1) / max(1, int(len(v))) ** 0.5 for v in seq_vals_all]
    fit = fit_rb_decay(depth_list, mean_vals, d=d, y_std=y_std, lock_B=True, add_m0=True)
    
    print("[DEBUG] Mean values (P̄ or fidelity) by depth:")
    for m, val in zip(depth_list, mean_vals):
        print(f"  m = {m:<3d} → value = {val:.6f}")


    return {
        "m_list": depth_list,
        "Pbar": mean_vals,
        "fit": {
            "model": fit.model,
            "params": fit.params,
            "param_stderr": fit.param_stderr,
        },
        "EPC": fit.EPC,
        "EPC_stderr": fit.EPC_stderr,
    }

def _find_clifford_index_by_matrix(
    target_u: np.ndarray,
    clifford_mats: Sequence[np.ndarray],
    *,
    atol: float = 1e-8,
) -> int:
    """在 24 个 1Q Clifford 中找到与 target_u 等价（忽略全局相位）的那个。"""
    for i, u in enumerate(clifford_mats):
        if cirq.linalg.allclose_up_to_global_phase(target_u, u, atol=atol):
            return int(i)
    raise ValueError("Recovery Clifford not found in 1Q Clifford library.")

def _resolve_rb_seed(seed: Optional[int]) -> int:
    if seed is None:
        return int(secrets.randbelow(2**32))
    return int(seed)

def _build_reverse_inverse_chain_indices(
    seq_indices: Sequence[int],
    clifford_mats: Sequence[np.ndarray],
    *,
    atol: float = 1e-8,
) -> List[int]:
    chain: List[int] = []
    for idx in reversed(seq_indices):
        u_inv = clifford_mats[int(idx)].conj().T
        chain.append(_find_clifford_index_by_matrix(u_inv, clifford_mats, atol=atol))
    return chain

def _sample_idle_after_flags(m: int, rng: np.random.Generator, p_idle: float) -> List[bool]:
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    if not (0.0 <= p_idle <= 1.0):
        raise ValueError(f"idle_insert_prob must be in [0, 1], got {p_idle}")
    if m == 0 or p_idle == 0.0:
        return [False] * m
    return [bool(x) for x in (rng.random(m) < p_idle)]

def _append_circuit_as_new_moments(dst: cirq.Circuit, src: cirq.Circuit) -> None:
    for moment in src:
        if moment.operations:
            dst.append(moment, strategy=cirq.InsertStrategy.NEW)

def build_1q_clifford_rb_circuit_compiled(
    m: int,
    qubit: cirq.Qid,
    *,
    seed: Optional[int] = None,
    measure: bool = True,
    key: str = "m",
    atol: float = 1e-8,
    compiler: Optional[Callable[[cirq.Circuit], cirq.Circuit]] = None,
    idle_insert_prob: float = 0.0,
    idle_duration_ns: float = 20.0,
    insert_idle_after_recovery: bool = False,
    recovery_mode: str = "single_clifford",
) -> Tuple[cirq.Circuit, cirq.Circuit, Dict[str, Any]]:
    """
    生成单比特 Clifford RB 电路，并返回其 weak-compiled 版本。

    参数
    ----
    m:
        随机 Clifford 序列长度（不含 recovery）
    qubit:
        目标 qubit
    seed:
        随机种子
    measure:
        是否在末尾加入测量
    key:
        测量 key
    atol:
        恢复 Clifford 匹配时的容差
    compiler:
        可选的 compile 函数；若为 None，则默认使用 rb_1q_weak_compile_blockwise

    返回
    ----
    abstract_circuit:
        Clifford 层的抽象 RB 电路（含 recovery，不含 measurement）
    compiled_circuit:
        弱编译后的 native-gate 电路（若 measure=True，则末尾含 measurement）
    meta:
        一些辅助信息，包括随机 Clifford 下标和 recovery 下标
    --------
    1. abstract_circuit 只表示“逻辑 Clifford 序列 + recovery”，不包含 idle；
    2. compiled_circuit 才表示真实执行线路：
       - 每个 Clifford 单独 weak compile 成一个 block
       - block 之间可按概率插入 cirq.wait(...)
       - 最后再 append measurement
    3. recovery_mode:
       - "single_clifford": 标准 RB，末尾只补一个总逆 Clifford
       - "reverse_inverse_chain": 调试/对照模式，补整串逆序逆门链
    """
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    if idle_duration_ns < 0:
        raise ValueError(f"idle_duration_ns must be non-negative, got {idle_duration_ns}")
    if recovery_mode not in ("single_clifford", "reverse_inverse_chain"):
        raise ValueError(
            f"Unsupported recovery_mode={recovery_mode!r}, "
            "expected 'single_clifford' or 'reverse_inverse_chain'."
        )

    seed = _resolve_rb_seed(seed)
    rng = np.random.default_rng(seed)

    clifford_mats, clifford_gates = get_1q_clifford_library()
    n_cliff = len(clifford_gates)

    # 1) 随机采样 m 个 Clifford
    seq_indices = rng.integers(0, n_cliff, size=m).tolist()

    # 2) 构造逻辑 abstract Clifford circuit（仅随机序列，不含 idle）
    abstract = cirq.Circuit()
    for idx in seq_indices:
        abstract.append(clifford_gates[int(idx)].on(qubit), strategy=cirq.InsertStrategy.NEW)

    # 3) recovery
    recovery_indices: List[int]
    if recovery_mode == "single_clifford":
        if len(abstract) == 0:
            u_total = np.eye(2, dtype=complex)
        else:
            u_total = cirq.unitary(abstract)

        u_recovery = u_total.conj().T
        recovery_idx = _find_clifford_index_by_matrix(
            u_recovery,
            clifford_mats,
            atol=atol,
        )
        recovery_indices = [int(recovery_idx)]
    else:
        recovery_indices = _build_reverse_inverse_chain_indices(
            seq_indices,
            clifford_mats,
            atol=atol,
        )

    for ridx in recovery_indices:
        abstract.append(clifford_gates[int(ridx)].on(qubit), strategy=cirq.InsertStrategy.NEW)

    # 4) 准备编译函数
    compile_fn = compiler if compiler is not None else (
        lambda c: rb_1q_weak_compile_blockwise(
            c,
            atol=atol,
            merge_within_block=False,
        )
    )

    # 5) physical compiled circuit：
    #    - 每个随机 Clifford 单独 weak compile
    #    - 在随机 Clifford block 后按概率插入 idle
    #    - recovery block 默认不插 idle（可选开启）
    compiled = cirq.Circuit()

    idle_after_random_flags = _sample_idle_after_flags(m, rng, idle_insert_prob)

    # 随机 Clifford blocks
    for i, idx in enumerate(seq_indices):
        block_abs = cirq.Circuit(cirq.Moment([clifford_gates[int(idx)].on(qubit)]))
        block_comp = compile_fn(block_abs)
        _append_circuit_as_new_moments(compiled, block_comp)

        if idle_after_random_flags[i] and idle_duration_ns > 0.0:
            compiled.append(
                cirq.wait(qubit, nanos=float(idle_duration_ns)),
                strategy=cirq.InsertStrategy.NEW,
            )

    # recovery blocks
    recovery_idle_inserted = False
    for j, ridx in enumerate(recovery_indices):
        block_abs = cirq.Circuit(cirq.Moment([clifford_gates[int(ridx)].on(qubit)]))
        block_comp = compile_fn(block_abs)
        _append_circuit_as_new_moments(compiled, block_comp)

        # 默认不在 recovery 后插 idle；若需要则只在整个 recovery 末尾考虑一次
        is_last_recovery = (j == len(recovery_indices) - 1)
        if (
            insert_idle_after_recovery
            and is_last_recovery
            and idle_duration_ns > 0.0
            and bool(rng.random() < idle_insert_prob)
        ):
            compiled.append(
                cirq.wait(qubit, nanos=float(idle_duration_ns)),
                strategy=cirq.InsertStrategy.NEW,
            )
            recovery_idle_inserted = True

    # 6) measurement 放在最后
    if measure:
        compiled.append(cirq.measure(qubit, key=key), strategy=cirq.InsertStrategy.NEW)

    meta: Dict[str, Any] = {
        "m": int(m),
        "seed": int(seed),
        "seq_indices": [int(x) for x in seq_indices],
        "recovery_mode": recovery_mode,
        "recovery_indices": [int(x) for x in recovery_indices],
        "recovery_idx": int(recovery_indices[0]) if len(recovery_indices) == 1 else None,
        "idle_insert_prob": float(idle_insert_prob),
        "idle_duration_ns": float(idle_duration_ns),
        "idle_after_random_flags": idle_after_random_flags,
        "n_idle_inserted_random": int(sum(idle_after_random_flags)),
        "idle_after_recovery": bool(recovery_idle_inserted),
        "n_random_cliffords": int(m),
        "n_recovery_cliffords": int(len(recovery_indices)),
        "compiled_n_ops": int(sum(1 for _ in compiled.all_operations())),
    }

    return abstract, compiled, meta
