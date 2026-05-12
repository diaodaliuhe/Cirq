from __future__ import annotations

import cirq
import numpy as np
from typing import Dict, List, Optional, Sequence, Any, Tuple, Iterable
import ast


from cirq.noise.utils.composite_noise_model import CompositeNoiseModel
from cirq.noise.utils.timed_circuit_context import TimedCircuitContext, GateTimingInfo, assign_timed_circuit_context

_ALLOWED_CONSTS = {
    "pi": np.pi,
    "e": np.e,
}
_ALLOWED_FUNCS = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "abs": np.abs,
}
_SAFE_ENV = {**_ALLOWED_CONSTS, **_ALLOWED_FUNCS, "np": np}

def _get(d: Dict[str, Any], path: str, default = None):
    """
        从嵌套 dict 中获取值，path 用 '.' 分隔。
        Args:
            d: 输入 dict。
            path: 用 '.' 分隔的路径字符串。
            default: 若路径无效则返回的默认值。
    """
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _max_segment_id(timing_map: Dict[cirq.Operation, List["GateTimingInfo"]]) -> int:
    """
        从 timing_map 中找最大 segment_id（若无则返回 0）。
        Args:
            timing_map: 来自 TimedCircuitContext 的 timing_map。
    """
    max_sid = -1
    for infos in timing_map.values():
        for info in infos:
            if hasattr(info, "segment_id"):
                max_sid = max(max_sid, info.segment_id)
    return max(0, max_sid)

# def _sample_delta_phis(
#     qubits: Sequence[cirq.Qid],
#     max_segment: int,
#     sigma: float,
#     seed: Optional[int] = None,
# ) -> Dict[int, Dict[cirq.Qid, float]]:
#     """
#         为每个 qubit 在每个 segment 采样 delta_phi。
#         Args:
#             qubits: 参与的 qubits 列表。
#             max_segment: 最大 segment_id。
#             sigma: 高斯分布标准差（弧度）。
#             seed: 随机种子，None 则使用全局随机数生成器。
#     """
#     rng = np.random.RandomState(seed) if seed is not None else np.random
#     return {seg: {q: float(rng.normal(0.0, sigma)) for q in qubits}
#             for seg in range(max_segment + 1)}

def _to_python_float(v):
    """
        把 numpy 数值类型转换为 Python float。
        Args:
            v: 输入数值, 支持 float / np.number / np.ndarray（0-d）。
    """
    if np.isscalar(v):
        return float(v)
    v = np.asarray(v)
    if v.shape == ():
        return float(v.item())
    raise TypeError(f"Expression must be scalar, got shape = {v.shape}")

def eval_number(x, *, default = None) -> float:
    """
        把 YAML 中的数值或受限表达式解析为 float。
        Args:
            x: 支持 int/float/str（受限表达式）；
    """
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        # 1) 先试 literal_eval（仅字面量）
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        except Exception:
            pass
        # 2) 受限 eval（只开放 _SAFE_ENV）
        try:
            v = eval(s, {"__builtins__": {}}, _SAFE_ENV)
            y = _to_python_float(v)
            # 可选：只接受有限实数
            if not np.isfinite(y):
                raise ValueError(f"Expression evaluated to non-finite value: {y}")
            return y
        except Exception as e:
            raise ValueError(f"Cannot evaluate numeric expression: {x!r}") from e
    if default is not None:
        return float(default)
    raise TypeError(f"Unsupported type for numeric eval: {type(x)} -> {x!r}")

def derive_rng(base_seed: int, *tags: int, pool: Optional[Iterable[int]] = None) -> np.random.Generator:
    """
        用 NumPy SeedSequence 派生一个独立 RNG。
        Args:
            base_seed: 顶层seed
            tags:      额外标签（如 m、seq_id、回合号等），用于稳定区分子流
            pool:      可选，进一步把一组标识加入 entropy
        Returns:
            np.random.Generator: 
    """
    if pool is None:
        ss = np.random.SeedSequence(int(base_seed), spawn_key = [int(t) for t in tags])
    else:
        entropy = [int(base_seed)] + [int(x) for x in pool]
        ss = np.random.SeedSequence(entropy = entropy, spawn_key = [int(t) for t in tags])
    return np.random.default_rng(ss)

# ---------- 单模型 builder（与类默认值严格对齐） ----------
def build_idle_from_yaml(timing_cfg: Dict, noise_cfg: Dict):
    """
        从 YAML 构建 IdleNoiseModel。
        Args:
            noise_cfg: 从顶层 "noise" 节点传入的 dict。
    """
    node = noise_cfg.get("idle", {})
    if not bool(node.get("enabled", True)):
        return None
    
    from cirq.noise.models.idle_noise_model import IdleNoiseModel
    return IdleNoiseModel(
        t1 = eval_number(timing_cfg.get("t1", 20)) * 1e-9,
        t2 = eval_number(timing_cfg.get("t2", 40)) * 1e-9,
        T1 = eval_number(node.get("T1", 30e-6)),
        Tphi = eval_number(node.get("Tphi", 60e-6)),
        scale = eval_number(node.get("scale", 1.0)),
    )

def build_ry_from_yaml(timing_cfg:Dict, noise_cfg: Dict):
    """
        从 YAML 构建 RyGateNoiseModel。
        Args:
            noise_cfg: 从顶层 "noise" 节点传入的 dict。
    """
    node = noise_cfg.get("ry_gate", {})
    if not bool(node.get("enabled", True)):
        return None
    
    from cirq.noise.models.Ry_gate_noise_model import RyGateNoiseModel
    return RyGateNoiseModel(
        t1 = eval_number(timing_cfg.get("t1", 20)) * 1e-9,
        t2 = eval_number(timing_cfg.get("t2", 40)) * 1e-9,
        T1 = eval_number(node.get("T1", 30e-6)),
        Tphi = eval_number(node.get("Tphi", 60e-6)),
        p_axis = eval_number(node.get("p_axis", 1e-4)),
        p_plane = eval_number(node.get("p_plane", 5e-4)),
        scale = eval_number(node.get("scale", 1.0)),
    )

def build_photon_decay_from_yaml(
    noise_cfg: Dict,
    timed_ctx: Optional[TimedCircuitContext] = None,
):
    """
        从 YAML 构建 PhotonDecayNoiseModel。
        Args:
            noise_cfg: 从顶层 "noise" 节点传入的 dict。
            timed_ctx: TimedCircuitContext，用于按门时间施加噪声。
    """
    node = noise_cfg.get("photon_decay", {})
    if not bool(node.get("enabled", True)):
        return None

    from cirq.noise.models.photon_decay_noise_model import PhotonDecayNoiseModel
    return PhotonDecayNoiseModel(
        ctx = timed_ctx,
        chi = eval_number(node.get("chi",    np.pi * (-2.6e6))),  # rad/s
        alpha_0 = eval_number(node.get("alpha_0", np.sqrt(0.8))),     # sqrt[n]
        kappa = eval_number(node.get("kappa",  4e6)),               # 1/s
        tau_m = eval_number(node.get("tau_m",  -590e-9)),            # s
        tau_g = eval_number(node.get("tau_g",  10e-9)),             # s
        scale = eval_number(node.get("scale", 1.0)),
    )

def is_flux_enabled(noise_cfg: Dict) -> bool:
    """
        判断 YAML 里是否启用 flux_quasistatic 噪声。
        Args:
            noise_cfg: 从顶层 "noise" 节点传入的 dict。
    """
    node = noise_cfg.get("flux_quasistatic", {})
    return bool(node.get("enabled", True))

def get_flux_sigma_seed(noise_cfg: Dict) -> Tuple[float, Any]:
    """
        从 YAML 里获取 flux_quasistatic 的 sigma 和 seed。
        Args:
            noise_cfg: 从顶层 "noise" 节点传入的 dict。
    """
    node = noise_cfg.get("flux_quasistatic", {}) or {}
    sigma = eval_number(node.get("sigma", 0.0))
    flux_seed = node.get("flux_seed", None)
    return float(sigma), flux_seed

# def build_flux_quasistatic_from_yaml(
#     cfg: Dict[str, Any],
#     timed_ctx: "TimedCircuitContext",
#     qubits: Sequence[cirq.Qid],
#     ) -> Optional[cirq.NoiseModel]:
#     """
#         从 YAML 构建 SegmentedFluxNoiseModel。
#         Args:
#             cfg: 从顶层传入的配置 dict。
#             timed_ctx: 由 assign_timed_circuit_context 生成的 TimedCircuitContext
#             qubits: 参与的 qubits 列表。
#     """
#     node = _get(cfg, "noise.flux_quasistatic", {})
#     if not node or not node.get("enabled", False):
#         return None
#     from cirq.noise.models.segmented_flux_noise_model import SegmentedFluxNoiseModel    
#     sigma = eval_number(node.get("sigma", 0.05))  # rad
#     seed  = node.get("seed", None)
#     timing_map = timed_ctx.timing_map
#     max_seg = _max_segment_id(timing_map)
#     delta_phis = _sample_delta_phis(qubits, max_seg, sigma, seed)
#     return SegmentedFluxNoiseModel(timing_map = timing_map, delta_phis = delta_phis)

def wrap_to_composite(models: List[cirq.NoiseModel]) -> cirq.NoiseModel:
    """
        把多个 NoiseModel 组合为一个 CompositeNoiseModel。
        Args:
            models: NoiseModel 列表。
    """
    for m in models:
        if not hasattr(m, "noise_model_type"):
            raise ValueError(f"Noise model {m} missing noise_model_type declaration.")
        t = m.noise_model_type()
        if t not in ("operation", "moment"):
            raise ValueError(f"Unknown noise model type: {t}")
    return CompositeNoiseModel(models)

def make_noise_model(
    circuit: cirq.Circuit,
    timing_cfg: Dict,
    noise_cfg: Dict,
    seed_base: int,           # 用于派生 flux 的随机种子
    tag_a: int = 0,           # 可传 (m) 或其它标签
    tag_b: int = 0,           # 可传 (seq_id) 或其它标签
) -> Tuple[CompositeNoiseModel, Dict]:
    """
        依据 YAML & Builder 组合噪声；返回 CompositeNoiseModel 与一些元信息。
        Args:
            circuit: cirq.Circuit 对象。
            timing_cfg: 从顶层 "timing" 节点传入的 dict。
            noise_cfg: 从顶层 "noise" 节点传入的 dict。
            seed_base: 用于派生 flux 的随机种子
            tag_a: 可传 (m) 或其它标签
            tag_b: 可传 (seq_id) 或其它标签
    """

    t1 = eval_number(_get(timing_cfg, "t1"))
    t2 = eval_number(_get(timing_cfg, "t2"))
    tau_c = eval_number(_get(timing_cfg, "tau_c"))

    # 分段时序
    ctx = assign_timed_circuit_context(circuit, t1 = t1, t2 = t2, tau_c = tau_c)
    max_seg = _max_segment_id(ctx.timing_map)
    qubits = list(circuit.all_qubits())

    # Builder：三个确定性模型
    models = []
    idle = build_idle_from_yaml(timing_cfg, noise_cfg);             idle and models.append(idle)
    ry   = build_ry_from_yaml(timing_cfg, noise_cfg);      ry   and models.append(ry)
    pd   = build_photon_decay_from_yaml(noise_cfg, ctx);     pd   and models.append(pd)
    
    # Flux（随机）：sigma/seed
    sigma, flux_seed = get_flux_sigma_seed(noise_cfg)
    if is_flux_enabled(noise_cfg):
        # seed 派生：seed_base + (tag_a, tag_b)；seed 未给则使用 seed_base
        base = int(flux_seed) if flux_seed is not None else int(seed_base)

        fx_node = noise_cfg.setdefault("flux_quasistatic", {})
        fx_node["seed"] = base

        rng = derive_rng(base, int(tag_a), int(tag_b))
        delta_phis = {
            seg: {q: float(rng.normal(loc=0.0, scale=sigma)) for q in qubits}
            for seg in range(max_seg + 1)
        }
        from cirq.noise.models.segmented_flux_noise_model import SegmentedFluxNoiseModel
        models.append(SegmentedFluxNoiseModel(ctx.timing_map, delta_phis))
    else:
        delta_phis = {}

    meta = {
        "ctx": ctx,
        "delta_phis": delta_phis,
        "flux_sigma": sigma,
        "flux_seed_effective": int(flux_seed) if flux_seed is not None else int(seed_base),
        "rng_tags": (int(tag_a), int(tag_b)),
    }

    return CompositeNoiseModel(models), meta

def get_flux_sampling_cfg(noise_cfg: Dict) -> Dict[str, Any]:
    fx = noise_cfg.get("flux_quasistatic", {}) or {}
    node = fx.get("sampling", {}) or {}

    mode = str(node.get("mode", "local_correlated")).strip().lower()
    if mode not in ("local_correlated", "global_trace_ar1"):
        raise ValueError(
            f"Unsupported flux sampling mode: {mode!r}. "
            f"Expected 'local_correlated' or 'global_trace_ar1'."
        )

    stride_raw = node.get("sample_stride_segments", "auto")
    if stride_raw is None or (isinstance(stride_raw, str) and stride_raw.strip().lower() == "auto"):
        stride = None
    else:
        stride = int(stride_raw)
        if stride <= 0:
            raise ValueError(
                f"'sample_stride_segments' must be positive or 'auto', got {stride_raw!r}"
            )

    rho_raw = node.get("rho", None)
    rho = None if rho_raw is None else float(eval_number(rho_raw))

    if mode == "global_trace_ar1":
        if rho is None:
            raise ValueError("flux_quasistatic.sampling.rho is required for mode='global_trace_ar1'.")
        if not (0.0 <= rho < 1.0):
            raise ValueError(f"rho must satisfy 0 <= rho < 1, got {rho}")
    else:
        rho = None

    return {
        "mode": mode,
        "sample_stride_segments": stride,
        "rho": rho,
    }