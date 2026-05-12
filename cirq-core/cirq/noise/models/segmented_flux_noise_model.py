import cirq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Optional

from cirq.noise.utils import NamedZGate, NamedMatrixGate, GateTimingInfo, TimedCircuitContext

def sample_flux_noise_segments(
    qubits: Sequence[cirq.Qid],
    max_segment: int,
    sigma: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, Dict[cirq.Qid, float]]:
    """
    为每个 segment 和每个 qubit 生成一个 δϕ 采样值，服从 N(0, σ²)
    """

    if rng is None:
        rng = np.random.default_rng()

    return {
        seg_id: {
            q: rng.normal(0, sigma)
            for q in qubits
        }
        for seg_id in range(max_segment + 1)
    }

class SegmentedFluxNoiseModel(cirq.NoiseModel):
    """
    - 以“操作签名”而非 Operation 实例来索引时序与游标，避免回调次数与时序长度失配。
    - 提供 overflow 策略（extend/cycle/error），默认 extend。
    - 保持对每一次回调都施加基于 (segment_id, qubit) 的 δϕ。
    """

    def __init__(
        self,
        timing_map: Dict[cirq.Operation, List[GateTimingInfo]],
        delta_phis: Dict[int, Dict[cirq.Qid, float]],
        *,
        overflow: str = "extend",        # "extend" | "cycle" | "error"
        ignore_exponent_for_1q: bool = True,
        normalize_cz_order: bool = True, # CZ 等对称二比特门对 qubits 排序
    ):
        self.delta_phis = delta_phis
        self.overflow = overflow
        self.ignore_exponent_for_1q = ignore_exponent_for_1q
        self.normalize_cz_order = normalize_cz_order

        # 1) 把原 timing_map 变成按“签名”的 map
        self._sig_map: Dict[Tuple, List[GateTimingInfo]] = defaultdict(list)
        for op, infos in timing_map.items():
            self._sig_map[self._op_signature(op)].extend(infos)

        # 2) 每个签名独立推进游标
        self._cursor_sig: Dict[Tuple, int] = {}

        # 缺签名只提示一次
        self._missing_once: set[Tuple] = set()

    # —— 签名函数：尽量稳定、可与 Cirq 的等价回写对齐 —— #
    def _op_signature(self, op: cirq.Operation) -> Tuple:
        g = op.gate
        name = type(g).__name__

        # 处理 1q gate 的 exponent：RB 里常见 Y**±0.5，可忽略指数以提升匹配鲁棒性
        if self.ignore_exponent_for_1q and len(op.qubits) == 1:
            exp = None
        else:
            exp = getattr(g, 'exponent', None)
            if isinstance(exp, (int, float)):
                # 规范到[-1,1] 左右（防止 1.0 与 -1.0 等价情况），并做轻微四舍五入以免浮点毛刺
                e = ((float(exp) + 2.0) % 2.0)
                if e > 1.0:
                    e -= 2.0
                exp = round(e, 6)

        # qubits：对称二比特门（如 CZ）排序，以避免 (q2,q3) 与 (q3,q2) 走不同桶
        qs = op.qubits
        if self.normalize_cz_order and name in ("CZ", "CZPowGate"):
            qs = tuple(sorted(qs, key=lambda q: getattr(q, 'x', str(q))))
        else:
            qs = tuple(qs)

        return (name, qs, exp)

    def _pick_info(self, sig: Tuple, k: int, infos: List[GateTimingInfo]) -> Optional[GateTimingInfo]:
        n = len(infos)
        if n == 0:
            return None

        if k < n:
            return infos[k]

        # 溢出策略
        if self.overflow == "extend":
            return infos[-1]
        elif self.overflow == "cycle":
            return infos[k % n]
        elif self.overflow == "error":
            raise RuntimeError(f"[FluxNoise] signature {sig} noisy calls {k+1} > timing entries {n}")
        else:
            # 默认退回到 extend
            return infos[-1]

    def _find_noise_ops(self, operation: cirq.Operation) -> cirq.OP_TREE:
        noise_ops: List[cirq.Operation] = []

        if isinstance(operation.gate, cirq.WaitGate):
            return []

        sig = self._op_signature(operation)
        infos = self._sig_map.get(sig)
        if not infos:
            # 若找不到签名（极少见），只提示一次然后跳过
            if sig not in self._missing_once:
                print(f"[SegmentedFluxNoiseModel] Note: timing missing for signature {sig}, no flux applied.")
                self._missing_once.add(sig)
            return []

        k = self._cursor_sig.get(sig, 0)
        info = self._pick_info(sig, k, infos)
        self._cursor_sig[sig] = k + 1

        if info is None:
            return []  # 无 timing，谨慎退出

        segment_id = info.segment_id
        qubits = operation.qubits

        if len(qubits) == 1:
            q = qubits[0]
            delta_phi = self.delta_phis.get(segment_id, {}).get(q, 0.0)
            noise_ops.append(NamedZGate(
                exponent = delta_phi / np.pi,
                name     = f"eD({delta_phi:.2e})"
            ).on(q))

        elif len(qubits) == 2:
            q0, q1 = qubits
            delta_phi_D = self.delta_phis.get(segment_id, {}).get(q0, 0.0)
            delta_phi_E = self.delta_phis.get(segment_id, {}).get(q1, 0.0)

            # 对第一个比特加 Z 偏置（半量纲）
            noise_ops.append(NamedZGate(
                exponent = delta_phi_D / (2 * np.pi),
                name     = f"eD({delta_phi_D:.2e})"
            ).on(q0))

            # 对二比特加 |11> 相位
            U_E = np.diag([1, 1, 1, np.exp(-1j * delta_phi_E)])
            noise_ops.append(NamedMatrixGate(matrix=U_E, name=f"eE({delta_phi_E:.2e})").on(q0, q1))

        return noise_ops

    def noisy_operation(self, operation: cirq.CircuitOperation) -> Sequence[cirq.CircuitOperation]:
        
        noise_ops = self._find_noise_ops(operation)

        if isinstance(operation, cirq.CircuitOperation):
            return cirq.CircuitOperation(cirq.FrozenCircuit(operation.circuit + noise_ops))
        
        elif isinstance(operation, cirq.Operation):
            return cirq.CircuitOperation(cirq.FrozenCircuit([operation] + noise_ops))

    def reset(self):
        self._cursor_sig.clear()
        self._missing_once.clear()

    def noise_model_type(self):
        return "operation"

    def __repr__(self):
        total_segments = len(self.delta_phis)
        return f"SegmentedFluxNoiseModel(segments={total_segments}, policy={self.overflow})"
