import cirq
from typing import Dict, Any, List

def _gate_duration_ns(op: cirq.Operation, t1: float, t2: float) -> float:
    gate = getattr(op, "gate", None)

    # 官方 delay / wait 门：按其自身 duration 计时
    if isinstance(gate, cirq.WaitGate):
        dur = gate.duration
        try:
            return float(dur.total_nanos())
        except TypeError as e:
            raise TypeError(
                f"WaitGate duration is not numeric: {dur!r}. "
                "compute_timing_summary 目前只支持数值型 delay duration。"
            ) from e
        
    n = len(op.qubits)
    return t1 if n == 1 else (t2 if n == 2 else t2)

def compute_timing_summary(circuit: cirq.Circuit, t1: float, t2: float) -> Dict[str, Any]:
    """
    计算电路的时序统计信息。

    参数：
        circuit: 待分析的量子电路
        t1: 单量子比特门的持续时间（ns）
        t2: 双量子比特门的持续时间（ns）
    
    返回:
        包含统计信息的字典，包括：
        - n_qubits: 量子比特数量
        - depth: 电路深度（时刻数）
        - n_ops_1q: 单量子比特门数量
        - n_ops_2q: 双量子比特门数量
        - n_ops_other: 其他多量子比特门数量
        - per_moment_durations_ns: 每个时刻的持续时间列表（ns）
        - total_duration_ns: 电路总持续时间（ns）
        - t1: 输入的单量子比特门持续时间
        - t2: 输入的双量子比特门持续时间

    """
    per_moment: List[float] = []
    n1 = n2 = no = 0
    for m in circuit:
        if m.operations:
            durs = []
            for op in m.operations:
                nq = len(op.qubits)
                if nq == 1: n1 += 1
                elif nq == 2: n2 += 1
                else: no += 1
                durs.append(_gate_duration_ns(op, t1, t2))
            per_moment.append(max(durs))
        else:
            per_moment.append(0.0)

    return {
        "n_qubits": len(circuit.all_qubits()),
        "depth": len(circuit),
        "n_ops_1q": n1, "n_ops_2q": n2, "n_ops_other": no,
        "per_moment_durations_ns": per_moment,
        "total_duration_ns": float(sum(per_moment)),
        "t1": t1, "t2": t2,
    }
