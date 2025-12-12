import numpy as np
import cirq
from typing import Dict, Optional


def identify_gate_name(gate: Optional[cirq.Gate]) -> str:
    """根据 gate 类型识别标准名称，适配目前的 RB gate set。"""
    if gate is None:
        return "IDLE"
    if isinstance(gate, cirq.YPowGate):
        e = float(((gate.exponent + 1) % 2) - 1)  # 归一到 (-1,1]
        if np.isclose(e, 0.5):  return "Y90+"
        if np.isclose(e,-0.5):  return "Y90-"
        return "Y"  # 其它角统一算 Y
    elif isinstance(gate, cirq.ZPowGate) and np.isclose(gate.exponent, 1.0):
        return "Z"
    elif isinstance(gate, cirq.ZPowGate):
        return "ZPow"
    elif isinstance(gate, cirq.XPowGate) and np.isclose(gate.exponent, 1.0):
        return "X"
    elif isinstance(gate, cirq.HPowGate) and np.isclose(gate.exponent, 1.0):
        return "H"
    elif gate == cirq.H:
        return "H"
    elif gate == cirq.CZ or isinstance(gate, cirq.CZPowGate):
        return "CZ"
    elif isinstance(gate, cirq.IdentityGate):
        return "IDLE"
    elif isinstance(gate, cirq.MeasurementGate):
        return "MEAS"
    return gate.__class__.__name__  # fallback


def estimate_circuit_fidelity(
    circuit: cirq.Circuit,
    gate_errors: Dict[str, float],
    default_error: float = 0.0,
    verbose: bool = False,
) -> float:
    """
    估算整个量子线路的 fidelity，基于给定的门误差率。

    参数：
    - circuit: cirq.Circuit
    - gate_errors: 每个门的 EPC，如 {"Y90+": 0.005, "CZ": 0.012}
    - default_error: 如果某个门没有在 gate_errors 中，默认用的误差值
    - verbose: 是否打印每个门的处理信息

    返回：
    - 估算的 fidelity（0~1）
    """
    log_fid = 0.0
    all_qubits = circuit.all_qubits()
    idle_error = gate_errors.get("IDLE", None)

    for moment in circuit:
        acted_qubits = set()
        for op in moment.operations:
            name = identify_gate_name(op.gate)
            if name in ("IDLE", "MEAS"):
                continue
            r = gate_errors.get(name, default_error)
            if verbose:
                print(f"Gate: {name:<6}  →  error = {r:.5f}")
            log_fid += np.log1p(-r)
            acted_qubits.update(op.qubits)

        if idle_error is not None:
            idle_qubits = all_qubits - acted_qubits
            for q in idle_qubits:
                if verbose:
                    print(f"Idle on qubit {q}  →  error = {idle_error:.5f}")
                log_fid += np.log1p(-idle_error)

    return np.exp(log_fid)
