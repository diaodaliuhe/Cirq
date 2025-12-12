import cirq, sympy, numpy as np
from typing import Tuple, List

def _U_block_QSIH(q0, q1, tx, ty, tz):
    return [
        cirq.CNOT(q0, q1),
        cirq.rz(tz)(q0),
        cirq.H(q1),
        cirq.rz(tx + np.pi/2)(q1),
        cirq.CNOT(q0, q1),
        cirq.rz(-ty)(q0),
        cirq.H(q1),
        cirq.CNOT(q0, q1),
        cirq.rx(np.pi/2)(q0),
        cirq.rx(-np.pi/2)(q1),
    ]

def _add_layer_QSIH(c: cirq.Circuit, qs: List[cirq.Qid], thetas, even=True):
    start = 0 if even else 1
    for i in range(start, len(qs)-1, 2):
        tx, ty, tz = thetas[i]
        c.append(_U_block_QSIH(qs[i], qs[i+1], tx, ty, tz))

def build_QSIH_circuits(n_qubits: int, n_layers: int, theta_value: float
                       ) -> Tuple[cirq.Circuit, cirq.Circuit, List[cirq.Qid]]:
    """
    构建具有自定义参数的 QSIH 电路。

    参数：
        n_qubits: 量子比特数量（至少 2）
        n_layers: QSIH 层数（至少 1）
        theta_value: 用于数值电路的构建参数

    返回:
        Tuple[cirq.Circuit, cirq.Circuit, List[cirq.Qid]]:
            符号电路、数值电路、量子比特列表
    """
    qs = cirq.LineQubit.range(n_qubits)
    c_sym, c_num = cirq.Circuit(), cirq.Circuit()

    def syms(l, i):
        return (sympy.Symbol(f"theta_x_{l}_{i}"),
                sympy.Symbol(f"theta_y_{l}_{i}"),
                sympy.Symbol(f"theta_z_{l}_{i}"))

    for l in range(n_layers):
        sym_layer = [syms(l, i) for i in range(n_qubits-1)]
        num_layer = [(theta_value, theta_value, theta_value)]*(n_qubits-1)
        _add_layer_QSIH(c_sym, qs, sym_layer, even=(l%2==0))
        _add_layer_QSIH(c_num, qs, num_layer, even=(l%2==0))

    return c_sym, c_num, qs

def build_circuit_from_config(cfg: dict):
    """
    根据配置字典构造量子电路。
    
    参数：
        cfg: 包含电路参数的字典，必须包含以下键：
            - type: 电路类型

    返回:
        Tuple[cirq.Circuit, cirq.Circuit, List[cirq.Qid]]:
            符号电路、数值电路、量子比特列表

    目前支持的电路类型：
        - "QSIH"
    """
    typ = cfg["type"].upper()
    if typ == "QSIH":
        return build_QSIH_circuits(cfg["n_qubits"], cfg["n_layers"], cfg["theta_value"])
    if typ == "HamiltonianSimulation":
        from supermarq.benchmarks import HamiltonianSimulation
        bc = HamiltonianSimulation(cfg["n_qubits"], cfg["time_step"], cfg["total_time"])
        circuit: cirq.Circuit = bc.circuit()[:-1] 
        return circuit
    raise ValueError(f"Unsupported circuit type: {typ}")
