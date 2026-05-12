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
    typ = cfg["type"]

    # 1. Old QSIH
    if typ == "QSIH":
        return build_QSIH_circuits(cfg["num_qubits"], cfg["num_layers"], cfg["theta_value"])
    
    # 2. Supermarq's HamiltonianSimulation
    if typ == "HamiltonianSimulation":
        from supermarq.benchmarks import HamiltonianSimulation
        bc = HamiltonianSimulation(cfg["num_qubits"], cfg["time_step"], cfg["total_time"])
        circuit: cirq.Circuit = bc.circuit()[:-1] 
        return circuit, circuit, list(circuit.all_qubits())
    
    # 3. Supermarq's random Bitcode
    if typ == "RandomBitcode":
        from supermarq.benchmarks import BitCode
        circuit = cirq.Circuit()
        for _ in range(cfg["runs"]):
            rng = np.random.default_rng()
            arr = rng.integers(low = 0, high = 2, size = 4)
            # print(f"Random bit string: {arr.tolist()}")
            bc = BitCode(num_data_qubits = cfg["num_data_qubits"], num_rounds = cfg["num_rounds"], bit_state=arr.tolist())
            c_unit: cirq.Circuit = bc.circuit()  # 新版直接 .circuit() 返回 Cirq Circuit
            c_unit = [moment for moment in c_unit if all(not (isinstance(op.gate, cirq.MeasurementGate) or isinstance(op.gate, cirq.ResetChannel)) for op in moment.operations)]
            circuit = circuit + c_unit
        return circuit, circuit, list(circuit.all_qubits())
    
    # 4. Randomized Benchmarking
#     def build_1q_clifford_rb_circuit_compiled(
#     m: int,
#     qubit: cirq.Qid,
#     *,
#     seed: Optional[int] = None,
#     measure: bool = True,
#     key: str = "m",
#     atol: float = 1e-8,
#     compiler: Optional[Callable[[cirq.Circuit], cirq.Circuit]] = None,
#     idle_insert_prob: float = 0.0,
#     idle_duration_ns: float = 0.0,
#     insert_idle_after_recovery: bool = False,
#     recovery_mode: str = "single_clifford",
# ) -> Tuple[cirq.Circuit, cirq.Circuit, Dict[str, Any]]:
    if typ == "RandomizedBenchmarking":
        from cirq.noise.utils.randomized_benchmarking import build_1q_clifford_rb_circuit_compiled
        m = cfg["depth"]
        idle_insert_prob = cfg["idle_insert_prob"]
        idle_duration_ns = cfg["idle_duration_ns"]

        abstract_c, compiled_c, meta = build_1q_clifford_rb_circuit_compiled(
            m=m,
            qubit=cirq.LineQubit(0),
            idle_insert_prob=idle_insert_prob,
            idle_duration_ns=idle_duration_ns,
        )
        circuit = compiled_c
        return circuit, circuit, list(circuit.all_qubits())
    
    # 5. CPMG
    if typ == "CPMG":
        num_qubits = cfg["num_qubits"]
        num_refocusing = cfg["num_refocusing"]
        wait_duration = cfg["wait_duration"]  

        qs = cirq.LineQubit.range(num_qubits)

        cpmg_c = cirq.Circuit()
        block = cirq.Circuit()

        cpmg_c += cirq.Circuit(cirq.Moment(cirq.H(q) for q in qs))
        cpmg_c += cirq.Circuit(cirq.Moment(cirq.WaitGate(duration = cirq.Duration(nanos = wait_duration / 2)).on(q) for q in qs))

        block += cirq.Circuit(cirq.Moment(cirq.X(q) for q in qs))
        block += cirq.Circuit(cirq.Moment(cirq.WaitGate(duration = cirq.Duration(nanos = wait_duration)).on(q) for q in qs))

        for _ in range(num_refocusing - 1):
            cpmg_c += block

        tail = cirq.Circuit()

        tail += cirq.Circuit(cirq.Moment(cirq.X(q) for q in qs))
        tail += cirq.Circuit(cirq.Moment(cirq.WaitGate(duration = cirq.Duration(nanos = wait_duration / 2)).on(q) for q in qs))
        tail += cirq.Circuit(cirq.Moment(cirq.H(q) for q in qs))

        cpmg_c += tail

        return cpmg_c, cpmg_c, list(cpmg_c.all_qubits())

    raise ValueError(f"Unsupported circuit type: {typ}")


