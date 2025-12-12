import cirq

def build_circuit(circuit_config: dict) -> cirq.Circuit:
    circuit_type = circuit_config.get("type", "").lower()
    measure = circuit_config.get("measure", False)
    n_qubits = circuit_config.get("qubits", 3)  # 可选字段

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()

    if circuit_type == "ghz":
        # 构造 GHZ 态制备电路
        circuit.append(cirq.H(qubits[0]))
        for i in range(n_qubits - 1):
            circuit.append(cirq.CX(qubits[i], qubits[i + 1]))
        if measure:
            circuit.append(cirq.measure(*qubits, key='m'))

    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")

    return circuit

