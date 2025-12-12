import cirq
import numpy as np

class NamedKrausChannel(cirq.Gate):
    def __init__(self, kraus_ops, name):
        self._kraus_channel = cirq.KrausChannel(kraus_ops)
        self.name = name

    def _num_qubits_(self):
        return self._kraus_channel.num_qubits()

    def _kraus_(self):
        return self._kraus_channel._kraus_()

    def _circuit_diagram_info_(self, args):
        return self.name


class NamedZGate(cirq.Gate):
    def __init__(self, exponent, name):
        self._exponent = exponent
        self.name = name

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return cirq.ZPowGate(exponent=self._exponent)._unitary_()

    def _circuit_diagram_info_(self, args):
        return self.name


class NamedMatrixGate(cirq.Gate):
    def __init__(self, matrix, name):
        self._matrix = matrix
        self.name = name

    def _num_qubits_(self):
        return int(np.log2(self._matrix.shape[0]))

    def _unitary_(self):
        return self._matrix

    def _circuit_diagram_info_(self, args):
        n = self._num_qubits_()
        if n != 2:
            raise ValueError("NamedMatrixGate 'errorE' style is only intended for 2-qubit gates.")

        return cirq.CircuitDiagramInfo(
            wire_symbols=(self.name, self.name),
            connected=True
        )

class NamedPhaseDampGate(cirq.Gate):
    def __init__(self, gamma: float, name: str):
        self._gamma = gamma
        self._name = name

    def _num_qubits_(self):
        return 1

    def _kraus_(self):
        return cirq.phase_damp(self._gamma)._kraus_()

    def _circuit_diagram_info_(self, args):
        return f"{self._name}({self._gamma:.2e})"

class NamedGate(cirq.Gate):
    def __init__(self, name: str, num_qubits: int = 1):
        self.name = name
        self._num_qubits = num_qubits

    def _num_qubits_(self):
        return self._num_qubits

    def _circuit_diagram_info_(self, args):
        if self._num_qubits == 1:
            return self.name
        else:
            return (self.name,) * self._num_qubits

class EPhaseDephasingGate(cirq.Gate):
    """两比特 'E' 平均通道：
    只衰减含 |11> 的 off-diagonal，相当于随机 U = exp(-i δφ |11><11|)
    做高斯平均后的 CPTP 通道。
    参数:
        phi_rms_E: δφ_E 的 rms（σ_E），单位: rad
    """

    def __init__(self, phi_rms_E: float, name: str = "Eavg"):
        self.phi_rms_E = float(phi_rms_E)
        self.name = name

        # coherence factor g_E = E[e^{i δφ_E}] = exp(-σ_E^2 / 2)
        self._gamma = float(1 - np.exp(- self.phi_rms_E ** 2))

    def _num_qubits_(self) -> int:
        return 2

    def _kraus_(self):
        gamma = self._gamma

        M0 = np.diag([1.0, 1.0, 1.0, np.sqrt(1 - gamma)])  # diag(1,1,1,sqrt(1-γ))
        M1  = np.diag([0.0, 0.0, 0.0, np.sqrt(gamma)])    # diag(0,0,0,sqrt(γ))

        return (M0, M1)

    def __repr__(self):
        return f"{self.name}({(self._gamma)/2:.2e})"