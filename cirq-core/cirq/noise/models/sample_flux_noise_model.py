import cirq
import numpy as np
from typing import Sequence

from cirq.noise.utils import NamedZGate, NamedMatrixGate

class FluxNoiseContext:
    def __init__(self, phi_rms: float, qubits: Sequence[cirq.Qid]):
        """
        Args:
            phi_rms: 标准差，用于 δϕ 的高斯采样
            qubits: 电路中的 qubits，用于建立索引映射
        """
        self.phi_rms = phi_rms
        self.qubits = qubits
        self.num_qubits = len(qubits)
        self.qubit_indices = {q: i for i, q in enumerate(qubits)}
        self.phi_samples = None

    def generate(self):
        """
        生成一次 δϕ 样本，每个 qubit 一个
        """
        self.phi_samples = np.random.normal(0, self.phi_rms, self.num_qubits)
        print(f"[FluxNoiseContext] δϕ samples: {self.phi_samples}")


class SampleFluxNoiseModel(cirq.NoiseModel):
    def __init__(self, context: FluxNoiseContext):
        self.context = context

    def noisy_operation(self, operation: cirq.Operation) -> Sequence[cirq.Operation]:
        if self.context.phi_samples is None:
            raise ValueError("FluxNoiseContext has no φ samples. Call context.generate() before simulation.")

        ops = [operation]

        q_list = operation.qubits
        qubit_idx = self.context.qubit_indices

        if len(q_list) == 1:
            q = q_list[0]
            idx = qubit_idx[q]
            delta_phi = self.context.phi_samples[idx]
            ops.append(NamedZGate(
                exponent=delta_phi / np.pi,
                name=f"eDs({delta_phi:.2e})"
                ).on(q))

        elif len(q_list) == 2:
            ctrl, target = q_list
            idx_c = qubit_idx[ctrl]
            idx_t = qubit_idx[target]
            delta_phi_D = self.context.phi_samples[idx_c]
            delta_phi_E = self.context.phi_samples[idx_t]

            ops.append(NamedZGate(
                exponent=delta_phi_D / (2 * np.pi),
                name=f"eDs({delta_phi_D:.2e})"
                ).on(ctrl))

            U_E = np.diag([1, 1, 1, np.exp(-1j * delta_phi_E)])
            
            ops.append(NamedMatrixGate(
                matrix=U_E,
                name=f"eEs({delta_phi_E:.2e})"
                ).on(ctrl, target))

        return ops

    def noise_model_type(self):
        return 'operation'

    def __repr__(self):
        return f"SampleFluxNoiseModel(context={self.context})"
