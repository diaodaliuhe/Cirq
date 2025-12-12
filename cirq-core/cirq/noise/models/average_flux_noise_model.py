import cirq
import numpy as np
from typing import Sequence, List

from cirq.noise.utils.named_wrappers import NamedPhaseDampGate, EPhaseDephasingGate

class AverageFluxNoiseModel(cirq.NoiseModel):
    def __init__(self, phi_rms_D: float = 0.01):
        # D 类噪声的 rms，相当于单比特 δϕ_D 的 σ_D
        self.phi_rms_D = float(phi_rms_D)

    def _gamma_from_var(self, rms: float) -> float:
        # var = <(δφ)^2>
        return float(1.0 - np.exp(- rms ** 2))

    # def noisy_operation(self, operation: cirq.Operation):
    #     ops = [operation]
    #     qs = operation.qubits

    #     # --- D 部分：单比特平均通道 ---
    #     var_D = self.phi_rms_D ** 2
    #     gamma_D = self._gamma_from_var(var_D)

    #     if len(qs) == 1:
    #         # D 噪声：对该 qubit 做 phase damping
    #         if gamma_D > 0.0:
    #             ops.append(NamedPhaseDampGate(gamma_D, "eDa").on(qs[0]))

    #     elif len(qs) == 2:
    #         ctrl, target = qs

    #         # D 噪声仍然只加在控制比特上（按你原来的物理模型）
    #         if gamma_D > 0.0:
    #             ops.append(NamedPhaseDampGate(gamma_D, "eDa").on(ctrl))

    #         # --- 精确 E 部分：相关两比特平均通道 ---
    #         # 文献: δϕ_E = δϕ_D / 2 => σ_E^2 = σ_D^2 / 4
    #         phi_rms_E = self.phi_rms_D / 2.0

    #         # 在两比特上加一个 correlated dephasing gate
    #         E_gate = EPhaseDephasingGate(phi_rms_E, name="eEa")
    #         ops.append(E_gate.on(ctrl, target))

    #     return ops

    def _find_noise_ops(self, operation: cirq.Operation) -> cirq.OP_TREE:
        noise_ops: List[cirq.Operation] = []

        qs = operation.qubits

        # --- D 部分：单比特平均通道 ---
        gamma_D = self._gamma_from_var(self.phi_rms_D)

        if len(qs) == 1:
            # D 噪声：对该 qubit 做 phase damping
            if gamma_D > 0.0:
                noise_ops.append(NamedPhaseDampGate(gamma_D, "eDa").on(qs[0]))

        elif len(qs) == 2:
            ctrl, target = qs

            # D 噪声仍然只加在控制比特上（按你原来的物理模型）
            if gamma_D > 0.0:
                noise_ops.append(NamedPhaseDampGate(gamma_D, "eDa").on(ctrl))

            # --- 精确 E 部分：相关两比特平均通道 ---
            # 文献: δϕ_E = δϕ_D / 2 => σ_E^2 = σ_D^2 / 4
            phi_rms_E = self.phi_rms_D / 2.0

            # 在两比特上加一个 correlated dephasing gate
            E_gate = EPhaseDephasingGate(phi_rms_E, name="eEa")
            noise_ops.append(E_gate.on(ctrl, target))

        return noise_ops
    
    def noisy_operation(self, operation: cirq.CircuitOperation) -> cirq.CircuitOperation:
        
        noise_ops = self._find_noise_ops(operation)

        if isinstance(operation, cirq.CircuitOperation):
            return cirq.CircuitOperation(cirq.FrozenCircuit(operation.circuit + noise_ops))
        
        elif isinstance(operation, cirq.Operation):
            return cirq.CircuitOperation(cirq.FrozenCircuit([operation] + noise_ops))


    def noise_model_type(self):
        return "operation"

    def __repr__(self):
        return f"AverageFluxNoiseModel(phi_rms_D={self.phi_rms_D})"