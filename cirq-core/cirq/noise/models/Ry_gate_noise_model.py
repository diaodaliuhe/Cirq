import cirq
import numpy as np
from typing import Sequence, Literal
from cirq.noise.utils import NamedKrausChannel

class RyGateNoiseModel(cirq.NoiseModel):
    def __init__(self,
                 # —— Y 脉冲（保持你原来默认）——
                 t1: float = 20e-9,
                 t2: float = 40e-9,
                 T1: float = 30e-6,
                 Tphi: float = 60e-6,
                 p_axis: float = 1e-4,     # 轴向（沿 Y）
                 p_plane: float = 5e-4,    # 平面（X/Z）
                 scale: float = 1.0):
        """
        Y: 仍只对 Y^±0.5（±π/2）加门噪声；Z: 默认 virtual，不加门噪声。
        若将 z_policy='physical'，则对 ZPow 也按同一范式加噪声（轴取 Z）。
        """
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.T1 = float(T1)
        self.Tphi = float(Tphi)
        self.p_axis= float(p_axis)
        self.p_plane= float(p_plane)

        self.scale = float(scale)

    # --------- helpers ---------
    def _compute_ap_params(self, duration: float):
        """两侧各 duration 的 AP 概率；保持原缩放规则。"""
        d = max(0.0, float(duration))
        p1 = 1 - np.exp(-d / self.T1)
        pphi = 1 - np.exp(-d / self.Tphi)
        # 同你原来的 nonlinearity
        p1 = 1 - (1 - p1) ** self.scale
        pphi = 1 - (1 - pphi) ** self.scale
        return p1, pphi

    def _anisotropic_pauli_kraus_axis(self, axis: str, p_plane: float, p_axis: float):
        """把‘轴向/平面’转成 Pauli 信道的 {pI,pX,pY,pZ}；沿用你原有的构造，并做坐标轴轮换。"""
        # 与原逻辑一致的缩放
        p_plane = 1 - (1 - p_plane) ** self.scale
        p_axis  = 1 - (1 - p_axis ) ** self.scale

        # 原版（轴=Y）用了：pX=p_plane/4, pY=(2*p_plane-p_axis)/4, pZ=p_plane/4
        p = {'X': p_plane/4, 'Y': p_plane/4, 'Z': p_plane/4}
        p[axis.upper()] = (2 * p_plane - p_axis) / 4

        pX, pY, pZ = p['X'], p['Y'], p['Z']
        pI = 1 - (pX + pY + pZ)
        return [
            (np.sqrt(pI), cirq.I),
            (np.sqrt(pX), cirq.X),
            (np.sqrt(pY), cirq.Y),
            (np.sqrt(pZ), cirq.Z),
        ]

    def _find_noise_ops(self, operation: cirq.Operation) -> cirq.OP_TREE:

        if isinstance(operation, cirq.CircuitOperation):
            if all(not (isinstance(op.gate, cirq.YPowGate) or isinstance(op.gate, cirq.CZPowGate)) for op in operation.circuit.all_operations()):
                return [], []
                
        elif isinstance(operation, cirq.Operation):
            if not (isinstance(operation.gate, cirq.YPowGate) or isinstance(operation.gate, cirq.CZPowGate)):
                return [], []
            
        if len(operation.qubits) == 1:
            q = operation.qubits[0]
            
            p1, pphi = self._compute_ap_params(self.t1 / 2)

            pre = [cirq.amplitude_damp(p1).on(q), cirq.phase_damp(pphi).on(q)]

            dep_kraus = self._anisotropic_pauli_kraus_axis(
                axis='Y', p_plane=self.p_plane, p_axis=self.p_axis
            )
            dep_name = f"Dep(axis=Y,a={self.p_axis:.1e},p={self.p_plane:.1e})"
            dep = [NamedKrausChannel([amp * cirq.unitary(P) for amp, P in dep_kraus], name=dep_name).on(q)]

            post = [cirq.amplitude_damp(p1).on(q), cirq.phase_damp(pphi).on(q)]

            return pre, dep + post

        elif len(operation.qubits) == 2:
            q0 = operation.qubits[0]
            q1 = operation.qubits[1]

            p1, pphi = self._compute_ap_params(self.t2 / 2)

            pre = [
                cirq.amplitude_damp(p1).on(q0), cirq.phase_damp(pphi).on(q0),
                cirq.amplitude_damp(p1).on(q1), cirq.phase_damp(pphi).on(q1),
            ]

            post = pre.copy()

            return pre, post


    def noisy_operation(self, operation: cirq.Operation) -> cirq.CircuitOperation:
        
        pre, dep_post = self._find_noise_ops(operation)

        if isinstance(operation, cirq.CircuitOperation):
            return cirq.CircuitOperation(cirq.FrozenCircuit(pre + list(operation.circuit.all_operations()) + dep_post))
        
        elif isinstance(operation, cirq.Operation):
            return cirq.CircuitOperation(cirq.FrozenCircuit(pre + [operation] + dep_post))
        

    def noise_model_type(self):
        return 'operation'

    def __repr__(self):
        return (
            "RyGateNoiseModel("
            f"t1={self.t1}, t2={self.t2}, T1={self.T1}, Tphi={self.Tphi}, "
            f"p_axis={self.p_axis}, p_plane={self.p_plane}, "
            f"scale={self.scale})"
        )

    