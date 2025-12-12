import cirq
import math
import numpy as np
from typing import Sequence, List


class IdleNoiseModel(cirq.NoiseModel):
    """NoiseModel that applies amplitude + phase damping noise to idling qubits."""

    def __init__(
            self,
            t1: float = 20e-9,
            t2: float = 40e-9,
            T1: float = 30e-6, 
            Tphi: float = 60e-6, 
            scale: float = 1.0,
            treat_z_as_virtual: bool = True,
            ):
        """
        Args:
            T1: Amplitude damping time constant.
            Tphi: Phase damping time constant.
            duration: Idling duration per moment (same units as T1, Tphi).
            scale: 外层缩放因子。
            treat_z_as_virtual: 若为 True，则 ZPowGate 视为 virtual-Z，
                                不让其阻止本 qubit 被判为 idle，
                                且纯 Z 层视为 0 时间层，不加 idle 噪声。
        """
        self.t1 = t1
        self.t2 = t2
        self.T1 = T1
        self.Tphi = Tphi
        self.scale = scale
        self.treat_z_as_virtual = treat_z_as_virtual

    def _is_virtual_op(self, op: cirq.Operation) -> bool:
        if not self.treat_z_as_virtual:
            return False
        return isinstance(op.gate, cirq.ZPowGate)
    
    def _find_noise_ops(
            self,
            moment: cirq.Moment,
            system_qubits: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        ops = list(moment.operations)
        has_multi:bool = False

        # 1) 过滤出“非 virtual”门，用来判断哪些 qubit 真正在工作
        if self.treat_z_as_virtual:
            non_virtual_ops = [op for op in ops if not self._is_virtual_op(op)]
        else:
            non_virtual_ops = ops

        # 2) 若整层只有 virtual 操作（如纯 ZPow），则视为 0 时间层，不加 idle 噪声
        if not non_virtual_ops:
            return False, [], []
        
        one_qubit_op_qubits = []

        for op in non_virtual_ops:
            if len(op.qubits) == 1:
                one_qubit_op_qubits.append(op.qubits[0])
            if len(op.qubits) > 1:
                has_multi = True

        # 3) 有非 virtual 门：这些门占用一个 duration 时间。
        #    active_qubits = 有真实门的比特；其它比特视为 idle
        active_qubits = {q for op in non_virtual_ops for q in op.qubits}
        idle_qubits = [q for q in system_qubits if q not in active_qubits]

        # 4) 计算噪声参数
        duration = self.t2 if has_multi else self.t1

        p1 = 1.0 - np.exp(- duration / self.T1)
        pphi = 1.0 - np.exp(- duration / self.Tphi)

        p1 = 1.0 - (1.0 - p1) ** self.scale
        pphi = 1.0 - (1.0 - pphi) ** self.scale

        # 5) 生成噪声 op
        idle_ops = []
        for q in idle_qubits:
            if math.isclose(p1, 0) or p1 > 0:
                idle_ops.append(cirq.amplitude_damp(p1).on(q))
            if math.isclose(pphi, 0) or pphi > 0:
                idle_ops.append(cirq.phase_damp(pphi).on(q))

        # 6) 生成 1q 后噪声
        if (self.t2 > self.t1 or math.isclose(self.t2, self.t1)) and has_multi:
            delta = self.t2 - self.t1

            p1 = 1.0 - np.exp(- delta / self.T1)
            pphi = 1.0 - np.exp(- delta / self.Tphi)

            p1 = 1.0 - (1.0 - p1) ** self.scale
            pphi = 1.0 - (1.0 - pphi) ** self.scale

            for q in one_qubit_op_qubits:
                if math.isclose(p1, 0) or p1 > 0:
                    idle_ops.append(cirq.amplitude_damp(p1).on(q))
                if math.isclose(pphi, 0) or pphi > 0:
                    idle_ops.append(cirq.phase_damp(pphi).on(q))
        # print(f"idle qubits:[{idle_qubits}], one_qubit_op_qubits:[{one_qubit_op_qubits}]")

        return has_multi, one_qubit_op_qubits, idle_ops

    def noisy_moment(
        self,
        moment: cirq.Moment,
        system_qubits: Sequence[cirq.Qid],
    ) -> cirq.Moment:
        
        new_ops: List[cirq.Operation] = []
        has_multi, one_qubit_op_qubits, idle_ops = self._find_noise_ops(moment, system_qubits)
        
        ops_temp: List[cirq.Operation] = []
        for i, idle_op in enumerate(idle_ops):
            ops_temp.append(idle_op)
            if i % 2 == 1:
                if ops_temp[0].qubits[0] in one_qubit_op_qubits:
                    one_qubit_op: cirq.Operation = None
                    for op in moment.operations:
                        if ops_temp[0].qubits[0] in op.qubits:
                            one_qubit_op = op
                            break
                    
                    if isinstance(one_qubit_op, cirq.CircuitOperation):
                        ops_temp = list(one_qubit_op.circuit.all_operations()) + ops_temp
                    elif isinstance(one_qubit_op, cirq.Operation):
                        ops_temp = [one_qubit_op] + ops_temp

                new_ops.append(cirq.CircuitOperation(cirq.FrozenCircuit(ops_temp)))
                ops_temp = []

        for op in moment.operations:
            if len(op.qubits) > (1 if has_multi else 0):
                new_ops.append(op)

        return cirq.Moment(new_ops)

    def noise_model_type(self):
        return "moment"

    def __repr__(self):
        return (
            f"IdleNoiseModel(T1={self.T1}, Tphi={self.Tphi}, "
            f"duration={self.duration}, scale={self.scale}, "
            f"treat_z_as_virtual={self.treat_z_as_virtual})"
        )