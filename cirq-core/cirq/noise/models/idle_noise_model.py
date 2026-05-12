import cirq
import math
import numpy as np
from typing import Sequence, List, Optional


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
        
        if isinstance(op,cirq.CircuitOperation):
            if all(isinstance(inner_op.gate, cirq.ZPowGate) for inner_op in op.circuit.all_operations()):
                return True
            return False
        else:
            if isinstance(op.gate, cirq.ZPowGate):
                return True
            return False

    def _unwrap_single_waitgate(self, op: cirq.Operation) -> Optional[cirq.Operation]:
        """If op is a WaitGate or a CircuitOperation wrapping a single WaitGate, return the outer-level WaitGate op."""

        # Case 1: direct WaitGate
        if isinstance(op.gate, cirq.WaitGate):
            return op

        # Case 2: CircuitOperation that contains exactly ONE op and it's WaitGate
        if isinstance(op, cirq.CircuitOperation):
            inner_ops = list(op.circuit.all_operations())
            if len(inner_ops) != 1:
                return None

            inner = inner_ops[0]
            if not isinstance(inner.gate, cirq.WaitGate):
                return None

            # duration (handle repetitions if it's an int)
            dur = inner.gate.duration
            rep = getattr(op, "repetitions", 1)
            if isinstance(rep, int):
                dur = dur * rep  # cirq.Duration supports * int
            # if rep is parameterized/sympy -> keep dur as-is (or raise if you prefer strict)

            # Map inner qubits to outer qubits using qubit_map if exists
            qmap = getattr(op, "qubit_map", None) or {}
            outer_qubits = tuple(qmap.get(q, q) for q in inner.qubits)

            # Return an equivalent outer WaitGate op
            return cirq.WaitGate(dur).on(*outer_qubits)

        return None
    
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
            return False, [], [], []
        
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
        delay_qubits_and_ops = {}
        for op in ops:
            wop = self._unwrap_single_waitgate(op)
            if wop is None:
                continue
            q = wop.qubits[0]
            for q in wop.qubits:
                delay_qubits_and_ops[q] = wop
        
        has_delay = True if delay_qubits_and_ops else False

        one_qubit_op_qubits = [q for q in one_qubit_op_qubits if q not in delay_qubits_and_ops.keys()]

        # print(f"active_qubits: {active_qubits},\nidle_qubits: {idle_qubits},\ndelay_qubits_and_ops: {delay_qubits_and_ops},\none_qubit_op_qubits: {one_qubit_op_qubits}")

        # 4) 计算噪声参数
        if has_delay:
            delay_max = max(wop.gate.duration.total_nanos() for wop in delay_qubits_and_ops.values()) / 1e9
            if has_multi:
                duration = max(self.t2, delay_max)
            else:
                duration = max(self.t1, delay_max)
        else:
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

        for q in list(delay_qubits_and_ops.keys()):
            delay_time = delay_qubits_and_ops[q].gate.duration.total_nanos() / 1e9
            p1 = 1.0 - np.exp(- delay_time / self.T1)
            pphi = 1.0 - np.exp(- delay_time / self.Tphi)

            p1 = 1.0 - (1.0 - p1) ** self.scale
            pphi = 1.0 - (1.0 - pphi) ** self.scale
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

        return has_multi, one_qubit_op_qubits, idle_ops, delay_qubits_and_ops

    def noisy_moment(
        self,
        moment: cirq.Moment,
        system_qubits: Sequence[cirq.Qid],
    ) -> cirq.Moment:
        
        new_ops: List[cirq.Operation] = []
        has_multi, one_qubit_op_qubits, idle_ops, delay_qubits_and_ops = self._find_noise_ops(moment, system_qubits)
        # print(f"has_multi: {has_multi}, one_qubit_op_qubits: {one_qubit_op_qubits}, idle_ops: {idle_ops}")
        ops_temp: List[cirq.Operation] = []
        ops_after_wrapped: List[cirq.Operation] = []
        for i, idle_op in enumerate(idle_ops):
            ops_temp.append(idle_op)
            if i % 2 == 1:
                q = ops_temp[0].qubits[0]
                if (q in one_qubit_op_qubits) or (q in delay_qubits_and_ops.keys()):
                    one_qubit_op: Optional[cirq.Operation] = None
                    for op in moment.operations:
                        if q in op.qubits:
                            one_qubit_op = op
                            break

                    if one_qubit_op is not None:
                        if isinstance(one_qubit_op, cirq.CircuitOperation):
                            ops_temp = list(one_qubit_op.circuit.all_operations()) + ops_temp
                        else:
                            ops_temp = [one_qubit_op] + ops_temp
                        ops_after_wrapped.append(one_qubit_op)

                new_ops.append(cirq.CircuitOperation(cirq.FrozenCircuit(ops_temp)))
                ops_temp = []

        for op in [op for op in moment.operations if op not in ops_after_wrapped]:
            if isinstance(op, cirq.CircuitOperation):
                new_ops.append(op)
            else:
                new_ops.append(cirq.CircuitOperation(cirq.FrozenCircuit([op])))

        return cirq.Moment(new_ops)

    def noise_model_type(self):
        return "moment"

    def __repr__(self):
        return (
            f"IdleNoiseModel(T1={self.T1}, Tphi={self.Tphi}, "
            f"duration={self.duration}, scale={self.scale}, "
            f"treat_z_as_virtual={self.treat_z_as_virtual})"
        )