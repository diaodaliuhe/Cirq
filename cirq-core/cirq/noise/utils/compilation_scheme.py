import numpy as np
from typing import List, Sequence, Union, Type, Any, Dict


import cirq
from cirq import ops, protocols

from cirq.transformers import (
    optimize_for_target_gateset, expand_composite,
    merge_single_qubit_gates_to_phased_x_and_z as merge_1q_into_phxz,
    eject_z, eject_phased_paulis, drop_negligible_operations, drop_empty_moments,
)

from cirq.transformers import drop_negligible_operations, drop_empty_moments
from cirq.transformers.analytical_decompositions import two_qubit_to_cz
from cirq.transformers.target_gatesets import compilation_target_gateset

from cirq.transformers import target_gatesets as tg

def _x_to_zyz(op: cirq.Operation, _):
    g = getattr(op, "gate", None)
    if isinstance(g, cirq.XPowGate):
        t = float(g.exponent); q = op.qubits[0]
        return [cirq.ZPowGate(exponent=-0.5).on(q),
                cirq.YPowGate(exponent=t).on(q),
                cirq.ZPowGate(exponent=+0.5).on(q)]
    return op

def compile_to_Rz_Ry_CZ_no_gauge(c: cirq.Circuit) -> cirq.Circuit:
    c = optimize_for_target_gateset(c, gateset=tg.CZTargetGateset())
    c = merge_1q_into_phxz(c); c = eject_z(c); c = eject_phased_paulis(c)
    c = expand_composite(c)
    c = cirq.map_operations(c, _x_to_zyz); c = expand_composite(c)
    c = drop_negligible_operations(c); c = drop_empty_moments(c)
    return c

def merge_rz_runs_no_advance(c: cirq.Circuit, *, mod_two=True, atol=1e-12) -> cirq.Circuit:
    """把同一比特的连续 Rz 合并为一个，并放在下一个非 Z 门之前；Rz 不新开 moment。"""
    pending = {}  # qubit -> exponent sum
    new_moments = []

    def flush_for_qubits(qs, moment_ops):
        for q in qs:
            if abs(pending.get(q, 0.0)) > atol:
                e = pending.pop(q)
                if mod_two:
                    # 规范到 (-1,1]，避免出现 1.999999 等
                    e = ((e + 1) % 2) - 1
                moment_ops.append(cirq.ZPowGate(exponent=e).on(q))

    for m in c:
        out_ops = []
        # 先把本层所有 Z 累加起来，但不立即输出
        for op in m:
            if isinstance(op.gate, cirq.ZPowGate):
                q = op.qubits[0]
                pending[q] = pending.get(q, 0.0) + float(op.gate.exponent)
            else:
                # 遇到非 Z：先冲掉涉及到这些 qubit 的 pending Z
                flush_for_qubits(op.qubits, out_ops)
                out_ops.append(op)
        if out_ops:
            new_moments.append(out_ops)

    # 电路结尾还有未冲掉的 Z：放到最后一层（若已有层则追加）
    tail = []
    flush_for_qubits(list(pending.keys()), tail)
    if tail:
        new_moments.append(tail)

    out = cirq.Circuit(new_moments)
    out = drop_negligible_operations(out, atol=atol); out = drop_empty_moments(out)
    return out

def compile_to_Rz_Ry_CZ(c: cirq.Circuit, *, mod_two=True, atol=1e-12) -> cirq.Circuit:
    """编译电路到 Rz-Ry-CZ 门集，并合并 Rz 门序列。"""
    c = compile_to_Rz_Ry_CZ_no_gauge(c)
    c = merge_rz_runs_no_advance(c, mod_two=mod_two, atol=atol)
    return c


class CZ_Y_Z_gateset(compilation_target_gateset.TwoQubitCompilationTargetGateset):
    """Target gateset: CZ + {Z^t, Y^t} + measurement + global phase."""

    def __init__(
        self,
        *,
        atol: float = 1e-8,
        allow_partial_czs: bool = False,
        additional_gates = (),
        preserve_moment_structure: bool = True,
        reorder_operations: bool = False,
    ) -> None:
        cz_gate = ops.CZPowGate if allow_partial_czs else ops.CZ
        super().__init__(
            cz_gate,
            ops.MeasurementGate,
            ops.ZPowGate,
            ops.YPowGate,
            ops.GlobalPhaseGate,
            *additional_gates,
            name="CZ_Y_Z_gateset",
            preserve_moment_structure=preserve_moment_structure,
            reorder_operations=reorder_operations,
        )
        self.additional_gates = tuple(
            g if isinstance(g, ops.GateFamily) else ops.GateFamily(gate=g)
            for g in additional_gates
        )
        self._additional_gates_repr_str = ", ".join(
            [ops.gateset._gate_str(g, repr) for g in additional_gates]
        )
        self.atol = atol
        self.allow_partial_czs = allow_partial_czs

    def _decompose_single_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        g = op.gate
        # 已经是目标门集里的 1q 门，直接接受
        if isinstance(g, (ops.ZPowGate, ops.YPowGate)):
            return op

        if not protocols.has_unitary(op):
            return NotImplemented

        u = protocols.unitary(op)
        phi0, phi1, phi2 = cirq.deconstruct_single_qubit_matrix_into_angles(u)  # radians
        q = op.qubits[0]

        ops_out = []
        if abs(phi0) > self.atol:
            ops_out.append(ops.ZPowGate(exponent=phi0 / np.pi).on(q))
        if abs(phi1) > self.atol:
            ops_out.append(ops.YPowGate(exponent=phi1 / np.pi).on(q))
        if abs(phi2) > self.atol:
            ops_out.append(ops.ZPowGate(exponent=phi2 / np.pi).on(q))

        # 如果三角度都 ~0，就返回空列表，相当于 identity。
        return ops_out

    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        if not protocols.has_unitary(op):
            return NotImplemented
        return two_qubit_to_cz.two_qubit_matrix_to_cz_operations(
            op.qubits[0],
            op.qubits[1],
            protocols.unitary(op),
            allow_partial_czs=self.allow_partial_czs,
            atol=self.atol,
            # 可选：为了避免在这一步就把 1q 门 merge 成 PhXZ，可以关掉 clean_operations
            # clean_operations=False,
        )
            
    @property
    def postprocess_transformers(self):
    # 覆盖掉父类默认的 (merge_single_qubit_moments_to_phxz, drop_negligible_operations, drop_empty_moments)
    # 只保留“清理”操作，不再把 1q 门并成 PhXZ。
        return (drop_negligible_operations, drop_empty_moments)
    
def _is_diagonal_gate(gate: Any) -> bool:
    """判断 gate 是否是 Z 方向的对角门（允许 Z 穿过它）."""
    return isinstance(
        gate,
        (
            cirq.ZPowGate,   # 单/多比特 Z^t（主要是单比特 Z^t）
            cirq.CZPowGate,  # CZ^t
            cirq.ZZPowGate,  # ZZ^t（如果你以后用到）
            cirq.DiagonalGate,  # 通用对角门（可选）
        ),
    )


def _is_zero_like(exp: Any, atol: float) -> bool:
    """粗略判断一个 exponent 是否“近似为 0”."""
    if exp is None:
        return True
    try:
        return abs(float(exp)) <= atol
    except Exception:
        # 符号参数之类：保守起见视为非零，不做裁剪
        return False


def merge_zpow_weak(
    circuit: cirq.Circuit,
    *,
    atol: float = 1e-8,
) -> cirq.Circuit:
    """
    只合并同一条线上连续或被对角门（Z、CZ 等）隔开的 ZPowGate，
    不跨越 YPow 等非对角门，也不会改写 Y 结构。

    规则（对每个 qubit 独立）：
        - 遇到单比特 ZPowGate：不立刻放入电路，而是把 exponent 累加到 pending[q]；
        - 遇到对角门（ZPow/CZPow/ZZPow/DiagonalGate）：
              允许 pending 的 Z 穿过它，不做 flush；
        - 遇到非对角门（例如 YPow）：
              先在该门之前单独插一个 moment，把涉及到的 qubit 上的 pending ZPowGate
              一次性“放出来”，然后再放这个 moment 的非对角门；
        - 电路结束后，仍然有 pending Z 的 qubit，会在尾部再加一个 moment 放这些 Z。

    参数：
        circuit: 输入的 Cirq 电路（建议已经编译到 {CZ, YPow, ZPow} 门集）。
        atol:   判断 |exponent| 是否近似为 0 的阈值；合并后若 |exp| <= atol，
                则不会插入对应的 ZPow（后续 drop_negligible_operations 也会再清一遍）。

    返回：
        一个新的 cirq.Circuit；ZPow 在每条线上被尽量往右推并合并，
        只穿过对角门，不穿过 YPow 等非对角门。
    """
    new_moments: List[cirq.Moment] = []

    # 每个 qubit 的累积 Z 指数
    pending: Dict[cirq.Qid, Any] = {}

    for moment in circuit.moments:
        flush_ops: List[cirq.Operation] = []  # 本 moment 前要 flush 的 Z
        main_ops: List[cirq.Operation] = []   # 本 moment 自己的门（去掉了被吃掉的 Z）

        for op in moment.operations:
            g = getattr(op, "gate", None)

            # 1) 单比特 ZPow：吸收进 pending，不直接输出
            if isinstance(g, cirq.ZPowGate) and len(op.qubits) == 1:
                q = op.qubits[0]
                if q in pending:
                    pending[q] = pending[q] + g.exponent
                else:
                    pending[q] = g.exponent
                continue

            # 2) 对角门：允许 Z 穿过，不触发 flush
            if g is not None and _is_diagonal_gate(g):
                main_ops.append(op)
                continue

            # 3) 非对角门：对参与 qubit 先 flush pending Z，再放这个门
            for q in op.qubits:
                if q in pending and not _is_zero_like(pending[q], atol):
                    flush_ops.append(cirq.ZPowGate(exponent=pending[q]).on(q))
                # 无论是否近零，flush 之后都清空 pending
                if q in pending:
                    pending.pop(q)
            main_ops.append(op)

        # 如果有需要 flush 的 Z，单独作为一个前置 moment
        if flush_ops:
            new_moments.append(cirq.Moment(flush_ops))
        # 当前 moment 自己的门（对角 + 非对角，已经去掉吃掉的 Z）
        if main_ops:
            new_moments.append(cirq.Moment(main_ops))

    # 电路末尾：还有没 flush 的 Z，全丢到最后一个 moment
    tail_ops: List[cirq.Operation] = []
    for q, exp in pending.items():
        if not _is_zero_like(exp, atol):
            tail_ops.append(cirq.ZPowGate(exponent=exp).on(q))
    if tail_ops:
        new_moments.append(cirq.Moment(tail_ops))

    new_circuit = cirq.Circuit(new_moments)
    # 最后做一点清理：去掉近似 identity 的门和空 moment
    new_circuit = cirq.drop_negligible_operations(new_circuit, atol=atol)
    new_circuit = cirq.drop_empty_moments(new_circuit)

    return new_circuit



def delete_gate_from_circuit(circuit:cirq.Circuit, moment_id: int, op: cirq.Operation):
    """
    从给定moment删除指定的量子操作。

    参数：
        circuit: 待修改的 cirq.Circuit 对象。
        moment_id: 要删除操作的 moment 索引。
        op: 要删除的量子操作。

    返回：
        None: 该函数直接修改输入电路对象，不返回值。
    """
    operations = list(circuit.moments[moment_id])
    operations = [o for o in operations if o != op]
    circuit.moments[moment_id] = cirq.Moment(operations)

def insert_gate_to_circuit(circuit:cirq.Circuit, moment_id: int, op: cirq.Operation):
    """
    在指定时刻插入一个量子操作。

    参数：
        circuit: 待修改的 cirq.Circuit 对象。
        moment_id: 要插入操作的 moment 索引。
        op: 要插入的量子操作。

    返回：
        None: 该函数直接修改输入电路对象，不返回值。
    """
    operations = list(circuit.moments[moment_id])
    operations.append(op)
    circuit.moments[moment_id] = cirq.Moment(operations)

def merge_zpow_subcircuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    将电路中的 ZPowGate 和实际执行的门合并为一个子电路，并返回修改后的电路。

    参数：
        circuit: 待修改的 cirq.Circuit 对象，其中包含了 ZPowGate 操作。

    返回：
        cirq.Circuit: 经过合并后的新电路。

    notes: 这是破坏性修改，将破坏原电路
    """

    num_moments = len(circuit.moments)
    all_qubits = circuit.all_qubits()
    new_circuit = cirq.Circuit()
    for _ in range(num_moments + 1):
        new_circuit.append(cirq.Moment())

    for q in all_qubits:
        new_circuit.append(cirq.Z(q))

    for i_m, moment in enumerate(circuit.moments):
        for op in moment.operations:
            qubits = op.qubits
            # print(qubits)
            if isinstance(op.gate, cirq.ZPowGate):
                q = qubits[0]
                next_ops: List[cirq.Operation] = [op]
                i_next = i_m + 1
                # 向后查找该qubit的后续门，直到遇到有意义的门
                while isinstance(next_ops[-1].gate, cirq.ZPowGate) and i_next < num_moments if next_ops else True:
                    next_moment = circuit.moments[i_next]
                    ops_on_q_temp = [o for o in next_moment.operations if q in o.qubits]
                    if not ops_on_q_temp:
                        i_next += 1
                        continue
                    
                    # 找到有门操作，先去除找到的门
                    delete_gate_from_circuit(circuit, i_next, ops_on_q_temp[0])
                    
                    if len(ops_on_q_temp[0].qubits) == 2:
                    # 如果是两比特门,从另一个 qubit 向前搜索连续的 ZPowGate
                        # 找另一个qubit
                        two_qubits = ops_on_q_temp[0].qubits
                        # print(two_qubits)
                        q1 = two_qubits[0] if two_qubits[1] == q else two_qubits[1]
                        # print(f"q1:{q1}\n")
                        next_ops_q1: List[cirq.Operation] = [cirq.Z(q1)]
                        # 向前滚动 1 moment
                        i_before = i_next - 1
                        while isinstance(next_ops_q1[0].gate, cirq.ZPowGate) and i_before >= i_m:
                            pre_moment = circuit.moments[i_before]
                            ops_on_another_temp = [o for o in pre_moment.operations if q1 in o.qubits]
                            # 还没找到，往前
                            if not ops_on_another_temp:
                                i_before -= 1
                                continue
                            
                            if isinstance(ops_on_another_temp[0].gate, cirq.ZPowGate):
                                # 找到了 Zpow
                                # print(f"backward search:{ops_on_another_temp[0]}")
                                next_ops_q1.insert(0, ops_on_another_temp[0])

                                delete_gate_from_circuit(circuit, i_before, ops_on_another_temp[0])
                                i_before -= 1
                            else:
                                # 找到了 非 Zpow
                                break
                        
                        next_ops_q1.pop()
                        next_ops.extend(next_ops_q1)

                    next_ops.append(ops_on_q_temp[0])
                    i_next += 1

                subcircuit = cirq.CircuitOperation(cirq.FrozenCircuit(next_ops))
                if i_next == num_moments and all(isinstance(o.gate, cirq.ZPowGate) for o in next_ops):
                    # 都是 ZPowGate，向前找有意义的块合并
                    find_operation:cirq.CircuitOperation = None
                    i_before = i_next - 1
                    while find_operation is None and i_before >= 0:
                        pre_moment = new_circuit.moments[i_before]
                        ops_on_q_temp = [o for o in pre_moment.operations if q in o.qubits]

                        if not ops_on_q_temp:
                            i_before -= 1
                            continue

                        find_operation = ops_on_q_temp[0]
                    
                    if find_operation is not None:
                        if isinstance(find_operation, cirq.CircuitOperation):
                            subcircuit_before = find_operation.circuit if isinstance(find_operation, cirq.CircuitOperation) else find_operation
                            subcircuit = cirq.CircuitOperation(cirq.FrozenCircuit(subcircuit_before + subcircuit.circuit))
                            # print(f"merge:\n{subcircuit}\ni_before:{i_before}\n")
                            delete_gate_from_circuit(circuit, i_m, op)
                            delete_gate_from_circuit(new_circuit, i_before, find_operation)
                            insert_gate_to_circuit(new_circuit, i_before, subcircuit)
                            continue
                # 合并next_ops
                
                # print(f"merge:\n{subcircuit}\ni_next:{i_next}\n")
                # circuit.moments[i_m] = cirq.Moment([o for o in moment.operations if o != op])

                delete_gate_from_circuit(circuit, i_m, op)
                insert_gate_to_circuit(new_circuit, i_next, subcircuit)
                # operations = list(new_circuit.moments[i_next - 1])
                # operations.append(subcircuit)
                # operations = [o for o in operations if o != op]
                # new_circuit.moments[i_next - 1] = cirq.Moment(operations)
            else:
                # print(f"merge:\n{op}\n")
                subcircuit = cirq.CircuitOperation(cirq.FrozenCircuit(op))
                delete_gate_from_circuit(circuit, i_m, op)
                insert_gate_to_circuit(new_circuit, i_m + 1, subcircuit)
                # operations = list(new_circuit.moments[i_m])
                # operations.append(op)
                # new_circuit.moments[i_m] = cirq.Moment(operations)
            
            # print(f"after delete:\n{circuit}")
            # print(f"new circuit:\n{new_circuit}")
    
    del new_circuit[0]
    return new_circuit

def asapize_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    asap_circuit = cirq.Circuit()

    for moment in circuit:
        for op in moment.operations:
            asap_circuit.append(op)
    
    return asap_circuit

def all_in_one_compile(circuit: cirq.Circuit) -> cirq.Circuit:
    p1 = optimize_for_target_gateset(circuit, gateset=CZ_Y_Z_gateset())
    p2 = merge_zpow_weak(p1)
    p3 = merge_zpow_subcircuit(p2)
    p4 = asapize_circuit(p3)
    return p4