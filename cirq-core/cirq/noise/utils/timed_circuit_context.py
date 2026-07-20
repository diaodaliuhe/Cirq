from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random

import cirq

@dataclass
class GateTimingInfo:
    start_time: float
    duration: float
    segment_id: int
    # 新增：定位和出现次序（用于 debug / deterministic seeding）
    moment_idx: int
    op_idx: int
    occurrence: int  # 同一个 Operation（按 == ）的第几次出现，从 0 开始
    op: cirq.Operation = None  # 可选：保存内部的有意义 operation

@dataclass
class TimedCircuitContext:
    circuit: cirq.Circuit
    timing_map: Dict[cirq.CircuitOperation, List[GateTimingInfo]] # 改为“一个 op → 多个出现的 timing 列表（按电路遍历顺序）”

    def print_timing_map(self):
        """
        按 segment、时间、qubit 顺序打印所有操作的 timing（对齐版）。
        """
        entries = []
        for op, infos in self.timing_map.items():
            for info in infos:
                qubit_ids = [int(q.x) if hasattr(q, 'x') else int(str(q)) for q in op.qubits]
                min_q = min(qubit_ids)
                entries.append((op, info, min_q))

        # 按 segment → start_time → qubit → moment → op_idx 排序
        entries.sort(
            key=lambda e: (
                e[1].segment_id,
                e[1].start_time,
                e[2],
                e[1].moment_idx,
                e[1].op_idx,
            )
        )

        # 打印表头
        header = (
            f"{'seg':>4} | {'t (ns)':>9} | {'dur (ns)':>9} | "
            f"{'qubits':>10} | {'occ':>4} | {'gate':>8}"
        )
        print("timing for operations (sorted by segment and qubit):\n")
        print(header)
        print("-" * len(header))

        # 逐条打印
        for op, info, _ in entries:
            q_str = "(" + ",".join(str(int(q.x)) for q in op.qubits) + ")"
            print(f"{info.segment_id:>4d} | "
                f"{info.start_time:>9.1f} | "
                f"{info.duration:>9.1f} | "
                f"{q_str:>10} | "
                f"{info.occurrence:>4d} | "
                f"{(str(info.op.gate) if info.op != None else 'None'):>8}")

# def assign_timed_circuit_context(
#     circuit: cirq.Circuit,
#     t1: float = 30.0,  # 单比特门的时间
#     t2: float = 60.0,  # 双比特门的时间
#     tau_c: float = 20.0,  # segment 的大小
#     treat_z_as_virtual: bool = True,
# ) -> TimedCircuitContext:
#     """
#     为给定 Cirq 电路中的每个操作分配开始时间、持续时间、所属 segment 编号，
#     并封装为 TimedCircuitContext。

#     参数：
#         circuit: cirq.Circuit 对象
#         t1: 单比特门的持续时间（单位 ns）
#         t2: 双比特门的持续时间（单位 ns）
#         tau_c: 截断时间（segment 大小，单位 ns）
#         treat_z_as_virtual: 若为 True，则 ZPowGate 视为 virtual-Z，duration=0

#     返回：
#         TimedCircuitContext，包括原始电路和其 timing map
#     """
#     two_q_slots = int(round(t2 / t1))  # 确保 t2 是 t1 的倍数，如果是的话，可以按照相同方式来调整
#     if abs(two_q_slots * t1 - t2) > 1e-9:
#         raise ValueError(f"t2={t2} 不是 t1={t1} 的整数倍，无法构造整齐 time slice。")

#     timing_map: Dict[cirq.Operation, List[GateTimingInfo]] = {}
#     seen_count: Dict[cirq.Operation, int] = {}
#     ready_slot: Dict[cirq.Qid, int] = {}  # 每个 qubit 的最早可用 slot
#     slot_has_2q: Dict[int, bool] = {}  # 记录每个 slot 是否有双比特门

#     # 遍历每个 moment 和其内部操作
#     for m_idx, moment in enumerate(circuit.moments):
#         for o_idx, op in enumerate(moment.operations):
#             qubits = op.qubits
#             op_copy = op

#             # 检查是否为虚拟 Z 操作
#             is_vz = treat_z_as_virtual and _is_virtual_z_op(op)

#             if isinstance(op, cirq.CircuitOperation):
#                 # 如果是 CircuitOperation，提取有效门
#                 internal_op = None
#                 for sub_op in op.circuit.all_operations():
#                     if not _is_virtual_z_op(sub_op):
#                         internal_op = sub_op
#                         break
#                 if internal_op:
#                     op = internal_op  # 使用有效门替代 CircuitOperation
#                 else:
#                     is_vz = treat_z_as_virtual
#                     continue  # 如果没有有效门，跳过

#             # 计算该门占用的 slot 数量
#             if len(qubits) == 0:
#                 gate_slots = 0
#             elif len(qubits) == 1:
#                 gate_slots = 0 if is_vz else 1
#             else:
#                 gate_slots = two_q_slots  # 双比特门使用 t2

#             # 计算该操作的开始时间和持续时间
#             if gate_slots == 0:
#                 start_slot = max(ready_slot.get(q, 0) for q in qubits) if qubits else 0
#                 start_time = start_slot * t1
#                 duration = 0.0
#             else:
#                 # 获取前一个 moment 的结束时间来确保时序正确
#                 if m_idx > 0:
#                     prev_moment_end_time = max(
#                         [timing_map.get(op, [])[0].start_time + timing_map.get(op, [])[0].duration for op in circuit.moments[m_idx - 1].operations]
#                     )
#                 else:
#                     prev_moment_end_time = 0

#                 # 当前 moment 的 start_time 是上一个 moment 的结束时间
#                 start_time = prev_moment_end_time
#                 duration = gate_slots * t1

#             # 更新 ready_slot 和 slot_has_2q
#             end_slot = start_time + duration
#             for q in qubits:
#                 ready_slot[q] = max(ready_slot.get(q, 0), end_slot)

#             # 将结果记录到 timing_map 中
#             occ = seen_count.get(op, 0)
#             seen_count[op] = occ + 1
#             segment_id = math.floor(start_time / tau_c)

#             # 更新 segment 划分规则，确保跨多个 segment 时，取占用时间较长的那个
#             if duration > tau_c:
#                 seg_end = math.floor((start_time + duration - 1e-9) / tau_c)
#                 segment_id = seg_end if seg_end != segment_id else segment_id

#             info = GateTimingInfo(
#                 start_time=start_time,
#                 duration=duration,
#                 segment_id=segment_id,
#                 moment_idx=m_idx,
#                 op_idx=o_idx,
#                 occurrence=occ,
#                 op=op
#             )
#             timing_map.setdefault(op_copy, []).append(info)

#     return TimedCircuitContext(circuit=circuit, timing_map=timing_map)

def assign_timed_circuit_context(
    circuit: cirq.Circuit,
    t1: float = 20.0,  # 单比特门的时间
    t2: float = 40.0,  # 双比特门的时间
    tau_c: float = 60.0,  # segment 的大小
    treat_z_as_virtual: bool = True,
    segment_origin_offset: float = 0.0,
    assignment_rule: str = "max_overlap",
    assignment_seed: Optional[int] = None,
) -> TimedCircuitContext:
    """
    为给定 Cirq 电路中的每个操作分配开始时间、持续时间、所属 segment 编号，
    并封装为 TimedCircuitContext。

    参数：
        circuit: cirq.Circuit 对象
        t1: 单比特门的持续时间（单位 ns）
        t2: 双比特门的持续时间（单位 ns）
        tau_c: 截断时间（segment 大小，单位 ns）
        treat_z_as_virtual: 若为 True，则 ZPowGate 视为 virtual-Z，duration=0
        segment_origin_offset: segment grid offset in ns. The segment id is computed
            from start_time + segment_origin_offset.
        assignment_rule: "max_overlap" (current default), "start_time", "midpoint",
            or "random_overlap".
        assignment_seed: deterministic seed for "random_overlap".

    返回：
        TimedCircuitContext，包括原始电路和其 timing map
    """
    assignment_rule = str(assignment_rule).lower()
    if assignment_rule not in ("max_overlap", "start_time", "midpoint", "random_overlap"):
        raise ValueError(f"Unknown assignment_rule: {assignment_rule!r}")

    segment_origin_offset = float(segment_origin_offset)
    assignment_rng = random.Random(assignment_seed)

    def _seg_at(t: float) -> int:
        return int((float(t) + segment_origin_offset) / tau_c)

    def _assign_segment(start: float, duration: float) -> int:
        shifted_start = float(start) + segment_origin_offset
        shifted_end = shifted_start + float(duration)

        if assignment_rule == "start_time":
            return _seg_at(start)

        if assignment_rule == "midpoint":
            return int((shifted_start + 0.5 * float(duration)) / tau_c)

        if assignment_rule == "random_overlap":
            start_seg = int(shifted_start / tau_c)
            end_seg = int(shifted_end / tau_c)
            if duration <= 0.0 or end_seg == start_seg:
                return start_seg

            segments = list(range(start_seg, end_seg + 1))
            weights = []
            for seg in segments:
                left = max(shifted_start, seg * tau_c)
                right = min(shifted_end, (seg + 1) * tau_c)
                weights.append(max(0.0, right - left))

            total = sum(weights)
            if total <= 0.0:
                return start_seg

            r = assignment_rng.random() * total
            acc = 0.0
            for seg, weight in zip(segments, weights):
                acc += weight
                if r <= acc:
                    return int(seg)
            return int(segments[-1])

        start_seg = int(shifted_start / tau_c)
        end_seg = int(shifted_end / tau_c)

        if end_seg == start_seg:
            return start_seg
        elif end_seg == start_seg + 1:
            divide_spot = (start_seg + 1) * tau_c
            pre = divide_spot - shifted_start
            post = shifted_end - divide_spot
            return end_seg if post > pre else start_seg
        elif end_seg > start_seg + 1:
            return start_seg + 1
        return start_seg

    two_q_slots = int(round(t2 / t1))  # 确保 t2 是 t1 的倍数，如果是的话，可以按照相同方式来调整
    if abs(two_q_slots * t1 - t2) > 1e-9:
        raise ValueError(f"t2={t2} 不是 t1={t1} 的整数倍，无法构造整齐 time slice。")

    timing_map: Dict[cirq.Operation, List[GateTimingInfo]] = {}
    seen_count: Dict[cirq.Operation, int] = {}

    start_time: float = 0.0
    # 遍历每个 moment 和其内部操作
    for m_idx, moment in enumerate(circuit.moments):
        moment_duration:float = 0.0
        op_times: List[float] = []

        for o_idx, op in enumerate(moment.operations):
            op_duration:float = 0.0
            op_true:cirq.Operation = None

            if len(op.qubits) == 1:
                if isinstance(op, cirq.CircuitOperation):
                    if all(isinstance(o.gate, cirq.ZPowGate) for o in list(op.circuit.all_operations())):
                        op_duration = 0.0
                        op_true = list(op.circuit.all_operations())[0]

                    else:
                        op_duration = t1
                        for o in list(op.circuit.all_operations()):
                            if isinstance(o.gate, cirq.YPowGate):
                                op_true = o
                                break
                else:
                    op_true = op
                    if isinstance(op.gate, cirq.ZPowGate):
                        op_duration = 0.0
                    elif isinstance(op.gate, cirq.WaitGate):
                        op_duration = op.gate.duration.total_nanos()
                    else:
                        op_duration = t1
            elif len(op.qubits) == 2:
                op_duration = t2
                if isinstance(op, cirq.CircuitOperation):
                    for o in op.circuit.all_operations():
                            if len(o.qubits) == 2:
                                op_true = o
                                break
                else:
                    op_true = op
            
            op_times.append(op_duration)

            segment_id = _assign_segment(start_time, op_duration)

            occ = seen_count.get(op, 0)
            seen_count[op] = occ + 1

            info = GateTimingInfo(
                start_time = start_time,
                duration = op_duration,
                segment_id = segment_id,
                moment_idx = m_idx,
                op_idx = o_idx,
                occurrence = occ,
                op = op_true
            )
            timing_map.setdefault(op, []).append(info)

        moment_duration = max(op_times)
        start_time += moment_duration

    return TimedCircuitContext(circuit=circuit, timing_map=timing_map)
