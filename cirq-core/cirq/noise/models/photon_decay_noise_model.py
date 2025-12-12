import cirq
import numpy as np
from typing import Sequence, Dict, List, Optional, Tuple
from collections import defaultdict
from cirq.noise.utils.timed_circuit_context import TimedCircuitContext, GateTimingInfo

class PhotonDecayNoiseModel(cirq.NoiseModel):
    def __init__(
        self, 
        ctx: Optional[TimedCircuitContext] = None,
        chi: float = np.pi * (-2.6e6),
        alpha_0: float = np.sqrt(0.8),
        kappa: float = 4e6,
        tau_m: float = -590e-9,   
        tau_g: float = 10e-9,     
        scale: float = 1.0,
    ):
        """
        chi:   qubit frequency shift per photon (rad/s)
        alpha_0: photon field amplitude at t = t_g (sqrt[photon number])
        kappa: resonator decay rate (1/s)
        tau_m: measurement pulse duration (s), 用在 exp(kappa * tau_m) 那个因子
        scale: 外部缩放，和你之前一样
        timed_context: 带 timing_map 的上下文，用于按门计算噪声
        """
        self.ctx = ctx
        self.timing_map = ctx.timing_map

        self.chi = chi
        self.alpha_0 = alpha_0
        self.kappa = kappa
        self.tau_m = tau_m
        self.tau_g = tau_g
        self.scale = scale

        (   self._moment_mid_s,   # List[float]
            self._moment_end_s,   # List[float]
        ) = self._build_moment_time_arrays(ctx)

        self._num_moments = len(self._moment_mid_s)
        # Cirq 按顺序调用 noisy_moment，这里用 cursor 对齐“第几个 moment”
        self._moment_cursor = 0

    def _build_moment_time_arrays(
        self,
        ctx: TimedCircuitContext,
    ) -> Tuple[List[float], List[float]]:
        """
        根据 ctx.timing_map 计算每个 moment 的：
          - 左端点 left_ns[m] = 本 moment 所有门 start_time 的最小值
          - 右端点 right_ns[m] = 本 moment 所有门 (start_time + duration) 的最大值
        然后得到：
          - mid_s[m]  = (left_ns[m] + right_ns[m]) / 2 * 1e-9
          - end_s[m]  = right_ns[m] * 1e-9
        若某个 moment 没有任何时序（极少见），退化成长度 0 的区间。
        """

        n_moments = len(ctx.circuit.moments)
        if n_moments == 0:
            return [], []

        # 初始化为 None，后面用 None 判断有没有门
        left_ns: List[Optional[float]] = [None] * n_moments
        right_ns: List[Optional[float]] = [None] * n_moments

        # 聚合所有 GateTimingInfo
        for infos in ctx.timing_map.values():
            for info in infos:
                m = info.moment_idx
                if m < 0 or m >= n_moments:
                    continue
                s = info.start_time              # ns
                e = info.start_time + info.duration  # ns

                if left_ns[m] is None or s < left_ns[m]:
                    left_ns[m] = s
                if right_ns[m] is None or e > right_ns[m]:
                    right_ns[m] = e

        # 把 None 的情况补齐（没有门的 moment 退化为上一个的结束点，首个则为 0）
        last_end_ns = 0.0
        for m in range(n_moments):
            if left_ns[m] is None and right_ns[m] is None:
                # 完全没有门：用上一 moment 的 end
                left_ns[m] = last_end_ns
                right_ns[m] = last_end_ns
            elif left_ns[m] is None:
                # 理论上不太会发生，保险起见
                left_ns[m] = right_ns[m]
            elif right_ns[m] is None:
                right_ns[m] = left_ns[m]

            last_end_ns = right_ns[m]

        # 计算中点和结束时间（秒）
        mid_s: List[float] = []
        end_s: List[float] = []
        for m in range(n_moments):
            mid_ns = 0.5 * (left_ns[m] + right_ns[m])
            mid_s.append(mid_ns * 1e-9)
            end_s.append(right_ns[m] * 1e-9)

        return mid_s, end_s

    # ------- 数学部分: 算出对应的 phase damping gamma -------

    def _F(self, t: float) -> float:
        """对应公式里的 F(t)，t 用秒."""
        k = self.kappa
        c = self.chi
        exp_term = np.exp(- k * t)
        denom = 4 * c**2 + k**2
        return exp_term / denom * (- k * np.sin(2 * c * t) - 2 * c * np.cos(2 * c * t))

    def _compute_gamma_interval(self, t1: float, t2: float) -> float:
        """
        根据论文公式，对任意时间区间 [t1, t2] (单位: 秒) 计算
        phase damping 的 gamma = 1 - exp(-Lambda)。

        这里假定 t_g = 0，t_m = tau_m。
        """
        if t2 <= t1:
            return 0.0

        prefactor = 2 * self.chi * self.alpha_0 * np.exp(self.kappa * (self.tau_m - self.tau_g))
        assist_diff = self._F(t2 - self.tau_g) - self._F(t1 - self.tau_g)
        Lambda = prefactor * assist_diff
        # (1) 物理上 Gamma_d(t) integration 不允许减少相干性——负值设为 0
        print
        if Lambda <= 0:
            return 0.0

        # (2) 计算 phase damping 参数 gamma
        gamma = 1 - np.exp(-Lambda)

        # (3) clip 防止浮点误差
        return float(np.clip(gamma, 0.0, 1.0))
    
    def _next_gamma(self) -> float:
        """
        依据当前 _moment_cursor 取区间：
          - 非最后一个 moment： [mid[m], mid[m+1]]
          - 最后一个 moment：   [mid[last], end[last]]
        并计算对应 gamma，随后自增 cursor。
        """
        if self._num_moments == 0:
            return 0.0

        idx = self._moment_cursor
        if idx >= self._num_moments:
            return 0.0

        self._moment_cursor += 1

        t_start = self._moment_mid_s[idx]
        if idx < self._num_moments - 1:
            t_end = self._moment_mid_s[idx + 1]
        else:
            t_end = self._moment_end_s[idx]

        gamma0 = self._compute_gamma_interval(t_start, t_end)
        gamma = 1.0 - (1.0 - gamma0) ** self.scale
        # print(f"Moment {idx}: t_start={t_start:.3e}s, t_end={t_end:.3e}s, gamma0={gamma0:.3e}, scaled gamma={gamma:.3e}")
        if gamma < 1e-5:
            gamma = 0.0
        return float(np.clip(gamma, 0.0, 1.0))

    def noisy_moment(
        self,
        moment: cirq.Moment,
        system_qubits: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        gamma = self._next_gamma()

        new_ops: List[cirq.Operation] = []
        used_qubits = set()  # 本 moment 中被某个 gate 使用到的 qubit

        # 1) 处理原来的门：每个门 → 一个 CircuitOperation
        for op in moment.operations:
            used_qubits.update(op.qubits)

            inner_ops: List[cirq.Operation] = []

            if isinstance(op, cirq.CircuitOperation):
                inner_ops.extend(op.circuit.all_operations())
            else:
                inner_ops.append(op)
            if gamma > 0.0:
                # 多比特门：对其所有参与 qubit 都绑上 phase_damp
                for q in op.qubits:
                    inner_ops.append(cirq.phase_damp(gamma).on(q))

            if gamma == 0.0:
                inner_ops.append(cirq.phase_damp(0.0).on(op.qubits[0]))

            new_ops.append(cirq.CircuitOperation(cirq.FrozenCircuit(inner_ops)))

        # 2) 对本 moment 中空闲的 qubit 也加一次 photon decay
        if gamma > 0.0:
            for q in system_qubits:
                if q not in used_qubits:
                    inner_ops = [cirq.phase_damp(gamma).on(q)]
                    new_ops.append(cirq.CircuitOperation(cirq.FrozenCircuit(inner_ops)))
        elif gamma == 0.0:
            for q in system_qubits:
                if q not in used_qubits:
                    # 如果 gamma 是 0.0，那么加入 PD(0.0000)
                    inner_ops = [cirq.phase_damp(0.0).on(q)]
                    new_ops.append(cirq.CircuitOperation(cirq.FrozenCircuit(inner_ops)))

        # 3) 返回一个新的 Moment，里面全是 CircuitOperation
        return cirq.Moment(new_ops)

    # ------- 其它小工具 -------

    def reset(self):
        self._moment_cursor = 0

    def with_timed_context(self, ctx: TimedCircuitContext) -> "PhotonDecayNoiseModel":
        return PhotonDecayNoiseModel(
            ctx=ctx,
            chi=self.chi,
            alpha_0=self.alpha_0,
            kappa=self.kappa,
            tau_m=self.tau_m,
            tau_g=self.tau_g,
            scale=self.scale,
        )

    def noise_model_type(self):
        return "moment"

    def __repr__(self):
        return (
            f"PhotonDecayNoiseModel(chi={self.chi}, alpha_0={self.alpha_0}, "
            f"kappa={self.kappa}, tau_m={self.tau_m}, tau_g={self.tau_g}, "
            f"scale={self.scale}, num_moments={self._num_moments})"
        )
