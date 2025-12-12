from __future__ import annotations
import dataclasses
from typing import Callable, List, Sequence, TYPE_CHECKING

from cirq import devices

if TYPE_CHECKING:
    import cirq

@dataclasses.dataclass
class MultiInsertionNoiseModel(devices.NoiseModel):
    """
    Noise model supporting inserting noise ops both before and after an operation.

    noise_rule: A function taking a cirq.Operation and returning a tuple:
                (list of ops before, list of ops after)
    """
    noise_rule: Callable[[cirq.Operation], tuple[List[cirq.Operation], List[cirq.Operation]]]

    def noisy_operation(self, operation: cirq.Operation) -> List[cirq.Operation]:
        pre_ops, post_ops = self.noise_rule(operation)
        return pre_ops + [operation] + post_ops

    def __repr__(self):
        return (
            f"cirq.noise.MultiInsertionNoiseModel("
            f"noise_rule={self.noise_rule})"
        )
