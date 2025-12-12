import cirq
from typing import List, Sequence, Generator, Union, Iterable

def flatten_ops(seq: Sequence[Union[cirq.Operation, Sequence]]) -> Generator[cirq.Operation, None, None]:
    """Recursively flatten a sequence of Operations and nested Sequences."""
    for item in seq:
        if isinstance(item, cirq.Operation):
            yield item
        elif isinstance(item, Sequence):
            yield from flatten_ops(item)
        else:
            raise TypeError(f"Expected cirq.Operation or Sequence, got {type(item)}: {item}")

def split_op_list(op_list, target_op):
    if target_op not in op_list:
        return [], op_list  # 全部是前部分，后面是空

    index = op_list.index(target_op)
    part1 = op_list[ : index ]  
    part2 = op_list[index + 1 : ]  
    return part1, part2

class CompositeNoiseModel(cirq.NoiseModel):

    def __init__(self, noise_models: List[cirq.NoiseModel]):
        self.op_models = []
        self.moment_models = []

        for model in noise_models:
            if hasattr(model, 'noise_model_type'):
                if model.noise_model_type() == 'operation':
                    self.op_models.append(model)
                elif model.noise_model_type() == 'moment':
                    self.moment_models.append(model)
                else:
                    raise ValueError(f"Unknown noise model type declared: {model}")
            else:
                raise ValueError(f"Noise model {model} missing noise_model_type declaration.")
            
    def print_models(self):
        print("Operation Models:")
        for model in self.op_models:
            print(f" - {model}")
        print("Moment Models:")
        for model in self.moment_models:
            print(f" - {model}")

    def noisy_moment(
        self, moment: Iterable[cirq.Moment], system_qubits: Sequence[cirq.Qid]
    ) -> cirq.Moment:
        
        final_moment:cirq.Moment = cirq.Moment()
        ops = moment.operations

        for op in ops:
            new_ops_left = []
            new_ops_right = []

            # 对该 op 施加所有 op 级 model, 得到新的 ops 列表
            for model in self.op_models:

                noisy_ops = model._find_noise_ops(op)

                if isinstance(noisy_ops, tuple):
                    pre_ops, post_ops = noisy_ops
                    new_ops_left.extend(pre_ops)
                    new_ops_right.extend(post_ops)
                else:
                    new_ops_right.extend(noisy_ops)

            if isinstance(op, cirq.CircuitOperation):
                final_moment += cirq.CircuitOperation(cirq.FrozenCircuit(new_ops_left + list(op.circuit.all_operations()) + new_ops_right))
            elif isinstance(op, cirq.Operation):
                final_moment += cirq.CircuitOperation(cirq.FrozenCircuit(new_ops_left + [op] + new_ops_right))

        # 处理 moment_models
        for model in self.moment_models:
            
            final_moment = model.noisy_moment(final_moment, system_qubits)
            
        return final_moment
