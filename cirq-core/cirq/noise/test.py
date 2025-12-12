import cirq
import numpy as np

class FluxNoiseModel(cirq.NoiseModel):
    def __init__(self, sigma, targets=None):
        self.sigma = sigma
        self.targets = targets

    def noisy_operation(self, operation):
        if isinstance(operation.gate, cirq.CZPowGate):
            delta_phi = np.random.normal(0, self.sigma)
            qubits = operation.qubits
            noise = cirq.ZPowGate(exponent=2 * delta_phi / np.pi).on_each(*qubits)
            return [operation, *noise]
        return operation
    
    def seen(self):
        print("changes can be seen!")
        return True
