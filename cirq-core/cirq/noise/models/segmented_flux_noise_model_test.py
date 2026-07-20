import cirq

from cirq.noise.models.segmented_flux_noise_model import SegmentedFluxNoiseModel
from cirq.noise.utils.timed_circuit_context import assign_timed_circuit_context


def _is_flux_marker(op: cirq.Operation) -> bool:
    name = getattr(getattr(op, "gate", None), "name", "")
    return isinstance(name, str) and name.startswith("eD(")


def _flatten_ops(circuit: cirq.Circuit):
    for op in circuit.all_operations():
        if isinstance(op, cirq.CircuitOperation):
            yield from op.circuit.all_operations()
        else:
            yield op


def _flux_counts_by_previous_gate(circuit: cirq.Circuit):
    counts = {}
    previous_non_flux = None
    for op in _flatten_ops(circuit):
        if _is_flux_marker(op):
            key = type(previous_non_flux.gate).__name__ if previous_non_flux is not None else "None"
            counts[key] = counts.get(key, 0) + 1
        else:
            previous_non_flux = op
    return counts


def test_segmented_flux_skips_virtual_z_and_wait_gate():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.ZPowGate(exponent=0.25).on(q),
        cirq.YPowGate(exponent=0.5).on(q),
        cirq.wait(q, nanos=40),
    )
    ctx = assign_timed_circuit_context(circuit, t1=20.0, t2=40.0, tau_c=60.0)
    delta_phis = {0: {q: 0.123}}

    noisy = circuit.with_noise(SegmentedFluxNoiseModel(ctx.timing_map, delta_phis))
    counts = _flux_counts_by_previous_gate(noisy)

    assert counts == {"YPowGate": 1}
