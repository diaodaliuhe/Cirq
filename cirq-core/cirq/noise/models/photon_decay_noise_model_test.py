import cirq

from cirq.noise.models.photon_decay_noise_model import PhotonDecayNoiseModel
from cirq.noise.utils.timed_circuit_context import assign_timed_circuit_context


def test_photon_decay_cursor_advances_and_fresh_context_resets():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.X(q), cirq.H(q))
    ctx = assign_timed_circuit_context(circuit, t1=20.0, t2=40.0, tau_c=100.0)

    model = PhotonDecayNoiseModel(ctx=ctx)
    assert model._moment_cursor == 0

    cirq.DensityMatrixSimulator().simulate(circuit.with_noise(model))
    assert model._moment_cursor == model._num_moments

    fresh = model.with_timed_context(ctx)
    assert fresh._moment_cursor == 0
    assert fresh._num_moments == model._num_moments
