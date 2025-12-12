from cirq.noise.utils.composite_noise_model import CompositeNoiseModel

def get_noise_model(config: dict):
    from cirq.noise.models import (
        idle_noise_model,
        photon_decay_noise_model,
        Ry_gate_noise_model,
        sample_flux_noise_model,
        segmented_flux_noise_model,
        average_flux_noise_model,
    )
    NAME_TO_MODEL = {
        "idle": idle_noise_model.IdleNoiseModel,
        "photon_decay": photon_decay_noise_model.PhotonDecayNoiseModel,
        "Ry": Ry_gate_noise_model.RyGateNoiseModel,
        "sample_flux": sample_flux_noise_model.SampleFluxNoiseModel,
        "segmented_flux": segmented_flux_noise_model.SegmentedFluxNoiseModel,
        "average_flux": average_flux_noise_model.AverageFluxNoiseModel,
    }
    noise_items = config.get("models", [])
    if not noise_items:
        raise ValueError("Noise model config must contain a non-empty 'models' list")

    models = []
    for item in noise_items:
        name = item["name"]
        params = item.get("params", {})
        if name not in NAME_TO_MODEL:
            raise ValueError(f"Unknown noise model: {name}")
        model_cls = NAME_TO_MODEL[name]
        models.append(model_cls(**params))

    return CompositeNoiseModel(models)