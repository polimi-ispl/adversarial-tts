"""Safely load acoustic weights and preserve the checkpoint's trained variance bins."""


def load_acoustic_state(path):
    import numpy as np
    import torch

    # Older optimizer metadata contains NumPy float scalars. Permit only their
    # known constructors/dtypes; never retry with unrestricted pickle loading.
    scalar = np.float64(0).__reduce__()[0]
    allowlist = [
        (scalar, "numpy.core.multiarray.scalar"),
        (scalar, "numpy._core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(allowlist):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Acoustic checkpoint must contain a model state dictionary")
    state = checkpoint["model"]
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Acoustic model state must contain named tensors only")
    return state


def constructor_stats(state, n_bins):
    """Recover only constructor endpoints; strict restoration supplies every bin."""
    import torch

    stats = {}
    for feature in ["pitch", "energy"]:
        key = f"variance_adaptor.{feature}_bins"
        values = state.get(key)
        if (
            not isinstance(values, torch.Tensor)
            or not values.is_floating_point()
            or values.shape != (n_bins - 1,)
            or values.numel() == 0
            or not torch.isfinite(values).all()
            or not torch.all(values[1:] >= values[:-1])
        ):
            raise ValueError(f"Checkpoint {key} must contain {n_bins - 1} finite sorted bins")
        stats[feature] = [values[0].item(), values[-1].item()]
    return stats


def restore_acoustic_state(model, state):
    import torch

    model.load_state_dict(state, strict=True)
    restored = model.state_dict()
    for feature in ["pitch", "energy"]:
        key = f"variance_adaptor.{feature}_bins"
        actual, expected = restored[key].detach().cpu(), state[key].detach().cpu()
        if actual.dtype != expected.dtype or not torch.equal(
            actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
        ):
            raise ValueError(f"Checkpoint {key} was not restored byte for byte")
