"""Legacy checkpoint compatibility must retain safe loading and exact trained bins."""

import numpy as np
import pytest
import torch


def forbidden_metadata(marker):
    marker.write_text("pickle executed")


class ForeignMetadata:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return forbidden_metadata, (self.marker,)


def test_legacy_numpy_scalars_do_not_require_unrestricted_pickle(tmp_path):
    from tts.acoustic_assets import load_acoustic_state

    path = tmp_path / "legacy.pt"
    torch.save(
        {"model": {"weight": torch.arange(3)}, "optimizer": [np.float64(0.1), np.float32(2)]},
        path,
    )
    assert torch.equal(load_acoustic_state(path)["weight"], torch.arange(3))


def test_foreign_pickle_metadata_is_rejected_without_execution(tmp_path):
    from tts.acoustic_assets import load_acoustic_state

    path = tmp_path / "foreign.pt"
    marker = tmp_path / "executed"
    torch.save({"model": {"weight": torch.zeros(1)}, "metadata": ForeignMetadata(marker)}, path)
    with pytest.raises(Exception, match="Weights only load failed"):
        load_acoustic_state(path)
    assert not marker.exists()


def bins():
    return {
        "variance_adaptor.pitch_bins": torch.tensor([-3.0, -0.5, 2.0, 9.0]),
        "variance_adaptor.energy_bins": torch.tensor([-1.0, 0.25, 0.5, 4.0]),
    }


def test_missing_stats_bootstraps_only_endpoints_from_complete_trained_bins():
    from tts.acoustic_assets import constructor_stats

    assert constructor_stats(bins(), 5) == {"pitch": [-3.0, 9.0], "energy": [-1.0, 4.0]}


@pytest.mark.parametrize("invalid", ["missing", "shape", "nan", "decreasing"])
def test_incomplete_or_invalid_checkpoint_bins_cannot_replace_stats(invalid):
    from tts.acoustic_assets import constructor_stats

    state = bins()
    key = "variance_adaptor.pitch_bins"
    if invalid == "missing":
        state.pop(key)
    elif invalid == "shape":
        state[key] = state[key][None, :]
    elif invalid == "nan":
        state[key][1] = float("nan")
    else:
        state[key] = state[key].flip(0)
    with pytest.raises(ValueError, match="pitch_bins"):
        constructor_stats(state, 5)


def test_strict_restore_keeps_each_trained_bin_including_nonuniform_intervals():
    from tts.acoustic_assets import restore_acoustic_state

    model = torch.nn.Module()
    model.variance_adaptor = torch.nn.Module()
    for feature in ["pitch", "energy"]:
        model.variance_adaptor.register_parameter(
            f"{feature}_bins", torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        )
    state = bins()
    restore_acoustic_state(model, state)
    for key, value in state.items():
        assert torch.equal(model.state_dict()[key].view(torch.uint8), value.view(torch.uint8))
    with pytest.raises(RuntimeError, match="Missing key"):
        restore_acoustic_state(
            model, {"variance_adaptor.pitch_bins": state["variance_adaptor.pitch_bins"]}
        )
