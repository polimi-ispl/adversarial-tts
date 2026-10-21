"""Validate and export acoustic postnet predictions without changing their scale."""

import numpy as np


def postnet_to_mels(postnet, lengths, num_mels):
    values = np.asarray(postnet)
    if values.ndim != 3 or values.shape[2] != num_mels or any(x == 0 for x in values.shape):
        raise ValueError(f"Expected postnet [batch, time, {num_mels}], got {values.shape}")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Acoustic predictions must be finite real numeric mels")
    lengths = list(lengths)
    if len(lengths) != values.shape[0] or any(
        type(n) is not int or not 0 < n <= values.shape[1] for n in lengths
    ):
        raise ValueError(
            "Acoustic lengths must be positive integers within the predicted frame count"
        )
    return [values[i, :length, :].T.copy() for i, length in enumerate(lengths)]
