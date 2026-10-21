#!/usr/bin/env python3
"""Inspect source mel configurations and explicitly reshape already exported numeric mels."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import yaml


def read_contract(acoustic_path, vocoder_path):
    acoustic = yaml.safe_load(acoustic_path.read_text())
    vocoder = json.loads(vocoder_path.read_text())
    p = acoustic["preprocessing"]
    a = {
        "sampling_rate": p["audio"]["sampling_rate"],
        "n_fft": p["stft"]["filter_length"],
        "hop_size": p["stft"]["hop_length"],
        "win_size": p["stft"]["win_length"],
        "num_mels": p["mel"]["n_mel_channels"],
        "fmin": p["mel"]["mel_fmin"],
        "fmax": p["mel"]["mel_fmax"],
    }
    v = {key: vocoder[key] for key in a}
    for label, values in [("acoustic", a), ("vocoder", v)]:
        for key in ["sampling_rate", "n_fft", "hop_size", "win_size", "num_mels"]:
            if type(values[key]) is not int or values[key] <= 0:
                raise ValueError(f"{label} {key} must be a positive integer")
        fmin, fmax = values["fmin"], values["fmax"]
        fmax = values["sampling_rate"] / 2 if fmax is None else fmax
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (fmin, fmax)):
            raise ValueError(f"{label} mel frequency bounds must be finite numbers")
        if not 0 <= fmin < fmax <= values["sampling_rate"] / 2:
            raise ValueError(f"{label} mel frequency bounds are invalid")
        if values["win_size"] > values["n_fft"] or values["hop_size"] > values["n_fft"]:
            raise ValueError(f"{label} window/hop cannot exceed n_fft")
    rates = vocoder["upsample_rates"]
    if not isinstance(rates, list) or not rates or any(type(x) is not int or x <= 0 for x in rates):
        raise ValueError("vocoder upsample_rates must be positive integers")
    mismatches = []
    for key in a:
        av = a[key]
        vv = v[key]
        if key == "fmax":
            av = a["sampling_rate"] / 2 if av is None else av
            vv = v["sampling_rate"] / 2 if vv is None else vv
        if av != vv:
            mismatches.append({"field": key, "acoustic": a[key], "vocoder": v[key]})
    factor = math.prod(rates)
    if factor != v["hop_size"]:
        mismatches.append(
            {
                "field": "upsample_factor_vs_hop",
                "upsample_factor": factor,
                "hop_size": v["hop_size"],
            }
        )
    return {
        "configuration_match": not mismatches,
        "acoustic": a,
        "vocoder": v,
        "mismatches": mismatches,
        "padding_samples_per_side": {
            "acoustic": a["n_fft"] // 2,
            "vocoder": (v["n_fft"] - v["hop_size"]) // 2,
        },
        "frontend_equivalent": False,
        "checkpoint_compatibility": "unverified",
        "source_evidence": {
            "acoustic": [
                "acoustic/audio/stft.py",
                "acoustic/audio/audio_processing.py",
                "acoustic/mel-inference.py",
            ],
            "vocoder": ["vocoder/meldataset.py", "vocoder/inference_utils.py"],
            "log": "ln(clamp(magnitude_mel, min=1e-5)), C=1; no mel mean/variance scaling found in this path",
            "magnitude_difference": "BigVGAN adds 1e-9 inside sqrt; AdaSpeech does not",
            "padding_difference": "AdaSpeech reflect-pads n_fft/2; BigVGAN reflect-pads (n_fft-hop)/2 with center=False",
        },
        "limits": [
            "Config equality is not proof that paper weights or arbitrary saved mels are compatible.",
            "Only the reviewed source paths establish these frontend conventions; inspect any other exporter independently.",
            "Batch mel outputs can contain padding; export with the true per-utterance mel lengths.",
            "No mel rescaling, frame shift, padding repair or checkpoint conversion is performed.",
        ],
    }


def convert_mels(source, output_dir, layout, num_mels, lengths_path=None, overwrite=False):
    if num_mels <= 0:
        raise ValueError("num-mels must be positive")
    try:
        values = np.load(source, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"Cannot read numeric mel file {source}: {error}") from error
    if not isinstance(values, np.ndarray):
        values.close()
        raise ValueError("Expected a NumPy mel array, not an archive")
    expected_rank = 3 if layout.startswith("B") else 2
    if values.ndim != expected_rank or any(size == 0 for size in values.shape):
        raise ValueError(f"Invalid mel shape {values.shape} for {layout}")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Mel values must be finite real numeric values without objects")
    batched = expected_rank == 3
    if not batched:
        values = values[np.newaxis, ...]
    if layout.endswith("TC"):
        values = values.transpose(0, 2, 1)
    if values.shape[1] != num_mels:
        raise ValueError(f"Mel shape {values.shape} does not have {num_mels} frequency channels")
    count, _, frames = values.shape
    if lengths_path is None:
        if batched:
            raise ValueError(
                "Batched mel conversion requires --lengths JSON with true frame lengths"
            )
        lengths = [frames]
    else:
        lengths = json.loads(lengths_path.read_text())
        if not isinstance(lengths, list) or len(lengths) != count:
            raise ValueError("Lengths must be a JSON list with one length per utterance")
        if any(type(length) is not int or not 0 < length <= frames for length in lengths):
            raise ValueError(f"Each true length must be an integer from 1 to {frames}")
    targets = [
        output_dir / (f"{source.stem}_{i:04d}.npy" if batched else f"{source.stem}.npy")
        for i in range(count)
    ]
    for target in targets:
        if target.resolve() == source.resolve():
            raise ValueError("Output cannot replace the input mel file")
        if target.is_symlink() or (target.exists() and (not overwrite or not target.is_file())):
            raise ValueError(
                f"Output already exists or is not a regular file: {target}; use --overwrite for files"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, target in enumerate(targets):
        fd, temporary = tempfile.mkstemp(prefix=".mel-", suffix=".npy", dir=output_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, values[i, :, : lengths[i]], allow_pickle=False)
            if overwrite:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {
        "files": [str(p) for p in targets],
        "layout": "CT",
        "lengths": lengths,
        "values_rescaled": False,
        "normalization_and_checkpoint_compatibility": "unverified",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Compare metadata; source frontend differences remain explicit"
    )
    inspect.add_argument("--acoustic-config", type=Path, required=True)
    inspect.add_argument("--vocoder-config", type=Path, required=True)
    convert = commands.add_parser(
        "convert", help="Reshape one saved array without changing its numeric scale"
    )
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--layout", choices=["CT", "TC", "BCT", "BTC"], required=True)
    convert.add_argument("--num-mels", type=int, required=True)
    convert.add_argument("--lengths", type=Path)
    convert.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            report = read_contract(args.acoustic_config, args.vocoder_config)
            status = 0 if report["configuration_match"] else 2
        else:
            report = convert_mels(
                args.input,
                args.output_dir,
                args.layout,
                args.num_mels,
                args.lengths,
                args.overwrite,
            )
            status = 0
        print(json.dumps(report, indent=2))
        return status
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
