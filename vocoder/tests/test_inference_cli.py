"""Check user-visible inference behavior with real CLI processes and audio I/O."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"wav": "inference.py", "mel": "inference_e2e.py"}
INPUT_FLAGS = {"wav": "--input_wavs_dir", "mel": "--input_mels_dir"}
OUTPUT_NAMES = {"wav": "voice_generated.wav", "mel": "voice_generated_e2e.wav"}


def run_cli(kind, bundle, inputs, output, *, cwd=None, extra=(), config=True):
    args = [
        sys.executable,
        str(ROOT / SCRIPTS[kind]),
        INPUT_FLAGS[kind],
        str(inputs),
        "--output_dir",
        str(output),
        "--checkpoint_file",
        str(bundle / "generator.pt"),
    ]
    if config:
        args.extend(["--config", str(bundle / "config.json")])
    args.extend(extra)
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("kind,frames", [("wav", 4096), ("mel", 2048)])
def test_inference_from_another_directory_filters_inputs_and_uses_configured_rate(
    kind, frames, model_bundle, sample_inputs, tmp_path
):
    output = tmp_path / "output"
    result = run_cli(
        kind, model_bundle, sample_inputs[kind], output, cwd=tmp_path, extra=("--device", "cpu")
    )
    assert result.returncode == 0, result.stderr
    rate, audio = wavfile.read(output / OUTPUT_NAMES[kind])
    assert rate == 24000
    assert audio.shape == (frames,)
    assert audio.dtype == np.int16
    assert np.isfinite(audio).all()
    assert len(list(output.glob("*.wav"))) == 1


@pytest.mark.parametrize("kind", ["wav", "mel"])
def test_completed_outputs_are_skipped_without_loading_model(
    kind, model_bundle, sample_inputs, tmp_path
):
    bundle = tmp_path / "broken-checkpoint"
    bundle.mkdir()
    (bundle / "config.json").write_bytes((model_bundle / "config.json").read_bytes())
    (bundle / "generator.pt").write_bytes(b"This file must not be loaded.")
    output = tmp_path / "output"
    output.mkdir()
    target = output / OUTPUT_NAMES[kind]
    target.write_bytes(b"completed")
    result = run_cli(kind, bundle, sample_inputs[kind], output)
    assert result.returncode == 0, result.stderr
    assert "No new files" in result.stdout
    assert target.read_bytes() == b"completed"


@pytest.mark.parametrize("kind", ["wav", "mel"])
def test_overwrite_regenerates_existing_output(kind, model_bundle, sample_inputs, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / OUTPUT_NAMES[kind]).write_bytes(b"old")
    result = run_cli(kind, model_bundle, sample_inputs[kind], output, extra=("--overwrite",))
    assert result.returncode == 0, result.stderr
    rate, audio = wavfile.read(output / OUTPUT_NAMES[kind])
    assert rate == 24000 and audio.size > 0


@pytest.mark.parametrize("shape", [(79, 8), (2, 80, 8), (80, 0)])
def test_invalid_mel_shape_is_reported_without_a_traceback(shape, model_bundle, tmp_path):
    inputs = tmp_path / "mels"
    inputs.mkdir()
    np.save(inputs / "bad.npy", np.zeros(shape, dtype=np.float32))
    result = run_cli("mel", model_bundle, inputs, tmp_path / "output")
    assert result.returncode == 2
    assert "mel" in result.stderr.lower() and "shape" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert not list((tmp_path / "output").glob("*.wav"))


def test_nonfinite_mels_are_rejected(model_bundle, tmp_path):
    inputs = tmp_path / "mels"
    inputs.mkdir()
    np.save(inputs / "bad.npy", np.full((80, 8), np.nan, dtype=np.float32))
    result = run_cli("mel", model_bundle, inputs, tmp_path / "output")
    assert result.returncode == 2
    assert "finite" in result.stderr.lower()
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("malformed", ["empty", "truncated", "archive"])
def test_malformed_mel_files_have_clear_errors(malformed, model_bundle, tmp_path):
    inputs = tmp_path / "mels"
    inputs.mkdir()
    path = inputs / "bad.npy"
    if malformed == "empty":
        path.write_bytes(b"")
    elif malformed == "truncated":
        np.save(path, np.zeros((80, 8), dtype=np.float32))
        path.write_bytes(path.read_bytes()[:30])
    else:
        with path.open("wb") as file:
            np.savez(file, mel=np.zeros((80, 8), dtype=np.float32))
    result = run_cli("mel", model_bundle, inputs, tmp_path / "output")
    assert result.returncode == 2
    assert str(path) in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", ["wav", "mel"])
@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_empty_input_directory_has_clear_error(kind, exists, model_bundle, tmp_path):
    inputs = tmp_path / "input"
    if exists:
        inputs.mkdir()
    result = run_cli(kind, model_bundle, inputs, tmp_path / "output")
    assert result.returncode == 2
    assert str(inputs) in result.stderr
    assert "Traceback" not in result.stderr


def test_mel_default_config_is_beside_checkpoint(model_bundle, sample_inputs, tmp_path):
    result = run_cli(
        "mel", model_bundle, sample_inputs["mel"], tmp_path / "output", cwd=tmp_path, config=False
    )
    assert result.returncode == 0, result.stderr


def test_unavailable_cuda_is_reported(model_bundle, sample_inputs, tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    result = run_cli(
        "mel", model_bundle, sample_inputs["mel"], tmp_path / "output", extra=("--device", "cuda")
    )
    assert result.returncode == 2
    assert "CUDA" in result.stderr and "available" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
@pytest.mark.parametrize("kind", ["wav", "mel"])
def test_explicit_mps_inference(kind, model_bundle, sample_inputs, tmp_path):
    result = run_cli(
        kind, model_bundle, sample_inputs[kind], tmp_path / "output", extra=("--device", "mps")
    )
    assert result.returncode == 0, result.stderr
    assert "Device: mps" in result.stdout
    rate, audio = wavfile.read(tmp_path / "output" / OUTPUT_NAMES[kind])
    assert rate == 24000 and audio.ndim == 1 and audio.size > 0


def test_shell_entry_point_runs_from_another_directory(model_bundle, sample_inputs, tmp_path):
    env = dict(os.environ, BIGVGAN_PYTHON=sys.executable)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/infer.sh"),
            "mel",
            "--input_mels_dir",
            str(sample_inputs["mel"]),
            "--output_dir",
            str(tmp_path / "output"),
            "--checkpoint_file",
            str(model_bundle / "generator.pt"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output" / OUTPUT_NAMES["mel"]).is_file()
