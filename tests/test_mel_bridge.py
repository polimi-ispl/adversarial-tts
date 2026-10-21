"""Validate observed config mismatches and preserve samples through explicit layout changes."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts/mel_bridge.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True, timeout=20
    )


def test_real_configs_match_fields_but_report_frontend_difference():
    result = run(
        "inspect",
        "--acoustic-config",
        ROOT / "acoustic/config/mel_infer/preprocess.yaml",
        "--vocoder-config",
        ROOT / "vocoder/configs/config.json",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["configuration_match"] is True
    assert report["acoustic"]["sampling_rate"] == 22050
    assert report["acoustic"]["num_mels"] == 80
    assert report["acoustic"]["hop_size"] == 256
    assert report["frontend_equivalent"] is False
    assert report["padding_samples_per_side"] == {"acoustic": 512, "vocoder": 384}
    assert report["checkpoint_compatibility"] == "unverified"


@pytest.mark.parametrize(
    "field,value",
    [("sampling_rate", 24000), ("num_mels", 100), ("fmax", 11000), ("upsample_rates", [4, 4])],
)
def test_config_mismatch_is_reported_without_claiming_compatibility(field, value, tmp_path):
    config = json.loads((ROOT / "vocoder/configs/config.json").read_text())
    config[field] = value
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    result = run(
        "inspect",
        "--acoustic-config",
        ROOT / "acoustic/config/mel_infer/preprocess.yaml",
        "--vocoder-config",
        p,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["configuration_match"] is False
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("layout,shape", [("TC", (3, 2)), ("CT", (2, 3))])
def test_single_layout_conversion_preserves_values(layout, shape, tmp_path):
    values = np.arange(6, dtype=np.float64).reshape(shape)
    source = tmp_path / "voice.npy"
    np.save(source, values)
    output = tmp_path / "converted"
    result = run(
        "convert", "--input", source, "--output-dir", output, "--layout", layout, "--num-mels", 2
    )
    assert result.returncode == 0, result.stderr
    expected = (
        [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]] if layout == "TC" else [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    )
    actual = np.load(output / "voice.npy")
    assert actual.tolist() == expected
    assert actual.dtype == np.float64


def test_batched_mels_require_and_apply_true_lengths(tmp_path):
    source = tmp_path / "batch.npy"
    np.save(source, np.arange(12, dtype=np.float32).reshape(2, 3, 2))
    args = (
        "convert",
        "--input",
        source,
        "--output-dir",
        tmp_path / "output",
        "--layout",
        "BTC",
        "--num-mels",
        2,
    )
    result = run(*args)
    assert result.returncode == 2
    assert "lengths" in result.stderr
    lengths = tmp_path / "lengths.json"
    lengths.write_text("[2, 1]")
    result = run(*args, "--lengths", lengths)
    assert result.returncode == 0, result.stderr
    assert np.load(tmp_path / "output/batch_0000.npy").tolist() == [[0.0, 2.0], [1.0, 3.0]]
    assert np.load(tmp_path / "output/batch_0001.npy").tolist() == [[6.0], [7.0]]


@pytest.mark.parametrize(
    "values,layout",
    [
        (np.zeros((3, 2)), "CT"),
        (np.zeros((2, 0)), "CT"),
        (np.full((2, 3), np.nan), "CT"),
        (np.array([["private"]], dtype=object), "CT"),
        (np.zeros((0, 2, 3)), "BCT"),
        (np.ones((2, 3), dtype=np.complex64), "CT"),
    ],
)
def test_invalid_arrays_fail_before_writing(values, layout, tmp_path):
    source = tmp_path / "bad.npy"
    np.save(source, values)
    result = run(
        "convert",
        "--input",
        source,
        "--output-dir",
        tmp_path / "output",
        "--layout",
        layout,
        "--num-mels",
        2,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr and "error:" in result.stderr
    assert not list((tmp_path / "output").glob("*.npy"))


@pytest.mark.parametrize("lengths", ["[0]", "[4]", "[true]", "[1.5]", "[1,2]"])
def test_invalid_true_lengths_are_rejected(lengths, tmp_path):
    source = tmp_path / "voice.npy"
    np.save(source, np.zeros((2, 3)))
    meta = tmp_path / "lengths.json"
    meta.write_text(lengths)
    result = run(
        "convert",
        "--input",
        source,
        "--output-dir",
        tmp_path / "output",
        "--layout",
        "CT",
        "--num-mels",
        2,
        "--lengths",
        meta,
    )
    assert result.returncode == 2
    assert "length" in result.stderr.lower()


def test_existing_outputs_are_not_overwritten_implicitly(tmp_path):
    source = tmp_path / "voice.npy"
    np.save(source, np.zeros((2, 3)))
    output = tmp_path / "output"
    output.mkdir()
    target = output / "voice.npy"
    target.write_bytes(b"preserve")
    args = ("convert", "--input", source, "--output-dir", output, "--layout", "CT", "--num-mels", 2)
    result = run(*args)
    assert result.returncode == 2 and target.read_bytes() == b"preserve"
    result = run(*args, "--overwrite")
    assert result.returncode == 0
    assert np.load(target).shape == (2, 3)


def test_conversion_never_replaces_the_input_even_with_overwrite(tmp_path):
    source = tmp_path / "voice.npy"
    np.save(source, np.zeros((2, 3)))
    before = source.read_bytes()
    result = run(
        "convert",
        "--input",
        source,
        "--output-dir",
        tmp_path,
        "--layout",
        "CT",
        "--num-mels",
        2,
        "--overwrite",
    )
    assert result.returncode == 2 and source.read_bytes() == before
    assert "input mel" in result.stderr


def test_corrupt_zip_has_clear_error_with_input_path(tmp_path):
    source = tmp_path / "corrupt mel.npy"
    source.write_bytes(b"PK\x03\x04bad")
    result = run(
        "convert",
        "--input",
        source,
        "--output-dir",
        tmp_path / "output",
        "--layout",
        "CT",
        "--num-mels",
        2,
    )
    assert result.returncode == 2
    assert str(source) in result.stderr
    assert "Traceback" not in result.stderr
