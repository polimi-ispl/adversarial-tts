"""Catch cwd/path regressions and collisions between the two component module roots."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def launch(component, *args, cwd=None, env=None):
    return subprocess.run(
        [PYTHON, str(ROOT / "scripts/run_component.py"), component, *args],
        cwd=cwd,
        env=dict(os.environ, ACOUSTIC_PYTHON=PYTHON, VOCODER_PYTHON=PYTHON, **(env or {})),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_doctor_keeps_module_roots_separate_even_with_polluted_pythonpath(tmp_path):
    # Removing the child's PYTHONPATH cleanup must expose the wrong module root here.
    for component, module in [("acoustic", "utils"), ("vocoder", "models")]:
        other = ROOT / ("vocoder" if component == "acoustic" else "acoustic")
        result = launch(component, "doctor", cwd=tmp_path, env={"PYTHONPATH": str(other)})
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert Path(report["cwd"]) == ROOT / component
        assert report["pythonpath"] is None
        assert str(ROOT / component) in report["modules"][module]
        assert Path(report["executable"]).resolve() == Path(PYTHON).resolve()


def test_vocoder_relative_paths_with_spaces_work_from_caller(tmp_path, model_bundle):
    import numpy as np

    work = tmp_path / "caller with spaces"
    work.mkdir()
    inputs = work / "my mels"
    inputs.mkdir()
    np.save(inputs / "voice.npy", np.zeros((80, 8), dtype=np.float32))
    result = launch(
        "vocoder",
        "mel",
        "--input_mels_dir=my mels",
        "--output_dir",
        "my output",
        "--checkpoint_file",
        os.path.relpath(model_bundle / "generator.pt", work),
        "--config",
        os.path.relpath(model_bundle / "config.json", work),
        "--device",
        "cpu",
        cwd=work,
    )
    assert result.returncode == 0, result.stderr
    rate, audio = wavfile.read(work / "my output/voice_generated_e2e.wav")
    assert rate == 24000 and audio.shape == (2048,)


def test_shell_wrapper_preserves_caller_directory(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/vocoder.sh"), "doctor"],
        cwd=tmp_path,
        env=dict(os.environ, VOCODER_PYTHON=PYTHON),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["cwd"] == str(ROOT / "vocoder")


def test_invalid_entry_cannot_escape_component():
    result = launch("acoustic", "../vocoder/train.py")
    assert result.returncode == 2
    assert "Unknown acoustic entry" in result.stderr


def test_invalid_interpreter_is_a_clear_error():
    result = subprocess.run(
        [PYTHON, str(ROOT / "scripts/run_component.py"), "vocoder", "doctor"],
        env=dict(os.environ, VOCODER_PYTHON="/not/a/python"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "VOCODER_PYTHON" in result.stderr and "Traceback" not in result.stderr


def test_missing_path_argument_is_not_reinterpreted_as_a_path():
    result = launch("vocoder", "mel", "--config", "--device", "cpu")
    assert result.returncode == 2
    assert "requires a path" in result.stderr


def test_training_list_paths_and_acoustic_positional_config_use_caller(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "component_launcher", ROOT / "scripts/run_component.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.absolute_arguments(["config.yaml"], "acoustic", tmp_path, "preprocess") == [
        str(tmp_path / "config.yaml")
    ]
    args = module.absolute_arguments(
        [
            "--input_training_file",
            "train.txt",
            "--list_input_unseen_wavs_dir",
            "first data",
            "second",
            "--checkpoint_path",
            "checkpoints",
            "--training_epochs",
            "5",
        ],
        "vocoder",
        tmp_path,
        "train",
    )
    assert args == [
        "--input_training_file",
        str(tmp_path / "train.txt"),
        "--list_input_unseen_wavs_dir",
        str(tmp_path / "first data"),
        str(tmp_path / "second"),
        "--checkpoint_path",
        str(tmp_path / "checkpoints"),
        "--training_epochs",
        "5",
    ]


def test_relative_interpreter_path_is_resolved_before_changing_cwd(tmp_path):
    work = tmp_path / "caller with spaces"
    work.mkdir()
    relative_python = os.path.relpath(sys.executable, work)
    result = subprocess.run(
        [PYTHON, str(ROOT / "scripts/run_component.py"), "vocoder", "doctor"],
        cwd=work,
        env=dict(os.environ, VOCODER_PYTHON=relative_python),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["executable"] == sys.executable


def test_time_major_saved_mel_bridge_feeds_real_vocoder(tmp_path, model_bundle):
    import numpy as np

    source = tmp_path / "saved acoustic mel.npy"
    np.save(source, np.zeros((8, 80), dtype=np.float32))
    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts/mel_bridge.py"),
            "convert",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "mels"),
            "--layout",
            "TC",
            "--num-mels",
            "80",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    result = launch(
        "vocoder",
        "mel",
        "--input_mels_dir",
        str(tmp_path / "mels"),
        "--output_dir",
        str(tmp_path / "audio"),
        "--checkpoint_file",
        str(model_bundle / "generator.pt"),
        "--config",
        str(model_bundle / "config.json"),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    rate, wave = wavfile.read(tmp_path / "audio/saved acoustic mel_generated_e2e.wav")
    assert rate == 24000 and wave.shape == (2048,) and np.isfinite(wave).all()


@pytest.mark.parametrize("style", ["separate", "equals"])
def test_reference_embedding_path_is_caller_relative_with_spaces(tmp_path, style):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "component_launcher", ROOT / "scripts/run_component.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    caller = tmp_path / "caller with spaces"
    caller.mkdir()
    args = (
        ["--reference_embedding", "speaker embeddings/a.npy"]
        if style == "separate"
        else ["--reference_embedding=speaker embeddings/a.npy"]
    )
    assert module.absolute_arguments(args, "acoustic", caller, "inference") == [
        "--reference_embedding",
        str(caller / "speaker embeddings/a.npy"),
    ]


@pytest.mark.parametrize(
    "flag,path",
    [
        ("-p", "configs/preprocess.yaml"),
        ("-m", "model.yaml"),
        ("-t", "train.yaml"),
        ("-p", "config/pre=process.yaml"),
    ],
)
def test_attached_short_config_paths_are_caller_relative(tmp_path, flag, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "component_launcher", ROOT / "scripts/run_component.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.absolute_arguments(
        [flag + path, "--pitch_control", "1.25"], "acoustic", tmp_path, "inference"
    ) == [flag, str(tmp_path / path), "--pitch_control", "1.25"]
