"""Protect actual acoustic-to-waveform data and the public text synthesis interface."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), *map(str, args)],
        cwd=cwd,
        env=dict(os.environ, **(env or {})),
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_help_is_available_without_loading_models():
    result = cli("--help")
    assert (
        result.returncode == 0 and "--text-file" in result.stdout and "--vocoder" in result.stdout
    )


def test_postnet_export_trims_true_lengths_and_preserves_channel_order():
    from tts.contracts import postnet_to_mels

    outputs = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    actual = postnet_to_mels(outputs, [2, 1], 3)
    assert actual[0].tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
    assert actual[1].tolist() == [[12.0], [13.0], [14.0]]


@pytest.mark.parametrize("lengths", [[0, 1], [5, 1], [True, 1], [1.5, 1], [1]])
def test_invalid_acoustic_lengths_are_not_sent_to_vocoder(lengths):
    from tts.contracts import postnet_to_mels

    with pytest.raises(ValueError, match="length"):
        postnet_to_mels(np.zeros((2, 4, 3)), lengths, 3)


def test_missing_bundle_is_clear_before_component_loading(tmp_path):
    result = cli(
        "--config",
        tmp_path / "missing.yaml",
        "--text",
        "hello",
        "--reference-embedding",
        "a.npy",
        "--reference-mel",
        "b.npy",
        "--output-dir",
        tmp_path / "output",
    )
    assert (
        result.returncode == 2
        and "missing.yaml" in result.stderr
        and "Traceback" not in result.stderr
    )


def test_unknown_device_has_clear_cli_error():
    result = cli("--acoustic-device", "not-a-device")
    assert (
        result.returncode == 2
        and "not-a-device" in result.stderr
        and "Traceback" not in result.stderr
    )


def assets(bundle, output):
    return [
        "--config",
        bundle / "tts.yaml",
        "--reference-embedding",
        bundle / "embedding.npy",
        "--reference-mel",
        bundle / "reference.npy",
        "--output-dir",
        output,
        "--acoustic-python",
        sys.executable,
        "--vocoder-python",
        sys.executable,
        "--acoustic-device",
        "cpu",
        "--vocoder-device",
        "cpu",
    ]


def test_actual_text_acoustic_postnet_and_selected_vocoder_produce_waveform(tts_bundle, tmp_path):
    output = tmp_path / "new speech"
    result = cli(
        *assets(tts_bundle, output),
        "--text",
        "hello",
        "--vocoder",
        "VR",
        "--save-mels",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    rate, wave = wavfile.read(output / "speech.wav")
    assert rate == 22050 and wave.shape == (2048,) and np.isfinite(wave).all()
    mel = np.load(output / "mels/speech.npy")
    assert mel.shape == (80, 8) and np.isfinite(mel).all()
    summary = json.loads((output / "synthesis.json").read_text())
    assert summary["utterances"] == [{"id": "speech", "frames": 8, "tokens": 4}]
    assert summary["vocoder_checkpoint"].endswith("VR-generator.pt")
    assert not (ROOT / "acoustic/output").exists()


def test_legacy_checkpoint_with_embedded_bins_needs_no_external_stats(tts_bundle, tmp_path):
    import torch
    import yaml

    config = yaml.safe_load((tts_bundle / "tts.yaml").read_text())
    for section in [config["acoustic"], *config["vocoders"].values()]:
        for key, value in section.items():
            section[key] = str(tts_bundle / value)
    config["acoustic"].pop("stats")
    state = torch.load(tts_bundle / "acoustic.pt", weights_only=True)
    # Keep nonuniform trained bins so endpoint-based regeneration cannot pass.
    state["model"]["variance_adaptor.pitch_bins"][2] += 0.25
    state["optimizer"] = {"legacy_loss": np.float64(1.25)}
    torch.save(state, tmp_path / "legacy.pt")
    config["acoustic"]["checkpoint"] = str(tmp_path / "legacy.pt")
    (tmp_path / "bundle.yaml").write_text(yaml.safe_dump(config))
    args = assets(tts_bundle, tmp_path / "no-stats")
    args[1] = tmp_path / "bundle.yaml"
    result = cli(*args, "--text", "hello")
    assert result.returncode == 0, result.stderr
    assert wavfile.read(tmp_path / "no-stats/speech.wav")[1].shape == (2048,)


def test_text_file_keeps_each_true_length_and_output_name(tts_bundle, tmp_path):
    texts = tmp_path / "sentences.txt"
    texts.write_text("hello\n\nhello hello\n", encoding="utf-8")
    output = tmp_path / "batch"
    result = cli(*assets(tts_bundle, output), "--text-file", texts, "--save-mels")
    assert result.returncode == 0, result.stderr
    assert np.load(output / "mels/000001.npy").shape == (80, 8)
    assert np.load(output / "mels/000002.npy").shape == (80, 16)
    assert wavfile.read(output / "000001.wav")[1].shape == (2048,)
    assert wavfile.read(output / "000002.wav")[1].shape == (4096,)


def test_existing_outputs_fail_before_loading_weights(tts_bundle, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "speech.wav").write_bytes(b"preserve")
    result = cli(*assets(tts_bundle, output), "--text", "hello")
    assert result.returncode == 2 and "Output exists" in result.stderr
    assert (output / "speech.wav").read_bytes() == b"preserve"


def test_reference_audio_uses_acoustic_frontend_without_speaker_encoder(tts_bundle, tmp_path):
    args = assets(tts_bundle, tmp_path / "audio-ref")
    i = args.index("--reference-mel")
    args[i : i + 2] = ["--reference-audio", tts_bundle / "reference.wav"]
    result = cli(*args, "--text", "hello")
    assert result.returncode == 0, result.stderr
    assert wavfile.read(tmp_path / "audio-ref/speech.wav")[1].shape == (2048,)


@pytest.mark.parametrize(
    "collision", ["reference_mel", "reference_audio", "bundle", "stats", "embedding", "text_file"]
)
def test_outputs_never_replace_input_assets_even_with_overwrite(collision, tmp_path):
    import yaml

    folder = tmp_path / "assets"
    folder.mkdir()
    output = tmp_path / "out"
    (output / "mels").mkdir(parents=True)
    for name in ["acoustic.pt", "generator.pt"]:
        (folder / name).write_bytes(b"must not be loaded")
    (folder / "stats.json").write_text('{"pitch":[0,1],"energy":[0,1]}')
    model = ROOT / "acoustic/config/mel_infer/model.yaml"
    preprocessing = ROOT / "acoustic/config/mel_infer/preprocess.yaml"
    config = {
        "acoustic": {
            "checkpoint": str(folder / "acoustic.pt"),
            "model": str(model),
            "preprocess": str(preprocessing),
            "stats": str(folder / "stats.json"),
            "lexicon": str(ROOT / "acoustic/lexicon"),
        },
        "vocoders": {
            "V": {
                "checkpoint": str(folder / "generator.pt"),
                "config": str(ROOT / "vocoder/configs/config.json"),
            }
        },
    }
    embedding = folder / "embedding.npy"
    reference = folder / "reference.npy"
    np.save(embedding, np.zeros(512))
    np.save(reference, np.zeros((80, 8)))
    source = ["--text", "hello"]
    conditioning = ["--reference-mel", reference]
    bundle = folder / "tts.yaml"
    target = output / "synthesis.json"
    if collision == "reference_mel":
        target = output / "mels/speech.npy"
        np.save(target, np.zeros((80, 8)))
        conditioning = ["--reference-mel", target]
    elif collision == "reference_audio":
        target = output / "speech.wav"
        target.write_bytes(b"private audio input")
        conditioning = ["--reference-audio", target]
    elif collision == "bundle":
        bundle = target
    elif collision == "stats":
        target.write_text('{"pitch":[0,1],"energy":[0,1]}')
        config["acoustic"]["stats"] = str(target)
    elif collision == "embedding":
        target = output / "mels/speech.npy"
        np.save(target, np.zeros(512))
        embedding = target
    elif collision == "text_file":
        target.write_text("hello\n")
        source = ["--text-file", target]
    bundle.write_text(yaml.safe_dump(config))
    before = target.read_bytes()
    result = cli(
        "--config",
        bundle,
        *source,
        "--reference-embedding",
        embedding,
        *conditioning,
        "--output-dir",
        output,
        "--save-mels",
        "--overwrite",
    )
    assert result.returncode == 2 and "overwrite input" in result.stderr.lower()
    assert target.read_bytes() == before
    assert "must not be loaded" not in result.stderr


def test_missing_selected_variant_is_an_asset_error(tts_bundle, tmp_path):
    result = cli(
        *assets(tts_bundle, tmp_path / "no-variant"), "--text", "hello", "--vocoder", "VRT"
    )
    assert (
        result.returncode == 2
        and "vocoders.VRT" in result.stderr
        and "Traceback" not in result.stderr
    )


def test_mismatched_model_configuration_is_rejected(tts_bundle, tmp_path):
    import yaml

    config = yaml.safe_load((tts_bundle / "tts.yaml").read_text())
    for section in [config["acoustic"], config["vocoders"]["V"]]:
        for key, value in section.items():
            section[key] = str(tts_bundle / value)
    v = json.loads((tts_bundle / "vocoder.json").read_text())
    v["hop_size"] = 128
    (tmp_path / "wrong.json").write_text(json.dumps(v))
    config["vocoders"]["V"]["config"] = str(tmp_path / "wrong.json")
    (tmp_path / "wrong.yaml").write_text(yaml.safe_dump(config))
    args = assets(tts_bundle, tmp_path / "out")
    args[1] = tmp_path / "wrong.yaml"
    result = cli(*args, "--text", "hello")
    assert result.returncode == 2 and "hop_size mismatch" in result.stderr


def test_bad_embedding_and_empty_text_file_have_clear_errors(tts_bundle, tmp_path):
    embedding = tmp_path / "bad.npy"
    np.save(embedding, np.zeros(256))
    args = assets(tts_bundle, tmp_path / "bad-out")
    args[args.index("--reference-embedding") + 1] = embedding
    result = cli(*args, "--text", "hello")
    assert (
        result.returncode == 2
        and "Reference embedding" in result.stderr
        and "Traceback" not in result.stderr
    )
    text = tmp_path / "empty.txt"
    text.write_text("\n ")
    result = cli(*assets(tts_bundle, tmp_path / "empty-out"), "--text-file", text)
    assert result.returncode == 2 and "nonempty" in result.stderr
