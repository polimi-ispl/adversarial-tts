#!/usr/bin/env python3
"""Predict unpadded channel-major mels using explicit acoustic model assets."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

# Keep acoustic imports first, and make the small shared TTS contracts importable
# both from a checkout and an installed wheel with separate Python environments.
sys.path.append(str(Path(__file__).resolve().parents[1]))


def numeric_array(path):
    import numpy as np

    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray):
        values.close()
        raise ValueError(f"Expected a NumPy array: {path}")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"Expected finite real numeric values: {path}")
    return values


def generate_embedding(reference_audio, device):
    try:
        from pyannote.audio import Model, Inference
    except ImportError as error:
        raise ValueError(
            "Speaker encoder requires requirements/speaker-encoder.txt; precomputed embeddings need no encoder package"
        ) from error
    import librosa
    import numpy as np
    import torch

    model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.environ.get("HF_TOKEN"))
    if model is None:
        raise ValueError("Cannot load pyannote/embedding; check permitted access and HF_TOKEN")
    encoder = Inference(model, window="whole", device=device)
    audio, rate = librosa.load(reference_audio, sr=22050, mono=True)
    audio = librosa.util.normalize(audio) * 0.95
    values = encoder({"waveform": torch.from_numpy(audio).float()[None, :], "sample_rate": rate})
    return np.asarray(values)


def predict(request):
    import numpy as np
    import torch
    import yaml
    from model.adaspeech_spkr_en_cn import AdaSpeech_spkr_en_cn
    import utils.tools
    import model.modules
    from frontend import preprocess_en_cn
    from tts.acoustic_assets import constructor_stats, load_acoustic_state, restore_acoustic_state
    from tts.contracts import postnet_to_mels

    name = request.get("device", "cpu")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ["cpu", "cuda"] or (name == "cuda" and not torch.cuda.is_available()):
        raise ValueError(
            f"Acoustic device {name} is unavailable; use cpu or an available CUDA device"
        )
    device = torch.device(name)
    utils.tools.device = device
    model.modules.device = device
    preprocessing = yaml.safe_load(Path(request["preprocess"]).read_text())
    configuration = yaml.safe_load(Path(request["model"]).read_text())
    if (
        configuration.get("spk-type", "spk") != "spk"
        or configuration["language_speaker"]["separated"]
    ):
        raise ValueError(
            "This runtime expects the reference-conditioned AdaSpeech_spkr_en_cn checkpoint family (spk, separated=false)"
        )
    channels = preprocessing["preprocessing"]["mel"]["n_mel_channels"]
    if channels != 80:
        raise ValueError("This acoustic checkpoint family has an 80-channel postnet")
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if request.get("reference_embedding"):
        embedding = numeric_array(request["reference_embedding"])
    else:
        if request.get("speaker_encoder") != "pyannote" or not request.get("reference_audio"):
            raise ValueError(
                "Supply a precomputed embedding or explicitly select the pyannote encoder with reference audio"
            )
        embedding = generate_embedding(request["reference_audio"], device)
    if embedding.shape not in [(512,), (1, 512)]:
        raise ValueError(
            f"Reference embedding must have shape [512] or [1,512], got {embedding.shape}"
        )
    embedding = torch.from_numpy(embedding.reshape(512).astype(np.float32)).to(device)
    if request.get("reference_mel"):
        reference = numeric_array(request["reference_mel"])
        if request.get("reference_mel_layout", "CT") == "TC":
            reference = reference.T
        if reference.ndim != 2 or reference.shape[0] != channels or reference.shape[1] == 0:
            raise ValueError(f"Reference mel must be [{channels},time], got {reference.shape}")
    else:
        import librosa
        import audio.stft
        from audio.tools import get_mel_from_wav

        audio.stft.device = torch.device("cpu")
        p = preprocessing["preprocessing"]
        waveform, _ = librosa.load(
            request["reference_audio"], sr=p["audio"]["sampling_rate"], mono=True
        )
        if waveform.size == 0:
            raise ValueError("Reference audio is empty")
        stft = audio.stft.TacotronSTFT(
            p["stft"]["filter_length"],
            p["stft"]["hop_length"],
            p["stft"]["win_length"],
            channels,
            p["audio"]["sampling_rate"],
            p["mel"]["mel_fmin"],
            p["mel"]["mel_fmax"],
        )
        reference, _ = get_mel_from_wav(waveform, stft)
    if not np.isfinite(reference).all():
        raise ValueError("Reference mel must contain finite values")
    reference = torch.from_numpy(reference.T.astype(np.float32))[None, :, :].to(device)
    preprocessing["path"]["lexicon_path"] = request["lexicon"]
    state = load_acoustic_state(request["checkpoint"])
    endpoints = constructor_stats(state, configuration["variance_embedding"]["n_bins"])
    records = []
    with tempfile.TemporaryDirectory(prefix="tts-stats-") as stats_dir:
        # The unchanged constructor reads endpoints from stats.json. When no
        # original stats are supplied, use checkpoint endpoints only to construct
        # the model, then restore and verify every trained bin byte for byte.
        # These endpoints do not recover training means/stds or dataset metadata.
        stats = (
            json.loads(Path(request["stats"]).read_text()) if request.get("stats") else endpoints
        )
        if not isinstance(stats, dict) or not all(k in stats for k in ["pitch", "energy"]):
            raise ValueError("Acoustic statistics require pitch and energy arrays")
        Path(stats_dir, "stats.json").write_text(json.dumps(stats))
        preprocessing["path"]["preprocessed_path"] = stats_dir
        preprocessing["path"]["combined_path"] = stats_dir
        model = AdaSpeech_spkr_en_cn(preprocessing, configuration).to(device).eval()
        model.device = device
        restore_acoustic_state(model, state)
        identifiers = set()
        for item in request["items"]:
            identifier = item["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in identifiers:
                raise ValueError("Utterance ids must be unique simple filenames")
            identifiers.add(identifier)
            sequence = preprocess_en_cn(item["text"], preprocessing)
            if sequence.ndim != 1 or sequence.size == 0:
                raise ValueError(f"Text has no supported phonemes: {identifier}")
            text = torch.from_numpy(sequence.astype(np.int64))[None, :].to(device)
            lengths = torch.tensor([sequence.size], device=device)
            with torch.inference_mode():
                prediction = model.inference(
                    [identifier],
                    [item["text"]],
                    [0],
                    text,
                    lengths,
                    sequence.size,
                    mels=reference,
                    languages=torch.tensor([0], device=device),
                    p_control=request.get("pitch_control", 1.0),
                    e_control=request.get("energy_control", 1.0),
                    d_control=request.get("duration_control", 1.0),
                    spk_emb=embedding,
                )
                true_lengths = prediction[10].detach().cpu().tolist()
                mels = postnet_to_mels(prediction[1].detach().cpu().numpy(), true_lengths, channels)
            np.save(output / f"{identifier}.npy", mels[0], allow_pickle=False)
            records.append(
                {"id": identifier, "frames": true_lengths[0], "tokens": int(sequence.size)}
            )
    (output / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request", type=Path, required=True, help="Explicit acoustic prediction request JSON"
    )
    args = parser.parse_args(argv)
    try:
        predict(json.loads(args.request.read_text(encoding="utf-8")))
    except Exception as error:
        detail = str(error)
        token = os.environ.get("HF_TOKEN")
        if token:
            detail = detail.replace(token, "[redacted]")
        parser.error(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
