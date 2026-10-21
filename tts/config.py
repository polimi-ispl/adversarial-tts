"""Model-bundle paths are relative to the bundle file; command-line inputs are caller-relative."""

import json
from pathlib import Path
import yaml


def load_bundle(path, variant):
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("acoustic"), dict):
        raise ValueError(f"Model bundle {path} requires an acoustic mapping")
    acoustic = {}
    for key in ["checkpoint", "preprocess", "model", "stats", "lexicon"]:
        value = data["acoustic"].get(key)
        if key == "stats" and value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"acoustic.{key} must be an explicit asset path in {path}")
        p = Path(value).expanduser()
        p = (path.parent / p).resolve() if not p.is_absolute() else p
        if not (p.is_dir() if key == "lexicon" else p.is_file()):
            raise ValueError(f"Acoustic {key} not found: {p}")
        acoustic[key] = str(p)
    selected = data.get("vocoders", {}).get(variant)
    if not isinstance(selected, dict):
        raise ValueError(f"Model bundle has no vocoders.{variant} mapping")
    vocoder = {}
    for key in ["checkpoint", "config"]:
        value = selected.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"vocoders.{variant}.{key} must be an explicit path")
        p = Path(value).expanduser()
        p = (path.parent / p).resolve() if not p.is_absolute() else p
        if not p.is_file():
            raise ValueError(f"Vocoder {key} not found: {p}")
        vocoder[key] = str(p)
    p = yaml.safe_load(Path(acoustic["preprocess"]).read_text())["preprocessing"]
    v = json.loads(Path(vocoder["config"]).read_text())
    a = {
        "sampling_rate": p["audio"]["sampling_rate"],
        "num_mels": p["mel"]["n_mel_channels"],
        "n_fft": p["stft"]["filter_length"],
        "hop_size": p["stft"]["hop_length"],
        "win_size": p["stft"]["win_length"],
        "fmin": p["mel"]["mel_fmin"],
        "fmax": p["mel"]["mel_fmax"],
    }
    for key, value in a.items():
        other = v[key]
        if key == "fmax":
            value = a["sampling_rate"] / 2 if value is None else value
            other = v["sampling_rate"] / 2 if other is None else other
        if value != other:
            raise ValueError(
                f"Acoustic/vocoder {key} mismatch: {value} versus {other}; use matching training configs"
            )
    import math

    if math.prod(v["upsample_rates"]) != v["hop_size"]:
        raise ValueError("Vocoder upsampling product does not match hop_size")
    return {
        "acoustic": acoustic,
        "vocoder": vocoder,
        "sampling_rate": v["sampling_rate"],
        "hop_size": v["hop_size"],
        "num_mels": v["num_mels"],
        "variant": variant,
    }
