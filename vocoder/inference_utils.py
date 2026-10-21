"""Shared file, configuration and device handling for vocoder inference."""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np
from scipy.io.wavfile import write
import torch

from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import BigVGAN


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint_file", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Matching vocoder JSON configuration.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="auto preserves CUDA-if-available, otherwise CPU. Select mps explicitly on Apple Silicon.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate existing output files."
    )


def load_config(path: Path) -> AttrDict:
    if not path.is_file():
        raise ValueError(
            f"Configuration file not found: {path}. Supply --config with the checkpoint's configuration."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    required = {
        "seed",
        "sampling_rate",
        "num_mels",
        "n_fft",
        "hop_size",
        "win_size",
        "fmin",
        "fmax",
        "resblock",
        "upsample_rates",
        "upsample_kernel_sizes",
        "upsample_initial_channel",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "activation",
        "snake_logscale",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Configuration {path} is missing keys: {', '.join(sorted(missing))}")
    return AttrDict(data)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this environment. Use --device cpu.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available in this environment. Use --device cpu.")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported inference device: {name}")
    return torch.device(name)


def collect_jobs(
    input_dir: Path, output_dir: Path, extension: str, suffix: str, overwrite: bool = False
) -> list[tuple[Path, Path]]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must be different.")
    inputs = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == extension)
    if not inputs:
        raise ValueError(f"No {extension} input files found in {input_dir}")
    jobs = []
    names = set()
    for source in inputs:
        target = output_dir / f"{source.stem}{suffix}.wav"
        if target.name.casefold() in names:
            raise ValueError(f"Input names would overwrite the same output: {target.name}")
        names.add(target.name.casefold())
        if target.exists() and not target.is_file():
            raise ValueError(f"Output path is not a regular file: {target}")
        if overwrite or not target.exists():
            jobs.append((source, target))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(inputs)} input files; skipping {len(inputs) - len(jobs)} completed outputs.")
    return jobs


def load_generator(checkpoint_file: Path, config: AttrDict, device: torch.device) -> BigVGAN:
    if not checkpoint_file.is_file():
        raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
    print(f"Loading '{checkpoint_file}'")
    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError(
            f"Could not load generator checkpoint {checkpoint_file}: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("generator"), Mapping):
        raise ValueError(
            f"Checkpoint must contain a 'generator' state dictionary: {checkpoint_file}"
        )
    generator = BigVGAN(config).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    generator.remove_weight_norm()
    print("Complete.")
    return generator


def prepare_inference(args, config, input_dir: Path, extension: str, suffix: str):
    if not Path(args.checkpoint_file).is_file():
        raise ValueError(f"Checkpoint file not found: {args.checkpoint_file}")
    device = select_device(args.device)
    jobs = collect_jobs(input_dir, Path(args.output_dir), extension, suffix, args.overwrite)
    if not jobs:
        print("No new files to process.")
        return device, jobs, None
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    print(f"Device: {device}")
    generator = load_generator(Path(args.checkpoint_file), config, device)
    return device, jobs, generator


def read_mel(path: Path, num_mels: int) -> np.ndarray:
    try:
        mel = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"Could not read mel array {path}: {error}") from error
    if isinstance(mel, np.lib.npyio.NpzFile):
        mel.close()
        raise ValueError(f"Expected a single NumPy mel array in {path}, not an archive.")
    if mel.ndim == 2:
        mel = mel[np.newaxis, ...]
    if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] != num_mels or mel.shape[2] < 1:
        raise ValueError(
            f"Invalid mel shape {mel.shape} in {path}; expected [{num_mels}, frames] or [1, {num_mels}, frames]."
        )
    if mel.dtype.kind not in "fiu":
        raise ValueError(f"Mel values must be real numbers: {path}")
    mel = np.asarray(mel, dtype=np.float32)
    if not np.isfinite(mel).all():
        raise ValueError(f"Mel values must all be finite: {path}")
    return mel


def write_waveform(path: Path, waveform: torch.Tensor, sampling_rate: int) -> None:
    audio = waveform.detach().squeeze(0).squeeze(0).cpu().numpy()
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Vocoder output must be a finite mono waveform.")
    pcm = (audio * MAX_WAV_VALUE).astype(np.int16)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".wav", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        write(temporary_path, sampling_rate, pcm)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(path)
