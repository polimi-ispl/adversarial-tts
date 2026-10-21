# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

"""Resynthesize WAV files with a matching BigVGAN configuration and checkpoint."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

from inference_utils import add_common_arguments, load_config, prepare_inference, write_waveform
from meldataset import mel_spectrogram


def inference(args, config):
    device, jobs, generator = prepare_inference(
        args, config, Path(args.input_wavs_dir), ".wav", "_generated"
    )
    if generator is None:
        return
    with torch.inference_mode():
        for source, target in tqdm(jobs):
            wav, _ = librosa.load(source, sr=config.sampling_rate, mono=True)
            minimum = max((config.n_fft - config.hop_size) // 2 + 1, config.hop_size)
            if wav.size < minimum:
                raise ValueError(
                    f"Audio is too short: {source}; need at least {minimum} samples at {config.sampling_rate} Hz."
                )
            if not np.isfinite(wav).all():
                raise ValueError(f"Audio values must all be finite: {source}")
            waveform = torch.from_numpy(wav).to(device).unsqueeze(0)
            mel = mel_spectrogram(
                waveform,
                config.n_fft,
                config.num_mels,
                config.sampling_rate,
                config.hop_size,
                config.win_size,
                config.fmin,
                config.fmax,
            )
            write_waveform(target, generator(mel), config.sampling_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_wavs_dir", type=Path, default=Path("test_files"))
    parser.add_argument("--output_dir", type=Path, default=Path("generated_files"))
    add_common_arguments(parser)
    args = parser.parse_args()
    config_path = args.config or Path(__file__).resolve().parent / "configs/config.json"
    try:
        inference(args, load_config(config_path))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
