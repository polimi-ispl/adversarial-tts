# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

"""Synthesize WAV files from saved mel arrays and a matching BigVGAN checkpoint."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from inference_utils import (
    add_common_arguments,
    load_config,
    prepare_inference,
    read_mel,
    write_waveform,
)


def inference(args, config):
    device, jobs, generator = prepare_inference(
        args, config, Path(args.input_mels_dir), ".npy", "_generated_e2e"
    )
    if generator is None:
        return
    with torch.inference_mode():
        for source, target in tqdm(jobs):
            mel = torch.from_numpy(read_mel(source, config.num_mels)).to(device)
            write_waveform(target, generator(mel), config.sampling_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_mels_dir", type=Path, default=Path("test_mel_files"))
    parser.add_argument("--output_dir", type=Path, default=Path("generated_files_from_mel"))
    add_common_arguments(parser)
    args = parser.parse_args()
    config_path = args.config or args.checkpoint_file.parent / "config.json"
    try:
        inference(args, load_config(config_path))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
