"""Small real BigVGAN checkpoints for inference integration tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from env import AttrDict
from models import BigVGAN


@pytest.fixture(scope="session")
def model_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-vocoder")
    source_config = Path(__file__).resolve().parents[1] / "configs/config.json"
    config = json.loads(source_config.read_text())
    config.update(
        upsample_initial_channel=32,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        sampling_rate=24000,
        fmax=11000,
    )
    (root / "config.json").write_text(json.dumps(config))
    torch.manual_seed(config["seed"])
    generator = BigVGAN(AttrDict(config))
    torch.save({"generator": generator.state_dict()}, root / "generator.pt")
    return root


@pytest.fixture
def sample_inputs(tmp_path):
    wavs, mels = tmp_path / "wavs", tmp_path / "mels"
    wavs.mkdir()
    mels.mkdir()
    t = np.arange(8192) / 48000
    wavfile.write(wavs / "voice.WAV", 48000, (np.sin(2 * np.pi * 220 * t) * 2000).astype(np.int16))
    np.save(mels / "voice.npy", np.zeros((80, 8), dtype=np.float32))
    for folder in (wavs, mels):
        (folder / ".DS_Store").write_bytes(b"ignored")
        (folder / "notes.txt").write_text("Not an inference input.")
        (folder / "nested").mkdir()
    return {"wav": wavs, "mel": mels}
