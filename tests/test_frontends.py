"""Characterize the real frontend boundary without acoustic weights or reference voices."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_retained_frontends_are_finite_but_have_different_frame_counts(tmp_path):
    wave = np.sin(2 * np.pi * 220 * np.arange(4096) / 22050).astype(np.float32) * 0.2
    np.save(tmp_path / "wave.npy", wave)
    acoustic = """
from pathlib import Path
import numpy as np,torch,sys
from audio.stft import TacotronSTFT
p=Path(sys.argv[1]); w=torch.from_numpy(np.load(p/'wave.npy'))[None,:]
stft=TacotronSTFT(1024,256,1024,80,22050,0,8000)
m,_=stft.mel_spectrogram(w)
np.save(p/'acoustic.npy',m[0].numpy())
"""
    vocoder = """
from pathlib import Path
import numpy as np,torch,sys
from meldataset import mel_spectrogram
p=Path(sys.argv[1]); w=torch.from_numpy(np.load(p/'wave.npy'))[None,:]
m=mel_spectrogram(w,1024,80,22050,256,1024,0,8000,center=False)
np.save(p/'vocoder.npy',m[0].numpy())
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for component, code in [("acoustic", acoustic), ("vocoder", vocoder)]:
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            cwd=ROOT / component,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
    a = np.load(tmp_path / "acoustic.npy")
    v = np.load(tmp_path / "vocoder.npy")
    assert a.shape == (80, 17)
    assert v.shape == (80, 16)
    assert np.isfinite(a).all() and np.isfinite(v).all()
    assert not np.allclose(a[:, :16], v, atol=1e-4)
