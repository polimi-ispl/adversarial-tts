"""Build actual reduced BigVGAN fixtures in its own interpreter/module namespace."""

import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def model_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("integration-model")
    code = """
import json,sys
from pathlib import Path
import torch
from env import AttrDict
from models import BigVGAN
p=Path(sys.argv[1]); c=json.loads(Path('configs/config.json').read_text())
c.update(upsample_initial_channel=32,upsample_rates=[8,8,2,2],upsample_kernel_sizes=[16,16,4,4],resblock_kernel_sizes=[3],resblock_dilation_sizes=[[1,3,5]],sampling_rate=24000)
(p/'config.json').write_text(json.dumps(c))
torch.manual_seed(1)
torch.save({'generator':BigVGAN(AttrDict(c)).state_dict()},p/'generator.pt')
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=ROOT / "vocoder",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return root


@pytest.fixture(scope="session")
def tts_bundle(tmp_path_factory, model_bundle):
    import json
    import shutil
    import numpy as np
    import yaml

    folder = tmp_path_factory.mktemp("tts assets with spaces")
    config = yaml.safe_load((ROOT / "acoustic/config/mel_infer/model.yaml").read_text())
    config["transformer"].update(encoder_layer=1, decoder_layer=1, conv_filter_size=64)
    config["variance_predictor"]["filter_size"] = 16
    config["PhonemeLevelEncoder"]["n_chans"] = 16
    config["PhonemeLevelPredictor"]["n_chans"] = 16
    config["variance_embedding"]["n_bins"] = 8
    preprocess = yaml.safe_load((ROOT / "acoustic/config/mel_infer/preprocess.yaml").read_text())
    preprocess["path"].update(
        preprocessed_path=str(folder), combined_path=str(folder), datasets_base_path=str(folder)
    )
    (folder / "model.yaml").write_text(yaml.safe_dump(config))
    (folder / "preprocess.yaml").write_text(yaml.safe_dump(preprocess))
    (folder / "stats.json").write_text(
        json.dumps({"pitch": [50, 500, 150, 30], "energy": [0, 10, 1, 1]})
    )
    lexicon = folder / "lexicon"
    lexicon.mkdir()
    (lexicon / "english_cjy.dict").write_text("hello HH AH0 L OW1\nworld W ER1 L D\n")
    (lexicon / "pinyin-lexicon-r.dict").write_text("ni3 n i3\n")
    np.save(folder / "embedding.npy", np.linspace(-1, 1, 512, dtype=np.float32))
    np.save(folder / "reference.npy", np.zeros((80, 12), dtype=np.float32))
    from scipy.io import wavfile

    t = np.arange(4096) / 22050
    wavfile.write(
        folder / "reference.wav", 22050, (np.sin(2 * np.pi * 220 * t) * 1000).astype(np.int16)
    )
    code = """
import json,math,sys
from pathlib import Path
import torch,yaml
from model.adaspeech_spkr_en_cn import AdaSpeech_spkr_en_cn
p=Path(sys.argv[1]);torch.manual_seed(7)
m=AdaSpeech_spkr_en_cn(yaml.safe_load((p/'preprocess.yaml').read_text()),yaml.safe_load((p/'model.yaml').read_text()))
with torch.no_grad():
    m.variance_adaptor.duration_predictor.linear_layer.weight.zero_()
    m.variance_adaptor.duration_predictor.linear_layer.bias.fill_(math.log(3))
torch.save({'model':m.state_dict()},p/'acoustic.pt')
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(folder)],
        cwd=ROOT / "acoustic",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    v = json.loads((model_bundle / "config.json").read_text())
    v.update(sampling_rate=22050, fmax=8000)
    (folder / "vocoder.json").write_text(json.dumps(v))
    shutil.copyfile(model_bundle / "generator.pt", folder / "V-generator.pt")
    shutil.copyfile(model_bundle / "generator.pt", folder / "VR-generator.pt")
    bundle = {
        "acoustic": {
            "checkpoint": "acoustic.pt",
            "model": "model.yaml",
            "preprocess": "preprocess.yaml",
            "stats": "stats.json",
            "lexicon": "lexicon",
        },
        "vocoders": {
            variant: {"checkpoint": f"{variant}-generator.pt", "config": "vocoder.json"}
            for variant in ["V", "VR"]
        },
    }
    (folder / "tts.yaml").write_text(yaml.safe_dump(bundle))
    return folder
