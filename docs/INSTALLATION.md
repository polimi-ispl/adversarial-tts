# Installation

## One environment

Install Python 3.10+ and an appropriate PyTorch CPU/CUDA/macOS build, then run `python -m pip install -e ".[acoustic,vocoder]"` from the project directory. The `adversarial-tts` console command, `python -m tts` and checkout `python inference.py` expose the same CLI. Wheels also include the acoustic/vocoder workers and required source/config/dictionary resources.

The acoustic inference requirements are in `requirements/acoustic-inference.txt`. They include modern supported bounds for PyTorch, NumPy, SciPy, librosa, matplotlib, PyYAML, pypinyin, inflect, Unidecode, g2p-en and NLTK. They exclude training-only preprocessing packages and optional speaker encoders. The historical `acoustic/requirements.txt` describes an older training stack and is not the inference installation recipe.

If an English word is absent from the dictionary, g2p-en requires NLTK's cmudict and perceptron taggers. Install them explicitly with `python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng`. Known dictionary words and Mandarin dictionary processing do not require that download. No resources are downloaded automatically by the new frontend.

## Separate environments

The orchestrator runs acoustic prediction and vocoding in separate processes, so different environments can be used:

```bash
python3.12 -m venv .venv-acoustic
.venv-acoustic/bin/python -m pip install -r requirements/acoustic-inference.txt
python3.12 -m venv .venv-vocoder
.venv-vocoder/bin/python -m pip install -r vocoder/requirements/inference-macos.txt
python3.12 -m venv .venv-cli
.venv-cli/bin/python -m pip install -e .

.venv-cli/bin/python inference.py --config configs/my-models.yaml --text "Hello." \
  --reference-embedding /path/to/speaker.npy --reference-mel /path/to/reference.npy \
  --output-dir outputs/example --acoustic-python .venv-acoustic/bin/python \
  --vocoder-python .venv-vocoder/bin/python
```

`ACOUSTIC_PYTHON` and `VOCODER_PYTHON` provide environment defaults; command-line interpreter options override them. Each variable is one executable, not a shell command. Relative interpreter paths are made caller-absolute without resolving away venv symlinks. Child processes clear `PYTHONPATH` and run in their component directory to isolate modules with common names.

Acoustic prediction supports `cpu` (default), `cuda` or `auto` (CUDA if available, otherwise CPU). The vocoder supports `cpu`, `cuda`, `mps` and `auto` (CUDA otherwise CPU). Selecting unavailable hardware gives an error; the program does not silently switch an explicitly selected device.

## Optional speaker encoder

Existing 512-dimensional embeddings require no additional encoder dependency. To generate one from reference audio, install `requirements/speaker-encoder.txt` in the acoustic environment and explicitly pass `--speaker-encoder pyannote --reference-audio ...` instead of an embedding/reference-mel pair. The retained conditioning family uses `pyannote/embedding`-style 512-dimensional vectors. The encoder can require Hugging Face access approval and `HF_TOKEN`; supply the token privately through the environment. No token is included in source/configs. See the [encoder model card](https://huggingface.co/pyannote/embedding) for its own access and use terms.

Optional model downloads happen only when that encoder is selected. The embedding-generation option has not been tested against a downloaded encoder checkpoint in the local test record; the precomputed-embedding path is exercised end to end.
