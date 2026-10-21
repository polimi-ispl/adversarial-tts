# Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis

by Jiayan Cui, Davide Salvi, Paolo Bestagini, and Stefano Tubaro.

Implementation and inference tools for *Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis*, accepted at **IEEE WIFS 2026**.

<p align="center">
  <a href="https://github.com/polimi-ispl/adversarial-tts/blob/main/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper PDF"></a>
  <a href="https://cocii.github.io/adversarial-tts/"><img src="https://img.shields.io/badge/Audio-Samples-1f6feb.svg" alt="Audio samples"></a>
  <a href="https://huggingface.co/Cocii/adversarial-tts/tree/main"><img src="https://img.shields.io/badge/Models-Hugging%20Face-ffcc00.svg" alt="Model checkpoints"></a>
</p>

**English** · [简体中文](README.zh-CN.md) · [Italiano](README.it.md)

## Audio samples

Compare six voices, three prompts, and the baseline / detector-guided vocoders on the **[interactive listening page](https://cocii.github.io/adversarial-tts/)**. The player supports playback-position locking for comparisons. The [original audio collection](https://my.feishu.cn/docx/FH7mdd1dco6xDkxFlhPchve8nyh) is also available.

[Overview](#overview) · [Method](#method) · [Results](#results) · [Quick start](#quick-start) · [Documentation](#documentation)

## Overview


We propose a Text-to-Speech (TTS) framework that generates speech which is
**intrinsically adversarial** to speech deepfake detectors. Instead of
applying post-hoc perturbations or noise injection, we **fine-tune the vocoder**
of a two-stage TTS system with one or more frozen detectors in the training loop.
The resulting synthesis pipeline lowers detector reliability with
**no post-processing and no extra cost at inference time**.

This release brings together the research method, pretrained models, and a reference-conditioned inference interface:

- **Three comparable vocoders:** baseline `V`, RawNet2-guided `VR`, and jointly guided `VRT`, sharing one acoustic model.
- **Transfer evaluation:** AASIST is excluded from fine-tuning and evaluates transfer to an unseen detector.
- **Usable research assets:** downloadable checkpoints, an interactive audio comparison, and one command-line interface for synthesis.

## Method

### Training and inference

```text
Text → frozen acoustic model → mel-spectrogram → BigVGAN vocoder → waveform
                                                     ↑              │
                                    update vocoder   └─ detector loss┘
                                                       (training only)
```

During fine-tuning, frozen RawNet2 and/or TSSD provide differentiable feedback. The acoustic model and detector weights stay fixed. At inference, the selected vocoder generates audio directly; detectors are outside the synthesis path.

The command-line names `V`, `VR`, and `VRT` correspond to the paper’s baseline, RawNet2-guided, and RawNet2 + TSSD-guided variants, respectively.

### Objective

Given a TTS generator composed of an acoustic model $`\mathcal{A}`$ (text → mel-spectrogram)
and a vocoder $`\mathcal{V}`$ (mel-spectrogram → waveform), we keep $`\mathcal{A}`$ fixed and
fine-tune only $`\mathcal{V}`$ with a composite objective:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\mathcal{V}} + \lambda \, \mathcal{L}_{\mathcal{D}}
```

where $`\mathcal{L}_{\mathcal{V}}`$ is the original vocoder loss, preserving synthesis quality,
and $`\mathcal{L}_{\mathcal{D}}`$ is the binary cross-entropy of the frozen detector(s) with
respect to the *authentic* class. The formulation extends naturally to multiple detectors:

```math
\mathcal{L}_{\mathcal{D}} = \sum_{k=1}^{K} \alpha_k \, \mathbb{E}_{\mathbf{t}} \left[ -\log\left(1 - \mathcal{D}_k(\mathcal{V}(\mathcal{A}(\mathbf{t})))\right) \right]
```

Here, $`\lambda`$ balances synthesis and detector objectives, while $`\alpha_k`$ weights the contribution of detector $`k`$ among $`K`$ frozen detectors.

Gradients flow through the frozen detectors and update only the vocoder parameters.

**Setup.**

- Acoustic model: enhanced FastSpeech2
- Vocoder: BigVGAN
- Detectors: RawNet2 and TSSD (white-box, used during fine-tuning), AASIST (black-box, test only)
- Data: LibriTTS

| Vocoder | Detector-guided training | Purpose |
| --- | --- | --- |
| `V` | None | Baseline synthesis |
| `VR` | RawNet2 | Single-detector attack |
| `VRT` | RawNet2 + TSSD | Multi-detector attack |

## Results

EER (%) of each detector on speech produced by each vocoder (higher is a stronger attack; 50% is chance):

| Vocoder | RawNet2 | TSSD | AASIST (unseen) |
| --- | --- | --- | --- |
| `V` | 5.80 | 3.40 | 0.85 |
| `VR` | 39.05 | 38.33 | 21.09 |
| `VRT` | 43.87 | 43.31 | 23.26 |

Detector-guided fine-tuning also **improves perceptual quality**: FAD (CLAP embeddings)
drops from 31.25 (`V`) to 22.95 (`VR`) and 17.81 (`VRT`),
and a MUSHRA-like listening test with 20 expert listeners rates the fine-tuned vocoders as
more realistic than the baseline. See the paper for the full evaluation.

## Quick start

Use Python **3.10+**; the commands below use Python 3.12. Install a hardware-appropriate PyTorch build when needed.

### 1. Install

```bash
git clone https://github.com/polimi-ispl/adversarial-tts.git
cd adversarial-tts
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[acoustic,vocoder]" huggingface_hub
```

### 2. Download models

```bash
.venv/bin/hf download Cocii/adversarial-tts --local-dir checkpoints
```

### 3. Synthesize

Provide your own `[512]` or `[1,512]` speaker embedding and channel-major `[80, time]` reference mel. A speaker embedding alone is not sufficient. To use reference audio instead, replace `--reference-mel ...` with `--reference-audio /path/to/reference.wav`.

```bash
.venv/bin/python inference.py \
  --config checkpoints/tts.yaml \
  --text "Hello, this is a speech synthesis example." \
  --reference-embedding /path/to/speaker.npy \
  --reference-mel /path/to/reference-mel.npy \
  --vocoder VRT --output-dir outputs/example \
  --acoustic-device cpu --vocoder-device cpu
```

The output is `outputs/example/speech.wav` with synthesis metadata. Select `V`, `VR`, or `VRT` with `--vocoder`. On supported Apple Silicon hardware, the vocoder can use `--vocoder-device mps`; acoustic inference uses CPU or CUDA.

### Outputs and common options

| Item | Behavior |
| --- | --- |
| `speech.wav` | Mono speech at 22,050 Hz for a single input text. |
| `synthesis.json` | Synthesis metadata, lengths, token counts, and selected checkpoint paths. |
| `--save-mels` | Also retain the unpadded predicted mel under `mels/`. |
| `--text-file sentences.txt` | Replace `--text` with one nonempty UTF-8 utterance per line; outputs are numbered WAV files. |
| `--pitch-control`, `--energy-control`, `--duration-control` | Positive acoustic controls; each defaults to `1.0`. |
| `--reference-mel-layout TC` | Accept a time-major `[time, 80]` reference mel instead of the default `[80, time]`. |

The frontend supports English and Mandarin. Existing outputs require `--overwrite`. See the [inference guide](docs/INFERENCE.md) for all arguments and separate acoustic/vocoder environments.

<details>
<summary>Use reference audio and install English frontend resources</summary>

Keep the speaker embedding and supply a WAV file for utterance conditioning:

```bash
.venv/bin/python inference.py \
  --config checkpoints/tts.yaml \
  --text "Hello, this is a speech synthesis example." \
  --reference-embedding /path/to/speaker.npy \
  --reference-audio /path/to/reference.wav \
  --vocoder VRT --output-dir outputs/reference-audio \
  --acoustic-device cpu --vocoder-device cpu
```

For English words outside the bundled dictionary, install the NLTK resources used by `g2p-en`:

```bash
.venv/bin/python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

The embedding represents speaker identity; the reference mel/audio supplies utterance conditioning. To extract an embedding with an optional speaker encoder, follow [installation](docs/INSTALLATION.md).

</details>

## Models

The **[Hugging Face model collection](https://huggingface.co/Cocii/adversarial-tts/tree/main)** contains the acoustic checkpoint, three vocoders, matching configurations, pronunciation dictionaries, and `SHA256SUMS`. The packaged `tts.yaml` uses relative paths and is ready for the synthesis command above.

| File | Purpose |
| --- | --- |
| `acoustic_inference.pt` | Acoustic weights for inference. |
| `g_V` | Baseline BigVGAN vocoder. |
| `g_VR` | RawNet2-guided vocoder. |
| `g_VRT` | RawNet2 + TSSD-guided vocoder. |
| `tts.yaml`, `model.yaml`, `preprocess.yaml`, `config_*.json` | Matching inference bundle and model configurations. |
| `96000.pth.tar` | Original acoustic training checkpoint, including optimizer state. |

All three vocoders use the same **22,050 Hz / 80 mel bands / hop 256** interface, with FFT/window 1024 and a 0–8,000 Hz mel range. Keep each checkpoint paired with its supplied configuration. Unified inference trims the predicted postnet mel to its true length and passes it to the vocoder in channel-major layout.

For a controlled comparison, keep the text, speaker embedding, reference, and acoustic controls fixed and change only `--vocoder`.

## Repository structure

| Path | Contents |
| --- | --- |
| [`inference.py`](inference.py), [`tts/`](tts/) | Unified CLI, model-bundle validation, and inference workers. |
| [`acoustic/`](acoustic/) | AdaSpeech acoustic model and bilingual text frontend. |
| [`vocoder/`](vocoder/) | BigVGAN vocoders, detector implementations, and retained research training code. |
| [`configs/`](configs/) | Model-bundle configuration example. |
| [`site/`](site/) | Static listening page and audio assets. |
| [`tests/`](tests/), [`docs/`](docs/) | Automated checks and detailed usage/research documentation. |

## Research and reproducibility

The EER, FAD, and listening-test results above are **reported in the paper**. Release validation checks checkpoint loading and executable text-to-WAV synthesis. The [testing guide](docs/TESTING.md) describes what was exercised; the [training guide](docs/TRAINING.md) records the original experiment resources and retained trainer requirements.

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Inference](docs/INFERENCE.md)
- [Models and conditioning](docs/MODELS.md)
- [Training](docs/TRAINING.md)
- [Testing](docs/TESTING.md)

The released inference interface uses the reference-conditioned AdaSpeech implementation and BigVGAN. The acoustic model is described as enhanced FastSpeech2 in the paper. Runtime checks establish checkpoint compatibility and executable synthesis; the results above are the evaluation reported in the paper. See the training guide for the retained research trainer and its experiment requirements.

## Responsible use

This work aims to expose weaknesses in current speech deepfake detectors and to support
the development of more robust countermeasures. The same techniques could be misused to
produce synthetic speech that is harder to detect. We release this work **exclusively for
defensive security research** and encourage further work on robust detection,
benchmarking protocols, and ethical guidelines for generative audio.

## Citation

```bibtex
@inproceedings{cui2026detectorguided,
  title     = {Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis},
  author    = {Cui, Jiayan and Salvi, Davide and Bestagini, Paolo and Tubaro, Stefano},
  booktitle = {IEEE International Workshop on Information Forensics and Security (WIFS)},
  year      = {2026}
}
```

## Acknowledgements

We thank [Microsoft’s AdaSpeech](https://speechresearch.github.io/adaspeech/) and [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN), together with the contributors of the supporting acoustic, vocoder, and detector implementations. Original attribution and applicable notices are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE.md](LICENSE.md).
