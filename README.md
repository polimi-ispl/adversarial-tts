# Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis
by Jiayan Cui, Davide Salvi, Paolo Bestagini, and Stefano Tubaro.

This repository will host the official implementation of *Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis*, accepted at IEEE WIFS 2026.

<p align="center">
  <!-- Replace XXXX.XXXXX with the arXiv ID once available -->
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://my.feishu.cn/docx/FH7mdd1dco6xDkxFlhPchve8nyh"><img src="https://img.shields.io/badge/Audio-Samples-1f6feb.svg" alt="Audio samples"></a>
  <img src="https://img.shields.io/badge/Code-Coming%20soon-lightgrey.svg" alt="Code coming soon">
</p>

## News
- 2026/10 - 📢 Repository created. Code and model weights are coming soon.

## Code release

> 🚧 **The code is not yet available.** We are cleaning up the training and
> inference pipeline and will release it in this repository soon.
> Star or watch the repo to be notified when it is published.

## Audio samples

Audio samples generated with the baseline and the detector-guided vocoders are
available **[here](https://my.feishu.cn/docx/FH7mdd1dco6xDkxFlhPchve8nyh)**.

## Overview

<!-- Optional: add the pipeline figure from the paper as docs/pipeline.png and uncomment -->
<!-- <p align="center"><img src="docs/pipeline.png" alt="Detector-guided TTS pipeline" width="550"></p> -->

We propose a Text-to-Speech (TTS) framework that generates speech which is
**intrinsically adversarial** to speech deepfake detectors. Instead of
applying post-hoc perturbations or noise injection, we **fine-tune the vocoder**
of a two-stage TTS system with one or more frozen detectors in the training loop.
The resulting synthesis pipeline lowers detector reliability with
**no post-processing and no extra cost at inference time**.

## Method

Given a TTS generator composed of an acoustic model $\mathcal{A}$ (text → mel-spectrogram)
and a vocoder $\mathcal{V}$ (mel-spectrogram → waveform), we keep $\mathcal{A}$ fixed and
fine-tune only $\mathcal{V}$ with a composite objective:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\mathcal{V}} + \lambda \, \mathcal{L}_{\mathcal{D}}$$

where $\mathcal{L}_{\mathcal{V}}$ is the original vocoder loss, preserving synthesis quality,
and $\mathcal{L}_{\mathcal{D}}$ is the binary cross-entropy of the frozen detector(s) with
respect to the *authentic* class. The formulation extends naturally to multiple detectors:

$$\mathcal{L}_{\mathcal{D}} = \sum_{k=1}^{K} \alpha_k \, \mathbb{E}_{\mathbf{t}} \left[ -\log\left(1 - \mathcal{D}_k(\mathcal{V}(\mathcal{A}(\mathbf{t})))\right) \right]$$

Gradients flow through the frozen detectors and update only the vocoder parameters.

**Setup.**
- Acoustic model: enhanced FastSpeech2
- Vocoder: BigVGAN
- Detectors: RawNet2 and TSSD (white-box, used during fine-tuning), AASIST (black-box, test only)
- Data: LibriTTS

| Vocoder | Detector-guided training | Purpose |
| --- | --- | --- |
| $\mathcal{V}$ | None | Baseline synthesis |
| $\mathcal{V}^{R}$ | RawNet2 | Single-detector attack |
| $\mathcal{V}^{RT}$ | RawNet2 + TSSD | Multi-detector attack |

## Results

EER (%) of each detector on speech produced by each vocoder (higher is a stronger attack; 50% is chance):

| Vocoder | RawNet2 | TSSD | AASIST (unseen) |
| --- | --- | --- | --- |
| $\mathcal{V}$ | 5.80 | 3.40 | 0.85 |
| $\mathcal{V}^{R}$ | 39.05 | 38.33 | 21.09 |
| $\mathcal{V}^{RT}$ | 43.87 | 43.31 | 23.26 |

Detector-guided fine-tuning also **improves perceptual quality**: FAD (CLAP embeddings)
drops from 31.25 ($\mathcal{V}$) to 22.95 ($\mathcal{V}^{R}$) and 17.81 ($\mathcal{V}^{RT}$),
and a MUSHRA-like listening test with 20 expert listeners rates the fine-tuned vocoders as
more realistic than the baseline. See the paper for the full evaluation.

## Responsible use

This work aims to expose weaknesses in current speech deepfake detectors and to support
the development of more robust countermeasures. The same techniques could be misused to
produce synthetic speech that is harder to detect. We release this work **exclusively for
defensive security research** and encourage further work on robust detection,
benchmarking protocols, and ethical guidelines for generative audio.

## Acknowledgements

Our work builds on [FastSpeech2](https://arxiv.org/abs/2006.04558),
[BigVGAN](https://github.com/NVIDIA/BigVGAN),
[RawNet2](https://arxiv.org/abs/2011.01108), [TSSD](https://doi.org/10.1109/LSP.2021.3089437),
[AASIST](https://github.com/clovaai/aasist), and the
[LibriTTS](https://www.openslr.org/60/) dataset. We thank their authors for making these
resources available.

## Citation

```bibtex
@inproceedings{cui2026detectorguided,
  title     = {Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis},
  author    = {Cui, Jiayan and Salvi, Davide and Bestagini, Paolo and Tubaro, Stefano},
  booktitle = {IEEE International Workshop on Information Forensics and Security (WIFS)},
  year      = {2026}
}
```
