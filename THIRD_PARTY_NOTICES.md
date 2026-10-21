# Third-party attribution

This project builds on existing research implementations. Source headers and original notices are retained; the project does not claim authorship of their underlying methods or implementations.

- **AdaSpeech acoustic modeling**: [Microsoft Research AdaSpeech](https://speechresearch.github.io/adaspeech/), with the reference-conditioned/bilingual implementation used here. AdaSpeech's original project-wide licensing grant and modifications require confirmation before public redistribution; source credits must remain.
- **BigVGAN**: [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN); retained NVIDIA notice in `vocoder/LICENSE` and source headers.
- **HiFi-GAN**: [jik876/hifi-gan](https://github.com/jik876/hifi-gan); MIT notices in `acoustic/hifigan/LICENSE` and `vocoder/incl_licenses/`.
- **Snake and alias-free activation building blocks**: retained notices in `vocoder/incl_licenses/LICENSE_1`, `LICENSE_2`, `LICENSE_3` and `LICENSE_5` accompany their corresponding adapted sources; preserve their original texts.
- **MelGAN**: [seungwonpark/melgan](https://github.com/seungwonpark/melgan), BSD 3-Clause text retained as `vocoder/incl_licenses/LICENSE_4`.
- **RawNet2**: [ASVspoof 2021 baseline](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2), MIT notice in `vocoder/incl_licenses/LICENSE_RawNet2`.
- **AASIST**: [clovaai/aasist](https://github.com/clovaai/aasist), MIT notice in `vocoder/incl_licenses/LICENSE_AASIST`.
- **TSSD**: [end-to-end-synthetic-speech-detection](https://github.com/ghua-ac/end-to-end-synthetic-speech-detection), GPL-3.0 notice in `vocoder/incl_licenses/LICENSE_TSSD`.
- **Text/STFT utilities**: retained source headers credit Keith Ito's Tacotron and Prem Seetharaman's PyTorch STFT implementation. Dictionary and externally obtained data/resources retain their own provenance and rights.
- **Public samples site**: Manrope fonts use SIL OFL 1.1 (`site/assets/fonts/OFL.txt`). `site/ASSET_SOURCES.md` records audio/character attribution and separates those materials from the font license. Public availability is not a blanket audio/model-training license.

Model checkpoints, speech/reference recordings and data resources require separate permission and license review. No unpublished paper, private voice references, speaker embeddings, checkpoint files or proprietary Times New Roman font are included in this source tree.
