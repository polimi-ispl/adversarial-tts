# BigVGAN vocoder

BigVGAN converts the acoustic model's channel-major log mels to waveform audio. The project-wide text-to-speech command selects V, VR or VRT through a model bundle containing the desired `generator` checkpoint and exact JSON config.

Use [the project inference guide](../docs/INFERENCE.md) for text-to-WAV, standalone saved-mel synthesis and WAV resynthesis. Both vocoder CLIs accept explicit config/checkpoint/input/output/device options, validate inputs and write WAVs atomically. CPU, CUDA and explicit Apple MPS are supported for inference. Default `auto` selects CUDA when available, otherwise CPU.

The model code and parameter names are retained for checkpoint compatibility. Experimental detector/training code is included; [training](../docs/TRAINING.md) describes the original-artifact requirements before any method or loss changes. See [model conditioning](../docs/MODELS.md) for padding/log-mel conventions and [testing](../docs/TESTING.md) for executable checks.

Original NVIDIA BigVGAN, HiFi-GAN, Snake/alias-free and detector notices are preserved under `incl_licenses/` and source headers. The TSSD component has GPL-3.0 provenance. See the project's [LICENSE.md](../LICENSE.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
