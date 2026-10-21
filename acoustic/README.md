# Acoustic model

Reference-conditioned AdaSpeech predicts 80-band postnet mels from English/Mandarin text, a 512-dimensional speaker embedding and an utterance reference mel. `predict.py` is the acoustic worker used by the project-wide `inference.py` / `adversarial-tts` command. It loads an explicit matching `model` checkpoint, applies the bilingual text frontend, trims true predicted lengths and writes channel-major mels for BigVGAN.

See [installation](../docs/INSTALLATION.md), [inference](../docs/INFERENCE.md) and [model assets](../docs/MODELS.md). Precomputed embeddings require no speaker-model download. Optional pyannote embedding generation from reference audio is selected explicitly.

Model and training code are under `model/`, `transformer/`, `train_en_cn.py` and `finetune.py`; see [training requirements](../docs/TRAINING.md). Current configs retain their training hyperparameters; the unified worker takes checkpoint/statistics/dictionary paths from the explicit model bundle instead of historical dataset directories.

The nested `vocoder/` helpers serve the included training code. The main text-to-WAV path uses the project's top-level BigVGAN component. Copyright/source headers and HiFi-GAN notices remain; see [third-party attribution](../THIRD_PARTY_NOTICES.md).
