# Models and conditioning

Download the acoustic model, V / VR / VRT vocoders, configurations and dictionaries from **[Hugging Face](https://huggingface.co/Cocii/adversarial-tts/tree/main)**:

```bash
python -m pip install huggingface_hub
hf download Cocii/adversarial-tts --local-dir checkpoints
```

Use `checkpoints/tts.yaml` with the synthesis CLI. The bundle resolves all packaged asset paths relative to itself. `SHA256SUMS` records the model and configuration hashes. For other checkpoints, keep model assets outside source control and supply their paths in your own bundle YAML.

## Bundle schema

See `configs/tts.example.yaml`. The `acoustic` mapping requires `checkpoint`, `preprocess`, `model` and `lexicon`; `stats` is optional when complete trained bins are stored in the checkpoint. The `vocoders` mapping contains any available V/VR/VRT entries, each requiring `checkpoint` and `config`. Only the selected variant is loaded. Relative paths are anchored at the bundle YAML, not the shell's cwd.

The acoustic checkpoint must contain the `model` state dictionary for the reference-conditioned `AdaSpeech_spkr_en_cn` implementation: `spk-type: spk`, `language_speaker.separated: false`, 80-channel mel/postnet and 512-dimensional reference embedding. Model layers, hidden sizes, symbol vocabulary, pitch/energy bins and checkpoint keys must match. This model accepts English or Mandarin text through its bilingual frontend; a different monolingual/XLSR architecture is not silently substituted.

When supplied, `stats.json` contains the original numerical pitch and energy statistics expected by the checkpoint, including the bin endpoints. Without it, the runtime reads the endpoints of `variance_adaptor.pitch_bins` and `variance_adaptor.energy_bins` solely to satisfy the unchanged model constructor in a private temporary directory. Strict state loading then restores **every stored bin**, including nonuniform intervals, and verifies them byte for byte. Missing, nonfinite, unsorted or incorrectly shaped bins are rejected. This does not recover training means/stds, dataset metadata or the original normalization procedure.

Dictionary paths are explicit, and all conditioning inputs are supplied independently of filenames or speaker maps. The acoustic loader requires PyTorch 2.6+ and always uses `weights_only=True`. Its limited NumPy float32/float64 scalar/dtype allowlist supports legacy optimizer metadata; arbitrary custom pickle objects remain rejected. The original checkpoint is read without modification, and no unrestricted-pickle fallback is used.

V/VR/VRT checkpoints each contain `generator`. The configuration must identify the exact architecture and conditioning convention. Resuming vocoder training additionally needs discriminator/optimizer/step state; a generator-only checkpoint is not a full training checkpoint.

## Mel conventions

The default acoustic configuration uses 22050 Hz, FFT/window 1024, hop 256, 80 bands and 0–8000 Hz. The selected vocoder must agree on these fields and on its upsampling product. Defaults alone do not identify a trained checkpoint.

The inspected acoustic reference frontend reflect-pads n_fft/2 samples per side (512 at the defaults); the BigVGAN WAV frontend pads (n_fft-hop)/2 (384) with `center=False`. BigVGAN also adds 1e-9 inside magnitude sqrt; the acoustic frontend does not. Both inspected paths use natural log with a 1e-5 floor, without a separate mel mean/variance normalization in the model's dataset/export path. External preprocessing may differ.

Unified text inference passes the acoustic **predicted postnet mel** directly to a vocoder trained for that conditioning convention. It trims by true predicted length and transposes BTC to CT. It does not rerun waveform extraction, force the reference and vocoder frontends to be numerically equivalent, or invent a padding/normalization repair. Use matching training configs and validate trained-checkpoint quality separately.

A reference mel supplies utterance conditioning to the acoustic model; it is not the generated utterance. A precomputed speaker embedding supplies identity conditioning and does not replace the reference mel. Reference audio can provide the former mel; optionally selected pyannote can provide its matching 512-dimensional identity vector.

## Paper checkpoint validation

Before claiming paper reproduction, verify exact acoustic/vocoder/detector weights and source revisions, environment, preprocessing/splits, detector class polarity and evaluation procedures. Random-checkpoint execution tests are not quality or detection measurements. Licenses and permissions for any distributed weights, training data and voice/reference material must be confirmed before publication.

Local validation matched the full-size acoustic state and all V/VR/VRT generator states strictly. All three vocoders then synthesized from the same predicted mel using full-size acoustic weights, synthetic identity conditioning and public reference audio, with acoustic CPU and explicit MPS vocoders (CPU fallback disabled). The inference-only acoustic checkpoint retains the original model tensors exactly and omits optimizer state. These checks do not establish original-speaker identity, training-statistics recovery, quality, detector metrics or training/resume reproduction.
