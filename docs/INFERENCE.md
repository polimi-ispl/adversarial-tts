# Inference

Run `python inference.py --help`, `python -m tts --help` or `adversarial-tts --help` for the same text-to-waveform interface.

## Inputs

`--config` selects a model-bundle YAML. `--vocoder V|VR|VRT` selects its vocoder entry. Supply either `--text` or `--text-file`; the latter is UTF-8, one nonempty line per sentence. Text is normalized with the bilingual pronunciation rules and mapped to the acoustic model's original symbol vocabulary.

Conditioning requires both a speaker embedding and an utterance reference:

- Use `--reference-embedding` with a numeric `[512]` or `[1,512]` `.npy` array.
- Use `--reference-mel` with `[80,T]` log mel, or add `--reference-mel-layout TC` for `[T,80]`.
- Alternatively, `--reference-audio` extracts the utterance mel using the configured acoustic frontend at its sampling rate.
- To generate a new embedding from that audio, omit `--reference-embedding` and explicitly add `--speaker-encoder pyannote`, with its optional dependency/access requirements installed. Do not pass a precomputed embedding and a speaker encoder together.

Use matching reference preprocessing and speaker-encoder conventions for your acoustic checkpoint. No reference filename format, speaker-ID lookup, mutable speaker map or fixed dataset directory is needed by this runtime.

`--pitch-control`, `--energy-control` and `--duration-control` are finite positive controls, each defaulting to 1.0. They are passed to the acoustic model without changing the training formulas.

## Data flow and outputs

The acoustic worker loads `checkpoint['model']` strictly, runs its `.inference()` method, takes postnet output `[B,T,80]` and the predicted true mel lengths, trims each output and writes `[80,true_T]`. Values are not normalized again, shifted or passed through a waveform frontend. Each sentence is handled separately, so padded frames from another sentence cannot leak into its waveform.

The top-level BigVGAN loader reads the selected `checkpoint['generator']` and matching JSON. Waveform sample rate/mono shape/frame count are checked before final publication. `--output-dir` receives `speech.wav` for `--text`, or numbered WAVs for `--text-file`, plus `synthesis.json` with lengths, token counts and selected checkpoint paths. `--save-mels` additionally keeps unpadded predictions under `mels/`.

Existing output files require `--overwrite`. Directories and symlink outputs are rejected. The program rejects an output path that aliases any input bundle, selected checkpoint/config/statistics/dictionary, text file, embedding, reference mel or reference audio, even with `--overwrite`. Worker computation occurs in a temporary workspace; validated files are published atomically per file. A multiple-file publication is not a transactional filesystem operation.

## Other synthesis entry points

`bash scripts/vocoder.sh mel ...` performs standalone saved-mel synthesis; `wav ...` performs analysis/resynthesis from WAV inputs. Both accept explicit matching `--config`, `--checkpoint_file` and `--device`, preserve their `_generated_e2e.wav`/`_generated.wav` suffixes, and skip completed outputs unless `--overwrite` is passed. Root launchers make supported CLI paths caller-relative before switching component cwd.

`scripts/mel_bridge.py convert` handles explicit CT/TC/BCT/BTC file layouts without rescaling, with mandatory true lengths for batch arrays. `inspect` compares preprocessing fields and reports known frontend differences. This file tool is optional for externally saved arrays; unified text synthesis does not need a separate conversion command.
