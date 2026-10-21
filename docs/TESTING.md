# Testing

Install the inference/development dependencies, then run:

```bash
python -m pip install -e ".[acoustic,vocoder]" -r requirements/development.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
(cd vocoder && python -m pytest -q)
node --test tests/position-lock.test.cjs
```

Root tests cover the public text-to-WAV CLI with actual reference-conditioned AdaSpeech and BigVGAN models, original symbol/frontend functions, synthetic reference inputs and randomly initialized smaller test checkpoints. The acoustic duration-predictor fixture is set to a known output so expected token/frame/WAV lengths can be checked independently. No downloaded voice model or private speech is used.

Coverage includes postnet true-length cropping and CT order; real text and multi-line files through acoustic prediction and selected vocoder; reference mel and waveform conditioning; missing model/config/reference/interpreter/device errors; invalid embeddings; path isolation/caller-relative paths; prevention of overwriting any input asset; standalone saved-mel/WAV synthesis contracts; array validation/overwrite handling; and a direct waveform-frontend characterization showing the expected padding difference.

The vocoder suite separately exercises the actual reduced architecture with real checkpoint/WAV/NumPy I/O. Both explicit Apple MPS cases were verified locally with CPU fallback disabled. Linux CI uses Python 3.10/3.12; hardware-dependent MPS cases skip there. The site retains three focused playback regressions. The new text pipeline is tested on CPU; optional downloaded pyannote encoder execution, trained-checkpoint quality and full acoustic MPS support are outside this test record.

A wheel smoke check installs the project into a separate import directory and invokes the installed console command from outside the checkout, including actual text-to-WAV synthesis. Tests prove executable data flow and resource packaging, not paper checkpoint quality, full training/resume or detector metrics. Exact local run results and upload checksums are kept in the maintainer's external handoff.

Legacy-checkpoint regressions exercise the limited NumPy scalar allowlist, rejection of arbitrary pickle code, strict state matching and byte-identical restoration of nonuniform pitch/energy bins when external stats are omitted.

Locally supplied full-size acoustic and V/VR/VRT checkpoints also passed strict state matching. Three end-to-end checks used the full acoustic weights on CPU and V, VR or VRT on Apple MPS with CPU fallback disabled, a synthetic unit 512-dimensional identity vector and a public reference WAV. For the test text, the same predicted CT mel [80,50] yielded three distinct non-silent mono PCM16 WAVs, each with 12800 samples at 22050 Hz. This verifies executable compatibility, not original-speaker conditioning, speech quality or detector metrics.
