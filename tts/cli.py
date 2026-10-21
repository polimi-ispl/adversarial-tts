"""Synthesize text with a reference-conditioned acoustic model and selected BigVGAN checkpoint."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]


def interpreter(value, label):
    found = shutil.which(value)
    if found is None:
        raise ValueError(f"{label} is not an executable Python interpreter: {value}")
    return os.path.abspath(found)


def run_worker(command, cwd, label):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        token = env.get("HF_TOKEN")
        if token:
            detail = detail.replace(token, "[redacted]")
        raise ValueError(f"{label} failed: {detail[-8000:]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Model bundle YAML; asset paths are bundle-relative",
    )
    texts = parser.add_mutually_exclusive_group(required=True)
    texts.add_argument("--text", help="One sentence")
    texts.add_argument("--text-file", type=Path, help="UTF-8 text, one nonempty line per utterance")
    parser.add_argument("--vocoder", choices=["V", "VR", "VRT"], default="V")
    parser.add_argument(
        "--reference-embedding", type=Path, help="512-dimensional NumPy speaker embedding"
    )
    parser.add_argument(
        "--reference-mel", type=Path, help="Reference conditioning .npy mel (default CT)"
    )
    parser.add_argument("--reference-mel-layout", choices=["CT", "TC"], default="CT")
    parser.add_argument(
        "--reference-audio", type=Path, help="Use the acoustic frontend to extract conditioning mel"
    )
    parser.add_argument(
        "--speaker-encoder",
        choices=["pyannote"],
        help="Explicitly generate embedding from reference audio",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-mels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--acoustic-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--vocoder-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--acoustic-python", default=os.environ.get("ACOUSTIC_PYTHON", sys.executable)
    )
    parser.add_argument(
        "--vocoder-python", default=os.environ.get("VOCODER_PYTHON", sys.executable)
    )
    parser.add_argument("--pitch-control", type=float, default=1.0)
    parser.add_argument("--energy-control", type=float, default=1.0)
    parser.add_argument("--duration-control", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        from .config import load_bundle

        bundle = load_bundle(args.config, args.vocoder)
        import math

        for name in ["pitch_control", "energy_control", "duration_control"]:
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f"{name.replace('_', '-')} must be finite and positive")
        if not args.reference_mel and not args.reference_audio:
            raise ValueError(
                "Supply --reference-mel or --reference-audio for utterance conditioning"
            )
        if args.reference_mel and args.reference_audio:
            raise ValueError("Choose --reference-mel or --reference-audio, not both")
        if not args.reference_embedding and not (args.speaker_encoder and args.reference_audio):
            raise ValueError(
                "Supply --reference-embedding, or --reference-audio with explicit --speaker-encoder pyannote"
            )
        if args.reference_embedding and args.speaker_encoder:
            raise ValueError("Choose an existing reference embedding or speaker-encoder generation")
        for name in ["reference_mel", "reference_audio", "reference_embedding"]:
            p = getattr(args, name)
            if p is not None and not p.is_file():
                raise ValueError(f"{name.replace('_', '-')} not found: {p}")
        if args.text is not None:
            items = [{"id": "speech", "text": args.text.strip()}]
        else:
            lines = [
                s.strip()
                for s in args.text_file.read_text(encoding="utf-8").splitlines()
                if s.strip()
            ]
            items = [{"id": f"{i:06d}", "text": line} for i, line in enumerate(lines, 1)]
        if not items or any(not i["text"] for i in items):
            raise ValueError("Text input must contain at least one nonempty utterance")
        output = args.output_dir.expanduser().resolve()
        targets = [output / f"{i['id']}.wav" for i in items]
        if args.save_mels:
            targets += [output / "mels" / f"{i['id']}.npy" for i in items]
        targets += [output / "synthesis.json"]
        protected = {args.config.expanduser().resolve()}
        for name in ["text_file", "reference_embedding", "reference_mel", "reference_audio"]:
            value = getattr(args, name)
            if value is not None:
                protected.add(value.expanduser().resolve())
        for section in ["acoustic", "vocoder"]:
            for value in bundle[section].values():
                path = Path(value).resolve()
                if path.is_dir():
                    protected.update(p.resolve() for p in path.rglob("*") if p.is_file())
                else:
                    protected.add(path)
        for target in targets:
            resolved = target.resolve()
            if resolved in protected or (
                target.exists() and any(p.exists() and target.samefile(p) for p in protected)
            ):
                raise ValueError(
                    f"Output would overwrite input asset: {target}; choose a different output directory"
                )
        for target in targets:
            if target.is_symlink() or (
                target.exists() and (not args.overwrite or not target.is_file())
            ):
                raise ValueError(
                    f"Output exists or is not a regular file: {target}; use --overwrite for files"
                )
        if output.exists() and not output.is_dir():
            raise ValueError(f"Output directory is not a directory: {output}")
        acoustic_python = interpreter(args.acoustic_python, "acoustic-python")
        vocoder_python = interpreter(args.vocoder_python, "vocoder-python")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tts-", dir=output.parent) as work:
            stage = Path(work)
            request = dict(
                bundle["acoustic"],
                items=items,
                output_dir=str(stage / "mels"),
                reference_embedding=str(args.reference_embedding.resolve())
                if args.reference_embedding
                else None,
                reference_mel=str(args.reference_mel.resolve()) if args.reference_mel else None,
                reference_audio=str(args.reference_audio.resolve())
                if args.reference_audio
                else None,
                reference_mel_layout=args.reference_mel_layout,
                speaker_encoder=args.speaker_encoder,
                device=args.acoustic_device,
                pitch_control=args.pitch_control,
                energy_control=args.energy_control,
                duration_control=args.duration_control,
            )
            request_file = stage / "request.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")
            run_worker(
                [
                    acoustic_python,
                    str(ROOT / "acoustic/predict.py"),
                    "--request",
                    str(request_file),
                ],
                ROOT / "acoustic",
                "Acoustic synthesis",
            )
            records = json.loads((stage / "mels/predictions.json").read_text())
            if len(records) != len(items) or [r["id"] for r in records] != [i["id"] for i in items]:
                raise ValueError(
                    "Acoustic prediction records do not match the requested utterances"
                )
            run_worker(
                [
                    vocoder_python,
                    str(ROOT / "vocoder/inference_e2e.py"),
                    "--input_mels_dir",
                    str(stage / "mels"),
                    "--output_dir",
                    str(stage / "audio"),
                    "--checkpoint_file",
                    bundle["vocoder"]["checkpoint"],
                    "--config",
                    bundle["vocoder"]["config"],
                    "--device",
                    args.vocoder_device,
                ],
                ROOT / "vocoder",
                "Vocoder synthesis",
            )
            publications = []
            for record in records:
                source = stage / "audio" / f"{record['id']}_generated_e2e.wav"
                with wave.open(str(source), "rb") as audio:
                    if (
                        audio.getframerate() != bundle["sampling_rate"]
                        or audio.getnchannels() != 1
                        or audio.getnframes() != record["frames"] * bundle["hop_size"]
                    ):
                        raise ValueError(
                            f"Vocoder WAV shape/rate differs from the predicted mel contract: {source.name}"
                        )
                publications.append((source, output / f"{record['id']}.wav"))
                if args.save_mels:
                    publications.append(
                        (
                            stage / "mels" / f"{record['id']}.npy",
                            output / "mels" / f"{record['id']}.npy",
                        )
                    )
            summary = {
                "vocoder": args.vocoder,
                "sampling_rate": bundle["sampling_rate"],
                "utterances": records,
                "acoustic_checkpoint": bundle["acoustic"]["checkpoint"],
                "vocoder_checkpoint": bundle["vocoder"]["checkpoint"],
            }
            metadata = stage / "synthesis.json"
            metadata.write_text(json.dumps(summary, indent=2) + "\n")
            publications.append((metadata, output / "synthesis.json"))
            for source, target in publications:
                target.parent.mkdir(parents=True, exist_ok=True)
                if args.overwrite:
                    os.replace(source, target)
                else:
                    os.link(source, target)
            print(
                json.dumps(
                    {
                        "output_dir": str(output),
                        "files": [str(output / f"{r['id']}.wav") for r in records],
                        "vocoder": args.vocoder,
                    },
                    indent=2,
                )
            )
        return 0
    except Exception as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
