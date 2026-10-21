#!/usr/bin/env python3
"""Run an original component entry with isolated imports and caller-relative paths."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    "acoustic": {
        "inference": "predict.py",
        "mel": "predict.py",
        "preprocess": "preprocess.py",
        "train": "train_en_cn.py",
        "finetune": "finetune.py",
        "evaluate": "evaluate.py",
    },
    "vocoder": {"wav": "inference.py", "mel": "inference_e2e.py", "train": "train.py"},
}
PATH_FLAGS = {
    "vocoder": {
        "--input_wavs_dir",
        "--input_mels_dir",
        "--output_dir",
        "--checkpoint_file",
        "--config",
        "--input_training_file",
        "--input_validation_file",
        "--checkpoint_path",
    },
    "acoustic": {
        "-p",
        "--preprocess_config",
        "-m",
        "--model_config",
        "-t",
        "--train_config",
        "--source",
        "--reference_audio",
        "--reference_embedding",
        "--vocoder_checkpoint",
        "--vocoder_config",
        "--pretrain_dir",
    },
}


def absolute_arguments(args, component, caller, entry=None):
    def absolute(value):
        if not value:
            raise ValueError("A path cannot be empty")
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else caller / path).resolve())

    result = []
    i = 0
    list_flags = (
        {"--list_input_unseen_wavs_dir", "--list_input_unseen_validation_file"}
        if component == "vocoder"
        else set()
    )
    while i < len(args):
        flag, equal, value = args[i].partition("=")
        if component == "acoustic" and len(flag) > 2 and flag[:2] in {"-p", "-m", "-t"}:
            flag, equal, value = flag[:2], "=", args[i][2:]
        if flag in PATH_FLAGS[component] or flag in list_flags:
            values = [value] if equal else []
            if not equal:
                i += 1
                if i >= len(args) or args[i].startswith("-"):
                    raise ValueError(f"{flag} requires a path; prefix a leading dash with ./")
                values.append(args[i])
            if flag in list_flags:
                while i + 1 < len(args) and not args[i + 1].startswith("-"):
                    i += 1
                    values.append(args[i])
            result.extend([flag, *(absolute(value) for value in values)])
        elif (
            component == "acoustic"
            and entry == "preprocess"
            and i == 0
            and not args[i].startswith("-")
        ):
            result.append(absolute(args[i]))
        else:
            result.append(args[i])
        i += 1
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=ENTRIES)
    parser.add_argument("entry", help="doctor or a named original entry; see README")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    try:
        component = options.component
        if options.entry != "doctor" and options.entry not in ENTRIES[component]:
            raise ValueError(
                f"Unknown {component} entry {options.entry!r}; choose doctor or "
                + ", ".join(ENTRIES[component])
            )
        key = component.upper() + "_PYTHON"
        requested = os.environ.get(key, "python3")
        executable = shutil.which(requested)
        if not executable:
            raise ValueError(
                f"{key} does not identify an executable Python interpreter: {requested}"
            )
        args = options.args
        if args[:1] == ["--"]:
            args = args[1:]
        executable = os.path.abspath(executable)
        args = absolute_arguments(args, component, Path.cwd(), options.entry)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        if options.entry == "doctor":
            if args:
                raise ValueError("doctor does not accept component arguments")
            modules = (
                ["model", "utils", "audio", "text", "vocoder"]
                if component == "acoustic"
                else ["models", "inference_utils", "meldataset"]
            )
            dependencies = (
                [
                    "torch",
                    "numpy",
                    "librosa",
                    "yaml",
                    "g2p_en",
                    "pypinyin",
                    "pyannote",
                    "transformers",
                    "inflect",
                    "unidecode",
                ]
                if component == "acoustic"
                else ["torch", "numpy", "librosa", "scipy"]
            )
            code = """
import importlib.util,json,os,sys
names=json.loads(sys.argv[1]); deps=json.loads(sys.argv[2])
def locate(name):
    spec=importlib.util.find_spec(name)
    return None if spec is None else spec.origin or ' | '.join(spec.submodule_search_locations or [])
print(json.dumps({'cwd':os.getcwd(),'executable':sys.executable,'pythonpath':os.environ.get('PYTHONPATH'),'modules':{n:locate(n) for n in names},'dependency_modules_present':{n:locate(n) is not None for n in deps},'scope':'Module discovery only; no checkpoint or runtime validation'},indent=2))
"""
            command = [executable, "-c", code, json.dumps(modules), json.dumps(dependencies)]
        else:
            command = [executable, str(ROOT / component / ENTRIES[component][options.entry]), *args]
        return subprocess.run(
            command, cwd=ROOT / component, env=environment, check=False
        ).returncode
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
