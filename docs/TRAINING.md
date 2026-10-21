# Training

The project includes acoustic and vocoder training implementations. Their model definitions, training forward methods, loss calculations and checkpoint parameter keys are retained. Changes to the supported inference path do not establish that the original paper's complete training experiment has been recovered.

## Acoustic model

The acoustic training entry is `acoustic/train_en_cn.py`, with model/preprocessing/train YAMLs under `acoustic/config/`. Fine-tuning is in `acoustic/finetune.py`. These programs expect an external preprocessed corpus containing mel, pitch, energy, durations, phoneme aggregates, speaker metadata/embeddings and matching statistics. Obtain the original checkpoint-associated environment and manifests before running/resuming the experiment.

Run training in its own environment and component directory; the inference dependency group is not a claim of complete preprocessing/training coverage. `acoustic/requirements.txt` is a historical dependency record with conflicting legacy pins, not the supported inference recipe. Review its training dependencies and configuration paths against the intended experiment rather than installing it over an inference environment.

## Vocoder and adversarial objectives

The vocoder training entry is `vocoder/train.py`. V denotes baseline synthesis, VR targets RawNet2, and VRT targets RawNet2/TSSD. AASIST is held out of vocoder fine-tuning in the intended method. The current experimental trainer contains inactive/commented detector paths and unresolved gradient/label/optimizer/loss/decay differences. It also imports a missing `audio_utils.py` through RawNet helpers.

The original V/VR/VRT training scripts, detector weights/configs, class order, data split/preprocessing records and logs are required to resolve those differences. Do not treat restoring commented code or changing losses as a verified reconstruction. Training behavior remains unchanged until the original experiment bundle is available.

## Resume and evaluation

Use exact model/config/checkpoint pairs. Acoustic resume may require the optimizer state as well as `model`. Vocoder resume needs its discriminator and optimizer state, epoch/steps and generator. Validate one small train/resume/evaluation path on the original supported hardware before a full run. Confirm score polarity and consistent AUC/EER/balanced-accuracy definitions before comparing against the manuscript. No training run or paper-quality result is claimed by inference tests.
