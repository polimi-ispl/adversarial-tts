# 安装

## 单一环境

安装 Python 3.10+ 和适合 CPU/CUDA/macOS 的 PyTorch 版本，然后在项目目录中运行 `python -m pip install -e ".[acoustic,vocoder]"`。`adversarial-tts` 控制台命令、`python -m tts` 和检出目录中的 `python inference.py` 提供相同的 CLI。wheel 包也包含声学模型/声码器工作进程，以及必需的源代码、配置和词典资源。

声学推理依赖列在 `requirements/acoustic-inference.txt` 中，其中为 PyTorch、NumPy、SciPy、librosa、matplotlib、PyYAML、pypinyin、inflect、Unidecode、g2p-en 和 NLTK 设定了现代受支持版本范围。这些依赖不包括仅用于训练的预处理包和可选说话人编码器。历史文件 `acoustic/requirements.txt` 描述的是较旧的训练依赖栈，并非推理安装方案。

如果英文单词不在词典中，g2p-en 需要 NLTK 的 cmudict 和感知机词性标注器。请运行 `python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng` 显式安装。已收录的词典单词和普通话词典处理不需要此次下载。新前端不会自动下载任何资源。

## 独立环境

流程编排器在不同进程中运行声学预测和声码器，因此可以使用不同环境：

```bash
python3.12 -m venv .venv-acoustic
.venv-acoustic/bin/python -m pip install -r requirements/acoustic-inference.txt
python3.12 -m venv .venv-vocoder
.venv-vocoder/bin/python -m pip install -r vocoder/requirements/inference-macos.txt
python3.12 -m venv .venv-cli
.venv-cli/bin/python -m pip install -e .

.venv-cli/bin/python inference.py --config configs/my-models.yaml --text "Hello." \
  --reference-embedding /path/to/speaker.npy --reference-mel /path/to/reference.npy \
  --output-dir outputs/example --acoustic-python .venv-acoustic/bin/python \
  --vocoder-python .venv-vocoder/bin/python
```

`ACOUSTIC_PYTHON` 和 `VOCODER_PYTHON` 提供环境默认值；命令行解释器选项会覆盖它们。每个变量只能指定一个可执行文件，不能是 shell 命令。相对解释器路径会转为相对于调用者目录的绝对路径，同时保留 venv 符号链接，不将其解析为最终目标。子进程会清除 `PYTHONPATH`，并在各自组件目录中运行，以隔离同名模块。

声学预测支持 `cpu`（默认）、`cuda` 或 `auto`（CUDA 可用时使用 CUDA，否则使用 CPU）。声码器支持 `cpu`、`cuda`、`mps` 和 `auto`（优先 CUDA，否则使用 CPU）。选择不可用的硬件会报错；程序不会静默切换显式选择的设备。

## 可选说话人编码器

已有的 512 维嵌入不需要额外的编码器依赖。如果要根据参考音频生成嵌入，请在声学环境中安装 `requirements/speaker-encoder.txt`，并显式传入 `--speaker-encoder pyannote --reference-audio ...`，以替代嵌入/参考梅尔频谱这一对输入。保留的条件输入模型系列使用 `pyannote/embedding` 风格的 512 维向量。编码器可能需要 Hugging Face 访问批准和 `HF_TOKEN`；请通过环境私密地提供令牌。源代码/配置中不包含令牌。该编码器自身的访问和使用条款请参阅[编码器模型卡](https://huggingface.co/pyannote/embedding)。

只有选择该编码器时，才会发生可选模型下载。本地测试记录中尚未使用下载的编码器检查点测试嵌入生成选项；预先计算嵌入的路径已完成端到端测试。
