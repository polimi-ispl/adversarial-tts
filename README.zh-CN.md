# Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis

作者： Jiayan Cui, Davide Salvi, Paolo Bestagini, and Stefano Tubaro.

论文 *Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis* 的配套实现与推理工具。论文已被 **IEEE WIFS 2026** 接收。

<p align="center">
  <a href="https://github.com/polimi-ispl/adversarial-tts/blob/main/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper PDF"></a>
  <a href="https://cocii.github.io/adversarial-tts/"><img src="https://img.shields.io/badge/Audio-Samples-1f6feb.svg" alt="Audio samples"></a>
  <a href="https://huggingface.co/Cocii/adversarial-tts/tree/main"><img src="https://img.shields.io/badge/Models-Hugging%20Face-ffcc00.svg" alt="Model checkpoints"></a>
</p>

[English](README.md) · **简体中文** · [Italiano](README.it.md)

## 音频试听

打开 **[交互式试听页](https://cocii.github.io/adversarial-tts/)**，比较六种声音、三段文本，以及基线和检测器引导的声码器。播放器支持锁定播放位置，方便切换对比。也可访问[原始音频集](https://my.feishu.cn/docx/FH7mdd1dco6xDkxFlhPchve8nyh)。

[概述](#概述) · [方法](#方法) · [结果](#结果) · [快速开始](#快速开始) · [详细文档](#详细文档)

## 概述

我们提出一种能够生成对语音深伪检测器具有**内在对抗性**的语音的文本转语音框架。该方法将一个或多个冻结的检测器纳入两阶段 TTS 系统的训练过程，直接**微调声码器**。合成时无需后处理或添加噪声，也不增加额外推理开销。

本仓库提供研究方法、预训练模型，以及基于参考条件的统一推理接口：

- **三种可对比的声码器：** 基线 `V`、RawNet2 引导的 `VR`、联合引导的 `VRT`，共享同一声学模型。
- **迁移评估：** AASIST 不参与微调，用于评估对未见检测器的迁移效果。
- **配套研究资源：** 可下载的检查点、交互式音频对比，以及统一的命令行合成接口。

## 方法

### 训练与推理流程

```text
文本 → 冻结的声学模型 → 梅尔频谱 → BigVGAN 声码器 → 波形
                                      ↑             │
                              更新声码器 └─ 检测器损失 ┘
                                            （仅训练）
```

微调时，冻结的 RawNet2 和/或 TSSD 提供可微反馈，声学模型和检测器权重保持固定。推理时，选定的声码器直接生成音频，合成流程无需运行检测器。

命令行名称 `V`、`VR`、`VRT` 分别对应论文中的基线、RawNet2 引导和 RawNet2 + TSSD 联合引导变体。

### 优化目标

TTS 生成器由声学模型 $`\mathcal{A}`$（文本 → 梅尔频谱）和声码器 $`\mathcal{V}`$（梅尔频谱 → 波形）组成。固定 $`\mathcal{A}`$，仅使用以下组合目标微调 $`\mathcal{V}`$：

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\mathcal{V}} + \lambda \, \mathcal{L}_{\mathcal{D}}
```

$`\mathcal{L}_{\mathcal{V}}`$ 为保持合成质量的原始声码器损失，$`\mathcal{L}_{\mathcal{D}}`$ 为冻结检测器针对真实类别的二元交叉熵。多个检测器时：

```math
\mathcal{L}_{\mathcal{D}} = \sum_{k=1}^{K} \alpha_k \, \mathbb{E}_{\mathbf{t}} \left[ -\log\left(1 - \mathcal{D}_k(\mathcal{V}(\mathcal{A}(\mathbf{t})))\right) \right]
```

其中，$`\lambda`$ 平衡语音合成目标和检测器目标，$`\alpha_k`$ 表示第 $`k`$ 个检测器的权重，$`K`$ 为冻结检测器的数量。

梯度穿过冻结的检测器，仅更新声码器参数。

**实验设置：** 增强版 FastSpeech2 声学模型、BigVGAN 声码器、LibriTTS 数据；RawNet2 与 TSSD 用于白盒微调，AASIST 仅用于黑盒测试。

| 声码器 | 检测器引导训练 | 用途 |
| --- | --- | --- |
| `V` | 无 | 基线合成 |
| `VR` | RawNet2 | 单检测器攻击 |
| `VRT` | RawNet2 + TSSD | 多检测器攻击 |

## 结果

论文报告的各检测器 EER（%）如下；数值越高，攻击越强，50% 表示随机水平：

| 声码器 | RawNet2 | TSSD | AASIST（未见） |
| --- | --- | --- | --- |
| `V` | 5.80 | 3.40 | 0.85 |
| `VR` | 39.05 | 38.33 | 21.09 |
| `VRT` | 43.87 | 43.31 | 23.26 |

检测器引导微调也改善了感知质量：基于 CLAP 嵌入的 FAD 从基线的 31.25 降至 `VR` 的 22.95 和 `VRT` 的 17.81。20 位专家参与的 MUSHRA 风格试听测试认为，微调后的声码器比基线更真实。完整评估见论文。

## 快速开始

需要 Python **3.10+**；以下命令使用 Python 3.12。请按硬件需求安装合适的 PyTorch 版本。

### 1. 安装

```bash
git clone https://github.com/polimi-ispl/adversarial-tts.git
cd adversarial-tts
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[acoustic,vocoder]" huggingface_hub
```

### 2. 下载模型

```bash
.venv/bin/hf download Cocii/adversarial-tts --local-dir checkpoints
```

### 3. 合成

请提供自己的 `[512]` 或 `[1,512]` 说话人嵌入，以及通道优先的 `[80, time]` 参考梅尔频谱；仅有嵌入无法提供完整条件输入。使用参考音频时，将 `--reference-mel ...` 替换为 `--reference-audio /path/to/reference.wav`。

```bash
.venv/bin/python inference.py \
  --config checkpoints/tts.yaml \
  --text "Hello, this is a speech synthesis example." \
  --reference-embedding /path/to/speaker.npy \
  --reference-mel /path/to/reference-mel.npy \
  --vocoder VRT --output-dir outputs/example \
  --acoustic-device cpu --vocoder-device cpu
```

输出为 `outputs/example/speech.wav` 和合成元数据。通过 `--vocoder` 选择 `V`、`VR` 或 `VRT`。支持的 Apple Silicon 硬件可使用 `--vocoder-device mps`；声学推理使用 CPU 或 CUDA。

### 输出与常用选项

| 项目 | 行为 |
| --- | --- |
| `speech.wav` | 单条文本对应的 22,050 Hz 单声道语音。 |
| `synthesis.json` | 合成元数据、长度、词元数量和所选检查点路径。 |
| `--save-mels` | 额外将去除填充后的预测梅尔频谱保存在 `mels/`。 |
| `--text-file sentences.txt` | 替换 `--text`；UTF-8 文件每个非空行是一条文本，生成编号 WAV。 |
| `--pitch-control`、`--energy-control`、`--duration-control` | 正数声学控制参数，默认均为 `1.0`。 |
| `--reference-mel-layout TC` | 使用时间优先的 `[time, 80]` 参考梅尔，替代默认 `[80, time]`。 |

文本前端支持英语与普通话。覆盖已有输出需使用 `--overwrite`。全部参数及声学模型、声码器分环境运行方式见[推理文档](docs/zh-CN/INFERENCE.md)。

<details>
<summary>使用参考音频与安装英文前端资源</summary>

保留说话人嵌入，使用 WAV 文件提供语句条件：

```bash
.venv/bin/python inference.py \
  --config checkpoints/tts.yaml \
  --text "Hello, this is a speech synthesis example." \
  --reference-embedding /path/to/speaker.npy \
  --reference-audio /path/to/reference.wav \
  --vocoder VRT --output-dir outputs/reference-audio \
  --acoustic-device cpu --vocoder-device cpu
```

遇到词典外英文词时，安装 `g2p-en` 使用的 NLTK 资源：

```bash
.venv/bin/python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

嵌入提供说话人身份条件，参考梅尔或音频提供语句条件。使用可选说话人编码器提取嵌入的方法见[安装文档](docs/zh-CN/INSTALLATION.md)。

</details>

## 模型

**[Hugging Face 模型集合](https://huggingface.co/Cocii/adversarial-tts/tree/main)** 包含声学模型、三种声码器、配套配置、发音词典和 `SHA256SUMS`。其中的 `tts.yaml` 使用相对路径，可直接用于上述合成命令。

| 文件 | 用途 |
| --- | --- |
| `acoustic_inference.pt` | 推理所用的声学模型权重。 |
| `g_V` | 基线 BigVGAN 声码器。 |
| `g_VR` | RawNet2 引导的声码器。 |
| `g_VRT` | RawNet2 + TSSD 引导的声码器。 |
| `tts.yaml`, `model.yaml`, `preprocess.yaml`, `config_*.json` | 配套推理资源包和模型配置。 |
| `96000.pth.tar` | 含优化器状态的原始声学训练检查点。 |

三种声码器共享 **22,050 Hz / 80 个梅尔频带 / hop 256** 接口，FFT 与窗长为 1024，梅尔频率范围为 0–8,000 Hz。每个检查点应使用配套配置。统一推理会按真实长度裁剪预测的 postnet 梅尔频谱，再以通道优先布局传给声码器。

进行受控对比时，保持文本、说话人嵌入、参考输入和声学控制参数相同，仅切换 `--vocoder`。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`inference.py`](inference.py)、[`tts/`](tts/) | 统一命令行、模型资源包校验和推理工作进程。 |
| [`acoustic/`](acoustic/) | AdaSpeech 声学模型与双语文本前端。 |
| [`vocoder/`](vocoder/) | BigVGAN 声码器、检测器实现与保留的研究训练代码。 |
| [`configs/`](configs/) | 模型资源包配置示例。 |
| [`site/`](site/) | 静态试听页与音频资源。 |
| [`tests/`](tests/)、[`docs/`](docs/) | 自动检查与详细使用、研究文档。 |

## 研究与复现

上述 EER、FAD 和听感测试结果均为**论文报告的实验结果**。发布验证检查模型加载与文本到 WAV 的可执行流程。[测试文档](docs/zh-CN/TESTING.md)说明已覆盖的验证范围；[训练文档](docs/zh-CN/TRAINING.md)记录原实验资源与保留训练器的运行要求。

## 详细文档

- [安装](docs/zh-CN/INSTALLATION.md)
- [推理](docs/zh-CN/INFERENCE.md)
- [模型与条件输入](docs/zh-CN/MODELS.md)
- [训练](docs/zh-CN/TRAINING.md)
- [测试](docs/zh-CN/TESTING.md)

发布的推理接口使用基于参考条件的 AdaSpeech 实现和 BigVGAN；论文中将声学模型描述为增强版 FastSpeech2。运行测试验证检查点兼容性和可执行合成流程，上述结果来自论文评估。保留的研究训练器及其所需实验资源见训练文档。

## 负责任地使用

本工作旨在揭示现有语音深伪检测器的弱点，并支持更稳健的防御方法研究。同类技术也可能被用于生成更难检测的合成语音。本工作**仅面向防御性安全研究**发布，鼓励对稳健检测、基准评估协议和生成式音频伦理规范开展进一步研究。

## 引用

```bibtex
@inproceedings{cui2026detectorguided,
  title     = {Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis},
  author    = {Cui, Jiayan and Salvi, Davide and Bestagini, Paolo and Tubaro, Stefano},
  booktitle = {IEEE International Workshop on Information Forensics and Security (WIFS)},
  year      = {2026}
}
```

## 致谢

感谢 [Microsoft AdaSpeech](https://speechresearch.github.io/adaspeech/) 和 [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN) 的作者，以及相关声学模型、声码器和检测器实现的贡献者。原始署名和适用声明保留在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [LICENSE.md](LICENSE.md) 中。
