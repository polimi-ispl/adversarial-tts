# 推理

运行 `python inference.py --help`、`python -m tts --help` 或 `adversarial-tts --help`，可查看相同的文本转波形接口。

## 输入

`--config` 选择模型资源包 YAML。`--vocoder V|VR|VRT` 选择其中的声码器条目。请提供 `--text` 或 `--text-file`；后者为 UTF-8 文件，每个非空行对应一句话。文本会按照双语发音规则规范化，并映射到声学模型原始的符号词表。

条件输入同时需要说话人嵌入和话语参考：

- 使用 `--reference-embedding` 提供数值型 `[512]` 或 `[1,512]` `.npy` 数组。
- 使用 `--reference-mel` 提供 `[80,T]` 对数梅尔频谱；如果形状为 `[T,80]`，则添加 `--reference-mel-layout TC`。
- 也可以使用 `--reference-audio`，按已配置声学前端的采样率提取话语梅尔频谱。
- 要根据该音频生成新嵌入，请省略 `--reference-embedding`，显式添加 `--speaker-encoder pyannote`，并满足其可选依赖和访问要求。不要同时传入预先计算的嵌入和说话人编码器。

请使用与声学检查点匹配的参考预处理和说话人编码器约定。此运行时不需要特定的参考文件命名格式、说话人 ID 查找、可变说话人映射或固定数据集目录。

`--pitch-control`、`--energy-control` 和 `--duration-control` 均为有限正数控制参数，默认值各为 1.0。它们会传递给声学模型，不会改变训练公式。

## 数据流与输出

声学工作进程以严格模式加载 `checkpoint['model']`，运行其 `.inference()` 方法，取得 postnet 输出 `[B,T,80]` 和预测的真实梅尔长度，裁剪每条输出，然后写出 `[80,true_T]`。数值不会再次规范化、偏移，也不会经过波形前端。每句话独立处理，因此其他句子的填充帧不会进入其波形。

顶层 BigVGAN 加载器读取所选 `checkpoint['generator']` 和配套 JSON。最终发布前会检查波形采样率、单声道形状和帧数。对于 `--text`，`--output-dir` 中写入 `speech.wav`；对于 `--text-file`，写入编号 WAV 文件。此外还会写入 `synthesis.json`，其中包含长度、token 数和所选检查点路径。`--save-mels` 还会将未填充的预测保留在 `mels/` 下。

覆盖现有输出文件需要 `--overwrite`。程序拒绝以目录或符号链接作为输出。即使指定 `--overwrite`，程序也会拒绝与任何输入资源包、所选检查点/配置/统计文件/词典、文本文件、嵌入、参考梅尔频谱或参考音频实际指向同一文件的输出路径。工作进程在临时工作区中计算；通过验证的文件会逐个以原子方式发布。多个文件的发布并非事务性文件系统操作。

## 其他合成入口

`bash scripts/vocoder.sh mel ...` 执行独立的已保存梅尔频谱合成；`wav ...` 对输入 WAV 执行分析/重合成。两者均接受显式匹配的 `--config`、`--checkpoint_file` 和 `--device`，保留各自的 `_generated_e2e.wav`/`_generated.wav` 后缀，并跳过已完成的输出，除非传入 `--overwrite`。根目录启动器会在切换组件 cwd 之前，将受支持 CLI 路径按相对于调用者目录的方式处理。

`scripts/mel_bridge.py convert` 显式处理 CT/TC/BCT/BTC 文件布局，不改变数值尺度；对于批次数组，必须提供真实长度。`inspect` 比较预处理字段，并报告已知前端差异。该文件工具可用于外部保存的数组，属于可选工具；统一文本合成无需单独执行转换命令。
