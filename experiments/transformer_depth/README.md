# 给小模型再接一个 Block

上一课的模型把最近六个字符送进一个 Transformer Block，再预测下一个字符。这次保留每个位置的八维表示和两个注意力头，比较一个 Block 与两个串联 Block。两个 Block 各有一套参数；字符和位置只在入口相加一次，最后的归一化和词表输出层也只执行一次。

运行前先想一想：第二个 Block 的输入是原来的字符向量，还是第一个 Block 的输出？如果只改第 5 个字符，前四个位置的输出会变化吗？

在仓库根目录，用 Python 3.10 或更新版本运行：

```sh
python3 -m unittest experiments.transformer_depth.test_depth
python3 -m experiments.transformer_depth.probe
# 可选：重新训练并保存到新目录
python3 -m experiments.transformer_depth.compare --output /tmp/my-depth-run --steps 1500
```

前两条命令直接检查代码并读取已训练权重，无需重新训练。可选的第三条命令使用 CPU 和 Python 标准库，要求输出目录不存在；它会分别训练一层和两层模型，保存权重、共同的批次顺序、损失与续写。运行时先看日志中的 `layers` 和 `step`；结束后打开输出目录的 `results.json`。要先试通命令，可以把步数改成 `--steps 2` 并使用另一个新目录，但这个短跑结果不能与完整训练混为一谈。

探针使用双层权重观察 `First `。`layers` 中每层的 `last_input`、`last_query` 和 `last_output` 是最后一个位置的八个数；第二层的输入应等于第一层的输出。默认把位置 4 的 `t` 换成 `s`，`causal_check` 比较各层及最终输出中位置 0 到 3 的数，并给出最大差值。可以用 `--position` 和 `--replace` 改变检查条件。

想知道改动传到了哪些位置，可以把位置 2 的 `r` 换成 `t`。先猜位置 0、1 和末尾空格各自会怎样，再运行：

```sh
python3 -m experiments.transformer_depth.probe --summary --position 2 --replace t
```

表中各层的数值来自对应位置的输出向量，`logits` 列比较模型在该位置给下一个字符候选的原始分数。每格取两次计算之间的最大绝对差值。位置 0 和 1 的差值为 0，它们只能读取自己和更早的字符。末尾仍是空格，但它能读取位置 2，本次输出的最大分数变化约为 0.3674。去掉 `--position 2 --replace t` 再跑一次，看看保持为 0 的范围怎样改变。

两个模型训练时看过相同数量的目标字符，不表示它们的运算量相同。比较 `results.json` 时，同时看训练与验证损失、生成结果、参数数目和分阶段耗时；多一层未必让这个小模型的续写更好。

随仓库保存的这次运行里，两者各处理 72,000 个目标字符；单层／双层验证集全位置 NLL 分别为 2.5501／2.5685，参数量分别为 1,881／2,729。两者 120 步贪心续写都陷入 `the` 重复。运行机器上的训练时间分别约 46／78 秒；这是本次程序和机器的计时。完整观测、生成和批次序列保存在 `results.json` 与 `schedule.json`；其中 `schedule_sha256` 对序列的紧凑 JSON 表示计算，而非对带空格的文件字节计算。

想进一步动手：

1. 对照第一层 `last_output` 和第二层 `last_input`，解释它们为什么相同；再比较两层 `last_query`，解释为什么不必相同。
2. 改变被替换字符的位置，先预测哪些位置必须保持不变，再运行探针核对。
3. 阅读 `test_shared_block_accumulates_both_uses_and_updates_once`：两次调用同一个 Block 时，为什么一次更新要使用两条路径的梯度之和？

本实验沿用仓库的短文本语料及其 `data-manifest.json`。训练集和验证集先按完整记录划分，再形成长度为六的窗口。代码和结果记录具体语料哈希、批次哈希及随机种子。
