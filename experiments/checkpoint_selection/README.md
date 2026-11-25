# 训练到哪一步，权重应该留下来？

同一个两层字符模型只反复使用 32 个训练窗口。到第 50 步，训练损失是 2.9898，验证损失是 3.3576。到第 1500 步，训练损失降到 0.1314，验证损失升到 9.1501。假设要用这个模型续写没见过的文本，你会保留哪一步？

在仓库根目录用 Python 3.10 或更新版本运行下面的命令。直接下载并解压源码也可以运行。

```sh
python3 -m experiments.checkpoint_selection.readback --case limited-data --choice best
python3 -m experiments.checkpoint_selection.readback --case limited-data --choice last
```

程序会分别载入第 50 步和第 1500 步。两次用相同前缀和随机种子续写，并核对权重以及分配给候选字符的原始分数（logits）。判断该保留哪份权重时，用同一批留出文本的平均预测损失来比较。两份权重都已保存，运行时无需训练。

## 把训练窗口放宽到 8,072 个

原训练可以从 8,072 个窗口抽批次，验证题和初始化与上面相同。先只看截至第 1000 步的记录，再看完第 1500 步。你会在什么时候更新自己的选择？

```sh
python3 -m experiments.checkpoint_selection.readback --last-observation 1000
python3 -m experiments.checkpoint_selection.readback
```

训练集和验证集按完整文本记录划分后各自生成窗口。有些窗口会重叠，验证集共计 13,284 个字符预测位置。验证时只算损失。完整数据见[四批轮训记录](limited-data-experiment/result.json)和[原训练记录](experiment/result.json)。

## 从初始化重跑

前面的命令读回现成权重。如果想验证训练本身，先在当前 Git 仓库旁检出第 12 课的固定版本，再分别执行重放脚本。请使用尚不存在的输出目录。每次在 CPU 上约需几分钟。

```sh
git worktree add --detach ../lesson12-source 3467b442a2ba1d944a97742eec0d4c4e25226b80
mkdir -p runs
python3 -B experiments/checkpoint_selection/checkpoint_probe.py --source ../lesson12-source --output runs/checkpoint-replay
python3 -B experiments/checkpoint_selection/limited_data_probe.py --source ../lesson12-source --output runs/limited-data-replay
```

若使用下载的源码压缩包，请在当前目录旁[克隆课程仓库](https://github.com/rarest/nblord-tiny-llm-course)，并将 `../lesson12-source` 切到上面这个提交。输出记录包含每步的损失和权重哈希，便于与随课记录比较。
