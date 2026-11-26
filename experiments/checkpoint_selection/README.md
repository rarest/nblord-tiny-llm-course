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

想亲眼看两条损失曲线怎样分开，可以用下载包里的代码和语料从头训练。在仓库根目录用 Python 3.10 或更新版本运行。下面两个命令各用一个新目录保存结果，每次在 CPU 上约需几分钟。

```sh
python3 -B experiments/checkpoint_selection/limited_data_probe.py --output run-32
python3 -B experiments/checkpoint_selection/checkpoint_probe.py --output run-full
```

先看 `run-32/result.json` 的 `rows`。找出验证损失最低的那一步，再看训练损失此后怎么走。打开 `run-full/result.json`，在相同的第 500、1000、1500 步比较两次训练的验证损失。两次使用同样的初始化和验证题，第一条命令只反复练四个批次，第二条从全部 8,072 个窗口取批次。

程序先按第 12 课固定版本核对模型源码、语料和批次顺序。已有该版本独立 Git 检出的读者仍可传 `--source <检出目录>`。再次运行时另取一个新输出目录。
