# NBLord 小模型课程实验

## 第 12 课　两层 Transformer

需要 Python 3.10 或更新版本，无第三方依赖。解压后在本目录执行：

```sh
python3 -m experiments.transformer_depth.probe
python3 -m experiments.transformer_depth.probe --position 2 --replace t
python3 -m unittest experiments.transformer_depth.test_depth
```

默认探针使用随包双层权重，无需重新训练。[实验说明](experiments/transformer_depth/README.md)记录了模型和运行结果。第 12 课的材料来自快照 `c44308cdaed4eff56e5e70c19d63ee66bc74614c`，语料来源与分割见[数据清单](experiments/short_text/data-manifest.json)。

## 第 13 课　选哪一步的权重

同一个两层模型，训练损失继续下降，验证损失却升了。先选一份权重，再[运行实验](experiments/checkpoint_selection/README.md)比较第 50 步和第 1500 步。
