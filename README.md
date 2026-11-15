# 第12课：两层 Transformer 实验

需要 Python 3.10 或更新版本，无第三方依赖。解压后在本目录执行：

```sh
python3 -m experiments.transformer_depth.probe
python3 -m experiments.transformer_depth.probe --position 2 --replace t
python3 -m unittest experiments.transformer_depth.test_depth
```

默认探针使用随包双层权重，不需要重新训练。详细说明见 experiments/transformer_depth/README.md。此包只包括运行本课实验所需源码、语料、权重与结果；固定来源快照 c44308cdaed4eff56e5e70c19d63ee66bc74614c，语料来源和分割见 experiments/short_text/data-manifest.json。
