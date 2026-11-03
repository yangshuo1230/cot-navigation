# Fixed DAG：FFN 是存储还是处理图信息

Direct 最佳 checkpoint，简单分层图与困难图各一张。冻结 FP32、CPU、math attention。不改训练代码，不写 checkpoint 或 W&B。

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
OMP_NUM_THREADS=4 .venv/bin/python research/ffn_role_20261023/study.py
```

结论与口径见 `REPORT.md`。图为 `ffn_role.png`。逐项数字在 `simple.json`、`hard.json`。
