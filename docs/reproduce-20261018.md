# 复现本轮实验

在仓库根目录使用已验证的项目 Python 环境。具体依赖与兼容范围见 environment.md。以下使用新结果目录，避免把已有结果误当重新训练。

## 完整限时批次

```powershell
.venv\Scripts\python.exe -u -m mis.benchmark --output results/reproduced-synthetic --budget 2
.venv\Scripts\python.exe scripts/summarize_benchmark.py results/reproduced-synthetic --report docs/reproduced-synthetic.md
```

默认生成 24 张图，每张分别研究强／弱定义。三个训练种子各自重新建模，结果按实例落盘；相同目录下可续跑，但配置或 MIS 源码哈希变化会拒绝续跑。

## 作者公开配置诊断

```powershell
.venv\Scripts\python.exe -u scripts/run_public_preset.py results/reproduced-synthetic --output results/reproduced-public-preset
.venv\Scripts\python.exe scripts/summarize_public_preset.py results/reproduced-synthetic results/reproduced-public-preset --report docs/reproduced-public-preset.md
```

固定取三个规模的 k=3、m=2n、生成种子 20261018，分别研究两种定义、三个训练种子、每次 1600 步。结果包含概率、阈值解、可行解与轨迹，原有公开源码没有修改。

## 结果核验

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

本轮 14 项测试全部通过。汇总脚本还会独立核验实例哈希、每个保存解的可行性、随机种子的完整性、贪心／后处理解的极大性及最优性声明是否与其他可行解矛盾。

精确复现随机结果还受 GPU 非确定性和限时预算内完成的训练轮数影响。已保存原始结果是本轮报告的事实依据，不承诺其他设备逐位重现。
