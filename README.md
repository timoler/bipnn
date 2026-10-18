# 基于 BIPNN 的超图最大独立集研究

本项目基于作者公开的 BIPNN 代码，开展超图最大独立集的毕业论文研究。已完成合成对照、独立验证／测试、三个公开真实数据版本及计算实现比较；方法继承与实验边界均明确记录。

## 实验进度

2026-10-18 最新一轮：4 张新验证图选择网络宽度与退火配置，锁定后评价 16 张新测试图及 Cora／High／Primary 三个真实数据版本，两种独立集定义均保留。新增 178 次训练、490 条方法运行记录、1068 组初始／轨迹／最终阶段证据，全部独立核验。

- [受控消融、锁定测试与真实数据报告](docs/controlled-20261018.md)
- [真实数据来源、编号和约束结构审计](docs/data-provenance.md)
- [损失计算的全部 12 组微基准与配图](results/loss-kernel-20261018/figures/caption.md)
- [毕业论文主线与后续完成标准](docs/thesis-workplan.md)

锁定合成测试中，弱独立集的实例平均解大小相对低度贪心提高约 2.74%，相对同解码器随机重启提高约 7.87%，仍低于 SCIP 限时可行解约 1.17%。强定义未整体超过贪心。真实数据时间超出约 5 秒目标的情况明显，必须结合实际耗时解释，不能声称统一等时优势。

此前的开发与公开配置诊断也完整保留：

- [限时小模型的完整对照报告](docs/experiments-20261018.md)
- [作者公开代码配置的补充实验](docs/public-preset-20261018.md)
- [对照结果图](results/synthetic-20261018/figures/paired-comparisons.pdf)

补充实验采用更长的固定训练预算，须与限时批次分别解释；详见报告中的适用范围。

## 来源

- [BIPNN 原论文](https://papers.nips.cc/paper_files/paper/2025/hash/8693ee1ea821666f8569228d1ab38baf-Abstract-Conference.html)
- [作者公开仓库](https://github.com/Classpi/BIPNN-Learning-to-Solve-Binary-Integer-Programming-via-Hypergraph-Neural-Networks)
- [继承与新增内容说明](PROVENANCE.md)
- [原始项目说明](docs/upstream-README.md)

保留原始 Git 历史及 GPL-3.0 许可证。原有 src 和 example 用于参考；新增 mis 提供显式区分强／弱独立集的验证入口。

## 当前实现

- 弱独立集：每条超边不能全部选中，使用高阶 PUBO 惩罚。
- 强独立集：每条超边最多选一个，使用二次节点对惩罚。
- HGNN+ 逐实例优化，显式记录阈值解与修复后的可行解。
- 小规模精确枚举、HiGHS 整数规划、低度优先贪心及正确性测试。
- SCIP 加贪心初解、相同解码器的随机单次／限时重启，以及未训练网络输出对照。
- 稠密／索引 PUBO 的数值梯度、存储与微内核计时比较；索引表示用于新受控训练，弱定义不保证提速。
- 关联计数式后处理的逐阶段等价实现与测试；本轮冻结训练保持旧解码器，新实现独立计时。
- JSON 结果包含实例、种子、配置、解、计时及源码版本信息。

建议论文以保留高阶约束的弱独立集为主线，强定义作为补充对照；两者分别报告。方法及计时边界见 [建模说明](docs/mis-method.md)。

## 运行

在仓库根目录、已配置 torch 和 dhg 的 Python 环境运行：

```powershell
python -m unittest discover -s tests -v
python -m mis.run --instance instances/toy_overlap.json --definition weak --device cuda --epochs 200 --seeds 0 1 2 --output results/smoke-weak.json
python -m mis.run --instance instances/toy_overlap.json --definition strong --device cuda --epochs 200 --seeds 0 1 2 --output results/smoke-strong.json
```

没有可用 CUDA 时可以显式改为 `--device cpu`。请求 CUDA 失败会报错，不会悄悄回退 CPU。Windows 项目虚拟环境可用 `.venv\Scripts\python.exe` 替换上述 `python`。

原仓库 environment.yaml 和 pyproject.toml 保留作历史参考，不应直接用其 CUDA 11.7 配置安装 RTX 50 系列环境。具体配置与 DHG 版本限制见 [环境说明](docs/environment.md)。

## 后续工作

1. 将等价的快速解码器统一用于网络和随机基线，按新协议重新比较时间与质量。
2. 扩展训练预算曲线和 1000／2000 节点的完整求解实验。
3. 有针对性地检查惩罚系数、退火及真实图冗余约束的作用。
4. 撰写建模、计算实现与实验分析章节，逐项引用继承方法及数据版本。

现有结果不构成全面优于传统求解器或大规模速度优势的证明。完整阶段可用 `scripts/run_controlled_all.ps1` 复现；该封装仅完成语法检查，本轮实际按报告中的同阶段命令顺序执行，干净环境安装尚未完整验证。
