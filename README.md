# 基于 BIPNN 的超图最大独立集研究

本项目基于作者公开的 BIPNN 代码，开展超图最大独立集的毕业论文研究。已完成合成对照、独立验证／测试、三个公开真实数据版本及计算实现比较；方法继承与实验边界均明确记录。

## 实验进度

2026-10-18 最新一轮已完成：统一使用等价快速解码器，在六张新合成图（500／1000／2000 节点、两种结构，弱定义）及 Cora／High／Primary 两种定义上评价 5／15／30 秒结果。新增 36 条神经训练轨迹、36 条随机轨迹、36 次独立 SCIP 求解，共 300 条主结果记录。1223 个网络候选、随机日志与 SCIP 界限通过全量独立核验；方法章节草稿已完成。

- [快速解码、严格时间截止与规模扩展报告](docs/fast-anytime-20261018.md)
- [时间／规模曲线与完整图注](results/fast-anytime-20261018/figures/caption.md)
- [方法章节草稿与原始文献](docs/thesis-method-draft.md)
- [受控消融、锁定测试与真实数据报告](docs/controlled-20261018.md)
- [真实数据来源、编号和约束结构审计](docs/data-provenance.md)
- [损失计算的全部 12 组微基准与配图](results/loss-kernel-20261018/figures/caption.md)
- [保持逐节点解一致的 CPU 解码对照](results/decode-kernel-20261018/README.md)
- [毕业论文主线与后续完成标准](docs/thesis-workplan.md)

新合成图的 30 秒结果按图先平均三个种子，再平均实例相对差：较低度贪心 +7.27%、同解码器随机重启 +11.74%、SCIP 截止可行解 +2.85%，六张图的种子均值均超过该预算 SCIP 解。5 秒时平均仍低于 SCIP 1.67%。三个真实数据版本上，神经结果的均值在两种定义下都低于 SCIP；Cora 弱定义为 1029，已证明最优值为 1030。结果限于这些实例，不能推断普遍优势。

神经与随机的三个检查点来自同一条 30 秒轨迹，SCIP 则独立按三个预算求解；只选截止前完整可用的候选，实际发出时间另列。本轮最大网络快照晚发约 0.319 秒。旧验证锁定配置继续沿用，但固定 30 秒退火及历史候选选优也有变化，不能把新旧端到端差异全部归因于解码提速。真实数据是已见版本的重新评价。

此前独立验证／测试批次完成 178 次训练、490 条记录、1068 组阶段证据，仍完整保留。其锁定合成测试中，弱定义对贪心 +2.74%、随机重启 +7.87%、SCIP 限时可行解 -1.17%；旧批次解码及时间边界不同，须分别解释。

本轮启动前 51 项正确性测试全部通过。独立 CPU 解码微基准的 18 个随机分数案例逐阶段完全一致，耗时比中位数约 186 倍；这是后处理微基准测量。本轮已统一应用快速解码，旧冻结训练保持原实现；该倍数不代表神经网络整网加速。

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
- 关联计数式后处理的逐阶段等价实现与测试；最新批次统一用于网络和随机基线，旧批次原样保留。
- 固定 30 秒退火、在线可行候选与严格截止前选优，记录实际发出时间及晚到诊断。
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

1. 根据现有阶段记录，有针对性地检查惩罚系数、退火及真实图冗余约束的作用；不以新测试结果重新选参。
2. 在已完成的方法草稿基础上撰写实验分析，补引言、相关工作和总结，并按学校模板整理。
3. 若需要更广的规模结论，再扩充独立图种子与分布；当前每个规模／结构仅一张图，三个网络种子是技术重复。

现有结果不构成全面优于传统求解器的证明。新批次复现入口及独立核验命令见最新报告。旧完整阶段可用 `scripts/run_controlled_all.ps1` 复现；该封装仅完成语法检查，旧批次实际按报告中的命令顺序执行，干净环境安装尚未完整验证。
