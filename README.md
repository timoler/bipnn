# 基于 BIPNN 的超图最大独立集研究

本项目基于作者公开的 BIPNN 代码，开展超图最大独立集的毕业论文研究。当前阶段为小实例建模与运行验证，不代表已完成正式基准实验。

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
- 小规模精确枚举、低度优先贪心及正确性测试。
- JSON 结果包含实例、种子、配置、解、计时及源码版本信息。

两种定义均保留用于验证，最终论文路线尚未确定。方法及计时边界见 [建模说明](docs/mis-method.md)。

## 运行

在仓库根目录、已配置 torch 和 dhg 的 Python 环境运行：

```powershell
python -m unittest discover -s tests -v
python -m mis.run --instance instances/toy_overlap.json --definition weak --device cuda --epochs 200 --seeds 0 1 2 --output results/smoke-weak.json
python -m mis.run --instance instances/toy_overlap.json --definition strong --device cuda --epochs 200 --seeds 0 1 2 --output results/smoke-strong.json
```

没有可用 CUDA 时可以显式改为 `--device cpu`。请求 CUDA 失败会报错，不会悄悄回退 CPU。Windows 项目虚拟环境可用 `.venv\Scripts\python.exe` 替换上述 `python`。

原仓库 environment.yaml 和 pyproject.toml 保留作历史参考，不应直接用其 CUDA 11.7 配置安装 RTX 50 系列环境。当前环境验证记录将在设备验证完成后保存到 docs/environment.md。

## 后续工作

1. 确定论文最终采用的独立集定义。
2. 核对真实数据的格式与原始节点数量，建立标准实例集。
3. 增加匹配定义的精确／启发式求解器基线，统一时间预算。
4. 分析后处理贡献、惩罚系数及随机种子影响。
5. 开展正式规模实验并撰写方法与实验章节。

当前小实例仅用于验证流程；不能据此声称超过传统求解器或具备大规模性能优势。
