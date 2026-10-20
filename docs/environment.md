# 运行环境与验证边界

## 2026-10-20：全新隔离安装已实际验证

已在不继承系统第三方包的 `.venv-clean-20261020-r3` 中完成安装、61 项测试（无跳过）、SCIP 与枚举对照，以及弱／强各三个种子、200 轮的 GPU 小图端到端复现。14 条保存记录独立核验通过。安装入口已补齐 PySCIPOpt 和上游包初始化所需依赖，完整安装及原生测试输出均保留，详见 [干净环境复现报告](clean-reproduction-20261020.md)。这是正确性与安装复现，不是全部正式性能实验在新环境中的重跑。

原项目 `.venv` 仍用于冻结的正式实验；以下保留其形成过程和较早验证边界。DHG 的旧 PyTorch 版本上界冲突仍显式存在，不由小图复现推断其所有功能兼容。

## 设备检查

本机识别到 NVIDIA GeForce RTX 5070 Ti，显存约 16 GB，驱动版本 595.79。nvidia-smi 中的 CUDA 13.2 表示驱动支持的 CUDA 版本，不能据此判断 Python 已安装 CUDA 版 PyTorch。

初始默认 Python 为 3.9.23，PyTorch 为 2.8.0+cpu，因此初始环境无法调用 GPU。CPU 环境已通过 8 项测试，并完成两种定义各三个种子的 200 轮小实例实验。

另一个已有 Python 环境的 CUDA 张量前向与反向运算通过，表明显卡和驱动能够工作；它不是本项目目前固定的运行环境。

## 项目环境

2026-10-18 快速解码与时间前缀批次启动前，冻结提交 `bcedc04` 对应的 51 项测试全部通过。在原 36 项基础上，新增 11 项快速基线测试和 4 项时间前缀测试，覆盖等价贪心、SCIP 截止前事件与晚到界隔离、候选选取及同轨迹快照。完整实验的解可行性与截止选择另由独立汇总审计核验，不能用单元测试代替。运行环境继续使用以下项目 `.venv`，未更换依赖。

本轮 manifest 包含原始文件字节哈希，自动换行转换也会改变哈希。仓库 `.gitattributes` 已按实测工作副本固定相关源码、验证选择文件及三份原始数据的 LF／CRLF 形式；不改变算法或数据内容。在 `core.autocrlf=false` 与 `true` 两种设置下从 Git 暂存版本分别重建 18 个关键文件，字节哈希均完全匹配；记录见 [换行重建核验](../results/checkout-verification-20261018.json)。复核历史结果应保留这些属性，不手工格式化冻结源码与选择文件。2026-10-20 已补做本机干净安装；跨设备完整训练尚未验证，换行规则检查也不等于环境可移植性证明。

2026-10-20 进一步固定收尾计划、父／子批次清单、输入文件及测试源码的换行，并在上述两种设置下逐文件比较 SHA-256，详见 [收尾批次检出核验](../results/checkout-verification-20261020.json)。中断归档的原始清单也保留其字节形式，保证中断说明中的原始哈希可复核。这项检查只验证文件检出的一致性。

2026-10-18 增量验证：本项目环境的 36 项测试全部通过，包括索引／稠密损失和梯度、公开网络短训练、分阶段解码、三份真实数据导入、SCIP 近似总时间预算以及计数式解码的逐阶段等价性。记录见 `results/checks-20261018.txt`。下面保留首次安装过程，较早的测试数量属于当时快照。

项目 .venv 从现有 Python 创建，复用已有的第三方支持包，在项目目录内单独安装 CUDA 版 torch 和 torchvision。该环境没有修改系统 Python 包，但它依赖基环境中的支持库，并非完全隔离的可移植环境。

已在项目 .venv 安装 torch 2.8.0+cu128、torchvision 0.23.0+cu128。CUDA 张量前向和反向计算通过，使用该环境的 10 项单元测试全部通过。GPU 小实例结果保存在 results 中。

PyTorch 官方说明从 2.7 开始提供 Blackwell 与 CUDA 12.8 支持：
https://pytorch.org/blog/pytorch-2-7/

## DHG 版本约束

当前使用 DHG 0.9.5，其包元数据仍声明 torch>=1.12.1,<2.0，与上述 GPU 环境存在版本约束冲突。这里明确采用绕过该旧版上界的方式，仅验证本项目实际使用的 HGNNPConv 操作；不能将有限测试解释为 DHG 全部功能与新 PyTorch 完全兼容。

不要直接对环境执行自动降级，也不要使用原始 environment.yaml 覆盖当前配置。

## 干净环境的安装入口

提供 `scripts/setup_mis.ps1`，以指定基础 Python 和 uv 创建不继承系统包的新环境。脚本安装固定版本的 torch、torchvision 和 `requirements-mis-support.txt`，再以 `--no-deps` 安装 DHG，拒绝覆盖已有虚拟环境。完整复现封装 `scripts/reproduce_clean.ps1` 还执行隔离检查、测试、SCIP 和 GPU 小图，并保存原生输出。

```powershell
.\scripts\reproduce_clean.ps1 -Python E:/Miniconda3/python.exe -EnvironmentName .venv-clean-verify -RunName clean-verify -Phase All
```

请为复现选择未使用过的环境名和结果名。2026-10-20 已实际从新目录执行 `Prepare` 与 `Smoke` 两阶段，证据和实际版本见上方报告；`All` 为顺序组合入口。原项目 `.venv` 及系统包均未更改，较早结果也未重写。

## 验证结果文件

- results/smoke-cpu-weak.json：CPU 弱独立集小实例。
- results/smoke-cpu-strong.json：CPU 强独立集小实例。
- results/smoke-gpu-weak.json：GPU 弱独立集小实例。
- results/smoke-gpu-strong.json：GPU 强独立集小实例。

独立检查已保存结果的命令：

```powershell
python -m mis.verify_report results/smoke-cpu-weak.json results/smoke-cpu-strong.json
python -m mis.verify_report results/smoke-gpu-weak.json results/smoke-gpu-strong.json
```

结果用于验证程序和记录流程，不能作为不同设备的性能基准；首次调用开销和初始化差异尚未做严格控制。
