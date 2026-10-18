# 运行环境与验证边界

## 设备检查

本机识别到 NVIDIA GeForce RTX 5070 Ti，显存约 16 GB，驱动版本 595.79。nvidia-smi 中的 CUDA 13.2 表示驱动支持的 CUDA 版本，不能据此判断 Python 已安装 CUDA 版 PyTorch。

初始默认 Python 为 3.9.23，PyTorch 为 2.8.0+cpu，因此初始环境无法调用 GPU。CPU 环境已通过 8 项测试，并完成两种定义各三个种子的 200 轮小实例实验。

另一个已有 Python 环境的 CUDA 张量前向与反向运算通过，表明显卡和驱动能够工作；它不是本项目目前固定的运行环境。

## 项目环境

2026-10-18 增量验证：本项目环境的 36 项测试全部通过，包括索引／稠密损失和梯度、公开网络短训练、分阶段解码、三份真实数据导入、SCIP 近似总时间预算以及计数式解码的逐阶段等价性。记录见 `results/checks-20261018.txt`。下面保留首次安装过程，较早的测试数量属于当时快照。

项目 .venv 从现有 Python 创建，复用已有的第三方支持包，在项目目录内单独安装 CUDA 版 torch 和 torchvision。该环境没有修改系统 Python 包，但它依赖基环境中的支持库，并非完全隔离的可移植环境。

已在项目 .venv 安装 torch 2.8.0+cu128、torchvision 0.23.0+cu128。CUDA 张量前向和反向计算通过，使用该环境的 10 项单元测试全部通过。GPU 小实例结果保存在 results 中。

PyTorch 官方说明从 2.7 开始提供 Blackwell 与 CUDA 12.8 支持：
https://pytorch.org/blog/pytorch-2-7/

## DHG 版本约束

当前使用 DHG 0.9.5，其包元数据仍声明 torch>=1.12.1,<2.0，与上述 GPU 环境存在版本约束冲突。这里明确采用绕过该旧版上界的方式，仅验证本项目实际使用的 HGNNPConv 操作；不能将有限测试解释为 DHG 全部功能与新 PyTorch 完全兼容。

不要直接对环境执行自动降级，也不要使用原始 environment.yaml 覆盖当前配置。

## 干净环境的安装入口

提供 scripts/setup_mis.ps1，以现有 Python 和 uv 创建新的 .venv-clean。该脚本安装指定版本的 torch、torchvision 和 requirements-mis-support.txt，再以 --no-deps 安装 DHG。脚本不会覆盖已有虚拟环境。

```powershell
.\scripts\setup_mis.ps1
.venv-clean\Scripts\python.exe -m unittest discover -s tests -v
```

该干净安装脚本尚未从空环境完整执行验证；本次实际验证使用上述项目 .venv 和既有 CPU 环境。安装完成仍需运行测试和 CUDA 小实例验证。

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
