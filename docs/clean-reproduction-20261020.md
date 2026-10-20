# 干净环境复现记录：2026-10-20

本机已从全新目录完成隔离安装、全部 **61 项单元测试（0 失败、0 错误、0 跳过）**、SCIP 精确小实例和弱／强独立集 GPU 端到端验证。最终使用的环境是仓库下的 **`.venv-clean-20261020-r3`**。原项目 `.venv` 和系统 Python 包均未修改。

本轮不是重新运行全部正式性能实验；它验证安装入口、实际需要的依赖、项目算法的 CPU 正确性及 GPU 小实例流程。小实例耗时不用于设备性能或算法优劣比较。

## 隔离与版本

环境由 Python 3.9.23 执行 `venv --without-pip` 创建，未使用 `--system-site-packages`。安装采用 uv 0.10.10，优先复用工作区已有缓存，并使用 `--link-mode copy`。torch 与 torchvision 的大包直接复用缓存。虚拟环境目录从不存在的状态创建，安装入口拒绝覆盖已有环境或结果目录。

[隔离证据](../results/clean-reproduction-20261020/environment.json)和 [GPU 阶段复核](../results/clean-reproduction-20261020/environment-smoke.json)实际检查了 `sys.prefix != sys.base_prefix`、`include-system-site-packages = false`、用户 site 禁用、site-packages 路径和模块来源。检查的全部 15 个直接支持库均从新环境内部导入。基解释器的标准库仍由 venv 正常使用，不等于继承它的第三方 site-packages。

| 组件 | 实际版本 |
|---|---|
| Python | 3.9.23 |
| torch／torchvision | 2.8.0+cu128／0.23.0+cu128 |
| PyTorch CUDA 运行时 | 12.8 |
| DHG | 0.9.5 |
| NumPy／SciPy | 1.26.4／1.13.1 |
| Matplotlib／Optuna | 3.9.4／4.9.0 |
| Requests／scikit-learn | 2.32.5／1.6.1 |
| PySCIPOpt | 5.3.0 |
| torch-geometric | 2.6.1 |
| dimod／dwave-neal／dwave-samplers／pyqubo | 0.12.18／0.6.0／1.6.0／1.5.0 |

全部 **60 个已安装发行包**的版本见 [requirements-freeze.txt](../results/clean-reproduction-20261020/requirements-freeze.txt)。这些新环境的依赖不来自系统第三方包。

## 实际修复的安装缺口

原支持清单缺少 PySCIPOpt。此外，`from src.models import ...` 会先执行上游 `src/__init__.py`，进一步导入 torch-geometric、dimod、dwave-neal 和 pyqubo 相关工具。原项目环境继承系统包，因此未暴露这些缺口。本轮将其按原环境版本补齐，保持上游模型和既有冻结算法源码不变。

DHG 0.9.5 的元数据要求 `torch>=1.12.1,<2.0`，与 RTX 5070 Ti 使用的新版 PyTorch 冲突；安装入口继续显式以 `--no-deps` 覆盖这个旧上界。实际 [uv pip check](../results/clean-reproduction-20261020/dependency-check.txt) 对 60 个包仅报告这一处已声明冲突，因此退出码为 1。不能将此环境描述为依赖元数据全部兼容，也不能由项目通过推断 DHG 的所有功能均兼容。

PySCIPOpt 创建模型时仍发出 `linked SCIP 9.02 ... use version 9.2.1` 的警告；同时 `Model.version()` 返回 `9.02`，而 `Model.printVersion()` 明确打印 **SCIP 9.2.1、SoPlex 7.1.3**。两种输出原样保存在 [scip-version.txt](../results/clean-reproduction-20261020/scip-version.txt)，不将数值 `9.02` 擅自解释成另一个完整版本号。对应的枚举对照、事件和截止时间测试均实际通过。

## CPU、SCIP 与 GPU 结果

完整单元测试实际执行 `unittest discover -s tests -v`，61 项全部通过，包含新增的 10 项实验补全测试、公开网络 CPU 短训练、损失及梯度、强弱定义、数据导入、解码逐阶段等价性、SCIP 事件和时间前缀检查。测试原生输出完整保存在 [native-prepare.txt](../results/clean-reproduction-20261020/native-prepare.txt)，测试源码哈希保存在环境证据中。

GPU 阶段在主实验整条暂停的窗口执行，无其他本任务 GPU 基准重叠。CUDA 张量前向、反向与同步检查通过，设备为 NVIDIA GeForce RTX 5070 Ti。随后对 `instances/toy_overlap.json`（12 个节点，含 2 个孤立节点）分别运行弱、强定义，各种子 0／1／2、每次 200 轮。

| 定义 | 枚举最优值 | SCIP 值／已证明最优 | GPU 神经训练三个种子的值 | 原始阈值解可行 |
|---|---:|---|---|---|
| 弱独立集 | 9 | 9／是 | 9，9，9 | 3／3 |
| 强独立集 | 6 | 6／是 | 6，6，6 | 3／3 |

SCIP 使用快速可行热启动和截止事件记录，两份报告各 1 条结果。GPU 两份报告各含贪心、精确枚举、HiGHS 和 3 次神经训练，共各 6 条结果。`mis.verify_report` 对这 **14 条保存记录**实际完成独立可行性与精确界核验。结果和原生验证输出见：

- [GPU 弱定义](../results/clean-reproduction-20261020/smoke-gpu-weak.json)、[GPU 强定义](../results/clean-reproduction-20261020/smoke-gpu-strong.json)。
- [SCIP 弱定义](../results/clean-reproduction-20261020/smoke-scip-weak.json)、[SCIP 强定义](../results/clean-reproduction-20261020/smoke-scip-strong.json)。
- [完整 GPU 原生输出](../results/clean-reproduction-20261020/native-smoke.txt)、[汇总 JSON](../results/clean-reproduction-20261020/summary.json)。

## 实际执行入口与日志

本次实际执行了下面两个 Windows PowerShell 阶段，均返回成功；不是只做脚本语法检查。实际宿主为 PowerShell 7.6.5。`Prepare` 创建全新环境、安装、隔离检查、冻结包版本、完整测试和 SCIP 小实例；`Smoke` 复核环境、执行 CUDA 反向、两定义 GPU 训练及结果验证。

```powershell
.\scripts\reproduce_clean.ps1 -Python E:/Miniconda3/python.exe `
  -EnvironmentName .venv-clean-20261020-r3 `
  -RunName clean-reproduction-20261020 `
  -CacheDirectory ../../work/uv-cache -Phase Prepare

.\scripts\reproduce_clean.ps1 -Python E:/Miniconda3/python.exe `
  -EnvironmentName .venv-clean-20261020-r3 `
  -RunName clean-reproduction-20261020 `
  -CacheDirectory ../../work/uv-cache -Phase Smoke
```

命令中的缓存相对路径与本次传入的工作区绝对路径指向同一位置。重新验证应选新的环境名和结果名，以保留现有证据。入口也支持 `-Phase All` 顺序执行两个阶段，本次为配合 GPU 计时窗口分阶段执行；没有单独声称测试另一台机器或其他 PowerShell 版本。

最终 [installation.txt](../results/clean-reproduction-20261020/installation.txt) 显式保存每个原生安装命令及合并后的 stdout／stderr，[native-prepare.txt](../results/clean-reproduction-20261020/native-prepare.txt) 和 [native-smoke.txt](../results/clean-reproduction-20261020/native-smoke.txt) 保存执行与验证输出。环境证据记录安装入口和支持清单的实际文件字节 SHA-256；汇总时全部与当前文件匹配。

首轮安装暴露缺依赖而失败；第二轮实际通过了 CPU／SCIP／GPU 检查，但整理时发现本宿主的 `Start-Transcript` 未捕获原生命令输出。因此保留 [首轮记录](../results/clean-reproduction-20261020-attempt1/attempt-notes.md)和 [第二轮记录](../results/clean-reproduction-20261020-attempt2/attempt-notes.md)，如实注明日志不足，并从第三个不存在的新目录完整重做。没有补造前两轮的完整原生安装日志。

最终保留 `.venv-clean-20261020-r3`；本任务创建的首轮和第二轮被替换环境在逐个确认解析绝对路径恰好为仓库下指定目录后清理，所有尝试的结果记录与 uv 缓存保留。安全核验和清理结果见 [cleanup.json](../results/clean-reproduction-20261020/cleanup.json)。本轮没有在新环境重跑正式大批次，也没有验证跨机器性能一致性；这不影响已完成的本机隔离安装和正确性复现结论。
