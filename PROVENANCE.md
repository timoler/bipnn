# 来源与开发边界

本仓库基于作者公开仓库：
https://github.com/Classpi/BIPNN-Learning-to-Solve-Binary-Integer-Programming-via-Hypergraph-Neural-Networks

起始提交：708c10a95a50fb2b06dd35bf1604e65a6ec5ebe5。

对应论文：Sen Bai, Chunqi Yang, Xin Bai, Xin Zhang, Zhengang Jiang. BIPNN: Learning to Solve Binary Integer Programming via Hypergraph Neural Networks. NeurIPS 2025.

论文页面：https://papers.nips.cc/paper_files/paper/2025/hash/8693ee1ea821666f8569228d1ab38baf-Abstract-Conference.html

保留公开仓库历史与 GPL-3.0 LICENSE。原有 src、数据、Notebook 和图片来自上述公开版本。原 README 另存于 docs/upstream-README.md，仅调整相对图片路径和行尾空格。

新增 mis、tests、instances 及相关文档用于明确两种独立集定义、验证建模、实现可行解后处理及记录小实例结果。继承 BIPNN 的无监督多项式优化思路，并使用 DHG 的 HGNN+ 层；不将这些既有方法声明为本项目原创。

当前成果属于开发验证，不是正式论文复现或性能优越性证明。

2026-10-18 增加 MIS 目标的索引式等价计算、分阶段可行解分析、独立验证／测试协议和公开真实数据的严格导入。索引计算沿用相同多项式，不能称为新的数学目标；损失微基准也不能代替完整网络的速度验证。受控训练保留公开网络与训练步，时间归一化退火是新增适配，与原 Notebook 固定步数日程分别记录。

真实数据导入保留上游文件及来源，新增 JSON 只是带哈希和编号映射的转换版本。Cora 的公开文件与论文表 4 数量不同；详见 [数据审计](docs/data-provenance.md)。
