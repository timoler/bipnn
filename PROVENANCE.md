# 来源与开发边界

本仓库基于作者公开仓库：
https://github.com/Classpi/BIPNN-Learning-to-Solve-Binary-Integer-Programming-via-Hypergraph-Neural-Networks

起始提交：708c10a95a50fb2b06dd35bf1604e65a6ec5ebe5。

对应论文：Sen Bai, Chunqi Yang, Xin Bai, Xin Zhang, Zhengang Jiang. BIPNN: Learning to Solve Binary Integer Programming via Hypergraph Neural Networks. NeurIPS 2025.

论文页面：https://papers.nips.cc/paper_files/paper/2025/hash/8693ee1ea821666f8569228d1ab38baf-Abstract-Conference.html

保留公开仓库历史与 GPL-3.0 LICENSE。原有 src、数据、Notebook 和图片来自上述公开版本。原 README 另存于 docs/upstream-README.md，仅调整相对图片路径和行尾空格。

新增 mis、tests、instances 及相关文档用于明确两种独立集定义、验证建模、实现可行解后处理及记录小实例结果。继承 BIPNN 的无监督多项式优化思路，并使用 DHG 的 HGNN+ 层；不将这些既有方法声明为本项目原创。

当前成果属于开发验证，不是正式论文复现或性能优越性证明。
