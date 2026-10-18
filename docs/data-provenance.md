# 真实超图数据来源、格式和建模审计

核查日期：2026-10-18。对象是 BIPNN 作者公开仓库固定版本 `708c10a95a50fb2b06dd35bf1604e65a6ec5ebe5` 中的三个文本文件。标准导入由 `mis/datasets.py` 实现，原始文件不改动。

## 来源能确认到什么程度

三个文件均继承自 [BIPNN 作者公开仓库的 data 目录](https://github.com/Classpi/BIPNN-Learning-to-Solve-Binary-Integer-Programming-via-Hypergraph-Neural-Networks/tree/708c10a95a50fb2b06dd35bf1604e65a6ec5ebe5/data)。本地检查表明，它们在 CRLF→LF 换行规范化后与该上游 Git 对象逐字节相同。导入元数据记录文件 URL、上游提交、原始字节哈希和 LF 规范化哈希。

正式论文附录 F.1 表 4 给出了三个名称的节点数、超边数和关联总数，可用于比对数量；它没有给出足以逐边确认三个文件原始采集身份及全部预处理步骤的信息。因此实验名称使用“BIPNN 公开仓库版本”，不把相同数量当作相同边集合的证明。[来源：NeurIPS 正式论文，PDF 第 15 页](https://papers.nips.cc/paper_files/paper/2025/file/8693ee1ea821666f8569228d1ab38baf-Paper-Conference.pdf)

| 名称 | 本文件节点数 | 本文件超边数 | 本文件关联总数 | 论文表 4：节点 / 超边 / 关联 | 结论 |
|---|---:|---:|---:|---|---|
| Cora | 1330 | 1413 | 4370 | 1330 / 1503 / 4599 | 边数、关联数不同，应明确标为公开仓库的 Cora 变体 |
| High | 327 | 7818 | 18192 | 327 / 7818 / 18192 | 三项统计一致；尚未证实边集合身份 |
| Primary | 242 | 12704 | 30729 | 242 / 12704 / 30729 | 三项统计一致；尚未证实边集合身份 |

Cora 差了 90 条边、229 个关联，不能以本文件的结果声称复现论文同一实例上的 Max-Cut 数值。这里接下来的任务是求解 MIS，本来也不能直接使用原论文的 Max-Cut 解值作为 MIS 上界或目标值。

作为原始采集来源的线索，XGI 维护的 [High School 数据说明](https://github.com/xgi-org/xgi-data/blob/main/datasheets/contact-high-school/README_contact-high-school.md) 与 [Primary School 数据说明](https://github.com/xgi-org/xgi-data/blob/main/datasheets/contact-primary-school/README_contact-primary-school.md) 说明了 SocioPatterns 接触记录及按同时接触的极大团生成超边的过程。XGI 版本还含时间重复记录，并有不同的原始边计数。当前没有把它们的边集合与 BIPNN 文件逐边对应，因此仅把它们列为候选来源，不把时间窗口、人物标签及清洗过程直接归属给这两个 BIPNN 文件。Cora 的具体共引／合作构造同样未知。上述未知项写入机器可读元数据。

## 文件格式与标准化政策

| 文件 | 必须显式选用的 header 政策 | 原始节点标签 | 实际记录数 | 重复超边 | 边内重复顶点 |
|---|---|---|---:|---:|---:|
| `cora_hypergraph.txt` | `none`：第一行 `88 115 275` 是超边 | 连续 0–1329 | 1413 | 0 | 0 |
| `high_hypergraph.txt` | `edges_nodes`：首行 `7818 327` 按“边记录数 节点数”解释并核验 | 连续 1–327 | 7818 | 0 | 0 |
| `primary_hypergraph.txt` | `edges_nodes`：首行 `12704 242` 同上 | 连续 1–242 | 12704 | 0 | 0 |

1. 不根据第一行恰有两个 token 自动判断 header。无 header 时，首条二节点边同样必须保留。
2. 按原始整数标签升序映射到 0 开始的索引，并保存完整双向可恢复的标签表。不根据最大标签猜节点总数。
3. 对带 header 的文件，声明边数必须等于实际边记录数，声明节点数必须等于完整标签集合大小；不符合即报错。声明边数在去重前核对。
4. 一个节点标签若在同一条边内重复出现，直接报错，避免把图自环擅自解释为单节点超边。合法的单节点超边可以保留，其弱／强 MIS 含义由现有定义决定。
5. 顶点在边内的排列不影响边身份。完全相同的超边去重，记录删除数；这是无权 MIS 的约束集合。当前三个文件没有重复超边，也没有进行边权推断。
6. 只有提供完整 `node_labels` 时，导入器才允许表示文件里未出现的孤立节点。数量差额不足以推断缺失节点的原始标签。当前 High／Primary 的 header 节点均出现过，声明范围内没有孤立节点；Cora 覆盖文件中观察到的 1330 个节点，是否存在文件范围外未提供的节点仍标为未知，不把它们补成新节点。
7. 仅忽略空行和整行 `#` 注释。原始字节 SHA-256 与 CRLF→LF 规范化 SHA-256 同时保存；固定上游校验使用后者，兼容不同平台的 Git 换行转换，其他任何字节变化仍会失败。

下表为固定上游 Git 对象的 LF 规范化 SHA-256。工作区 CRLF 文件的原始字节哈希会不同，实际值由每次导入记录，不能把这种换行差异解释为实例内容变化。

| 文件 | LF 规范化 SHA-256 |
|---|---|
| Cora | `8b6632b64bd139eb2e66a20904e3acd1c7202ac214272271f182688c1e893be5` |
| High | `81de2ca7ce7f4d00b33403241ff048359b8c8040af4fcd2697b79a481a279e60` |
| Primary | `9cdbe0fdc2638b8ad2357b9cbe608b2219d2119535d063d14f5cdac49e752eb4` |

## 结构统计与高阶约束的实际作用

度数指去重后，一个节点所属的超边条数。所有边大小均为 2–5。

| 文件 | k=2 | k=3 | k=4 | k=5 | 度最小 / 最大 / 平均 |
|---|---:|---:|---:|---:|---|
| Cora | 516 | 426 | 295 | 176 | 1 / 125 / 3.286 |
| High | 5498 | 2091 | 222 | 7 | 2 / 148 / 55.633 |
| Primary | 7748 | 4600 | 347 | 9 | 28 / 261 / 126.979 |

弱独立集要求每条超边不能全部选中。如果一条高阶边包含某条文件中已有的二节点边，该二节点约束已经保证这条高阶边不会被全部选中。因此这条高阶可行性约束是冗余的；这是对文件集合的直接检查，不依赖神经网络结果。

| 文件 | k>2 超边数 | 含已有二节点子边的高阶边 | 所有节点对均已作为二节点边出现的高阶边 | 强 MIS 团展开的独特 pair 数 | 相比原二节点边新增 pair 数 |
|---|---:|---:|---:|---:|---:|
| Cora | 897 | 300 | 2 | 4144 | 3628 |
| High | 2320 | 2320 | 1995 | 5818 | 320 |
| Primary | 4956 | 4951 | 4416 | 8317 | 569 |

High 的全部高阶弱可行性约束已被二节点约束蕴含，Primary 仅有 5 条高阶边不含这样的二节点子边。High／Primary 仍适合真实数据求解比较，但不能单靠它们证明高阶可行性建模必不可少。Cora 和纯 k=3、5 的合成图能提供不同结构的检查。

导入保留全部合法超边，只统计上述冗余，不自动删除。删除冗余约束虽然保持弱 MIS 二进制可行集合，却会改变连续 PUBO 损失和消息传播结构，属于需要单独标注的实验。强 MIS 使用全部超边的节点对约束，不能只使用原文件中已有的二节点边。

## 使用与验证

在仓库根目录运行；输出文件已存在时拒绝覆盖，避免悄悄改变已跑过的实例。

```powershell
python -m mis.datasets --dataset cora --output instances/real/cora-upstream.json
python -m mis.datasets --dataset high --output instances/real/high-upstream.json
python -m mis.datasets --dataset primary --output instances/real/primary-upstream.json
python -m unittest discover -s tests -p test_datasets.py -v
```

每个输出均包含 `n`、`edges` 和 `import_metadata`，可以直接交给现有 MIS 入口。Python 接口 `load_upstream_dataset("high")` 返回对象的 `.problem`、`.metadata` 及 `.to_dict()` 可用于实验驱动。

新增 8 个导入测试覆盖首边保留、header 计数、标签映射、去重与关联数、边内重复顶点、单节点边、显式孤立节点、原始／规范化哈希、LF／CRLF 等价、冗余统计以及三个固定上游文件的数量核验；2026-10-18 在本地项目环境全部通过。数据导入与这些测试不使用 GPU。
