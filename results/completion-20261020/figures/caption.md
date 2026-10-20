# 收尾实验图表、图注与源数据

以下两套图使用联合核验通过的正式结果。主分析纳入 222 条神经训练轨迹、429 条结果记录；发生中断的首批补充已整批归档排除，未混入绘图数据。

| 图表 | 预览 | 论文用矢量文件 | 中英文图注 |
|---|---|---|---|
| 退火、惩罚系数与时域消融 | [PNG，400 dpi](ablation-controls.png) | [PDF](ablation-controls.pdf)／[SVG](ablation-controls.svg) | [完整图注](ablation-controls.caption.md) |
| 十二张新图的规模重复验证 | [PNG，400 dpi](independent-graphs.png) | [PDF](independent-graphs.pdf)／[SVG](independent-graphs.svg) | [完整图注](independent-graphs.caption.md) |

消融图的范围表示三个配对种子差的最小值至最大值，单位为解大小除以节点数后差值的百分点；新图曲线先对每张图平均模型种子，再展示同条件三张图的均值及图间范围。两者均不是置信区间。零点、负结果及较密结构下低于 SCIP 的结果完整保留。15 秒时域主比较采用独立匹配采样补充，原未匹配结果仅作诊断。

源数据入口：

- [完整正式汇总及绘图源数据 JSON](source-data.json)。
- [主批次 303 条神经快照](source-main-neural.csv)与[正式补充 30 条快照](source-schedule-match.csv)。快照数不等于训练次数。
- [十二张新图的各方法结果](source-replication-graphs.csv)与[48 条图／方法均值](source-graph-means.csv)。
- [138 条配对种子差](source-paired-differences.csv)，包含罚项基准的零差。

两套最终 PNG 已分别由绘图代理与主代理检查。尺寸、字体、对齐、碰撞、源数据绑定和已解释的自动预检警告见 [质量核验说明](QA.md)及[机器可读 SHA 记录](QA-record.json)。完整实验解释见[收尾报告](../../../docs/completion-20261020.md)，历史批次见[实验总索引](../../../docs/experiment-index-20261020.md)。
