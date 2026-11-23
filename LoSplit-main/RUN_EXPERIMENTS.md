# 实验入口说明

修改 `run_experiments.py` 的 `SETTINGS` 后可直接运行。默认方法为
`none abp abp_sparse prune rigbd`，ABP Sparse 的 multiplier=0.95、strength=0.25。
数据集支持 Cora、Citeseer、Pubmed、Physics；分类器支持 GCN、GAT、GraphSage。
命令行名称不区分大小写。每次调用选择一个数据集和一个分类器。

```bash
python run_experiments.py --dataset citeseer --model gat
python run_experiments.py --dataset pubmed --model graphsage
python run_experiments.py --dataset physics --model gcn
python run_experiments.py --stage defense --dataset cora --model gcn --attack-root runs/my_cora_attacks3 --defense-root runs/comparison_new
python run_experiments.py --dataset physics --model gat --dry-run
```

默认目录按数据集、分类器、seed 分开。已有输出会拒绝覆盖；重跑时指定新的根目录。
生成器仍采用原有 GCN 代理，`--model` 选择最终评估分类器，并同步传给样本生成入口的
`--test_model`。没有为新增组合声称经过调参；沿用现有配置。三个样本生成流程保持不变。
严格确定性配置支持扩展入口，但新增组合未做数值复现核验；若服务器算子不支持严格确定性，
可使用 `--no-deterministic`，并保留环境信息和日志。

## 结果与计时

输出 `results.md` 和 `results.json`。每个方法的模型旁保存 `.pth.meta.json`。
JSON 保存原始比例，Markdown 将 CA、ASR、Precision、Recall 转成百分数。

- CA：干净测试集合准确率；ASR：真实类别不等于目标类别的测试集合上的目标类比例。
- Precision/Recall：在相同的训练锚点集合上，与固定快照的标记集合比较。
  ABP Sparse 和 RIGBD 输出节点检测结果；none、ABP、prune 没有节点判别接口，记为 null / —。
  不将被删除边的端点自行转换成节点判别结果。
- `Seconds`（方法耗时）：预处理、模型构造、训练、评估全部完成，不含快照读取和输出文件写入。
- `DetectionSeconds`（检测耗时）：从方法预处理开始，到节点判别或图过滤完成。
  RIGBD 包括初始分类器建模、训练和扰动检测，不含随后防御模型重训和测试。
  ABP Sparse、ABP、prune 先检测或过滤，后训练，因此该列截止其训练前的检测/过滤阶段；none 为 null。
- 所有内部时间均在边界同步所选 CUDA 设备。另记进程耗时，包含导入、数据加载和写入。
- RIGBD 最佳候选的两个内部耗时均含一次共用初始模型训练；`GridSeconds` 为整个扫描实际耗时。

## RIGBD 扫描

恢复本地原版源文件中的范围：

| 流程 | K | b |
|---|---|---|
| GTA | 10, 20, 25, 50 | 0.3, 0.5, 0.7, 0.9 |
| UGBA | 25, 50, 100 | 0.3, 0.5, 0.7, 0.9 |
| DPGBA | 2, 5, 25, 50, 100 | 0.1, 0.3, 0.5, 0.7, 0.9 |

最佳参数采用防御后分类器的最小 ASR，平局依次选择更小 K、更小 b。
该结果表示按评估集 ASR 选出的网格最优值；验证集准确率继续作为候选指标保存。
最终 Markdown 只展示最佳候选；JSON 元数据保留候选指标与计时以便核对。
共用的初始检测模型保持固定，不将上一个候选重训后的模型当成下一个候选的检测模型。
保留原代码按快照标记节点数截取排序结果的规则；这是已知数量条件，解释结果时需注明。
保存的权重对应最佳候选，不是循环的最后一组。

## 接口与适用范围

- 修正 DPGBA 原入口固定 GCN 的行为，统一处理 GCN 的元组输出和 GAT/GraphSage 的张量输出。
- 修正 DPGBA 的 GAT/GraphSage 在 RIGBD 重训路径中错误解包输出的问题。
- 修正 GAT 在 RIGBD 重训后无条件再做普通训练的问题；GAT 支持关闭显式自环。
- RIGBD 重训传入整数节点标识，保证集合差按节点值计算。
- 当前流程的原图边权为 1，GTA/UGBA/DPGBA 生成边权经阈值化后为 0 或 1。
  GAT/GraphSage 排除零权边后使用保留边的拓扑，因此当前方法没有连续边权幅值丢失问题。
  统一入口也会在拓扑分解、随机删边及评估图过滤前排除零权边，避免把未启用的边当作实际连接。
  ABP 按拓扑过滤，ABP Sparse 同时读取原始节点特征，prune 按特征相似度删边，
  RIGBD 按随机删边前后的模型预测变化排序；这些步骤均可作用于当前 GAT/GraphSage。
  ABP Sparse 的 0.95 倍数作用于检测阈值、0.25 强度作用于训练损失，二者均作为浮点参数保留。
  将来若加入 0.2、0.8 等正连续边权，再补充相应加权聚合。GraphSage 自身的根节点分支保留，
  与 GCN/GAT 显式自环不是同一结构。
- Pubmed/Physics 的 prune 采用原有分块相似度计算，避免一次展开全部边特征。
- ABP Sparse 参数沿用指定值；其跨数据集与跨模型效果需要服务器实验确认。
  Physics 的特征矩阵和 GAT 中间表示较大，需要结合服务器显存选择运行组合。

本次仅修改代码，执行启动器单元检查和语法检查；没有训练模型、生成新样本或运行完整实验。
