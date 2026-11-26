# 实验入口说明

2026-11-26 新增 `abp_feature_reference`，用于当前 Cora/GCN 的特征一致性检测扩展。
默认新增自身特征检查并启用共有值修复，训练一个普通模型和一个参考模型。
四种方式统一使用 hidden=32、train_lr=0.01。该设置与作者各入口的模型宽度设置分别记录。
使用 `--feature-repair none` 可运行保留原特征的对照，`--feature-members 3` 可比较三成员配置。
完整定义与证明见 [方法与证明](FEATURE_REFERENCE_THEORY.md)，实测见 [新实验结果](FEATURE_REFERENCE_RESULTS.md)。

```text
python run_experiments.py --stage defense --attacks GTA UGBA DPGBA SPEAR --methods abp_feature_reference --dataset Cora --model GCN --seed 31 --attack-root runs/feature_reference_20261126/confirmation_inputs/seed31 --defense-root runs/my_feature_reference
```

修改 `run_experiments.py` 的 `SETTINGS` 后可直接运行。默认方法为
`none losplit abp abp_sparse abp_reference prune rigbd rigbd_selfloop`，ABP Sparse 的 multiplier=0.95、strength=0.25。
生成入口支持 GTA、UGBA、DPGBA、SPEAR。SPEAR 与 LoSplit 的复现说明见 [SPEAR_LOSPLIT.md](SPEAR_LOSPLIT.md)。
数据集支持 Cora、Citeseer、Pubmed、Physics；分类器支持 GCN、GAT、GraphSage。
命令行名称不区分大小写。每次调用选择一个数据集和一个分类器。

参考判别版已作为 `abp_reference` 加入可选方法。设置
`SETTINGS["methods"] = ["abp_reference"]` 后即可使用原启动器。
四个方法参数为 `abp_reference_multiplier=2.0`、`abp_reference_ratio=0.2`、
`abp_reference_risk=0.005`、`abp_reference_members=3`。
固定人数复现可设置 `abp_reference_count=120`，覆盖比例设置。
默认共训练六个模型；参考随机种子由 `abp_reference_seed` 控制。
完整说明见 [研究方案与证明](REFERENCE_METHOD.md)、[数据及文献核对](REFERENCE_AUDIT.md)
和 [本轮结果](REFERENCE_RESULTS.md)。

```text
python run_experiments.py --stage defense --methods abp_reference --model GCN --attack-root runs/abp_sparse_20261123/seed16/inputs --defense-root runs/my_reference_gcn
python run_experiments.py --stage defense --methods abp_reference --model GAT --attack-root runs/abp_sparse_20261123/seed16/inputs --defense-root runs/my_reference_gat
python run_experiments.py --stage defense --methods abp_reference --model GraphSage --attack-root runs/abp_sparse_20261123/seed16/inputs --defense-root runs/my_reference_sage
```

参数消融入口为 `run_reference_ablation.py`。省略 `--execute` 生成实验清单，
添加该参数后执行全部已列出的配置，并保存逐次状态、日志及五项指标。
参考版可添加 `--reference-device cpu` 显式选择 CPU。本机 GAT 在严格确定性模式下
自动使用 CPU 并记录原因；采用常规 GPU 运算可添加 `--no-deterministic`。

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
JSON 保存原始比例，Markdown 将 CA、ASR、Precision、Recall、FPR 转成百分数。

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
| SPEAR | 50, 100 | 0.5, 0.7, 0.9 |

2026-11-24 修订后，最佳参数采用防御后分类器的最高验证集准确率，平局依次选择更小 K、更小 b。
只有显式指定 `--rigbd-grid-selection min_asr` 才按测试 ASR 选择；该选项仅用于回溯诊断。
最终 Markdown 只展示最佳候选；JSON 元数据保留候选指标与计时以便核对。
共用的初始检测模型保持固定，不将上一个候选重训后的模型当成下一个候选的检测模型。
默认根据排序首节点的训练标签推断目标类别，以首次连续两个异类标签确定严格分界，不读取真实挂接数量。
`--rigbd-selection losplit` 保留 GTA 已知数量 / UGBA、DPGBA 已知目标类别的对照规则，但修正负下标与分界偏移。
`--rigbd-selection fixed_count` 是显式使用真实挂接数量的诊断对照，不能当作未知数量检测结果。
保存的权重对应最佳候选，不是循环的最后一组。

## 稀疏结构诊断与自环变体

原两层无自环检测器对部分叶状结构没有有效的两跳特征传播路径。修复随机数或增加采样次数不能消除该限制。
`rigbd` 保留无自环结构；新增的 `rigbd_selfloop` 显式开启检测器自环，是单独标记的变体，不等同于原论文结构。
两种模式都按 PyG 消息接收端 `edge_index[1]` 恢复失去全部邻居的节点，并保留原边权。
重训仍用 LoSplit 的两个均值损失项，不声称实现了原论文 Eq.6 的对数置信度目标。
检测器、采样器、恢复分类器使用独立可复现的随机流；元数据记录检测模型状态哈希、选中名单、目标类别 CA。

固定 Cora 图上比较两个模式，不重新生成数据：

```text
python run_experiments.py --stage defense --dataset Cora --model GCN --methods rigbd rigbd_selfloop --rigbd-fixed --attack-root runs/comparison_inputs_GCN2/Cora/GCN/seed10 --defense-root runs/rigbd_fixed_comparison_new
```

自环变体当前限定 GCN。完整诊断、作者图复跑、消融和复现记录见 `research/rigbd_diagnosis_20261124/REPORT.md`。

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

2026-11-24 已执行 Cora/GCN 的固定图实验、作者资产复跑及回归核验；其他数据集/模型未在本次训练验证。
