# 实验参数与选点数量核对

核对日期：2026-11-27。原版来源：`D:/GNNAttackReaserch/LoSplit-main.zip`。

## 配置入口

实验参数统一维护于 `run_experiments.py`：

- `SETTINGS`：运行阶段、算法、方法、数据集、骨架、运行环境与方法参数。
- `ALGORITHM_DEFAULTS`：四个算法原版 Python 入口的生成与训练参数。
- `DATASET_OVERRIDES`：原版 `defense.sh` 对四个数据集的显式覆盖。
- `ATTACK_SETTINGS`：由上述两层合并得到的 16 组最终配置。
- `LOSPLIT_SETTINGS`、`PRUNE_SETTINGS`、`RIGBD_GRIDS`：对应方法的专属参数。
- `REFERENCE_MODEL_SETTINGS`：现有 `range_reference`、`abp_feature_reference` 的固定骨架配置。

四个生成入口与四个评估入口从该文件读取默认值，并按数据集解析配置。生成阶段的全部算法超参数由启动器显式传入。当前核验的 16 组算法配置、LoSplit 三个参数和 prune 阈值均与原版对应入口一致。

`SETTINGS['seed'] = None` 表示采用原版种子：GTA、UGBA、SPEAR 为 10，DPGBA 为 12。显式设置 `SETTINGS['seed']` 或传入 `--seed` 会统一覆盖。`epochs=None` 采用原版轮数；显式 `--epochs` 统一覆盖分类器训练轮数。逐算法输入和输出路径使用实际种子；混合种子默认汇总报告放在 `seedauthor/results.md`。

当前保留的工程设置包括确定性运行、配对快照流程，以及既有 DPGBA 分布判别器实现修正。DPGBA 的 `range` 参数恢复为 LoSplit 原版的 `0.01`；该值控制重构分数参考集合的筛选宽度。现有实现继续执行参考样本数量检查。此次仅验证配置与选点数量，生成器训练结果待后续实验获得。

## vs_number 与实际保留数量

统计使用本地原始数据标签、固定数据划分、NumPy 随机抽样和已有候选缓存。执行过程包含张量读取与索引过滤，模型训练、生成器训练和推理均为零次。

计数流程为：按 `vs_number` 截取候选，再剔除原始标签等于 `target_class` 的节点。下表每格为“候选预算 → 保留的唯一节点数”。

| 算法 | Cora，目标 40 | Citeseer，目标 40 | Pubmed，目标 160 | Physics，目标 160 |
|---|---:|---:|---:|---:|
| GTA | 48 → 40 | 43 → 40 | 201 → 160 | 200 → 160 |
| UGBA | 44 → 40 | 42 → 40 | 162 → 160* | 394 → 160 |
| DPGBA | 50 → 40 | 45 → 40 | 194 → 160 | 200 → 160 |
| SPEAR | 50 → 40 | 47 → 40 | 222 → 160 | 207 → 160 |

全部组合的唯一节点数与保留节点数一致，且候选均位于原代码允许的未标记集合。

GTA、DPGBA 使用固定划分和原版随机种子直接重现抽样；UGBA、SPEAR 使用当前项目中的候选顺序缓存。SPEAR–Pubmed 的 `target_class=2`，其他组合均为 0。以上结果与当前标签、缓存和种子绑定。

*UGBA–Pubmed：读取 `selected_nodes/Pubmed/Seperate/seed10/class_1.txt` 与 `class_2.txt`，各截取 81 项，剔除目标类后共 160 项。原选点函数会先拟合编码器，再根据预测类别是否出现决定读取哪些类别缓存；这里保留“两个非目标类别均出现”的条件。此次仅核对缓存分支的计数。

GTA、UGBA、DPGBA 的数量对应接收节点数及附加子图份数；`trigger_size=3` 表示每份新增 3 个节点。SPEAR 的数量对应被改写特征的原图节点数。

本次标签读取路径位于 `D:/GNNAttackReaserch/abp_based_on_losplit/data`；候选缓存来自当前 `work/LoSplit-main`。详细 JSON 记录数据路径、文件 SHA256、参数、候选数、保留数与节点名单：

- `research/parameter_audit_20261127/selection_counts.json`
- `research/parameter_audit_20261127/author_parameter_comparison.json`

可复查脚本为 `audit_selection_counts.py`；它只读取数据和候选缓存。

## hidden 的作用

`hidden` 对应骨架模型的隐藏表示宽度：GCN 和 GraphSage 使用 `nhid=args.hidden`；GAT 使用每个注意力头的 `nhid`，首层拼接后宽度为 `heads × hidden`，当前构造器设置 8 个头。

生成阶段的影子 GCN 也共用 `args.hidden`；DPGBA 的分布判别 MLP 同样使用该值。SPEAR 还使用影子 GCN 的嵌入作为生成器输入。因此修改该参数会同时影响多个共用它的模块。

| 算法 | Cora | Citeseer | Pubmed | Physics |
|---|---:|---:|---:|---:|
| GTA | 32 | 32 | 32 | 32 |
| UGBA | 32 | 32 | 32 | 64 |
| DPGBA | 32 | 32 | 16 | 128 |
| SPEAR | 80 | 64 | 64 | 64 |

上表用于生成阶段及常规分类器/LoSplit/prune。现有 `range_reference`、`abp_feature_reference` 继续采用 `REFERENCE_MODEL_SETTINGS` 中的固定 `hidden=32`、`train_lr=0.01`。

## 验证

- 对压缩包原版逐项核验 16 组算法配置、实际生成命令、LoSplit 参数及 prune 阈值。
- `python -m unittest test_run_experiments -q`：17 项检查通过，覆盖逐数据集参数、直接入口参数解析、路径、配对任务与显式覆盖。
- 使用 `--dry-run` 检查命令；生成入口和模型训练均保持零次执行。
