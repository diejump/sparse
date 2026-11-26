# SPEAR / LoSplit 接入与 Cora 复现

更新日期：2026-11-26（Asia/Hong_Kong）。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-11-26
- Verification Status: VERIFIED
- Version Label: spear_losplit_integration_v1

## 接入结果

`run_experiments.py` 已支持四种生成流程 `GTA UGBA DPGBA SPEAR`，以及
`none losplit abp abp_sparse abp_reference prune rigbd rigbd_selfloop`。

- `SPEAR_LoSplit.py`：原选点顺序、非目标类筛选、特征维度选择、影子模型与生成器交替优化、最优生成器选择、原图特征写入、评估、保存版本 2 快照。
- `SPEAR_RIGBD.py`：读取固定快照，恢复生成器与影子模型，交给公共防御入口。选点和训练图直接使用快照。
- `spear_snapshot.py`：保存影子模型、生成器、特征维度及特征文件摘要；核对外部生成器与快照权重；按生成阶段的隐藏宽度恢复影子模型。
- `experiment_defenses.py`：恢复 LoSplit 的 RCE 早期训练、逐轮目标类推断与二分 GMM、最大损失间隔选轮、全新分类器 `finetune2` 恢复训练。保存选中节点、保留节点、逐轮间隔、推断类别、计时和指标。
- GTA、UGBA、DPGBA 的防御入口同步接受 `--defense_mode=losplit`。DPGBA 既有 split 参数与公共参数注册已合并。
- LoSplit 保留原训练节点顺序与输入边权，恢复阶段使用标签副本。GCN、GAT、GraphSage 沿用各自发布实现的恢复目标；设备转移和 DPGBA 的张量返回值已对齐。

`models/backdoor_SPEAR.py` 与本地上游副本的 AST 完全一致。
`select_sample.py` 仅补齐首次生成缓存时的数据集子目录。

## 作者参数

参数来源为本项目 `defense.sh` 的 SPEAR 命令和四种方法的 LoSplit 参数。
SPEAR 的生成轮数为 200，外层样本数为 512，分类器学习率为 0.01。

| 数据集 | hidden | vs_number | alpha_int | shadow_lr / trojan_lr | homo_loss_weight | target_loss_weight | target_class | split_lr | split_epoch | gamma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 80 | 50 | 30 | 0.0002 | 0.10 | 1 | 0 | 0.008 | 10 | 0.05 |
| Citeseer | 64 | 47 | 10 | 0.0002 | 0.05 | 1 | 0 | 0.08 | 20 | 0.8 |
| Pubmed | 64 | 222 | 10 | 0.005 | 0.15 | 2 | 2 | 0.03 | 15 | 0.8 |
| Physics | 64 | 207 | 15 | 0.0005 | 0.15 | 2 | 0 | 0.001 | 20 | 0.7 |

Cora 的 `vs_number=50` 经原有目标类别筛选后得到 40 个节点。
数据划分沿用 `get_split` 中固定的 RandomState(10)；运行 seed 控制模型训练随机性。
SPEAR 特征重要性文件和节点排序缓存继续沿用项目现有文件。

## Cora / GCN 实测

环境：Python 3.10、PyTorch 2.0.0+cu118、PyG 2.5.3、RTX 3050 Laptop 4 GB。
主实验开启严格确定性，生成、分类器和恢复训练均按 200 轮执行。
CA 使用 270 个干净测试节点；ASR 使用 225 个真实类别异于目标类别的测试节点。

| seed | SPEAR ASR (%) | SPEAR CA (%) | LoSplit ASR (%) | LoSplit CA (%) | Precision (%) | Recall (%) | FPR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | 96.4444 | 83.3333 | 0.0000 | 84.8148 | 100.0000 | 100.0000 | 0.0000 |
| 12 | 97.3333 | 83.7037 | 0.0000 | 84.4444 | 100.0000 | 100.0000 | 0.0000 |
| 10 独立重跑 | 96.4444 | 83.3333 | 0.0000 | 84.8148 | 100.0000 | 100.0000 | 0.0000 |

两种种子均达到 SPEAR ASR > 90%、LoSplit ASR < 10% 的预期。
seed=10 的独立重跑中，生成器、影子模型、投毒图、索引、普通分类器和 LoSplit 分类器权重逐张量一致。
每轮普通分类器的权重均与同轮生成入口中的评估分类器一致；磁盘重载后的 CA / ASR 与日志一致。
快照文件摘要随保存路径等元数据变化，数值一致性核验逐项比较张量。

本地论文文本 `D:/GNNAttackReaserch/project/research/sources/LoSplit.txt` 表 1 给出
Cora / SPEAR / GCN 的 ASR=100.00%、CA=81.85%，对应 LoSplit 的 ASR=0.00%、CA=84.44%。
本轮采用新训练生成器与独立分类器种子，表中同时保留实测结果和论文参照。

### SPEAR 对全部防御入口的验证

同一份 seed=12 快照，全部方法完成训练、写出模型和元数据。

| 方法 | CA (%) | ASR (%) |
|---|---:|---:|
| none | 83.7037 | 97.3333 |
| losplit | 84.4444 | 0.0000 |
| abp | 81.4815 | 98.2222 |
| abp_sparse | 81.4815 | 98.2222 |
| abp_reference | 84.0741 | 95.5556 |
| prune | 78.5185 | 100.0000 |
| rigbd | 84.0741 | 72.4444 |
| rigbd_selfloop | 84.8148 | 97.7778 |

RIGBD 使用 SPEAR 原脚本的 K=[50,100]、b=[0.5,0.7,0.9] 网格，按现有启动器的验证准确率规则选参。
其余选项沿用启动器既有参数。以上对照展示同一快照上的实际效果。

### 其余三种流程的 LoSplit 验证

输入为已有的 `runs/cora_gcn_strict_20261121` 固定快照，seed=10。
这些输入保持原版本与张量，单独用于检查 LoSplit 接入。

| 流程 | none ASR (%) | LoSplit ASR (%) | none CA (%) | LoSplit CA (%) |
|---|---:|---:|---:|---:|
| GTA | 96.8889 | 4.8889 | 80.0000 | 82.9630 |
| UGBA | 96.8889 | 5.7778 | 82.5926 | 83.3333 |
| DPGBA | 100.0000 | 2.2222 | 74.8148 | 80.0000 |

## 原因与还原口径

1. 原 SPEAR 两个入口调用 `fit(..., test=True)`，只加载生成器权重；其影子模型仍是新初始化状态。`inject_trigger` 读取影子模型嵌入，因此成对保存与恢复这两套权重是固定快照复算的必要环节。
2. 统一入口原先只有三种流程，并统一传入 hidden=32。新接入为 SPEAR 使用作者逐数据集配置，Cora 为 hidden=80、alpha_int=30。
3. 原 SPEAR 防御入口会重新选点并构造训练图；现在从快照直接读取固定索引与图张量。
4. 其余生成入口的 LoSplit 段落处于停用状态。恢复流程集中到 `--methods losplit`，复用作者 `LoSplit.fit` 与分类器 `finetune2`。
5. 旧 GAT / GraphSage 恢复代码中的随机标签留在 CPU；现已转移到标签设备。DPGBA 的两个对应模型返回预测张量，旧恢复代码按二元组解包；现已对齐其实际返回值。相关 CPU / CUDA 测试均通过。

**实现口径：沿用作者发布代码。** 作者 `finetune2` 在循环前执行一次随机重标记，
`reshuffle_labels` 与局部 `labels` 共用张量，遗忘项使用 `relu(-NLL)`，在当前 log-softmax 输出下为零。
论文文字描述为逐轮随机重标记与梯度上升；这里将两者差异明确记录。
本轮保留作者恢复目标和时序，调用方传入标签副本以维持快照原始标签。
与上游逐函数比较时，除设备转移和张量返回接口适配外，恢复代码一致。

## 运行命令

在配置好依赖的 `LoSplit-main` 目录执行，使用新输出目录：

```bash
python run_experiments.py --stage all --attacks SPEAR --methods none losplit --dataset Cora --model GCN --seed 10 --attack-root runs/my_spear_cora/inputs --defense-root runs/my_spear_cora/outputs
```

对已有成对快照单独运行 LoSplit：

```bash
python run_experiments.py --stage defense --attacks SPEAR --methods losplit --dataset Cora --model GCN --seed 10 --attack-root runs/spear_losplit_20261126/seed10/inputs --defense-root runs/my_spear_losplit_replay
```

四种流程一起预览：

```bash
python run_experiments.py --attacks GTA UGBA DPGBA SPEAR --methods none losplit --dry-run
python -m unittest test_run_experiments test_spear_integration -v
```

本机使用的解释器为 `D:/GNNAttackReaserch/gnn_defense_study_20261029/.venv/Scripts/python.exe`。
完整运行参数以各报告的 JSON 参数数组为准。
论文比较的完整数值验证范围为 Cora / GCN；GAT、GraphSage 完成小图恢复训练的 CPU / CUDA 接口测试。

## 结果文件

- [seed=10 原始报告](runs/spear_losplit_20261126/seed10/outputs/results.md)
- [seed=12 全部防御报告](runs/spear_losplit_20261126/seed12/outputs/results.md)
- [seed=10 独立重跑](runs/spear_losplit_20261126/seed10_repeat/outputs/results.md)
- [其余三种流程报告](runs/spear_losplit_20261126/other_methods/results.md)
- [重载、张量与上游代码核验](runs/spear_losplit_20261126/verification.json)
- [核验脚本](runs/spear_losplit_20261126/verify_results.py)

已完成 18 项回归测试，其中恢复接口测试覆盖 4 个 GAT / GraphSage 实现各自的 CPU 与 CUDA 路径。
SPEAR 单元测试覆盖注入结果往返一致性、图结构保持、影子权重缺失、特征维度变更和生成器配对。
LoSplit 单元测试覆盖原图/节点顺序保持、标签副本隔离、最大损失间隔选轮、DPGBA 参数注册兼容。
