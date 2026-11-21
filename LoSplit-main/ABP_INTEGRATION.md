# 固定快照防御与 ABP 接入

2026-11-20。原结构版比较：`D:/GNNAttackReaserch/work/cora_fixed_20261120`；优化版与DPGBA核查：`D:/GNNAttackReaserch/work/abp_upgrade_20261120`。完整命令见 [USAGE.md](USAGE.md)。

## 入口

继续直接运行作者的 `GTA_LoSplit.py`、`UGBA_LoSplit.py`、`DPGBA/DPGBA_LoSplit.py` 生成攻击。当前仓库的这些入口在攻击评估结束后保存版本2快照，其中包含：投毒图特征、边、边权、观测标签、固定训练/验证/测试索引、接回测试边、原始评估标签及生成器参数。作者 LoSplit 恢复阶段在本轮开始前已经停用，保留当前设置。

防御继续使用三个原 `*_RIGBD.py` 入口，读取上述快照。生成器仅用于给攻击测试节点添加触发器；防御训练图直接来自快照，不再选点或生成。

以GTA为例，工作目录为本仓库根目录：

```powershell
$py = 'D:\GNNAttackReaserch\gnn_defense_study_20261029\.venv\Scripts\python.exe'
$env:MPLBACKEND = 'Agg'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

# 无防御对照：固定快照上的普通作者GCN
& $py GTA_RIGBD.py --dataset=Cora --seed=10 --hidden=32 --epochs=200 --train_lr=0.01 --defense_mode=none --skip_rigbd --pre_train_param=D:/GNNAttackReaserch/work/cora_fixed_20261120/GTA/poison.pt --trigger_generator_address=D:/GNNAttackReaserch/work/cora_fixed_20261120/GTA/generator.pth

# 原prune单独运行
& $py GTA_RIGBD.py --dataset=Cora --seed=10 --hidden=32 --epochs=200 --train_lr=0.01 --defense_mode=prune --prune_thr=0.8 --skip_rigbd --pre_train_param=D:/GNNAttackReaserch/work/cora_fixed_20261120/GTA/poison.pt --trigger_generator_address=D:/GNNAttackReaserch/work/cora_fixed_20261120/GTA/generator.pth

# 固定RIGBD：K=20，初始独立删边概率b=0.5；保留原孤立节点恢复
& $py GTA_RIGBD.py --dataset=Cora --seed=10 --hidden=32 --epochs=200 --train_lr=0.01 --defense_mode=none --rigbd_k=20 --rigbd_b=0.5 --pre_train_param=D:/GNNAttackReaserch/work/cora_fixed_20261120/GTA/poison.pt --trigger_generator_address=D:/GNNAttackReaserch/work/cora_fixed_20261120/GTA/generator.pth

# ABP独立防御
& $py GTA_RIGBD.py --dataset=Cora --seed=10 --hidden=32 --epochs=200 --train_lr=0.01 --defense_mode=abp --pre_train_param=D:/GNNAttackReaserch/work/cora_fixed_20261120/GTA/poison.pt --trigger_generator_address=D:/GNNAttackReaserch/work/cora_fixed_20261120/GTA/generator.pth
```

UGBA使用 `UGBA_RIGBD.py`；DPGBA使用 `DPGBA/DPGBA_RIGBD.py`，并显式设置 `--device_id=0 --hidden=32 --target_class=0`。各命令可通过 `--model_save_path=<路径>` 保存最终分类器。所有方法均检查外部生成器参数与快照内参数逐张量一致，并输出快照SHA256。

`none` 沿用作者入口的RIGBD模式语义；添加 `--skip_rigbd` 后运行普通GCN对照。`prune --skip_rigbd` 独立剪边训练及评估；未添加该参数时仍保留作者原来的“剪边后继续RIGBD”行为。`abp` 自动启用独立图防御模式。

## ABP规则与三次投影

对方向 `u→v`，移去接收节点v后，若u所在连通部分仍含观测训练节点，就保留该方向；否则删除。结构锚为作者实际监督索引 `cat(idx_train, idx_attach)`，规则只读取这些节点的训练身份。新增模块不读取真实目标类、原始测试标签或附着真值来选择传递方向。

`models/ABP.py`、`DPGBA/models/ABP.py` 为同一参考结构实现的本地副本。骨架和训练继续调用原 `model_construct` / GCN.fit。

1. 训练：在原投毒训练图上投影，使用全部原观测训练标签训练作者GCN。
2. CA：先将原投毒训练图与保存的测试边合并，再对这张完整图投影并评估。
3. ASR：在未投影的完整评估图上调用原生成器注入测试触发器，再对整张触发图投影并评估。

保留原始训练边用于第2、3步，避免在已投影图上重复分解导致结构信息丢失。ABP不对称保留信息方向，投影后直接交给原GCN归一化，保留原模型参数结构。

## 本轮统一设置

- Cora固定划分seed10；生成与分类模型seed10。
- 攻击预算沿用作者Cora配置：GTA48、UGBA44、DPGBA50，再由原代码排除原目标类候选。
- 生成GTA/UGBA400轮、DPGBA300轮。DPGBA攻击阶段lr/train_lr=0.015，其他攻击参数沿用defense.sh。
- 四种比较方法最终GCN：隐藏32、普通fit200轮、dropout0.5、lr0.01、weight_decay0.0005。
- RIGBD检测fit200轮，恢复保持作者400轮；K=20、b=0.5。选择规则、KL评分、恢复目标保持当前作者实现。
- prune阈值保留各入口默认：GTA0.8、UGBA0.2、DPGBA0.8。
- `abp`为原结构版；新增`abp_dual`和`abp_assisted`分别接入双模型标签复核和教师扩展，参数、候选及路由规则见USAGE.md。三者作为独立防御选项运行。

本轮不修改已有攻击函数、作者GCN类和损失。入口改动包括参数化K/b、独立运行开关、检查点记录及ABP图处理分支。实验开始前的完整源码保存在实验目录 `source_before/`。


## 优化版接入

`models/ABP_Learning.py`与`DPGBA/models/ABP_Learning.py`提供相同的fit/forward/test接口，内部普通模型、复核模型和教师均调用原model_construct/fit。学习模块只使用观测训练/验证标签和公开类别数；结构候选与推理路由使用正权边，避免把零权边误计为实际连通。原ABP图投影函数本身保持原内容，追加的branch_profiles来自原参考函数。

优化版不会自动保证每项指标优于原ABP；各攻击的实测覆盖、CA、ASR与目标类召回率见新结果报告。
