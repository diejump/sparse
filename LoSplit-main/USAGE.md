# 项目使用命令

推荐使用跨平台 Python 入口：修改 `run_experiments.py` 顶部 `SETTINGS` 后直接运行，
或执行 `python run_experiments.py`。详细说明见 [PYTHON_RUN.md](PYTHON_RUN.md)。
预览命令：`python run_experiments.py --dry-run`。

工作目录：`D:/GNNAttackReaserch/work/LoSplit-main`。全部训练仍从作者入口启动。

## 1. 环境与GPU

```powershell
Set-Location 'D:\GNNAttackReaserch\work\LoSplit-main'
$py = 'D:\GNNAttackReaserch\gnn_defense_study_20261029\.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:MPLBACKEND = 'Agg'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

`--device_id=0` 使用当前可见GPU。`--no-cuda` 可显式切换CPU。入口打印的Namespace中 `cuda=True` 表示本次使用GPU。

GPU执行：干净GCN、攻击代理模型/生成器、投毒分类器、prune的张量相似度计算、RIGBD检测/恢复，以及ABP优化版的普通模型、复核模型、教师和恢复训练。NetworkX的ABP结构分解和分支统计在CPU上，投影后的GCN和推理在所选设备上。prune的分块评估会将相似度结果汇总到CPU。

11月20日 `cora_fixed_20261120` 的clean、3个攻击、12个防御均使用GPU；11月19日另一次实验曾用CPU运行Clean/GTA。本机显卡为RTX3050 Laptop 4GB；本文命令已在Cora/GCN上核验，大图完整生成的显存需求按原模型计算。

## 2. 直接复用已有固定快照

以下选择一个攻击，随后所有方法使用同一份快照和生成器。把 `$Attack` 改为 `GTA`、`UGBA` 或 `DPGBA` 即可。

```powershell
$Attack = 'DPGBA'
$Entry = @{ GTA='GTA_RIGBD.py'; UGBA='UGBA_RIGBD.py'; DPGBA='DPGBA/DPGBA_RIGBD.py' }[$Attack]
$InputRoot = 'D:/GNNAttackReaserch/work/cora_fixed_20261120'
$OutputRoot = 'D:/GNNAttackReaserch/work/my_defense_run'
New-Item -ItemType Directory -Force -Path "$OutputRoot/$Attack" | Out-Null
$Inputs = @("--pre_train_param=$InputRoot/$Attack/poison.pt", "--trigger_generator_address=$InputRoot/$Attack/generator.pth")
$ClassifierLR = @{ GTA=0.01; UGBA=0.01; DPGBA=0.015 }[$Attack]
$Common = @('--dataset=Cora','--model=GCN','--test_model=GCN','--device_id=0','--seed=10','--hidden=32','--epochs=200',"--train_lr=$ClassifierLR",'--target_class=0')
$PruneThreshold = @{ GTA=0.8; UGBA=0.2; DPGBA=0.8 }[$Attack]

# 无防御：普通GCN
& $py $Entry @Common @Inputs --defense_mode=none --skip_rigbd "--model_save_path=$OutputRoot/$Attack/none.pth"

# 作者prune独立防御
& $py $Entry @Common @Inputs --defense_mode=prune --skip_rigbd "--prune_thr=$PruneThreshold" "--model_save_path=$OutputRoot/$Attack/prune.pth"

# 修正后的无自环RIGBD：K=20，b=0.5
& $py $Entry @Common @Inputs --defense_mode=none --rigbd_k=20 --rigbd_b=0.5 "--model_save_path=$OutputRoot/$Attack/rigbd.pth"

# 显式自环探测变体（GCN）；与原论文无自环结构分别报告
& $py $Entry @Common @Inputs --defense_mode=none --rigbd_probe_self_loops --rigbd_k=20 --rigbd_b=0.5 "--model_save_path=$OutputRoot/$Attack/rigbd_selfloop.pth"

# 原ABP：结构投影 + 作者GCN.fit
& $py $Entry @Common @Inputs --defense_mode=abp "--model_save_path=$OutputRoot/$Attack/abp.pth"

# 优化候选一：普通模型 + ABP复核模型 + 负标签恢复
& $py $Entry @Common @Inputs --defense_mode=abp_dual "--model_save_path=$OutputRoot/$Attack/abp_dual.pth"

# 优化候选二：另加GCN教师辅助监督
& $py $Entry @Common @Inputs --defense_mode=abp_assisted "--model_save_path=$OutputRoot/$Attack/abp_assisted.pth"
```

`none`保持原RIGBD入口的历史含义：未带`--skip_rigbd`时执行RIGBD；带该开关时运行普通分类器。
当前统一入口的 `prune` 直接运行图过滤与分类器；ABP 选项也独立运行。
RIGBD 默认采用训练标签分界和接收端恢复边。自环变体需显式启用，详细诊断见
`research/rigbd_diagnosis_20261124/REPORT.md`。

三个防御入口读取相同版本2快照，检查外部生成器与快照内权重逐张量一致，并打印快照SHA256。`pre_train_param`在此处是包含固定索引的投毒快照，不是分类器权重。分类器训练分别独立执行。

## 3. 重新训练攻击、生成新快照

使用新目录保存，不覆盖用于比较的既有快照。下面显式固定seed10和作者Cora攻击参数；三个攻击生成命令均为原作者入口。

```powershell
$NewInputs = 'D:/GNNAttackReaserch/work/new_attack_run'
New-Item -ItemType Directory -Force -Path "$NewInputs/GTA","$NewInputs/UGBA","$NewInputs/DPGBA" | Out-Null

& $py GTA_LoSplit.py --dataset=Cora --test_model=GCN --device_id=0 --seed=10 --hidden=32 --epochs=200 --train_lr=0.01 --vs_number=48 "--trigger_generator_address=$NewInputs/GTA/generator.pth" "--pre_train_param=$NewInputs/GTA/poison.pt" --split_lr=0.01 --split_epoch=10 --gamma=0.3

& $py UGBA_LoSplit.py --dataset=Cora --test_model=GCN --device_id=0 --seed=10 --hidden=32 --epochs=200 --train_lr=0.01 --vs_number=44 --homo_loss_weight=50 --homo_boost_thrd=0.5 "--trigger_generator_address=$NewInputs/UGBA/generator.pth" "--pre_train_param=$NewInputs/UGBA/poison.pt" --split_lr=0.01 --split_epoch=15 --gamma=0.2

& $py DPGBA/DPGBA_LoSplit.py --dataset=Cora --test_model=GCN --device_id=0 --seed=10 --vs_number=50 --k=50 --hidden=32 --weight_target=1 --weight_ood=1 --weight_targetclass=2 --train_lr=0.015 --lr=0.015 --target_class=0 --trojan_epochs=300 --epochs=200 "--trigger_generator_address=$NewInputs/DPGBA/generator.pth" "--pre_train_param=$NewInputs/DPGBA/poison.pt" --split_lr=0.01 --split_epoch=15 --gamma=1.0
```

生成后，将第2节 `$InputRoot` 改为 `$NewInputs`。当前攻击入口在攻击评估后保存快照；LoSplit 动态分割与恢复训练通过 `run_experiments.py --methods losplit` 独立执行，详见 `SPEAR_LOSPLIT.md`。DPGBA还会按原代码写出仓库根目录`model_weights.pth`；需要保留该文件时，先复制备份。

干净参考模型单独运行：

```powershell
& $py Clean.py --dataset=Cora --model=GCN --test_model=GCN --device_id=0 --seed=10 --hidden=32 --epochs=200 --train_lr=0.01
```

## 4. ABP优化版配置与检查点

默认设置来自原ABP学习方案：

```text
--abp_min_branch=2
--abp_recovery_epochs=400
--abp_negative_weight=3.0
--abp_teacher_hidden=32
--abp_teacher_epochs=200
--abp_confidence=0.8
```

两种优化版均调用作者原骨架和fit；恢复阶段使用方案中的置信加权负标签目标。教师扩展多训练一个GCN，预测时使用普通模型和复核模型。

多节点分支判定使用`edge_weight>0`的有效结构；零权边不增加分支节点数。普通模型F保留作者输入，复核模型R使用ABP投影。测试时按当前图的有效分支结构路由。这一规则会使只有一个实际连通触发节点的查询保留F输出，实际覆盖见新结果报告。

保存文件包括`.pth`和`.pth.meta.json`，后者记录结构候选、保留监督、推断类别、恢复设置和检查点指标。元数据中的真实目标类或测试真值未用于学习选择。当前学习分支已在三种Cora攻击/GCN上完成GPU训练、磁盘重载、CA/ASR复算和路由核对。

## 5. prune阈值

阈值由入口的常数默认值或`--prune_thr`决定，不随数据集自动学习。

| 入口 | 默认prune_thr |
|---|---:|
| GTA_LoSplit.py / GTA_RIGBD.py | 0.8 |
| UGBA_LoSplit.py | 0.5 |
| UGBA_RIGBD.py | 0.2 |
| DPGBA/DPGBA_LoSplit.py / DPGBA_RIGBD.py | 0.8 |

同一个入口中，Cora/Citeseer/Pubmed/Physics未显式覆盖时使用相同默认值。`defense.sh`没有为这些数据集指定`prune_thr`。过滤规则为保留余弦相似度严格大于阈值的边；该阈值与RIGBD的随机删边概率`rigbd_b`独立。

## 6. 历史DPGBA强攻击复核

本机旧生成器与旧快照已经导入版本2格式用于诊断；转换保留原张量和原生成器，不重新生成图。以下组合已复算CA80.74%、ASR100%。

```powershell
$OldInputs = 'D:/GNNAttackReaserch/work/abp_upgrade_20261120/dpgba_audit/old18'
& $py DPGBA/DPGBA_RIGBD.py --dataset=Cora --model=GCN --test_model=GCN --device_id=0 --seed=10 --hidden=32 --epochs=200 --train_lr=0.015 --target_class=0 --defense_mode=none --skip_rigbd "--pre_train_param=$OldInputs/poison.pt" "--trigger_generator_address=$OldInputs/generator.pth" "--classifier_rng_state=$OldInputs/rng.pt"
```

`classifier_rng_state`仅用于复核旧分类器初始化；新实验按seed设置运行。生成器、投毒图和固定索引成对使用。两次差距的8组控制与生成器交换结果见 `work/abp_upgrade_20261120/DPGBA_GAP_AUDIT.md`。
