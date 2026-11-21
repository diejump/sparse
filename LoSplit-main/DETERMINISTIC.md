# GTA、UGBA的可选严格确定性配置

新增开关：`--deterministic`。覆盖四个原入口：

- `GTA_LoSplit.py`：GTA生成与攻击评估。
- `UGBA_LoSplit.py`：UGBA生成与攻击评估。
- `GTA_RIGBD.py`：固定GTA快照上的分类与防御。
- `UGBA_RIGBD.py`：固定UGBA快照上的分类与防御。

新增配置不修改数学公式、模型、损失、优化器、采样或权重选择。GTA仍使用末轮参数；UGBA仍按原训练目标记录并恢复最佳参数。后端归约顺序和TF32精度设置可能使结果不同于旧的非确定性运行；目标是在相同环境和配置下重复一致。

开关默认关闭；不传入时，原计算路径保持原样。严格模式使用`torch.use_deterministic_algorithms(True, warn_only=False)`，不支持确定性的运算会报错。

## 环境

```powershell
Set-Location 'D:\GNNAttackReaserch\work\LoSplit-main'
$py = 'D:\GNNAttackReaserch\gnn_defense_study_20261029\.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:PYTHONHASHSEED = '10'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$env:PYTHONUTF8 = '1'
$Output = 'D:/GNNAttackReaserch/work/my_strict_gta_ugba'
New-Item -ItemType Directory -Force -Path "$Output/GTA","$Output/UGBA" | Out-Null
```

`PYTHONHASHSEED`必须在Python启动前设置，并与`--seed`一致。当前实测范围为本机Cora/GCN、作者默认选择策略；严格入口会检查Cora/GCN范围。

## 原攻击入口

```powershell
& $py GTA_LoSplit.py --deterministic --dataset=Cora --model=GCN --test_model=GCN --device_id=0 --seed=10 --hidden=32 --epochs=200 --trojan_epochs=400 --train_lr=0.01 --vs_number=48 "--trigger_generator_address=$Output/GTA/generator.pth" "--pre_train_param=$Output/GTA/poison.pt" --split_lr=0.01 --split_epoch=10 --gamma=0.3

& $py UGBA_LoSplit.py --deterministic --dataset=Cora --model=GCN --test_model=GCN --device_id=0 --seed=10 --hidden=32 --epochs=200 --trojan_epochs=400 --train_lr=0.01 --vs_number=44 --homo_loss_weight=50 --homo_boost_thrd=0.5 "--trigger_generator_address=$Output/UGBA/generator.pth" "--pre_train_param=$Output/UGBA/poison.pt" --split_lr=0.01 --split_epoch=15 --gamma=0.2
```

## 原防御入口

以GTA为例：

```powershell
$Inputs = @("--trigger_generator_address=$Output/GTA/generator.pth", "--pre_train_param=$Output/GTA/poison.pt")
$Common = @('--deterministic','--dataset=Cora','--model=GCN','--test_model=GCN','--device_id=0','--seed=10','--hidden=32','--epochs=200','--train_lr=0.01')

# 普通分类器对照
& $py GTA_RIGBD.py @Common @Inputs --defense_mode=none --skip_rigbd "--model_save_path=$Output/GTA/none.pth"
# 独立prune
& $py GTA_RIGBD.py @Common @Inputs --defense_mode=prune --prune_thr=0.8 --skip_rigbd "--model_save_path=$Output/GTA/prune.pth"
# 原RIGBD固定参数
& $py GTA_RIGBD.py @Common @Inputs --defense_mode=none --rigbd_k=20 --rigbd_b=0.5 "--model_save_path=$Output/GTA/rigbd.pth"
# 原ABP
& $py GTA_RIGBD.py @Common @Inputs --defense_mode=abp "--model_save_path=$Output/GTA/abp.pth"
```

UGBA改用`UGBA_RIGBD.py`及`$Output/UGBA/`输入路径，prune阈值沿用该入口默认0.2。学习防御仍可通过已有`abp_dual`、`abp_assisted`选项调用。

本次完成两算法各两次400轮攻击生成和普通分类器入口的重复性验证；不是重新运行一张完整防御性能矩阵。完整的既有使用说明见[USAGE.md](USAGE.md)，DPGBA配置见[DPGBA/DETERMINISTIC.md](DPGBA/DETERMINISTIC.md)。

## 实现与记录

根目录`deterministic_runtime.py`与DPGBA同名配置模块内容相同，均只设置随机状态和后端开关。计算代码通过AST及文件哈希核对保持原样。

本次验证记录目录：`D:/GNNAttackReaserch/work/gta_ugba_deterministic_side_20261120`。
