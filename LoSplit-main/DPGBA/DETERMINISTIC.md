# DPGBA严格确定性运行配置

确定性运行开关不更换 GPU 算子或加入运行时函数替换。

2026-11-23 实现修订：对照 DPGBA 官方固定版本，恢复 `range=1.0` 和双类别分布判别标签；
参考样本不足时明确报错。模型结构、优化器、末轮保存方式保持原实现。
以下早期逐位复现记录对应修订前版本；修订后结果见
`../runs/dpgba_official_fix_20261123/FIX_REPORT.md`，不得用早期哈希验证新版输出。

已实测范围：本机RTX3050 Laptop / PyTorch2.0.0+cu118 / PyG2.5.3、Cora、GCN、随机挂接、outter_size4096。两次独立完整300轮训练得到逐位一致的末轮生成器和投毒快照；四种分类器及预测也逐位一致。结果见 `D:/GNNAttackReaserch/work/dpgba_deterministic_side_20261120/RESULTS.md`。

## 运行环境

`PYTHONHASHSEED`在Python启动前设置，且与`--seed`一致。严格模式使用`warn_only=False`；遇到PyTorch识别的不支持确定性的运算时直接报错，不降级或忽略。

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
$Output = 'D:/GNNAttackReaserch/work/my_deterministic_run'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
```

## 原入口生成攻击

```powershell
& $py DPGBA/DPGBA_LoSplit.py --deterministic --dataset=Cora --model=GCN --test_model=GCN --device_id=0 --seed=10 --vs_number=50 --k=50 --hidden=32 --weight_target=1 --weight_ood=1 --weight_targetclass=2 --train_lr=0.015 --lr=0.015 --target_class=0 --trojan_epochs=300 --epochs=200 --split_lr=0.01 --split_epoch=15 --gamma=1.0 "--trigger_generator_address=$Output/generator.pth" "--pre_train_param=$Output/poison.pt"
```

## 读取同一快照运行防御

```powershell
$Common = @('--deterministic','--dataset=Cora','--model=GCN','--test_model=GCN','--device_id=0','--seed=10','--hidden=32','--epochs=200','--train_lr=0.015','--target_class=0',"--trigger_generator_address=$Output/generator.pth","--pre_train_param=$Output/poison.pt")

& $py DPGBA/DPGBA_RIGBD.py @Common --defense_mode=none --skip_rigbd "--model_save_path=$Output/none.pth"
& $py DPGBA/DPGBA_RIGBD.py @Common --defense_mode=prune --prune_thr=0.8 --skip_rigbd "--model_save_path=$Output/prune.pth"
& $py DPGBA/DPGBA_RIGBD.py @Common --defense_mode=none --rigbd_k=20 --rigbd_b=0.5 "--model_save_path=$Output/rigbd.pth"
& $py DPGBA/DPGBA_RIGBD.py @Common --defense_mode=abp "--model_save_path=$Output/abp.pth"
```

本轮沿用已有的固定快照加载、独立防御开关和ABP接入。普通分类器fit200轮；RIGBD保持原检测200轮、恢复400轮，K20/b0.5；prune阈值0.8。确定性配置不会改变这些参数或将末轮生成器改成best生成器。

## 范围与复现记录

严格开关目前限定上述Cora/GCN配置。Cora候选数小于4096，作者代码不进入独立无种子`np.random.RandomState().choice`分支。其他数据集若进入该分支，单靠全局seed不能控制该随机对象；此次按要求保留原采样代码，不将开关扩展为所有DPGBA配置的保证。

`DETERMINISTIC_RUNTIME`日志列出是否启用、warn_only、设备、线程数、cuBLAS配置和版本。旧入口不传`--deterministic`时保持原有运行行为。新生成结果与旧非确定性结果使用不同输出目录。
