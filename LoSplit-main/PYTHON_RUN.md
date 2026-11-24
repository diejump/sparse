# 直接用 Python 运行

使用已装好项目依赖的 Python 3.10 环境，打开 `run_experiments.py`，修改顶部
`SETTINGS` 后点击 IDE 的运行按钮即可。脚本自动定位项目目录，所有子进程使用当前
Python 解释器；Windows 和 Ubuntu 使用同一个文件，不需要写 Bash 或 PowerShell 脚本。

默认 `stage="all"`：依次生成 GTA、UGBA、DPGBA，再分别运行 none、prune、RIGBD、
ABP、ABP dual、ABP assisted。参数对应此前的 Cora/GCN、seed10 严格模式命令。
这是完整训练，可能耗时较长。每个任务在控制台输出，同时保存 `.log`。

## 最终 Markdown 报告

默认在 `defense_root/results.md` 保存统一报告；只生成攻击（`stage="attack"`）时，
保存到 `attack_root/results.md`。可在 SETTINGS 中设置 `"report": "runs/experiment_01.md"`，
或传入 `--report runs/experiment_01.md` 自定义位置。相对路径以启动脚本所在目录为基准。

报告包含攻击与防御任务的 CA、ASR 百分比，RIGBD 的 Precision、Recall、FPR，
以及任务状态、耗时、输入输出链接、日志链接、执行参数和日志中的快照 SHA256。
攻击入口与 none 对照分别训练分类器，指标分别列出。缺失指标显示 `—`，不记成零。
只运行 defense 时，报告列出本次选择的防御任务，历史攻击指标仍在对应的 attack.log 中。

报告在启动检查通过后建立，随后逐任务更新；训练失败或 Ctrl+C 中断时，也会保留
已得到的结果、失败原因和未运行任务。CUDA 检查失败也会写入报告。
目标文件冲突、缺少输入等训练前参数检查错误会直接退出，不创建报告。
已有报告不会被覆盖，需改用新路径。`--dry-run` 只预览，不生成报告。

也可在项目目录执行一行命令（Ubuntu 可将 python 换成 python3）：

```text
python run_experiments.py --dry-run
python run_experiments.py
python run_experiments.py --stage attack --attacks GTA UGBA DPGBA
python run_experiments.py --stage defense --attacks DPGBA --methods abp_dual
```

只运行防御：将 SETTINGS 的 `stage` 改成 `"defense"`，`attacks` 改成例如
`["DPGBA"]`，`methods` 改成例如 `["abp_dual"]`。`attack_root` 指向包含
`DPGBA/generator.pth` 和 `DPGBA/poison.pt` 的目录，二者必须来自同一次生成。
或使用 `--attack-root`、`--defense-root` 参数指定路径。

路径默认相对于本脚本所在目录，复制整个项目到 Ubuntu 后无需修改 Windows 盘符；
自定义的绝对路径需要按目标系统修改。`gpu` 为物理 GPU 编号，子进程统一使用可见设备 0。
修改 `seed` 会同时更新训练参数和子进程启动前的 PYTHONHASHSEED。

## 分类器独立随机种子

GTA、UGBA、DPGBA 的 LoSplit/RIGBD 两类入口在启动时保留原有全局种子/确定性设置，
并在评估分类器构造前额外调用 `seed_classifier`，重置 Python、NumPy、PyTorch CPU/CUDA RNG。
默认分类器种子等于 `--seed`，所以现有启动器无需新增参数。
如直接调用两个入口，也可给两边传入相同的 `--classifier_seed=123`。
classifier_seed 可以不同于全局 seed，不要求修改进程启动时的 PYTHONHASHSEED。

要比较攻击入口与 none，需两边都使用新代码、相同 classifier_seed 和分类器训练超参数，
并加载同一份投毒图/生成器。none 需使用 `--defense_mode=none --skip_rigbd`。
新规则不会复现修改前的历史攻击评估数值，需要重新运行攻击评估。
只把防御入口的全局种子设置移动到 GCN 前，并不能使它对齐旧攻击入口。
三个攻击均已接入；重置位置与 UGBA 一致，位于攻击评估分类器和防御入口的
none/prune/ABP 分类器构造前。RIGBD 的检测与恢复模型也分别使用独立分类器种子，
删边使用独立 CPU 生成器，不依赖前一个候选的训练调用数。

RIGBD 的采样方向、标签分界和模型身份已在 2026-11-24 复核。原无自环检测器保留为 `rigbd`，
叶状结构的显式自环探测变体为 `rigbd_selfloop`（GCN）。后者默认固定 K=20、b=0.5；
两者默认都不读取真实挂接数量。使用 `--rigbd-fixed` 可让 `rigbd` 也固定参数；
默认网格按验证集准确率选参。详细命令和适用范围见 `RUN_EXPERIMENTS.md`。
DPGBA 的 `--classifier_rng_state` 历史回放参数优先于独立种子，指定该参数时按保存状态回放。
2026-11-23 已按官方实现恢复 DPGBA 的参考池默认范围（range=1.0）和双类别分布判别标签，
并将启动器中 DPGBA 的分类器学习率统一为 0.015；GTA、UGBA 仍为 0.01。
epochs、dropout、生成阶段学习率、挂接节点预算及目标类别保持既有设置。
修订后的新实验与旧版记录应分开保存；旧快照不会因代码修改而自动恢复分布约束。
源代码对比和本次实测见 `runs/dpgba_official_fix_20261123/FIX_REPORT.md`。

输出模型或日志已存在时会停止，改用新输出目录再运行。失败会立即停止并保留日志，
不会自动跳过可能不完整的攻击文件。DPGBA 作者代码还会写入项目根目录
`model_weights.pth`；若该文件已存在，先备份或移走，再运行攻击生成。

启动器不安装依赖。实际运行前会检查当前解释器中的 PyTorch/CUDA；严格确定性已审计
范围仍为 Cora/GCN，跨平台数值一致性需在目标环境另行验证。
