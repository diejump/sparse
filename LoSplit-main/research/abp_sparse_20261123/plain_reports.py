"""Generate clear Chinese reports directly from the verified result files."""
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[2]
DOC=Path(__file__).parent
OUT=ROOT/'runs/abp_sparse_20261123'
STUDY=OUT/'parameter_study'
KINDS=['GTA','UGBA','DPGBA']

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def pct(value):return '—' if value is None else f'{100*value:.2f}'
def mean(rows,key,section='test'):return statistics.mean(r[section][key] for r in rows)

def main_comparison(summary):
    lines=['| 生成方法 | 设置 | CA（%） | ASR（%） | Precision（%） | Recall（%） | FPR（%） |',
           '|---|---|---:|---:|---:|---:|---:|']
    for kind in KINDS:
        for m,l,name in [(2.,0.,'原参数：倍数2、强度0'),(.95,.25,'兼顾正常分类：倍数0.95、强度0.25'),(2.,.5,'更低ASR：倍数2、强度0.5')]:
            r=next(x for x in summary if x['kind']==kind and x['multiplier']==m and x['penalty_weight']==l)
            lines.append(f'| {kind} | {name} | {pct(r["CA"])} | {pct(r["ASR"])} | {pct(r["precision"])} | {pct(r["recall"])} | {pct(r["FPR"])} |')
    return '\n'.join(lines)

def write_reports():
    audited=read(STUDY/'metric_audit.json')
    result=read(STUDY/'parameter_summary.json')
    grid=read(STUDY/'development_test_metrics.json')
    threshold=read(STUDY/'threshold_sweep.json')
    assert audited['status']==result['status']=='VERIFIED'
    chosen=result['winner'];m=chosen['multiplier'];l=chosen['penalty_weight']
    comparison=main_comparison(result['summary'])
    report=f'''# ABP-Sparse：指标核对与参数实验

更新日期：2026-11-23。

## 1. 主要结果

原报告的Precision、Recall、FPR、ASR和CA计算正确。本轮重新读取原始图数据和27个已保存模型，逐项重算人数及比例，全部与原记录一致。

降低ASR有两个可调方向：改变检测阈值，允许多识别少量干净节点；增加训练惩罚，降低已识别节点原记录类别的预测概率。后一个方向即使保持检测精确率和召回率100%，也能降低ASR。

下表使用生成随机种子11、12、13，每份图数据分别用分类器随机种子10、20、30训练。每次先按指标定义计算，再对9次结果取平均。生成种子13的数据是本轮新生成的。

{comparison}

## 2. 指标是否算错

本报告的目标节点是实际挂接触发器的原图节点。训练阶段的检测指标在全部带训练标签的节点上计算。

| 指标 | 计算方法 |
|---|---|
| Recall，召回率 | 正确识别的目标节点数 ÷ 全部真实目标节点数 |
| Precision，精确率 | 正确识别的目标节点数 ÷ 全部识别候选节点数 |
| FPR，误报率 | 被误报为目标的干净节点数 ÷ 全部干净节点数 |
| CA，干净准确率 | 干净测试节点中预测正确的数量 ÷ 干净测试节点总数 |
| ASR | 在原本不属于指定类别的测试节点中，添加触发器后被预测为指定类别的比例 |

以原GTA数据、生成种子10为例：40个真实目标全部识别正确，541个干净训练节点均未误报。因此Precision=40/40=100%，Recall=40/40=100%，FPR=0/541=0%。

CA沿用项目划分，在270个干净测试节点上计算。另一组添加触发器的测试节点共有271个，其中225个原本不属于指定类别；主ASR以225为分母。重新核对确认，这两组测试节点均未参与训练或验证选模。

核对文件还保存了全部541个测试节点的干净准确率，以及以271为分母的另一种ASR统计，便于核对口径。正文保持项目原有的270和225两个分母。

独立核对程序先用节点集合确定每个节点是否识别正确，再用另一套计数实现重算结果；模型部分重新加载权重执行预测。完整记录见 `parameter_study/metric_audit.json`。

## 3. 有几个超参数

原ABP-Sparse新增1个方法参数：检测阈值倍数。加入本轮训练项后，共有2个方法参数。

| 参数 | 作用 | 本轮测试值 |
|---|---|---|
| 检测阈值倍数 \(m\) | 越小，识别候选越多；越大，候选越少 | 检测扫描24档，范围0.25至32；重点重训0.9、0.95、2和10 |
| 训练惩罚强度 \(\lambda\) | 控制对已识别节点原记录类别的抑制程度；0表示关闭 | 0、0.05、0.1、0.25、0.5、1、2、4 |

实际检测阈值由“倍数×参考节点最大特征分数”计算得到，所以阈值本身不再算一个独立参数。内部至少一个有效挂接节点是方法结构要求。

普通图卷积网络另有6项训练设置，本轮保持固定：2层、隐藏层32维、随机丢弃比例0.5、权重衰减0.0005、训练200轮，以及学习率。GTA和UGBA学习率为0.01，DPGBA为0.015。随机种子用于重复实验。

此前只检查了阈值对检测结果的影响。本轮补充了逐档重训和测试集ASR、CA比较：72个初筛模型、24个重复确认模型、3个较高阈值模型、108个固定参数验证模型，以及18个较大惩罚强度模型，共225次分类器训练。

## 4. 如何降低ASR

### 改变检测阈值

初始倍数2能准确识别当前数据中的目标节点。把倍数从1改到8，检测集合保持相同，因此在其他条件相同时，分类器训练也相同。

把倍数降到0.95，开发数据中每种生成方式都多识别了1个干净节点，精确率约97.6%，召回率100%。降到0.9时，多识别2至3个干净节点，精确率约93%至95%，召回率仍为100%。这些设置满足你提出的两个指标均大于90%的要求，是否值得采用由重训后的ASR与CA共同判断。

### 增加训练惩罚

初始方法移除候选节点的训练标签。本轮进一步降低候选节点原记录类别的预测概率。若该概率为 \(q\)，新增损失为 \(-\lambda\log(1-q)\)。

这样既保留普通节点的正常分类训练，又利用已识别节点提供的信息。本轮较大强度设置取得了更低的ASR，同时指定类别正常节点的分类准确率下降。完整结果因此同时列出这项类别准确率。

## 5. 参数怎样选择

先在生成种子10上筛选参数。24组主要参数组合各训练GTA、UGBA、DPGBA模型，再用两个额外的分类器种子复核4组候选配置。

选择使用验证集。为公平比较，各配置都在同一批验证节点上评分，这批节点排除了所有候选配置识别出的目标。关注的类别由候选节点中最常见的记录标签确定。选择程序检查验证集中其他类别被误分到该类的比例。

兼顾正常分类的选择条件为：Precision和Recall均大于90%；平均验证准确率比原参数下降不超过2个百分点；关注类别的验证准确率下降不超过5个百分点。在这些条件内，优先选择误分比例较低的配置。

三次分类器重复后选出倍数{m}、惩罚强度{l}。该选择在计算参数网格的测试集结果前已保存。随后固定参数，在另外三组生成数据上测试。

更强调ASR时，可比较更大的惩罚强度。结果表单独列出强度2和4在新生成数据上的表现，包括正常分类的实际代价。

## 6. 方法步骤与证明

1. 根据实际有权重的连接执行ABP：移除一个接收节点后，若邻居仍能连到其他带训练标签的节点，就保留该方向，否则删除。
2. 对被切断的邻居计算特征分数。分数表示特征分散在多少个有效维度上。
3. 邻居分数超过阈值时，把接收节点识别为候选。
4. 从训练和验证标签中移除候选节点；按选定强度增加训练惩罚。
5. 预测时始终按ABP规则处理整张图。

单点挂接且新增节点没有训练标签时，ABP会切断新增节点向原图传递信息的全部方向。当前图卷积网络固定参数后，原图输出因而保持不变。特征检测则在正常邻居和触发器邻居分数可分时精确找到目标。

完整证明、训练项导数及数值实现说明见 [数学说明](theory_review.md)。理论覆盖单节点挂接以及触发器内部完全没有边的情况。

## 7. 文件与运行方式

- [完整结果与参数实验](../../runs/abp_sparse_20261123/RESULTS.md)。
- [指标独立核对](../../runs/abp_sparse_20261123/parameter_study/metric_audit.json)。
- [逐次验证结果](../../runs/abp_sparse_20261123/parameter_study/validation_metrics.csv)。
- [检测实现](../../models/ABP_Sparse.py)与[本轮训练实现](../../models/ABP_Sparse_Training.py)。
- 实验入口：`tune_sparse.py`；汇总入口：`summarize_parameters.py`。

在项目目录下，使用已安装依赖的Python执行下例。输出目录对应的阶段名称应使用新的名称，以保留历史结果。

```powershell
& 'D:/GNNAttackReaserch/gnn_defense_study_20261029/.venv/Scripts/python.exe' research/abp_sparse_20261123/tune_sparse.py --phase my_parameter_run --generations 13 --kinds GTA UGBA DPGBA --multipliers {m} --penalties {l} --seeds 10 20 30 --evaluate
```

## 8. 结果适用范围

本轮使用当前Cora数据和固定节点划分，通过改变生成器和分类器的随机种子检查重复结果。检测依赖特征分数分离；若新增节点与正常节点的特征和连接完全一致，单靠这些观测无法区分来源。全部节点计数、参数选择记录及重复实验均已保存。
'''
    (DOC/'REPORT.md').write_text(report,encoding='utf-8')

    lines=['# ABP-Sparse：完整结果与参数实验','',
        '本表只比较ABP-Sparse自身的参数设置。各百分比由保存的原始计数计算。',
        '目标节点指实际挂接触发器的原图节点。Precision、Recall、FPR在全部带训练标签的节点中计算；CA使用270个干净测试节点，ASR使用225个原本不属于指定类别的测试节点。',
        '', '## 1. 固定参数后的比较','',
        '生成随机种子11、12、13，每份图数据采用分类器随机种子10、20、30。每行平均9次结果。',
        '',comparison,'',
        '“指定类别正确率”表示：在不添加测试触发器的完整测试集中，真实属于指定类别的90个节点被正确分类的比例。它用于说明降低ASR对这一正常类别的影响。',
        '', '| 生成方法 | 阈值倍数 | 惩罚强度 | CA（%） | ASR（%） | 指定类别正确率（%） |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        lines.append(f'| {r["kind"]} | {r["multiplier"]} | {r["penalty_weight"]} | {pct(r["CA"])} | {pct(r["ASR"])} | {pct(r["normal_target_class_accuracy"])} |')
    lines+=['','## 2. 更大惩罚强度的取舍','',
        '以下全部使用本轮新生成的种子13数据，每行平均三个分类器种子。这样可以在同一份图数据上比较不同强度。',
        '', '| 生成方法 | 阈值倍数 | 惩罚强度 | CA（%） | ASR（%） | Precision（%） | Recall（%） | 指定类别正确率（%） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['priority_summary']:
        lines.append(f'| {r["kind"]} | {r["multiplier"]} | {r["penalty_weight"]} | {pct(r["CA"])} | {pct(r["ASR"])} | {pct(r["precision"])} | {pct(r["recall"])} | {pct(r["normal_target_class_accuracy"])} |')
    lines+=['','## 3. 只改变检测阈值','',
        '生成种子10、分类器种子10；训练惩罚强度固定为0。每档均重新训练后计算测试指标。',
        '', '| 生成方法 | 阈值倍数 | CA（%） | ASR（%） | Precision（%） | Recall（%） | FPR（%） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for kind in KINDS:
        for r in sorted([r for r in grid if r['kind']==kind and r['penalty_weight']==0],key=lambda r:r['multiplier']):
            lines.append(f'| {kind} | {r["multiplier"]} | {pct(r["test"]["CA"])} | {pct(r["test"]["ASR"])} | {pct(r["detection"]["precision"])} | {pct(r["detection"]["recall"])} | {pct(r["detection"]["fpr"])} |')
    lines+=['','## 4. 只改变训练惩罚强度','',
        '生成种子10、分类器种子10；阈值倍数固定为2。各行精确率和召回率均为100%，误报率为0%。',
        '', '| 生成方法 | 惩罚强度 | CA（%） | ASR（%） | 指定类别正确率（%） |',
        '|---|---:|---:|---:|---:|']
    for kind in KINDS:
        for r in sorted([r for r in grid if r['kind']==kind and r['multiplier']==2],key=lambda r:r['penalty_weight']):
            lines.append(f'| {kind} | {r["penalty_weight"]} | {pct(r["test"]["CA"])} | {pct(r["test"]["ASR"])} | {pct(r["test"]["normal_target_class_accuracy"])} |')
    lines+=['','## 5. 两个参数共同变化','',
        '下表平均GTA、UGBA、DPGBA三种生成方式的开发结果，每种方式各训练一个分类器。精确率和召回率列出三种方式中的最低值。参数选择使用验证集，测试集结果在配置选择完成后计算。',
        '', '| 阈值倍数 | 惩罚强度 | 平均CA（%） | 平均ASR（%） | 最低Precision（%） | 最低Recall（%） |',
        '|---:|---:|---:|---:|---:|---:|']
    for multiplier in [.9,.95,2.]:
        for penalty in [0,.05,.1,.25,.5,1,2,4]:
            rr=[r for r in grid if r['multiplier']==multiplier and r['penalty_weight']==penalty]
            assert len(rr)==3
            lines.append(f'| {multiplier} | {penalty} | {pct(mean(rr,"CA"))} | {pct(mean(rr,"ASR"))} | {pct(min(r["detection"]["precision"] for r in rr))} | {pct(min(r["detection"]["recall"] for r in rr))} |')
    lines+=['','## 6. 选定参数的实际检测人数','',
        f'阈值倍数{m}、惩罚强度{l}。检测由图和阈值决定，因此每份生成数据列一次人数。',
        '', '| 生成随机种子 | 方法 | 正确识别目标 | 误报干净节点 | 漏检目标 | 正确识别干净节点 | Precision（%） | Recall（%） | FPR（%） |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['selected_detection_counts']:
        lines.append(f'| {r["generation_seed"]} | {r["kind"]} | {r["tp"]} | {r["fp"]} | {r["fn"]} | {r["tn"]} | {pct(r["precision"])} | {pct(r["recall"])} | {pct(r["fpr"])} |')
    lines+=['','## 7. 完整性与复现核对','',
        '- 原27个模型的五项指标均已独立重算，人数和结果一致。',
        '- 关闭新增训练项时，九个复算模型的全部权重与原方法逐项相同。',
        '- 本轮225次分类器训练全部保存结果和权重；75个开发模型的测试评价在参数配置选定后执行。',
        '- 单独保存阈值的24档检测扫描，覆盖九份原始图数据。',
        '- 所有重复使用同一张Cora图及固定节点划分。表中均值用于描述这些重复实验。',
        '', '## 8. 原始文件','',
        '- `parameter_study/metric_audit.json`：指标独立核对。',
        '- `parameter_study/threshold_sweep.json`：24档阈值的检测结果。',
        '- `parameter_study/development_test_metrics.json`：75个开发模型的测试结果。',
        '- `parameter_study/confirmation_selection.json`：验证集参数选择。',
        '- `parameter_study/validation_metrics.csv`：108次固定参数验证。',
        '- `parameter_study/asr_priority/`：18次较大惩罚强度实验。',
        '- `parameter_study/parameter_summary.json`：汇总和精确复现核对。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    audit='''# 实现与指标复核

本轮重新计算27个原模型的指标，并核对225次参数实验的结果。

## 检测指标

目标节点是实际挂接触发器的原图节点。精确率以全部识别候选为分母，召回率以全部真实目标为分母，误报率以全部干净节点为分母。主检测范围是全部带训练标签的节点。

核对程序使用完整节点集合，并要求真实目标与检测候选都处于声明的评价范围内。另一套计数实现得到相同结果。原GTA数据中，40个目标全部正确识别，541个干净节点均未误报。

## 测试指标

CA使用270个干净测试节点。ASR使用另一组测试节点中原本不属于指定类别的225个节点。核对程序重新加载模型权重并执行预测，正确数量、指定类别预测数量和两个分母均与原记录一致。

这两组测试节点都与训练节点和验证节点分开。部分训练目标同时出现在原验证集中，方法根据自己的检测结果同步筛除相应训练标签和验证标签。

## 参数选择与重复实验

参数筛选使用生成种子10的验证集结果。三次分类器重复后保存配置，再计算参数探索模型的测试结果。另用生成种子11、12、13验证，其中13的数据为本轮新生成。全部实验沿用项目固定的Cora节点划分。

关闭新增训练项时，九个复算模型的权重与原方法逐项相同。225次分类器训练均保留记录；正式结果汇总包括全部规定配置。

## 实现检查

原12项性质检查覆盖单节点、路径、树、环、零权边、节点编号变化、特征幅度变化和指标手算。新增4项检查验证训练项导数、极端输出值下的稳定性和阈值参数处理。另有1,022种小图配置及100个随机图检查。

证明和适用条件统一见 `theory_review.md`；详细统计见 `runs/abp_sparse_20261123/parameter_study/parameter_summary.json`。
'''
    (DOC/'audit.md').write_text(audit,encoding='utf-8')
    print(json.dumps({'report':str(DOC/'REPORT.md'),'results':str(OUT/'RESULTS.md'),'models_verified':result['total_new_classifier_fits']},ensure_ascii=False))

if __name__=='__main__':write_reports()
